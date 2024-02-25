import gc
from logging import config
from secrets import token_bytes
import einops
from numpy import pad
import transformers
import arena_capstone.scripts.llamatokenize as llamatokenize

from dataclasses import dataclass, field
from typing import List, Optional, Set, Union
from arena_capstone.soft_suffix.gumbel_softmax import GumbelSoftmaxConfig
import pandas as pd
import torch
import wandb
from colorama import Back, Fore, Style
from jaxtyping import Bool, Float, Int
from torch import Tensor, embedding
from tqdm import tqdm

import time
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedModel,
    LogitsProcessor,
    LogitsProcessorList,
    LlamaModel,
)
from arena_capstone.scripts.run_with_llama import get_llama

from arena_capstone.rewards.reward_generator import (
    RewardGenerator,
    get_reward_generator,
    RewardModelOutput,
)
from arena_capstone.algorithm.embedding_model import (
    EmbeddingFriendlyForCausalLM,
    EmbeddingFriendlyModel,
    MaskedChunk,
    EmbeddedBatch,
)
from arena_capstone.algorithm.token_gradients import TokenGradients
from arena_capstone.rewards.dataset_preprocess import proc_data

from arena_capstone.soft_suffix.optim import OptimCfg
from arena_capstone.soft_suffix.sched_config import SchedConfig
from arena_capstone.soft_suffix.suffix import Suffix, SuffixConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SoftOptPromptConfig(SchedConfig):
    suffix: Int[Tensor, "batch seq"]
    post_suffix: Int[Tensor, "batch seq"]
    batch_size: int
    suffix_config: "SuffixConfig"
    generate_gumbel_config: "GumbelSoftmaxConfig"
    T: int = 1000
    T_greedy: int = 100
    device: str = DEVICE
    use_wandb: bool = True
    beta1: float = 0.91
    beta2: float = 0.99
    do_print: bool = True
    generate_length: int = 6
    loss_use_end_rewards: bool = False
    k: int = 256
    rand_generate_length: int = 6
    rand_search_early_stopping: bool = False
    early_stop_min_i: int = 1


class SoftOptPrompt:
    def __init__(
        self,
        cfg: SoftOptPromptConfig,
        model: AutoModelForCausalLM,
        reward_model: RewardGenerator,
        tokenizer: AutoTokenizer,
        embedding_model: Optional[EmbeddingFriendlyModel],
    ):
        assert callable(model)
        self.cfg = cfg
        self.model: LlamaForCausalLM = model
        assert model.dtype == torch.bfloat16
        assert model.model.dtype == torch.bfloat16
        print(model.model.config)
        print(model.config)
        self.embedding_model = (
            EmbeddingFriendlyForCausalLM(self.model)
            if embedding_model is None
            else embedding_model
        )
        self.reward_model = reward_model
        self.token_gradient_generator = TokenGradients(model, self.embedding_model)
        self.tokenizer = tokenizer
        self.suffix = Suffix(
            cfg.suffix_config,
            cfg.suffix,
        )
        self.tau_zero_table = (
            wandb.Table(columns=["prefix", "suffix", "completion", "step"])
            if self.cfg.use_wandb
            else None256
        )
        self.table = (
            wandb.Table(columns=["prefix", "suffix", "completion", "step"])
            if self.cfg.use_wandb
            else None
        )

        pd = proc_data(tokenizer)
        self.pd = pd
        self.data = [next(pd) for _ in range(103)]
        self.grad_acc = None
        self.gradsquares = None
        self.cached_prefixes_seqchunk = MaskedChunk(
            seq=torch.zeros(1, 0, device=DEVICE, dtype=torch.int64),
            mask=torch.zeros(1, 0, device=DEVICE, dtype=torch.bool),
        )
        self.run_num = 1

        def dataset():
            i = 0
            while True:
                i += 1
                yield self.data[(i) % len(self.data)]

        self.dataset = dataset()

    def train(self):
        for run_num in tqdm(range(1, self.cfg.T + 1)):
            prefixes_seqchunk = self.get_next_prefixes()
            suffix = self.suffix(self.cfg.batch_size)
            prompt_seqchunk = self.embedding_model.embed_nice(
                prefixes_seqchunk, suffix, self.cfg.post_suffix
            )
            generated_probs = self.generate_fn(prompt_seqchunk)
            print("generated_probs", generated_probs.shape)
            reward_seqchunk = self.reward_model.embedding_model.embed_nice(
                prefixes_seqchunk, self.cfg.post_suffix, generated_probs
            )
            rewards = self.reward_model.embedding_model.forward_from_embed(
                reward_seqchunk
            )
            self.trainstep(rewards)

    def generate_fn(
        self,
        prompt_seqchunk: MaskedChunk,
        tau=None,
    ) -> List[Tensor]:  # not 100% on the return type
        output_probs = []

        for _ in range(self.cfg.generate_length):
            logits_next = self.embedding_model.forward_from_embed(
                prompt_seqchunk
            ).logits[:, -1:, :]
            # logits_next is shape (batch, 1, vocab_size)

            next_token_probs = self.cfg.generate_gumbel_config.gumbel_softmax(
                logits_next, tau=tau
            )
            output_probs.append(next_token_probs)

            prompt_seqchunk = self.embedding_model.embed_nice(
                prompt_seqchunk, next_token_probs
            )

        return torch.cat(output_probs, dim=1)

    def get_next_prefixes(self, batch_size=None):
        prefix_strs = [
            next(self.dataset) for _ in range(batch_size or self.cfg.batch_size)
        ]
        self.cached_prefixes_seqchunk = self.tokenize(prefix_strs, prefix=True)
        return self.cached_prefixes_seqchunk

    @property
    def optim(self):
        return self.suffix.optim

    def tokenize(self, strings: Union[List[str], str], prefix: bool) -> MaskedChunk:
        if isinstance(strings, str):
            strings = [strings]
        tokenized = self.tokenizer(strings, padding=True, return_tensors="pt")
        tokens = tokenized.input_ids.to(DEVICE)
        masks = tokenized.attention_mask.to(device=DEVICE, dtype=torch.bool)
        if not prefix:
            tokens = tokens[:, 1:]
            masks = masks[:, 1:]
        assert tokens.ndim == 2 == masks.ndim
        return MaskedChunk(seq=tokens, mask=masks)

    def generate_printable(self, tau=None):
        prefixes_seqchunk = self.cached_prefixes_seqchunk
        suffix = self.suffix(1, tau=tau)
        prompt_seqchunk = self.embedding_model.embed_nice(
            prefixes_seqchunk, suffix, self.cfg.post_suffix
        )

        with torch.inference_mode():
            completion = self.generate_fn(prompt_seqchunk, tau=0)
        completion_tokens = torch.argmax(completion, dim=-1)
        suffix_tokens = torch.argmax(suffix, dim=-1)

        # completion is shape (batch, generate_length, vocab_size)
        return (
            llamatokenize.detokenize_many(
                self.tokenizer, list(self.cached_prefixes_seqchunk.seq)
            ),
            llamatokenize.detokenize_many(self.tokenizer, list(suffix_tokens)),
            llamatokenize.detokenize_many(self.tokenizer, list(completion_tokens)),
        )

    def log(self, run_num: int):
        if run_num % 2 != 1:
            return
        if self.cfg.use_wandb:
            self.suffix.log(run_num, loghists=run_num % 50 == 1)
            prefixes, suffixes, completions = self.generate_printable()
            for prefix, suffix, completion in zip(prefixes, suffixes, completions):
                self.table.add_data(
                    prefix, suffix, completion, run_num
                )  # prefix, suffix, completion, step
                if self.cfg.do_print:
                    print(
                        f"tau {self.cfg.generate_gumbel_config.tau:.2f}",
                        prefix,
                        suffix,
                        completion,
                    )
                    time.sleep(0.1)

            prefixes, suffixes, completions = self.generate_printable(tau=1e-8)
            for prefix, suffix, completion in zip(prefixes, suffixes, completions):
                self.tau_zero_table.add_data(
                    prefix, suffix, completion, run_num
                )  # prefix, suffix, completion, step
                if self.cfg.do_print:
                    print("tau zero", prefix, suffix, completion)
                    time.sleep(0.1)

    def suffix_only_train_test(self):
        for run_num in tqdm(range(1, self.cfg.T + 1)):
            suffix = self.suffix(self.cfg.batch_size)
            reward_seqchunk = self.reward_model.embedding_model.embed_nice(
                torch.tensor(
                    self.tokenizer.bos_token_id, device=DEVICE, dtype=torch.int64
                ).unsqueeze(0),
                suffix,
            )
            rewards = self.reward_model.embedding_model.forward_from_embed(
                reward_seqchunk
            )
            self.trainstep(rewards, run_num)

    def get_loss(self, rewards: RewardModelOutput):
        if not isinstance(rewards, RewardModelOutput):
            loss = rewards.mean()
        elif self.cfg.loss_use_end_rewards:
            loss = rewards.end_rewards.mean()
        else:
            loss = rewards.rewards.mean()
        if self.cfg.use_wandb:
            wandb.log({"loss": loss.item()}, step=self.run_num)
        return loss

    def trainstep(self, rewards: RewardModelOutput, **kwargs):
        loss = self.get_loss(rewards)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        self.cfg.scheduler_step(self.run_num, loss=loss)
        self.log(run_num=self.run_num)
        self.run_num += 1

    def megatrain(self, build_reward_context, build_generate_context=None): ...
    def suffix_only_full_train(self, optim=None):  # GBRT paper setup, basically
        self.cached_prefixes_seqchunk = self.cached_prefixes_seqchunk[:, 0:0]
        for run_num in tqdm(range(1, self.cfg.T + 1)):
            suffix = self.suffix(self.cfg.batch_size)
            prompt_seqchunk = self.embedding_model.embed_nice(
                suffix,
            )

            generated_probs = self.generate_fn(prompt_seqchunk)
            reward_seqchunk = self.reward_model.embedding_model.embed_nice(
                generated_probs
            )

            rewards = self.reward_model.embedding_model.forward_from_embed(
                reward_seqchunk
            )
            self.trainstep(rewards)

    def get_rand_suffixes_vectorized(self, suffix, batch_size, d_vocab):
        suffix_len = suffix.size(0)
        # Clone the original suffix `batch_size` times
        print("suffix", suffix.shape)
        rand_suffixes = suffix.unsqueeze(0).repeat(batch_size, 1)

        # Generate random indices for each suffix in the batch
        rand_indices = torch.randint(suffix_len, size=(batch_size, 1), device=DEVICE)
        # Generate random tokens for each suffix in the batch
        rand_tokens = torch.randint(d_vocab, size=(batch_size, 1), device=DEVICE)

        # Use torch.arange to generate a batch of indices [0, 1, ..., batch_size-1] and use it along with rand_indices
        # to index into rand_suffixes and replace the tokens at the random indices with rand_tokens
        batch_indices = torch.arange(batch_size, device=DEVICE).unsqueeze(1)
        rand_suffixes[batch_indices, rand_indices] = rand_tokens

        return rand_suffixes

    def get_topk_suffixes_vectorized(self, suffix, batch_size, suffix_probs, local_k):
        suffix_len, d_vocab = suffix_probs.shape
        # Clone the original suffix `batch_size` times
        topk_suffixes = suffix.unsqueeze(0).repeat(batch_size, 1)

        # For each position in the suffix, identify the top k probabilities and their indices
        topk_values, topk_indices = torch.topk(input=suffix_probs, k=local_k, dim=1)

        # Sample uniformly from these top k indices for each position in the suffix
        # This gives us a tensor of shape (suffix_len, batch_size) after sampling
        sampled_indices = torch.stack(
            [
                indices[torch.randint(len(indices), (batch_size,))]
                for indices in topk_indices
            ],
            dim=1,
        )

        # Use torch.arange to generate a batch of indices [0, 1, ..., batch_size-1]
        # and a repeat of range for each position in the suffix to correctly assign the sampled tokens
        batch_indices = torch.arange(batch_size, device=suffix.device).unsqueeze(1)
        position_indices = torch.arange(suffix_len, device=suffix.device).repeat(
            batch_size, 1
        )

        # Replace the tokens in topk_suffixes with the sampled tokens
        topk_suffixes[batch_indices, position_indices] = sampled_indices

        return topk_suffixes

    def suffix_only_part_random_test(self):
        self.run_num = 1
        inference_batch_size = self.cfg.k
        num_cycles = 200
        for run_num in tqdm(range(100)):
            suffix = self.suffix(self.cfg.batch_size)
            reward_seqchunk = self.reward_model.embedding_model.embed_nice(
                torch.tensor(
                    self.tokenizer.bos_token_id, device=DEVICE, dtype=torch.int64
                ).unsqueeze(0),
                suffix,
            )
            rewards = self.reward_model.embedding_model.forward_from_embed(
                reward_seqchunk
            )

            self.trainstep(rewards)

        for _ in range(num_cycles):
            # random search:
            num_rand_per_cycle = 100
            one_hot_best = self.suffix_only_random_test(
                inference_batch_size, T=num_rand_per_cycle, return_one_hot=True
            )
            self.suffix.update_suffix_from_probs(one_hot_best)
            # modify suffix according to the best one-hot
            for run_num in tqdm(range(1, self.cfg.T + 1)):
                suffix = self.suffix(self.cfg.batch_size)
                reward_seqchunk = self.reward_model.embedding_model.embed_nice(
                    torch.tensor(
                        self.tokenizer.bos_token_id, device=DEVICE, dtype=torch.int64
                    ).unsqueeze(0),
                    suffix,
                )
                rewards = self.reward_model.embedding_model.forward_from_embed(
                    reward_seqchunk
                )

                self.trainstep(rewards)

    def suffix_only_random_test(
        self, inference_batch_size=None, T=None, return_one_hot=False
    ):
        with torch.inference_mode():
            if T is None:
                T = self.cfg.T
            if inference_batch_size is None:
                inference_batch_size = self.cfg.batch_size

            suffix_probs = self.suffix(1, noise_scale=0)
            vocab_size = suffix_probs.shape[-1]

            rand_suffixes = torch.multinomial(
                suffix_probs.view(-1, vocab_size),
                num_samples=inference_batch_size,
                replacement=True,
            )

            rand_suffixes = rand_suffixes.view(inference_batch_size, -1)

            for run_num in tqdm(range(1, T + 1)):
                rewards = self.reward_model(
                    input_ids=rand_suffixes,
                    attention_mask=torch.ones_like(rand_suffixes, dtype=torch.bool),
                )

                best_suffix_idx = torch.argmin(
                    self.process_rewards(rewards)
                )  # [:2] maybe? bc the dumb reward batch thing
                best_suffix = rand_suffixes[best_suffix_idx]
                self.get_loss(rewards)  # does logging
                rand_suffixes = self.get_rand_suffixes_vectorized(
                    best_suffix, inference_batch_size, vocab_size
                )
                self.run_num += 1

            if return_one_hot:
                return F.one_hot(best_suffix, vocab_size)
            else:
                return best_suffix

    def full_rand_get_initial_sampled_suffix(self, batch_size=None):
        suffix_probs = self.suffix(batch_size or self.cfg.k)
        rand_suffixes = torch.multinomial(
            suffix_probs.view(-1, suffix_probs.shape[-1]),
            num_samples=1,
            replacement=True,
        )

        return rand_suffixes.view(self.cfg.k, -1)

    def evaluate_suffix(
        self, prefixes_seqchunk: MaskedChunk, suffix: Int[Tensor, "batch seq"]
    ) -> Tensor:
        # print("prefixes_seqchunk", prefixes_seqchunk.seq.shape)
        # print("suffix", suffix.shape)
        # print("self", self.cfg.post_suffix.shape)
        prefix = prefixes_seqchunk.seq.expand(suffix.shape[0], -1)
        post_suffix = self.cfg.post_suffix.expand(suffix.shape[0], -1)
        tokens = torch.cat((prefix, suffix, post_suffix), dim=1)
        mask = torch.cat(
            (
                prefixes_seqchunk.mask.expand(suffix.shape[0], -1),
                torch.ones_like(suffix, dtype=torch.bool),
                torch.ones_like(post_suffix, dtype=torch.bool),
            ),
            dim=1,
        )
        generated = self.model.generate(
            inputs=tokens,
            attention_mask=mask,
            do_sample=True,
            temperature=1e-10,
            max_length=tokens.shape[1] + self.cfg.rand_generate_length,
        )
        print(generated.shape)

        rewards = self.reward_model(
            input_ids=generated,
            attention_mask=torch.cat(
                (
                    mask,
                    torch.ones(
                        (1, self.cfg.rand_generate_length),
                        device=DEVICE,
                        dtype=torch.bool,
                    ).expand(mask.shape[0], -1),
                ),
                dim=1,
            ),
        )
        return self.process_rewards(rewards)

    def process_rewards(self, rewards: RewardModelOutput, batch_only: bool = True):
        if self.cfg.loss_use_end_rewards:
            r = rewards.end_rewards
        else:
            r = rewards.rewards.mean(dim=-2)
        if batch_only:
            return r.mean(-1)
        return rewards.rewards

    def full_rand_test_search(self, return_one_hot):
        vocab_size = 32_001
        best_reward = float("inf")

        with torch.inference_mode():
            rand_suffixes = self.full_rand_get_initial_sampled_suffix()
            for run_num in tqdm(range(1, self.cfg.T_greedy + 1)):
                # this is extremely inefficself.cfg.k, ient but gotta test it quick
                sub_batch_size = 256
                assert self.cfg.k % sub_batch_size == 0
                prefixes = self.get_next_prefixes(1)
                best_reward = (
                    best_reward if self.cfg.rand_search_early_stopping else float("inf")
                )
                best_suffix_idx = None
                best_so_far = float("inf")
                for i in range(self.cfg.k // sub_batch_size):
                    r = self.evaluate_suffix(
                        prefixes,
                        rand_suffixes[sub_batch_size * i : sub_batch_size * (i + 1)],
                    )
                    # print("rand_suffixes", rand_suffixes.shape)
                    # print("r", r.shape)
                    rminidx = torch.argmin(r)
                    rmre = r[rminidx]
                    if rmre < best_so_far:
                        best_suffix_idx = rminidx + sub_batch_size * i
                        best_so_far = rmre
                    if (
                        self.cfg.rand_search_early_stopping
                        and i >= self.cfg.early_stop_min_i
                        and best_so_far < best_reward
                    ):
                        break
                while best_suffix_idx.size().numel() > 1:
                    print("best_suffix_idx", best_suffix_idx)
                    best_suffix_idx = best_suffix_idx[0]

                if self.cfg.rand_search_early_stopping:
                    wandb.log({"searched_i": i})
                best_reward = best_so_far
                # rewards.append(r)
                # rewards = torch.cat(rewards, dim=0)
                best_suffix = rand_suffixes[best_suffix_idx]
                self.get_loss(r)
                # self.get_loss(rewards[best_suffix_idx])
                self.run_num += 1
                if run_num == self.cfg.T_greedy:
                    if return_one_hot:
                        return F.one_hot(best_suffix, vocab_size)
                    else:
                        return best_suffix

                rand_suffixes = self.get_rand_suffixes_vectorized(
                    best_suffix, self.cfg.k, vocab_size
                )

    def full_rand_mixed_train(self):
        self.run_num = 1
        num_cycles = 200

        for _ in range(num_cycles):
            one_hot_best = self.full_rand_test_search(return_one_hot=True)
            self.suffix.update_suffix_from_probs(one_hot_best)
            self.train()

    def run(self, train_fn=None):
        train_fn = train_fn or self.train

        if self.cfg.use_wandb:

            wandb.init(project=f"soft-opt-prompt-nice-panels", config=self.cfg)
        try:
            train_fn()
        except Exception as e:
            if self.cfg.use_wandb:
                wandb.log({"table": self.table})
                wandb.finish()
            raise e


def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = True
    # num_prompts = 16
    model, embedding_model, tokenizer = get_llama()

    # harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    # harmful_behavior_data.head()
    # prefix_strs = harmful_behavior_data["goal"].tolist()[2 : 2 + num_prompts]

    # prefixes = [
    #     torch.tensor(tokens, device=DEVICE, dtype=torch.long)
    #     for tokens in tokenizer(prefix_strs).input_ids
    # ]

    reward_model: RewardGenerator = get_reward_generator()

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(DEVICE)
    post_suffix = post_suffix[1:].unsqueeze(0)
    # while True:
    #     s2e = input("next:")
    #     toks = tokenizer(
    #         s2e,
    #         return_tensors="pt",
    #     )
    #     print(
    #         "bad reward:",
    #         reward_model(
    #             input_ids=toks.input_ids.to(DEVICE),
    #             attention_mask=toks.attention_mask.to(DEVICE),
    #         ),
    #     )

    # GBRT paper hyperparams
    import math

    sgd = OptimCfg(
        optim_name="SGD",
        lr=3e1,
        betas=(0.9, 0.99),
        momentum=0.75,
        weight_decay=1e-5,
    )

    adam = OptimCfg(
        optim_name="RAdam",
        lr=0.13,
        betas=(0, 0.99),
        momentum=0.9,
        weight_decay=1e-6,
    )
    suffix_config = SuffixConfig(
        gumbel_config=GumbelSoftmaxConfig(
            tau=math.e / math.pi,
            hard=False,
            tau_backward=None,
            noise_scale=1 / 7,
            min_tau=0.1,
            max_tau=math.pi,
            tau_annealing_rate=1,
            harden_range=None,
            noise_in_hard=5,
            noise_annealing=1,
            tau_hard=18,
            scale_noise=True,
            max_scaled_noise=1,
            loss_threshold=10,
        ),
        suffix_len=5,
        optim=adam,
        update_size_from_probs=1,
        update_reset_optim=True,
    )

    generate_gumbel_config = GumbelSoftmaxConfig(
        tau=math.e / math.pi ** (1 + 1 / math.e),
        hard=False,
        tau_backward=None,
        noise_scale=math.e / math.pi**math.e,
        min_tau=0.04,
        tau_annealing_rate=0.9999,
        loss_threshold=10,
    )

    test_str = "concerns conflicts formally隆 Tell I I fightingages:@ yes"

    tokenized = tokenizer(test_str, return_tensors="pt")
    test_tokens = tokenized.input_ids.to(DEVICE)
    test_mask = tokenized.attention_mask.to(DEVICE, dtype=torch.bool)
    test_reward = reward_model(input_ids=test_tokens, attention_mask=test_mask)

    print("rewards:", test_reward.end_rewards.mean(), test_reward.rewards.mean())

    cfg = SoftOptPromptConfig(
        suffix=None,  # torch.randint(5, 1000, (6,), device=DEVICE),
        post_suffix=post_suffix,
        batch_size=4,
        suffix_config=suffix_config,
        generate_gumbel_config=generate_gumbel_config,
        T=250,
        T_greedy=100,
        use_wandb=True,
        generate_length=6,
        rand_generate_length=4,
        loss_use_end_rewards=True,
        k=2048,
        rand_search_early_stopping=True,
        early_stop_min_i=1,
    )

    upo = SoftOptPrompt(
        cfg=cfg,
        model=model,
        reward_model=reward_model,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
    )

    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    upo.run(upo.full_rand_mixed_train)


if __name__ == "__main__":
    main()
