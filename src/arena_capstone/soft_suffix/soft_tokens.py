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
    device: str = DEVICE
    use_wandb: bool = True
    beta1: float = 0.91
    beta2: float = 0.99
    do_print: bool = True
    generate_length: int = 6
    loss_use_end_rewards: bool = True
    k: int = 256


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
            else None
        )
        self.table = (
            wandb.Table(columns=["prefix", "suffix", "completion", "step"])
            if self.cfg.use_wandb
            else None
        )

        pd = proc_data(tokenizer)
        self.pd = pd
        self.data = [next(pd) for _ in range(100)]
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
                yield self.data[i % len(self.data)]

        self.dataset = dataset()

    def train(self):
        for run_num in tqdm(range(1, self.cfg.T + 1)):
            prefixes_seqchunk = self.get_next_prefixes()
            suffix = self.suffix(self.cfg.batch_size)
            prompt_seqchunk = self.embedding_model.embed_nice(
                prefixes_seqchunk, suffix, self.cfg.post_suffix
            )
            generated_probs = self.generate_fn(prompt_seqchunk)
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

    def get_next_prefixes(self):
        prefix_strs = [next(self.dataset) for _ in range(self.cfg.batch_size)]
        self.cached_prefixes_seqchunk = self.tokenize(prefix_strs, prefix=True)
        return self.cached_prefixes_seqchunk

    @property
    def optim(self):
        return self.suffix.optim

    def run(self, train_fn=None):
        train_fn = train_fn or self.train

        if self.cfg.use_wandb:
            wandb.init(project=f"soft-opt-prompt-{train_fn.__name__}", config=self.cfg)

        try:
            train_fn()
        except Exception as e:
            if self.cfg.use_wandb:
                wandb.log({"table": self.table})
                wandb.finish()
            raise e

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
        if run_num % 5 != 1:
            return
        if self.cfg.use_wandb:
            self.suffix.log(run_num, loghists=run_num % 50 == 0)
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

    def trainstep(self, rewards: RewardModelOutput, **kwargs):
        if self.cfg.loss_use_end_rewards:
            loss = rewards.end_rewards.mean()
        else:
            loss = rewards.rewards.mean()
        loss.backward()
        wandb.log({"loss": loss.item()}, step=self.run_num)
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
        for _ in range(num_cycles):
            # random search:
            num_rand_per_cycle = 50
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

            suffix_probs = self.suffix(inference_batch_size)
            vocab_size = suffix_probs.shape[-1]

            rand_suffixes = torch.multinomial(
                suffix_probs.view(-1, vocab_size), num_samples=1, replacement=True
            )

            rand_suffixes = rand_suffixes.view(inference_batch_size, -1)

            for run_num in tqdm(range(1, T + 1)):
                rewards = self.reward_model(
                    input_ids=rand_suffixes,
                    attention_mask=torch.ones_like(rand_suffixes, dtype=torch.bool),
                )

                best_suffix_idx = torch.argmin(
                    rewards.end_rewards
                )  # [:2] maybe? bc the dumb reward batch thing
                best_suffix = rand_suffixes[best_suffix_idx]

                if self.cfg.use_wandb:
                    if self.cfg.loss_use_end_rewards:
                        loss = rewards.end_rewards.mean()
                    else:
                        loss = rewards.rewards.mean()
                    wandb.log({"loss": loss.item()}, step=self.run_num)

                rand_suffixes = self.get_rand_suffixes_vectorized(
                    best_suffix, inference_batch_size, vocab_size
                )

                # maybe log idk
                # if run_num % 5 == 0:
                # one_hot_best = F.one_hot(best_suffix, vocab_size)
                # self.suffix.update_suffix_from_probs(one_hot_best)
                # self.log(run_num=self.run_num)

                self.run_num += 1

            if return_one_hot:
                return F.one_hot(best_suffix, vocab_size)
            else:
                return best_suffix


def main():
    torch.set_default_dtype(torch.bfloat16)

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
        lr=3e-1,
        betas=(0.9, 0.99),
        momentum=0.9,
        weight_decay=0,
    )

    adam = OptimCfg(
        optim_name="RAdam",
        lr=3e-1,
        betas=(0.9, 0.99),
        momentum=0.9,
        weight_decay=3e-6,
    )
    suffix_config = SuffixConfig(
        gumbel_config=GumbelSoftmaxConfig(
            tau=12,
            hard=False,
            tau_backward=None,
            noise_scale=1 / 7,
            min_tau=0.001,
            max_tau=10,
            tau_annealing_rate=0.995,
            harden_range=None,
            noise_in_hard=5,
            noise_annealing=0.99,
            tau_hard=18,
            scale_noise=True,
            max_scaled_noise=1,
            loss_threshold=-2,
        ),
        suffix_len=7,
        optim=adam,
        update_size_from_probs=10,
        update_reset_optim=False,
    )

    # suffix_config = SuffixConfig(
    #     gumbel_config=GumbelSoftmaxConfig(
    #         tau=4,
    #         hard=False,
    #         tau_backward=None,
    #         noise_scale=7,
    #         min_tau=0.001,
    #         tau_annealing_rate=1,
    #         harden_range=(200, 400),
    #         noise_in_hard=None,
    #         noise_annealing=0.999,
    #         tau_hard=math.e**math.pi,
    #         scale_noise=False,
    #         max_scaled_noise=10,
    #     ),
    #     suffix_len=7,
    #     optim=OptimCfg(
    #         optim_name="RAdam",
    #         lr=3e-1,
    #         betas=(0.9, 0.99),
    #         momentum=0.9,
    #         weight_decay=0,
    #     ),
    # )

    generate_gumbel_config = GumbelSoftmaxConfig(
        tau=100,
        hard=False,
        tau_backward=None,
        noise_scale=1,
        min_tau=0.01,
        tau_annealing_rate=0.995,
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
        batch_size=64,
        k=256,
        suffix_config=suffix_config,
        generate_gumbel_config=generate_gumbel_config,
        T=50,
        use_wandb=True,
        generate_length=6,
        beta1=0.5,
        beta2=0.99,
    )

    upo = SoftOptPrompt(
        cfg=cfg,
        model=model,
        reward_model=reward_model,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
    )

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        upo.run(upo.suffix_only_part_random_test)


if __name__ == "__main__":
    main()
