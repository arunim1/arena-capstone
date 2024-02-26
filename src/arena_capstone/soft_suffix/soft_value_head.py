import gc
from logging import config
from secrets import token_bytes
import einops
from numpy import pad
import transformers
import arena_capstone.scripts.llamatokenize as llamatokenize
from nqgl.mlutils.time_gpu import profilefunc_wrapper, time_methods
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
from arena_capstone.algorithm.topk_gradients import (
    top_k_substitutions,
    sample_replacements,
)

import time
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    PreTrainedModel,
    LogitsProcessor,
    LogitsProcessorList,
    LlamaModel,
)
from arena_capstone.scripts.run_with_llama import get_value_head

from arena_capstone.rewards.reward_generator import (
    RewardGenerator,
    get_reward_generator,
    RewardModelOutput,
)
from arena_capstone.algorithm.embedding_model import (
    EmbeddingFriendlyForCausalLM,
    EmbeddingFriendlyValueHeadForCausalLM,
    EmbeddingFriendlyModel,
    MaskedChunk,
    EmbeddedBatch,
)
from arena_capstone.algorithm.token_gradients import TokenGradients
from arena_capstone.rewards.dataset_preprocess import proc_data

from arena_capstone.soft_suffix.optim import OptimCfg
from arena_capstone.soft_suffix.sched_config import SchedConfig
from arena_capstone.soft_suffix.search_vecd import (
    get_rand_suffixes_vectorized,
)
from arena_capstone.soft_suffix.suffix import Suffix, SuffixConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RO:
    rewards: float
    end_rewards: float
    # attention_mask: Tensor


@dataclass
class SoftOptPromptConfig(SchedConfig):
    suffix: Int[Tensor, "batch seq"]
    post_suffix: Int[Tensor, "batch seq"]
    batch_size: int
    suffix_config: "SuffixConfig"
    generate_gumbel_config: "GumbelSoftmaxConfig"
    num_prompts: int = 16
    T: int = 1000
    T_greedy: int = 100
    device: str = DEVICE
    use_wandb: bool = True
    beta1: float = 0.91
    beta2: float = 0.99
    do_print: bool = True
    generate_length: int = 6
    loss_use_end_rewards: bool = False
    k: int = 128
    search_batch_size: int = 256
    rand_generate_length: int = 6
    rand_search_early_stopping: bool = False
    early_stop_min_i: int = 1
    search_mode_topk: bool = False
    search_average_over_batches: int = 4
    search_sub_batch_size: int = 256


# @time_methods
class VHSoftOptPrompt:
    def __init__(
        self,
        cfg: SoftOptPromptConfig,
        # model: AutoModelForCausalLM,
        embedding_model: EmbeddingFriendlyValueHeadForCausalLM,
        tokenizer: LlamaTokenizer,
    ):
        model = embedding_model.model
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
        self.token_gradient_generator = TokenGradients(model, self.embedding_model)
        self.tokenizer = tokenizer
        self.suffix: Suffix = Suffix(
            cfg.suffix_config,
            suffix_logits=cfg.suffix,
            tokenizer=tokenizer,
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
        self.data = [next(pd) for _ in range(cfg.num_prompts)]
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
            generated_probs, rewards = self.generate_fn(prompt_seqchunk)
            self.trainstep(rewards)

    def generate_fn(
        self,
        prompt_seqchunk: MaskedChunk,
        tau=None,
    ) -> List[Tensor]:  # not 100% on the return type
        output_probs = []

        def soft_sample(logits):
            probs = self.cfg.generate_gumbel_config.gumbel_softmax(logits, tau=tau)
            output_probs.append(probs)
            return self.embedding_model.embed_nice(probs)

        n = self.embedding_model.forward_from_embed(prompt_seqchunk, use_cache=True)
        rewards = []
        end_rewards = []

        for _ in range(self.cfg.generate_length):
            embed_next = soft_sample(n.logits[..., -1:, :])
            n = self.embedding_model.forward_from_embed(
                embed_next.seq,
                attention_mask=embed_next.mask,
                use_cache=True,
                past_key_values=n.past_key_values,
            )
            rewards.append(n.rewards)
            end_rewards.append(n.end_rewards)

        ro = RO(
            rewards=torch.stack(rewards, dim=1),
            end_rewards=torch.stack(end_rewards, dim=1),
        )
        setattr(ro, "rewards", torch.stack(rewards, dim=1))
        setattr(ro, "end_rewards", torch.stack(end_rewards, dim=1))

        return torch.cat(output_probs, dim=1), ro

    def generate_fn_old(
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

    def log_best_suffix(self):
        f = open("best_suffixex.txt", "a")
        s = f"""
        ---{wandb.run.name}:{self.run_num}: {loss}
                {llamatokenize.detokenize_many(
                self.tokenizer, list(self.best_search_suffix))}
        &&&
        {self.best_search_suffix}
        \n\n\n
        """
        f.write(s)

    def log(self, run_num: int):
        if run_num % 11 == 0:
            self.log_best_suffix()
        if run_num % 5 != 1:
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
                        "<<<post-suffix>>>",
                        completion,
                    )
                    time.sleep(0.1)

            prefixes, suffixes, completions = self.generate_printable(tau=1e-8)
            for prefix, suffix, completion in zip(prefixes, suffixes, completions):
                self.tau_zero_table.add_data(
                    prefix, suffix, completion, run_num
                )  # prefix, suffix, completion, step
                if self.cfg.do_print:
                    print("tau zero", prefix, suffix, "<<<post-suffix>>>", completion)
                    time.sleep(0.1)

    def generate_printable(self, tau=None):
        prefixes_seqchunk = self.cached_prefixes_seqchunk
        suffix = self.suffix(1, tau=tau)
        prompt_seqchunk = self.embedding_model.embed_nice(
            prefixes_seqchunk, suffix, self.cfg.post_suffix
        )

        with torch.inference_mode():
            completion, rewards = self.generate_fn(prompt_seqchunk, tau=0)
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

    def get_loss(self, rewards: RewardModelOutput, process=True):
        reward = self.process_rewards(rewards) if process else rewards
        loss = reward.mean()
        if self.cfg.use_wandb:
            wandb.log({"loss": loss.item()}, step=self.run_num)
        return loss

    def process_rewards(self, rewards: RewardModelOutput, batch_only: bool = True):
        end_rewards = []

        if isinstance(rewards, Tensor):
            print("WARNING: ASSUMING REWARD PASSED WAS CORRECT IN process_rewards")
            r = rewards.mean()
        elif self.cfg.loss_use_end_rewards is True:
            r = rewards.end_rewards
        elif isinstance(self.cfg.loss_use_end_rewards, int):
            if isinstance(rewards, RO):
                r = rewards.rewards[:, -self.cfg.loss_use_end_rewards :].mean(dim=-1)
            else:
                selected_rewards = []
                for i in range(rewards.attention_mask.shape[0]):
                    indicies = rewards.attention_mask[i].nonzero()[
                        -self.cfg.loss_use_end_rewards :
                    ]
                    selected_rewards.append(rewards.rewards[i, indicies].mean())
                r = torch.stack(selected_rewards, dim=0).unsqueeze(-1)  # size = (B, D)

        else:
            print("suggested to select an integer number of rewards instead!")
            r = rewards.rewards.mean(dim=-2)

        if batch_only:
            return r.mean(-1)
        return r

    def trainstep(self, rewards: RewardModelOutput, **kwargs):
        loss = self.get_loss(rewards)
        loss.backward()
        self.suffix.pre_step(self.run_num)
        self.optim.step()
        self.optim.zero_grad()
        self.cfg.scheduler_step(self.run_num, loss=loss)
        # self.log(run_num=self.run_num)
        self.run_num += 1

    def fullrand_get_suffix(self, suffix, batch_size, d_vocab):
        is_topk = (
            self.run_num % 2 == 0
            if self.cfg.search_mode_topk != "both"
            else self.cfg.search_mode_topk
        )
        if is_topk:
            k = (
                self.cfg.k
                if isinstance(self.cfg.k, int)
                else self.cfg.k[self.run_num % len(self.cfg.k)]
            )
            replacements = top_k_substitutions(
                -1 * self.suffix.suffix_logits.squeeze(0), k=k
            )
            return sample_replacements(replacements, suffix, batch_size)

            # return get_topk_suffixes_vectorized(suffix, batch_size, self.suffix(1), 128)
        else:
            return get_rand_suffixes_vectorized(suffix, batch_size, d_vocab)

    def fullrand_get_initial_suffix_sample(self, batch_size=None):
        suffix_probs = self.suffix(batch_size or self.cfg.search_batch_size)
        rand_suffixes = torch.multinomial(
            suffix_probs.view(-1, suffix_probs.shape[-1]),
            num_samples=1,
            replacement=True,
        )

        return rand_suffixes.view(self.cfg.search_batch_size, -1)

    def evaluate_suffix(  # TODO
        self,
        suffix: Int[Tensor, "batch seq"],
        prefixes: Optional[List[MaskedChunk]] = None,
    ) -> Tensor:
        if prefixes is None:
            prefixes = [
                self.get_next_prefixes(1)
                for _ in range(self.cfg.search_average_over_batches)
            ]
        reward_summed = 0

        for prefixes_seqchunk in prefixes:
            prefix_output = self.model(
                input_ids=prefixes_seqchunk.seq,
                attention_mask=prefixes_seqchunk.mask,
                use_cache=True,
            )
            past_key_values = [
                [kv.expand(suffix.shape[0], *kv.shape[1:]) for kv in layer]
                for layer in prefix_output.past_key_values
            ]

            post_suffix = self.cfg.post_suffix.expand(suffix.shape[0], -1)
            tokens = torch.cat(
                (
                    prefixes_seqchunk.seq.expand(suffix.shape[0], -1),
                    suffix,
                    post_suffix,
                ),
                dim=1,
            )
            mask = torch.ones_like(tokens, dtype=torch.bool)
            mask[:, : prefixes_seqchunk.mask.shape[1]] = prefixes_seqchunk.mask
            generated = self.model.generate(
                tokens,
                attention_mask=mask,
                past_key_values=past_key_values,
                max_new_tokens=self.cfg.rand_generate_length,
                output_hidden_states=True,
                do_sample=False,
            )
            mask = torch.ones_like(generated, dtype=torch.bool)
            mask[:, : prefixes_seqchunk.mask.shape[1]] = prefixes_seqchunk.mask
            mask[:, -self.cfg.rand_generate_length :] = torch.where(
                generated[:, -self.cfg.rand_generate_length :]
                == self.tokenizer.pad_token_id,
                0,
                1,
            )

            # generated = torch.cat(generated, dim=1)
            print("generated", generated.shape, tokens.shape)
            # rewards = )!)(input_ids=generated, attention_mask=mask)
            reward_summed = reward_summed + self.process_rewards(rewards)
        return reward_summed / len(prefixes)

    @torch.inference_mode()
    def full_rand_test_search(self, return_one_hot):
        vocab_size = 32_001
        best_reward = float("inf")
        average_over_batches = self.cfg.search_average_over_batches
        # with :
        rand_suffixes = self.fullrand_get_initial_suffix_sample()
        for run_num in tqdm(range(1, self.cfg.T_greedy + 1)):
            # this is extremely inefficient but gotta test it quick
            assert self.cfg.search_batch_size % self.cfg.search_sub_batch_size == 0
            best_reward = (
                best_reward if self.cfg.rand_search_early_stopping else float("inf")
            )
            best_suffix_idx = None
            best_so_far = float("inf")
            prefixes = [self.get_next_prefixes(1) for _ in range(average_over_batches)]
            for i in range(
                self.cfg.search_batch_size // self.cfg.search_sub_batch_size
            ):
                r = self.evaluate_suffix(
                    rand_suffixes[
                        self.cfg.search_sub_batch_size
                        * i : self.cfg.search_sub_batch_size
                        * (i + 1)
                    ],
                    prefixes=prefixes,
                )
                # print("rand_suffixes", rand_suffixes.shape)
                # print("r", r.shape)
                rminidx = torch.argmin(r)
                rmre = r[rminidx]
                if rmre < best_so_far:
                    best_suffix_idx = rminidx + self.cfg.search_sub_batch_size * i
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
            if i % 10 > 5:
                self.get_loss(r, process=False)
            else:
                self.get_loss(  # this logs it
                    self.evaluate_suffix(suffix=best_suffix.unsqueeze(0)),
                    process=False,
                )

            # self.get_loss(rewards[best_suffix_idx])
            self.run_num += 1
            if run_num == self.cfg.T_greedy:
                if return_one_hot:
                    return F.one_hot(best_suffix, vocab_size)
                else:
                    return best_suffix

            rand_suffixes = self.fullrand_get_suffix(
                best_suffix, self.cfg.search_batch_size, vocab_size
            )

    def full_rand_mixed_train(self):
        self.run_num = 1
        num_cycles = 200

        for _ in range(num_cycles):
            self.train()
            one_hot_best = self.full_rand_test_search(return_one_hot=True)
            self.suffix.update_suffix_from_probs(one_hot_best)

    def to_profile_train(self):
        self.cfg.T = 2
        self.cfg.T_greedy = 1
        self.cfg.search_batch_size = 512
        self.run_num = 1
        num_cycles = 2

        for _ in range(num_cycles):
            self.train()
            one_hot_best = self.full_rand_test_search(return_one_hot=True)
            self.suffix.update_suffix_from_probs(one_hot_best)

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
    model, embedding_model, tokenizer = get_value_head()

    # reward_model: RewardGenerator = get_reward_generator()

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(DEVICE)
    post_suffix = post_suffix[1:].unsqueeze(0)
    import math

    sgd = OptimCfg(
        optim_name="SGD",
        lr=3e1,
        betas=(0.9, 0.99),
        momentum=0.7,
        nesterov=True,
        weight_decay=1e-6,
    )

    adam = OptimCfg(
        optim_name="RAdam",
        lr=3,
        betas=(0.9, 0.99),
        momentum=0.9,
        weight_decay=0,
    )

    rprop = OptimCfg(
        optim_name="RProp",
    )

    suffix_config = SuffixConfig(
        gumbel_config=GumbelSoftmaxConfig(
            tau=math.e,
            hard=False,
            tau_backward=None,
            noise_scale=1 / 7,
            min_tau=0.1,
            max_tau=math.e,
            tau_annealing_rate=0.995,
            harden_range=None,
            noise_in_hard=5,
            noise_annealing=1,
            tau_hard=18,
            scale_noise=True,
            max_scaled_noise=1,
            loss_threshold=10,
            sine_tau=(50, 0.4, 0.65),
        ),
        suffix_len=15,
        optim=sgd,
        update_size_from_probs=math.e,
        update_reset_optim=True,
        update_const_scale_logits=0.8,
        ceil_scale=50,
        l1_coeff=0,
    )

    generate_gumbel_config = GumbelSoftmaxConfig(
        tau=math.e / math.pi ** (1 + 1 / math.e),
        hard=False,
        tau_backward=None,
        noise_scale=1,
        min_tau=0.04,
        tau_annealing_rate=0.9999,
        loss_threshold=10,
        sine_tau=(33, 0.25, 0.8),
    )

    test_str = "concerns conflicts formally隆 Tell I I fightingages:@ yes"

    tokenized = tokenizer(test_str, return_tensors="pt")
    test_tokens = tokenized.input_ids.to(DEVICE)
    test_mask = tokenized.attention_mask.to(DEVICE, dtype=torch.bool)
    # test_reward = reward_model(input_ids=test_tokens, attention_mask=test_mask)

    # print("rewards:", test_reward.end_rewards.mean(), test_reward.rewards.mean())

    cfg = SoftOptPromptConfig(
        num_prompts=1600,
        suffix=None,  # torch.randint(5, 1000, (6,), device=DEVICE),
        post_suffix=post_suffix,
        batch_size=32,
        suffix_config=suffix_config,
        generate_gumbel_config=generate_gumbel_config,
        generate_length=8,
        T=1500,
        T_greedy=10,
        rand_generate_length=8,
        search_batch_size=2048,
        search_average_over_batches=4,
        rand_search_early_stopping=False,
        search_mode_topk="both",
        k=256,  # [32] * 4 + [128] * 4 + [512] * 4 + [2048] * 2 + [4096] * 2,
        early_stop_min_i=1,
        use_wandb=True,
        loss_use_end_rewards=3,
    )

    upo = VHSoftOptPrompt(
        cfg=cfg,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
    )

    # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    upo.run(upo.train)

    # upo.run(profilefunc_wrapper()(lambda: upo.to_profile_train()))


if __name__ == "__main__":
    main()
