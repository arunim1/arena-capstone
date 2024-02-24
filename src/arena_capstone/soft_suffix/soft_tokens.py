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
    optim: "OptimCfg"
    T: int = 1000
    device: str = DEVICE
    use_wandb: bool = True
    beta1: float = 0.91
    beta2: float = 0.99
    do_print: bool = True
    generate_length: int = 6


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
        self.cached_prefixes_seqchunk = None

        def dataset():
            i = 0
            while True:
                i += 1
                yield self.data[i % len(self.data)]

        self.dataset = dataset()

    def train(self, optim=None):

        # optim = optim or torch.optim.RAdam(
        #     self.suffix.parameters(), lr=3e-1, betas=(self.cfg.beta1, self.cfg.beta2)
        # )
        optim = optim or torch.optim.SGD(self.suffix.parameters(), lr=3e-1, momentum=0)
        for run_num in tqdm(range(1, self.cfg.T + 1)):
            prefixes_seqchunk = self.get_next_prefixes()
            suffix = self.suffix()
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

            loss = rewards.end_rewards.mean()  # possibly change to rewards.rewards
            loss.backward()
            # # printing gradient:
            # print("suffix grad", self.suffix.suffix_logits.grad)
            wandb.log({"loss": loss.item()}, step=run_num)
            optim.step()
            optim.zero_grad()
            self.cfg.scheduler_step(run_num)
            if run_num % 2 == 0:
                self.log(run_num=run_num)

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
        if self.cfg.use_wandb:
            self.suffix.log_historgram(run_num)
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

        optim = torch.optim.RAdam(
            self.suffix.parameters(),
            lr=3e0,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=0.001,
        )
        # optim = torch.optim.SGD(self.suffix.parameters(), lr=1e2, momentum=0.0)

        self.get_next_prefixes()
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

            # loss = rewards.end_rewards.mean()  # possibly change to rewards.rewards
            loss = rewards.rewards.mean()
            loss.backward()
            # printing gradient:
            # print("suffix grad", self.suffix.suffix_logits.grad)
            wandb.log({"loss": loss.item()}, step=run_num)
            optim.step()
            optim.zero_grad()
            self.cfg.scheduler_step(run_num, loss=loss)
            if run_num % 50 == 10:
                self.log(run_num=run_num)

    def suffix_only_full_train(self, optim=None):  # GBRT paper setup, basically

        optim = optim or torch.optim.RAdam(
            self.suffix.parameters(),
            lr=3e-2,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=0.1,
        )
        self.get_next_prefixes()
        self.cached_prefixes_seqchunk = self.cached_prefixes_seqchunk[:, 0:0]
        for run_num in tqdm(range(1, self.cfg.T + 1)):
            suffix = self.suffix()
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

            loss = rewards.end_rewards.mean()  # possibly change to rewards.rewards
            # loss = rewards.reward.mean()
            loss.backward()
            self.cfg.scheduler_step(run_num)

            # printing gradient:
            print("suffix grad", self.suffix.suffix_logits.grad)
            wandb.log({"loss": loss.item()}, step=run_num)
            optim.step()
            optim.zero_grad()
            if run_num % 25 == 0:
                self.log(run_num=run_num)


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

    # suffix_config = SuffixConfig(
    #     gumbel_config=GumbelSoftmaxConfig(
    #         tau=2,
    #         hard=False,
    #         tau_backward=None,
    #         noise_scale=1 / 7,
    #         min_tau=0.001,
    #         tau_annealing_rate=0.995,
    #         harden_range=(10, 10),
    #         noise_in_hard=5,
    #         noise_annealing=0.99,
    #         tau_hard=math.e**math.pi,
    #         scale_noise=True,
    #         max_scaled_noise=1,
    #     ),
    #     suffix_len=7,
    # )

    suffix_config = SuffixConfig(
        gumbel_config=GumbelSoftmaxConfig(
            tau=4,
            hard=False,
            tau_backward=None,
            noise_scale=7,
            min_tau=0.001,
            tau_annealing_rate=1,
            harden_range=(200, 400),
            noise_in_hard=None,
            noise_annealing=0.999,
            tau_hard=math.e**math.pi,
            scale_noise=False,
            max_scaled_noise=10,
        ),
        suffix_len=7,
    )

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
        suffix_config=suffix_config,
        generate_gumbel_config=generate_gumbel_config,
        T=4000,
        use_wandb=True,
        generate_length=6,
        beta1=0.5,
        beta2=0.99,
        optim=None,
    )

    upo = SoftOptPrompt(
        cfg=cfg,
        model=model,
        reward_model=reward_model,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
    )

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        upo.run(upo.suffix_only_train_test)


if __name__ == "__main__":
    main()
