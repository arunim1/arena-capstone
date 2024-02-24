import gc
from logging import config
from secrets import token_bytes
import einops
from numpy import pad
import transformers
import arena_capstone.scripts.llamatokenize as llamatokenize

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Union
from arena_capstone.rewards.gumbel_softmax import gumbel_softmax
import pandas as pd
import torch
import wandb
from colorama import Back, Fore, Style
from jaxtyping import Bool, Float, Int
from torch import Tensor, embedding
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

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
import arena_capstone.algorithm.topk_gradients as topkgrad
from arena_capstone.algorithm.gcg import GCGConfig
from arena_capstone.algorithm.token_gradients import TokenGradients
from arena_capstone.rewards.dataset_preprocess import proc_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def isinfnan(t):
    return torch.any(torch.isnan(t) | torch.isinf(t))


class Config:
    pass

    def update_config_and_log(self, cfgname, d: dict, run_num: int):
        for k, v in d.items():
            setattr(self, k, v)
        wandb.log({f"{cfgname}/{k}": v for k, v in d.items()}, step=run_num)

    def scheduler_step(self, run_num, name="cfg", **kwargs):
        for attrname in dir(self):
            if attrname.startswith("_"):
                continue
            attr = getattr(self, attrname)
            if hasattr(attr, "schedule"):
                attr.update_config_and_log(
                    f"{name}.{attrname}", attr.schedule(run_num, **kwargs), run_num
                )
            if isinstance(attr, Config):
                attr.scheduler_step(run_num, f"{name}.{attrname}", **kwargs)


# def update_config_and_log(cfg, cfgname, d: dict):
#     for k, v in d.items():
#         setattr(cfg, k, v)
#     wandb.log({f"{cfgname}.{k}": v for k, v in d.items()})


# def schedule(cfg, name=""):
#     for attrname in cfg.__annotations__.keys():
#         attr = getattr(cfg, attrname)
#         if hasattr(attr, "schedule"):
#             update_config_and_log(attr, f"{name}.{attr}", attr.schedule())
#         if isinstance(attr, Config):
#             schedule(attr, f"{name}.{attr}")


@dataclass
class SoftOptPromptConfig(Config):
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


@dataclass
class OptimCfg(Config):
    optim: str = "RAdam"
    lr: float = 3e-1
    betas: Tuple[float, float] = (0.9, 0.99)
    momentum: float = 0.9
    weight_decay: float = 0
    eps: float = 1e-8

    def get(self, params):
        if self.optim == "RAdam":
            return torch.optim.RAdam(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                eps=self.eps,
            )
        if self.optim == "SGD":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optim}")


@dataclass
class GumbelSoftmaxConfig(Config):
    tau: float = 1
    hard: bool = False
    tau_backward: float = None
    noise_scale: float = 1
    bad_words_ids: Optional[Tuple[int]] = (1, 2, 32_000)  # TODO check this
    min_tau: Optional[float] = 0.01
    tau_annealing_rate: Optional[float] = 0.95
    harden_range: Optional[Tuple[int, int]] = None
    noise_in_hard: float = 0
    tau_hard: float = None
    temp_tau_soft: float = None
    temp_noise: float = None
    scale_noise: bool = False
    max_scaled_noise: float = 1
    max_tau: float = 20
    noise_annealing: float = 0.99

    def gumbel_softmax(self, logits, tau=None, noise_scale=None, hard=None):
        if self.bad_words_ids is not None:
            logit_mask = torch.zeros(logits.shape[-1], device=logits.device)
            logit_mask[torch.tensor(self.bad_words_ids, dtype=torch.int64)] = torch.inf
            logits = logits - logit_mask
        return gumbel_softmax(
            logits,
            tau=tau or self.tau,
            hard=hard if hard is not None else self.hard,
            tau_backward=self.tau_backward,
            noise_scale=noise_scale
            or (
                self.noise_scale
                if not self.scale_noise
                else min(self.noise_scale * self.tau, self.max_scaled_noise)
            ),
            dim=-1,
        )

    def __post_init__(self):
        self.temp_tau_soft = self.temp_tau_soft or self.tau
        self.temp_noise = self.noise_scale

    def schedule(self, run_num: int, **kwargs) -> dict:
        d = {}
        d["hard"] = self.harden_range is None or (
            self.harden_range[1] - self.harden_range[0]
        ) <= (run_num % self.harden_range[1])

        d["noise_scale"] = self.noise_scale * self.noise_annealing
        if d["hard"]:
            if self.noise_in_hard is not None:
                d["noise_scale"] = self.noise_in_hard
            d["tau"] = self.tau_hard or self.tau

        else:
            if self.noise_in_hard is not None:
                d["noise_scale"] = self.temp_noise
            loss = kwargs["loss"]
            if loss < -4.5:
                d["temp_tau_soft"] = min(
                    self.max_tau,
                    max(self.min_tau, self.temp_tau_soft * self.tau_annealing_rate),
                )
            else:
                d["temp_tau_soft"] = min(
                    self.max_tau,
                    max(self.min_tau, self.temp_tau_soft / (self.tau_annealing_rate)),
                )
            d["tau"] = d["temp_tau_soft"]

        # if self.tau < 4:
        #     d["tau_backward"] = 2 + self.tau / 2
        return d


@dataclass
class SuffixConfig(Config):
    gumbel_config: GumbelSoftmaxConfig
    suffix_len: int = 5


DEBUG = False


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class Suffix(nn.Module):
    def __init__(
        self,
        cfg: SuffixConfig,
        suffix_logits=None,
    ):
        super().__init__()
        if suffix_logits is None:
            suffix_logits = torch.zeros(1, cfg.suffix_len, 32001, device=DEVICE)
        else:
            if suffix_logits.ndim == 2:
                suffix_logits = suffix_logits.unsqueeze(0)
        self.suffix_logits = nn.Parameter(suffix_logits.clone())
        self.tau = cfg.gumbel_config.tau
        self.tau_backward = cfg.gumbel_config.tau_backward
        self.hard = cfg.gumbel_config.hard
        self.noise_scale = cfg.gumbel_config.noise_scale
        self.cfg = cfg

    def forward(self, batch_size, tau=None) -> Tensor:
        return self.cfg.gumbel_config.gumbel_softmax(
            self.suffix_logits.expand(batch_size, -1, -1), tau=tau
        )

    def log_historgram(self, run_num: int):
        suffix = self(1)
        suffix_softmax = F.softmax(self.suffix_logits, dim=-1)
        suffix_softmax = self.cfg.gumbel_config.gumbel_softmax(
            self.suffix_logits, noise_scale=0, hard=False
        )

        for i in range(suffix.shape[1]):
            max_probs_g = torch.max(suffix[:, i, :], dim=-1)
            max_probs = torch.max(suffix_softmax[:, i, :], dim=-1)
            mean_max_g = max_probs_g.values.mean()
            mean_max_sm = max_probs.values.mean()
            std_max_g = max_probs_g.values.std()
            wandb.log(
                {
                    f"suffix/probs/hists/gumbel/{i}": wandb.Histogram(
                        suffix[:, i, :].float().detach().cpu().numpy()
                    ),
                    f"suffix/probs/hists/softmax/{i}": wandb.Histogram(
                        suffix_softmax[:, i, :].float().detach().cpu().numpy()
                    ),
                    f"suffix/probs/max/means/gumbel/{i}": mean_max_g,
                    f"suffix/probs/max/means/softmax/{i}": mean_max_sm,
                    # f"suffix/maxprob/sts/gumbel/{i}": std_max_g,
                },
                step=run_num,
            )


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
            weight_decay=0.1,
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
            harden_range=(200, 250),
            noise_in_hard=None,
            noise_annealing=0.999,
            tau_hard=math.e**math.pi**2,
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
        beta1=0.9,
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
