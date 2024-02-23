import gc
from logging import config
from secrets import token_bytes
import einops
from numpy import pad
import transformers
import arena_capstone.scripts.llamatokenize as llamatokenize

from dataclasses import dataclass
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


@dataclass
class SoftOptPromptConfig:
    prefixes: List[Int[Tensor, "prefix_lens"]]
    suffix: Int[Tensor, "batch seq"]
    post_suffix: Int[Tensor, "batch seq"]
    # k: int
    batch_size: int
    suffix_config: "SuffixConfig"
    generate_gumbel_config: "GumbelSoftmaxConfig"
    # threshold: float = 1
    optim: "OptimCfg"
    T: int = 1000
    device: str = DEVICE
    use_wandb: bool = True
    generate_length: int = 32
    # use_end_reward_selecting: bool = False
    # use_end_reward_gradients: bool = False
    # generate_targets_at_run_num: int = False
    num_prompts: int = 1
    # m_c_start: int = 1
    # print_text: bool = True
    # eos_penalty_grad: float = 1
    # temperature: float = 0.0402107123903543
    beta1: float = 0.9
    beta2: float = 0.99
    do_print: bool = True


@dataclass
class OptimCfg:
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

    def log(self, parent_name):
        return wandb.log(
            {
                f"{parent_name}.lr": self.lr,
                f"{parent_name}.betas": self.betas,
                f"{parent_name}.momentum": self.momentum,
                f"{parent_name}.weight_decay": self.weight_decay,
                f"{parent_name}.eps": self.eps,
            }
        )


@dataclass
class GumbelSoftmaxConfig:
    tau: float = 1
    hard: bool = False
    tau_backward: float = None
    noise_scale: float = 1

    def gumbel_softmax(self, logits, tau=None):
        return gumbel_softmax(
            logits,
            tau=tau or self.tau,
            hard=self.hard,
            tau_backward=self.tau_backward,
            noise_scale=self.noise_scale,
            dim=-1,
        )

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def updated(self, **kwargs):
        return GumbelSoftmaxConfig(
            **{k: v if v is not None else getattr(self, k) for k, v in kwargs.items()}
        )

    def log(self, parent_name):
        return wandb.log(
            {
                f"{parent_name}.{attr}": getattr(self, attr)
                for attr in self.__annotations__
            }
        )


@dataclass
class SuffixConfig:
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

    def forward(self, tau=None) -> Tensor:
        return self.cfg.gumbel_config.gumbel_softmax(self.suffix_logits, tau=tau)

    def log_historgram(self, run_num: int):
        suffix = self()
        suffix_softmax = F.softmax(self.suffix_logits, dim=-1)

        for i in range(suffix.shape[1]):
            wandb.log(
                {
                    f"suffix_gumbel_hist_{i}": wandb.Histogram(
                        suffix[:, i, :].float().detach().cpu().numpy()
                    ),
                    f"suffix_softmax_hist_{i}": wandb.Histogram(
                        suffix_softmax[:, i, :].float().detach().cpu().numpy()
                    ),
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
        for run_num in tqdm(range(self.cfg.T)):
            prefixes_seqchunk = self.get_next_prefixes()
            suffix = self.suffix()
            prompt_seqchunk = self.embedding_model.embed_seqchunks(
                prefixes_seqchunk, suffix, self.cfg.post_suffix
            )
            generated_probs = self.generate_fn(prompt_seqchunk)
            reward_seqchunk = self.reward_model.embedding_model.embed_seqchunks(
                prefixes_seqchunk, self.cfg.post_suffix, generated_probs
            )
            rewards = self.reward_model.embedding_model.forward_from_embed(
                reward_seqchunk
            )

            loss = rewards.end_rewards.mean()  # possibly change to rewards.rewards
            loss.backward()
            # printing gradient:
            print("suffix grad", self.suffix.suffix_logits.grad)
            wandb.log({"loss": loss.item()})
            optim.step()
            optim.zero_grad()
            if run_num % 2 == 0:
                self.log(run_num=run_num + 1)

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

            prompt_seqchunk = self.embedding_model.embed_seqchunks(
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
        suffix = self.suffix(tau=tau)
        prompt_seqchunk = self.embedding_model.embed_seqchunks(
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
            self.suffix.log_historgram(run_num=run_num + 1)
            prefixes, suffixes, completions = self.generate_printable()
            for prefix, suffix, completion in zip(prefixes, suffixes, completions):
                self.table.add_data(
                    prefix, suffix, completion, run_num + 1
                )  # prefix, suffix, completion, step
                if self.cfg.do_print:
                    print(prefix, suffix, completion)

    def suffix_only_train_test(self):

        optim = torch.optim.RAdam(
            self.suffix.parameters(), lr=3e-1, betas=(self.cfg.beta1, self.cfg.beta2)
        )
        # optim = torch.optim.SGD(self.suffix.parameters(), lr=3e1, momentum=0.9)

        self.get_next_prefixes()
        for run_num in tqdm(range(self.cfg.T)):
            suffix = self.suffix()
            reward_seqchunk = self.reward_model.embedding_model.embed_seqchunks(
                suffix,
            )
            rewards = self.reward_model.embedding_model.forward_from_embed(
                reward_seqchunk
            )

            loss = rewards.end_rewards.mean()  # possibly change to rewards.rewards
            loss.backward()
            # printing gradient:
            print("suffix grad", self.suffix.suffix_logits.grad)
            wandb.log({"loss": loss.item()})
            optim.step()
            optim.zero_grad()
            if run_num % 2 == 0:
                self.log(run_num=run_num + 1)

    def suffix_only_full_train(self, optim=None):  # GBRT paper setup, basically

        optim = optim or torch.optim.RAdam(
            self.suffix.parameters(),
            lr=1e-2,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay=0.01,
        )
        self.get_next_prefixes()
        self.cached_prefixes_seqchunk = self.cached_prefixes_seqchunk[:, 0:0]
        for run_num in tqdm(range(self.cfg.T)):
            suffix = self.suffix()
            prompt_seqchunk = self.embedding_model.embed_seqchunks(
                suffix,
            )

            generated_probs = self.generate_fn(prompt_seqchunk)
            reward_seqchunk = self.reward_model.embedding_model.embed_seqchunks(
                generated_probs
            )

            rewards = self.reward_model.embedding_model.forward_from_embed(
                reward_seqchunk
            )

            loss = rewards.end_rewards.mean()  # possibly change to rewards.rewards
            # loss = rewards.reward.mean()
            loss.backward()
            # printing gradient:
            print("suffix grad", self.suffix.suffix_logits.grad)
            wandb.log({"loss": loss.item()})
            optim.step()
            optim.zero_grad()
            if run_num % 2 == 0:
                self.log(run_num=run_num + 1)


def main():
    torch.set_default_dtype(torch.bfloat16)

    num_prompts = 16
    model, embedding_model, tokenizer = get_llama()

    harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    harmful_behavior_data.head()
    prefix_strs = harmful_behavior_data["goal"].tolist()[2 : 2 + num_prompts]

    prefixes = [
        torch.tensor(tokens, device=DEVICE, dtype=torch.long)
        for tokens in tokenizer(prefix_strs).input_ids
    ]

    reward_model: RewardGenerator = get_reward_generator()

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(DEVICE)
    post_suffix = post_suffix[1:].unsqueeze(0)

    suffix_config = SuffixConfig(
        gumbel_config=GumbelSoftmaxConfig(
            tau=5,
            hard=False,
            tau_backward=None,
            noise_scale=1.5,
        ),
        suffix_len=5,
    )

    generate_gumbel_config = GumbelSoftmaxConfig(
        tau=10,
        hard=False,
        tau_backward=None,
        noise_scale=1,
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
        prefixes=prefixes,
        suffix_config=suffix_config,
        generate_gumbel_config=generate_gumbel_config,
        T=4000,
        use_wandb=True,
        generate_length=6,
        num_prompts=num_prompts,
        beta1=0.9,
        beta2=0.998,
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
