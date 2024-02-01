# Glen and Arumin

import gc
from dataclasses import dataclass
from logging import config
from typing import List, Optional, Set, Tuple, Union

import einops
import pandas as pd
import torch
import transformers
import wandb
from colorama import Back, Fore, Style
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

import topk_gradients as topkgrad
from embedding_model import (
    EmbeddedBatch,
    EmbeddingFriendlyForCausalLM,
    EmbeddingFriendlyModel,
)
from gcg import GCGConfig
from token_gradients import TokenGradients

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class UPOConfig:
    prefixes: List[Int[Tensor, "prefix_lens"]]
    targets: List[Int[Tensor, "target_lens"]]
    suffix: Int[Tensor, "batch seq"]
    post_suffix: Int[Tensor, "batch seq"]
    k: int
    batch_size: int
    threshold: float = 1
    T: int = 1000
    modelname: str = "gpt2"
    device: str = DEVICE
    use_wandb: bool = False


class UPO:
    def __init__(
        self,
        cfg: UPOConfig,
        model: AutoModelForCausalLM,
        embedding_model: Optional[EmbeddingFriendlyModel] = None,
    ):
        assert callable(model)
        self.cfg = cfg
        self.model = model
        self.embedding_model = (
            EmbeddingFriendlyForCausalLM(self.model)
            if embedding_model is None
            else embedding_model
        )
        self.token_gradient_generator = TokenGradients(model, self.embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.modelname)
        self.suffix = cfg.suffix.clone()
        self.table = (
            wandb.Table(columns=["prefix", "suffix", "completion", "step"])
            if self.cfg.use_wandb
            else None
        )

    def upo(self, print_between=False):
        """
        datasource: List[List[str]]
        T: int "repeat T times"
        """
        if self.cfg.use_wandb:
            wandb.init(project="upo", config=self.cfg)

        prefixes = self.cfg.prefixes
        targets = self.cfg.targets
        m = len(prefixes)
        m_c = 1
        for run_num in range(self.cfg.T):  # repeat T times
            token_grad_batch = self.token_gradient_generator.get_token_gradients(
                prefixes[:m_c], self.suffix, targets[:m_c]
            )
            replacements = topkgrad.top_k_substitutions(token_grad_batch, self.cfg.k)
            # del token_grad_batch
            # gc.collect()
            next_suffixes = topkgrad.sample_replacements(
                replacements, self.suffix, self.cfg.batch_size
            )
            maxes_over_batch = torch.full(
                (self.cfg.batch_size,), -torch.inf, device=self.cfg.device
            )
            sum_over_batch = torch.zeros(self.cfg.batch_size, device=self.cfg.device)

            with torch.inference_mode():
                # the pog for loop
                for i in range(m_c):
                    tokens_batch = self.embedding_model.splice_tokens_batch(
                        prefixes[i], next_suffixes, targets[i], get_logits=True
                    )

                    losses = self.token_gradient_generator.get_loss_looping(
                        batch=tokens_batch,
                        target=targets[i],
                    )

                    sum_over_batch += losses
                    assert maxes_over_batch.shape == losses.shape
                    maxes_over_batch = torch.max(maxes_over_batch, losses)
                    # del tokens_batch

            # losses_batch_reshaped          ->
            # losses_batch_mean_over_prompt  [num_batches]   -> argmin

            best_suffix_idx = torch.argmin(sum_over_batch)
            best_suffix = next_suffixes[best_suffix_idx]

            self.suffix = best_suffix
            if maxes_over_batch[best_suffix_idx].max() < self.cfg.threshold and m_c < m:
                m_c += 1

            if print_between:
                if run_num % 10 == 0:
                    if run_num % 20 == 0:
                        generate(self)
                    print(Back.BLUE + "    ", self.tokenizer.decode(best_suffix))
                    print(
                        "loss opt:",
                        maxes_over_batch[best_suffix_idx].item(),
                        sum_over_batch.mean() / m_c,
                    )
                    print("m_c:", m_c)
                    print(Style.RESET_ALL)
            if self.cfg.use_wandb:
                wandb.log(
                    {"loss": maxes_over_batch[best_suffix_idx].item()}, step=run_num + 1
                )
                wandb.log({"m_c": m_c}, step=run_num + 1)
                if run_num % 50 == 0:
                    completions = get_completions(self)
                    for prefix, suffix, completion in completions:
                        self.table.add_data(prefix, suffix, completion, run_num + 1)

        if self.cfg.use_wandb:
            wandb.log(
                {"loss": maxes_over_batch[best_suffix_idx].item()}, step=run_num + 1
            )
            wandb.log({"m_c": m_c}, step=run_num + 1)
            wandb.log({"table": self.table})
            wandb.finish()

    def run(self):
        try:
            self.upo(print_between=not self.cfg.use_wandb)
        except Exception as e:
            if self.cfg.use_wandb:
                wandb.log({"table": self.table})
                wandb.finish()


def get_completions(upo: UPO):
    preplussuffixes = [torch.cat([prefix, upo.suffix]) for prefix in upo.cfg.prefixes]
    output = []
    for i, (tokens, target) in enumerate(zip(preplussuffixes, upo.cfg.targets)):
        all_ones_mask = torch.ones_like(tokens).bool()

        gen = upo.model.generate(
            tokens.unsqueeze(0),
            max_length=tokens.shape[0] + target.shape[0],
            attention_mask=all_ones_mask.unsqueeze(0),
            pad_token_id=upo.tokenizer.pad_token_id,
        ).squeeze()
        prefix_text = upo.tokenizer.decode(tokens[: -upo.suffix.shape[0]])
        suffix_text = upo.tokenizer.decode(tokens[-upo.suffix.shape[0] :])
        generated_text = upo.tokenizer.decode(gen[tokens.shape[0] :])

        output.append(
            (
                prefix_text,
                suffix_text,
                generated_text,
            )
        )
    return output


def generate(upo: UPO):
    preplussuffixes = [torch.cat([prefix, upo.suffix]) for prefix in upo.cfg.prefixes]
    for i, (tokens, target) in enumerate(zip(preplussuffixes, upo.cfg.targets)):
        all_ones_mask = torch.ones_like(tokens).bool()

        gen = upo.model.generate(
            tokens.unsqueeze(0),
            max_length=tokens.shape[0] + target.shape[0],
            attention_mask=all_ones_mask.unsqueeze(0),
            pad_token_id=upo.tokenizer.pad_token_id,
        ).squeeze()
        prefix_text = upo.tokenizer.decode(tokens[: -upo.suffix.shape[0]])
        suffix_text = upo.tokenizer.decode(tokens[-upo.suffix.shape[0] :])
        generated_text = upo.tokenizer.decode(gen[tokens.shape[0] :])

        print(
            f"{i} goal:     "
            + Fore.BLUE
            + prefix_text
            + Fore.RED
            + suffix_text
            + Fore.GREEN
            + upo.tokenizer.decode(target)
        )
        print(Style.RESET_ALL, end="")

        print(
            f"{i} generated:"
            + Fore.BLUE
            + prefix_text
            + Fore.RED
            + suffix_text
            + Fore.GREEN
            + generated_text
        )
        print(Style.RESET_ALL, end="")


def main():
    prefix_strs = [
        "A cat ",
        "The cat ",
        "That dog over there ",
        "A dog ",
    ]
    target_strs = [
        "is a dawg",
        "is a dawg",
        "is a cat",
        "is a cat and",
    ]

    harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    harmful_behavior_data.head()
    prefix_strs = harmful_behavior_data["goal"].tolist()[:1]
    target_strs = harmful_behavior_data["target"].tolist()[:1]

    model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    targets = [
        torch.tensor(tokens, device=DEVICE, dtype=torch.long)
        for tokens in tokenizer(target_strs).input_ids
    ]

    prefixes = [
        torch.tensor(tokens, device=DEVICE, dtype=torch.long)
        for tokens in tokenizer(prefix_strs).input_ids
    ]

    print(prefixes)
    print(targets)

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str).input_ids
    print(post_suffix)
    assert False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = UPOConfig(
        suffix=torch.randint(0, 50257, (10,), device=device),
        batch_size=128,
        prefixes=prefixes,
        post_suffix=post_suffix,
        targets=targets,
        T=500,
        k=100,
        use_wandb=False,
        threshold=1,
    )

    upo = UPO(cfg=cfg, model=model)
    upo.run()


if __name__ == "__main__":
    main()
