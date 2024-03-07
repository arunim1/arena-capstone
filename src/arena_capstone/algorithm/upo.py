# Glen and Arumin

import gc
from dataclasses import dataclass
from logging import config
from typing import List, Optional, Set, Tuple, Union

import einops
import pandas as pd
from requests import post
import torch
import transformers
import wandb
from colorama import Back, Fore, Style
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    LlamaForCausalLM,
    LlamaTokenizer,
)
import time
import arena_capstone.algorithm.topk_gradients as topkgrad
from arena_capstone.algorithm.embedding_model import (
    EmbeddedBatch,
    EmbeddingFriendlyForCausalLM,
    EmbeddingFriendlyModel,
)
from arena_capstone.algorithm.gcg import GCGConfig
from arena_capstone.algorithm.token_gradients import TokenGradients

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
    wandb_project_name: str = "upo"
    do_print: bool = False
    subbatch_size: int = 128
    # banned_tokens: List[int] = None
    starting_m_c: int = 1
    early_search_exit_min_improvement: bool = False
    extra_max_emphasis: float = 0.0
    extra_sampled_twice: int = 0
    num_prompts_per_cycle: int = 4


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
        self.tensor_table = (
            wandb.Table(columns=["suffix_tensor", "step"])
            if self.cfg.use_wandb
            else None
        )
        self.banned_tokens = []
        self.t0 = time.time()

    def upo(self, print_between=False):
        """
        datasource: List[List[str]]
        T: int "repeat T times"
        """
        if self.cfg.use_wandb:
            wandb.init(project=self.cfg.wandb_project_name, config=self.cfg)
        prev_mm_loss = -float("inf")
        prefixes = self.cfg.prefixes
        targets = self.cfg.targets
        m = len(prefixes)
        m_c = self.cfg.starting_m_c

        def get_prompt_selector(m_c):
            while True:
                perm = torch.randperm(m_c)
                for perm_i in range(0, m_c, self.cfg.num_prompts_per_cycle):
                    yield set(
                        perm[perm_i : perm_i + self.cfg.num_prompts_per_cycle].tolist()
                    )

        prompt_selector = get_prompt_selector(m_c)

        bad_loss_extra_sample_id = None
        for run_num in tqdm(range(self.cfg.T)):  # repeat T times
            gc.collect()
            token_grad_batch = self.token_gradient_generator.get_token_gradients(
                prefixes[:m_c], self.suffix, self.cfg.post_suffix, targets[:m_c]
            )
            replacements = topkgrad.top_k_substitutions(
                token_grad_batch, self.cfg.k, exclude=self.banned_tokens
            )
            del token_grad_batch
            gc.collect()
            next_suffixes = topkgrad.sample_replacements(
                replacements, self.suffix, self.cfg.batch_size
            )
            if self.cfg.extra_sampled_twice > 0:
                # assert self.cfg.extra_sampled_twice == 2
                nsl = []
                for i in range(next_suffixes.shape[0]):
                    nsl.append(
                        topkgrad.sample_replacements(
                            replacements, next_suffixes[i], self.cfg.extra_sampled_twice
                        )
                    )
                next_suffixes = torch.cat(nsl, dim=0)
                del nsl
            maxes_over_batch = torch.full(
                (self.cfg.batch_size,), -torch.inf, device=self.cfg.device
            )
            sum_over_batch = torch.zeros(self.cfg.batch_size, device=self.cfg.device)
            idx = None
            prompts = next(prompt_selector) | (
                {bad_loss_extra_sample_id}
                if bad_loss_extra_sample_id is not None
                else set()
            )
            with torch.inference_mode():

                # the pog for loop
                assert self.cfg.batch_size % self.cfg.subbatch_size == 0
                for j in range(0, self.cfg.batch_size, self.cfg.subbatch_size):
                    for i in prompts:
                        gc.collect()
                        print(f"selected_prompt{i}")
                        next_suffixes_batch = next_suffixes[
                            j : j + self.cfg.subbatch_size
                        ]

                        torch.cuda.empty_cache()
                        tokens_batch = self.embedding_model.splice_tokens_batch(
                            prefixes[i],
                            next_suffixes_batch,
                            self.cfg.post_suffix,
                            targets[i],
                            get_logits=True,
                        )

                        losses = self.token_gradient_generator.get_loss_looping(
                            batch=tokens_batch,
                            target=targets[i],
                        )

                        # sum_over_batch += losses
                        sum_over_batch[j : j + self.cfg.subbatch_size] += losses

                        assert (
                            maxes_over_batch[j : j + self.cfg.subbatch_size].shape
                            == losses.shape
                        )
                        maxes_over_batch[j : j + self.cfg.subbatch_size] = torch.max(
                            maxes_over_batch[j : j + self.cfg.subbatch_size], losses
                        )
                        del tokens_batch
                    gc.collect()
                    sum_over_batch[j : j + self.cfg.subbatch_size] += (
                        maxes_over_batch[j : j + self.cfg.subbatch_size]
                        * self.cfg.extra_max_emphasis
                        * m_c
                    )

                    if self.cfg.early_search_exit_min_improvement:
                        min_improvement = (
                            self.cfg.early_search_exit_min_improvement
                            if isinstance(
                                self.cfg.early_search_exit_min_improvement, float
                            )
                            else 0.0
                        )
                        idx = j + torch.argmin(
                            sum_over_batch[j : j + self.cfg.subbatch_size]
                        )
                        if maxes_over_batch[idx] < prev_mm_loss - min_improvement:
                            break

                best_suffix_idx = idx or torch.argmin(sum_over_batch)
                best_suffix = next_suffixes[best_suffix_idx]
                self.suffix = best_suffix
                losses_per_prompt = torch.zeros(m_c, device=self.cfg.device)
                for i in range(m_c):
                    tokens_batch = self.embedding_model.splice_tokens_batch(
                        prefixes[i],
                        best_suffix.unsqueeze(0),
                        self.cfg.post_suffix,
                        targets[i],
                        get_logits=True,
                    )

                    losses = self.token_gradient_generator.get_loss_looping(
                        batch=tokens_batch,
                        target=targets[i],
                    )
                    losses_per_prompt[i] = losses
                    del tokens_batch
                torch.cuda.empty_cache()
                gc.collect()
            prev_mm_loss = maxes_over_batch[best_suffix_idx].item()

            worst_prompt_ids = torch.argsort(losses_per_prompt, descending=True)
            zipf = 1 / torch.arange(1, m_c + 1, device=self.cfg.device)
            zipf /= zipf.sum()

            weighted_avg = torch.sum(zipf * losses_per_prompt[worst_prompt_ids])
            bad_loss_extra_sample_id = worst_prompt_ids[
                torch.multinomial(zipf, 1).item()
            ]

            max_loss = losses_per_prompt[worst_prompt_ids[0]].item()
            if weighted_avg < self.cfg.threshold and m_c < m:
                m_c += 12
                prompt_selector = get_prompt_selector(m_c)

            if print_between:
                if run_num % 10 == 0:
                    if run_num % 20 == 0:
                        generate(self)
                    print(Back.BLUE + "    ", self.tokenizer.decode(best_suffix))
                    print("m", m)
                    print(
                        "loss opt:",
                        maxes_over_batch[best_suffix_idx].item(),
                        sum_over_batch.mean() / len(prompts),
                    )
                    print("m_c:", m_c)
                    print(Style.RESET_ALL)

            if self.cfg.use_wandb:
                self.tensor_table.add_data(self.suffix, run_num + 1)
                wandb.log(
                    {
                        "loss": maxes_over_batch[best_suffix_idx].item(),
                        "time elapsed:": time.time() - self.t0,
                        "max_loss": max_loss,
                        "weighted_avg": weighted_avg,
                        "sum_loss": sum_over_batch[best_suffix_idx].item() / m_c,
                    },
                    step=run_num + 1,
                )
                wandb.log({"m_c": m_c}, step=run_num + 1)
                if run_num % 50 + 1 == 0:
                    completions = get_completions(self)
                    for prefix, suffix, completion in completions:
                        self.table.add_data(prefix, suffix, completion, run_num + 1)

        if self.cfg.use_wandb:
            wandb.log(
                {
                    "loss": maxes_over_batch[best_suffix_idx].item(),
                    "max_loss": max_loss,
                },
                step=run_num + 1,
            )
            wandb.log({"m_c": m_c}, step=run_num + 1)
            wandb.log({"table": self.table})
            wandb.log({"suffix_tensor": self.tensor_table})
            wandb.finish()

    def run(self):
        try:
            self.upo(print_between=self.cfg.do_print)
        except Exception as e:
            print("suffix_tensor", self.suffix.detach().cpu().tolist())
            if self.cfg.use_wandb:
                wandb.log({"table": self.table})
                wandb.log({"suffix_tensor": self.tensor_table})
                wandb.finish()
            raise e
        finally:
            print("suffix_tensor", self.suffix.detach().cpu().tolist())


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

        post_suffix = upo.cfg.post_suffix
        post_suffix_str = upo.tokenizer.decode(post_suffix)

        print(
            f"{i} goal:     "
            + Fore.BLUE
            + prefix_text
            + Fore.RED
            + suffix_text
            + post_suffix_str
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
            + post_suffix_str
            + Fore.GREEN
            + generated_text
        )
        print(Style.RESET_ALL, end="")


def main():
    torch.set_default_dtype(torch.bfloat16)
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

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(DEVICE)
    print(post_suffix.shape)

    cfg = UPOConfig(
        suffix=torch.randint(0, 50257, (10,), device=DEVICE),
        post_suffix=post_suffix,
        batch_size=4,
        prefixes=prefixes,
        targets=targets,
        T=500,
        k=100,
        use_wandb=False,
        threshold=1,
        do_print=True,
        subbatch_size=2,
    )

    upo = UPO(cfg=cfg, model=model)
    upo.run()


if __name__ == "__main__":
    main()
