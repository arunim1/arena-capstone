import gc
from dataclasses import dataclass
from logging import config
from typing import List, Optional, Set, Tuple, Union

import einops
import pandas as pd
from sympy import true
import torch
import transformers
import wandb
from colorama import Back, Fore, Style
from jaxtyping import Bool, Float, Int
from torch import Tensor, embedding
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from arena_capstone.scripts.run_with_llama import get_llama

from arena_capstone.rewards.reward_generator import (
    RewardGenerator,
    get_reward_generator,
)
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
class RewardUPOConfig:
    prefixes: List[Int[Tensor, "prefix_lens"]]
    targets: List[Int[Tensor, "target_lens"]]
    suffix: Int[Tensor, "batch seq"]
    post_suffix: Int[Tensor, "batch seq"]
    k: int
    batch_size: int
    threshold: float = 1
    T: int = 1000
    device: str = DEVICE
    use_wandb: bool = False
    generate_length: int = 100
    use_end_reward_selecting: bool = False
    use_end_reward_gradients: bool = False
    


class RewardUPO:
    def __init__(
        self,
        cfg: RewardUPOConfig,
        model: AutoModelForCausalLM,
        reward_model: RewardGenerator,
        tokenizer: AutoTokenizer,
        embedding_model: Optional[EmbeddingFriendlyModel],
    ):
        assert callable(model)
        self.cfg = cfg
        self.model = model
        self.embedding_model = (
            EmbeddingFriendlyForCausalLM(self.model)
            if embedding_model is None
            else embedding_model
        )
        self.reward_model = reward_model
        self.token_gradient_generator = TokenGradients(model, self.embedding_model)
        self.tokenizer = tokenizer
        self.suffix = cfg.suffix.clone()
        self.table = (
            wandb.Table(columns=["prefix", "suffix", "completion", "step"])
            if self.cfg.use_wandb
            else None
        )

    def get_prompt(self):
        # Add batch dim to suffix
        reshaped_suffix = self.suffix.unsqueeze(0).expand(self.prefix.shape[0], self.suffix.shape[0])
        return torch.cat((self.prefix, reshaped_suffix, self.post_suffix), dim=1)

    def upo_over_rewards(self, print_between=False):
        """
        datasource: List[List[str]]
        T: int "repeat T times"
        """
        if self.cfg.use_wandb:
            wandb.init(project="upo", config=self.cfg)

        prefixes = self.cfg.prefixes

        # TODO replace with generated strings maybe?
        # prompt = self.get_prompt()
        # targets = self.model.generate(
        #     prompt, max_length=self.cfg.max_new_tokens, do_sample=True
        # )

        targets = self.cfg.targets

        m = len(prefixes)
        m_c = 1
        for run_num in tqdm(range(self.cfg.T)):  # repeat T times
            # token_grad_batch = self.token_gradient_generator.get_token_gradients()

            reward_grad_batch = self.embedding_model.splice_embedded_batch(
                prefixes=prefixes[:m_c],
                suffix_tokens=self.suffix,
                post_suffix_tokens=self.cfg.post_suffix,
                targets=targets[:m_c],
                get_logits=True,
            )
        
            rewards = self.reward_model.logit_rewards_from_embedded_batch(
                batch=reward_grad_batch,
            )
            if self.cfg.use_end_reward_gradients:
                loss = torch.sum(rewards.end_rewards)
            else:
                loss = torch.sum(rewards.rewards[reward_grad_batch.target_mask])
            mean_reward = torch.mean(rewards.end_rewards)
            loss.backward()
            # print(reward_grad_batch.suffix_tensor.grad)
            
            # assert False
            del rewards
            # does anything here on need to change? (before the inference mode)
            replacements = topkgrad.top_k_substitutions(reward_grad_batch, self.cfg.k)
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
                        prefixes[i],
                        next_suffixes,
                        self.cfg.post_suffix,
                        targets[i],
                        get_logits=True,
                    )

                    rewards = self.reward_model.logit_rewards_from_tokens_batch(
                        batch=tokens_batch
                    )

                    low, high = tokens_batch.target_bounds

                    if self.cfg.use_end_reward_selecting:
                        losses = rewards.end_rewards
                    else:
                        losses = torch.sum(rewards.rewards[:, low:high], dim=(-1, -2))

                    sum_over_batch += losses
                    assert maxes_over_batch.shape == losses.shape
                    maxes_over_batch = torch.max(maxes_over_batch, losses)
                    del rewards
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
                    print("mean_reward:", mean_reward.item())
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
            self.upo_over_rewards(print_between=not self.cfg.use_wandb)
        except Exception as e:
            if self.cfg.use_wandb:
                wandb.log({"table": self.table})
                wandb.finish()
            raise e


def get_completions(upo: RewardUPO):
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


def generate(upo: RewardUPO):
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

        # print(
        #     f"{i} goal:     "
        #     + Fore.BLUE
        #     + prefix_text
        #     + Fore.RED
        #     + suffix_text
        #     + Fore.GREEN
        #     + upo.tokenizer.decode(target)
        # )
        # print(Style.RESET_ALL, end="")

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
    num_prompts = 2
    model, embedding_model, tokenizer = get_llama()

    harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    harmful_behavior_data.head()
    prefix_strs = harmful_behavior_data["goal"].tolist()[:num_prompts]
    target_strs = harmful_behavior_data["target"].tolist()[:num_prompts]

    targets = [
        torch.tensor(tokens[1:], device=DEVICE, dtype=torch.long)
        for tokens in tokenizer(target_strs).input_ids
    ]

    prefixes = [
        torch.tensor(tokens, device=DEVICE, dtype=torch.long)
        for tokens in tokenizer(prefix_strs).input_ids
    ]

    reward_model = get_reward_generator()

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(DEVICE)
    post_suffix = post_suffix[1:]
    print(post_suffix.shape)

    cfg = RewardUPOConfig(
        suffix=torch.randint(0, model.config.vocab_size, (12,), device=DEVICE),
        post_suffix=post_suffix,
        batch_size=128,
        prefixes=prefixes,
        targets=targets,
        T=500,
        k=100,
        use_wandb=False,
        threshold=1,
        use_end_reward_selecting=True,
        use_end_reward_gradients=True,
    )

    upo = RewardUPO(
        cfg=cfg,
        model=model,
        reward_model=reward_model,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
    )
    with torch.cuda.amp.autocast():
        upo.run()


if __name__ == "__main__":
    main()
