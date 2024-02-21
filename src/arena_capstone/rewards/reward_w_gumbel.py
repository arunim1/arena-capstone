import gc
from dataclasses import dataclass
from logging import config
from secrets import token_bytes
from typing import List, Optional, Set, Tuple, Union

import einops
from numpy import pad
import pandas as pd
import torch
import transformers
import wandb
from colorama import Back, Fore, Style
from jaxtyping import Bool, Float, Int
from torch import Tensor, embedding
from tqdm import tqdm
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    LlamaForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)
from arena_capstone.scripts.run_with_llama import get_llama

from arena_capstone.rewards.reward_generator import (
    RewardGenerator,
    get_reward_generator,
    RewardModelOutput,
)
import arena_capstone.algorithm.topk_gradients as topkgrad
from arena_capstone.algorithm.embedding_model import (
    EmbeddedBatch,
    EmbeddingFriendlyForCausalLM,
    EmbeddingFriendlyModel,
)
from arena_capstone.algorithm.gcg import GCGConfig
from arena_capstone.algorithm.token_gradients import TokenGradients

from arena_capstone.rewards.dataset_preprocess import proc_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def isinfnan(t):
    return torch.any(torch.isnan(t) | torch.isinf(t))


import arena_capstone.scripts.llamatokenize as llamatokenize


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
    generate_length: int = 32
    use_end_reward_selecting: bool = False
    use_end_reward_gradients: bool = False
    generate_targets_at_run_num: int = False
    num_prompts: int = 1
    m_c_start: int = 1
    print_text: bool = True
    eos_penalty_grad: float = 1
    temperature: float = 0.0402107123903543


DEBUG = False


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


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
        self.model: LlamaForCausalLM = model
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
        self.suffix_table = (
            wandb.Table(columns=["suffix", "step"]) if self.cfg.use_wandb else None
        )
        pd = proc_data(tokenizer)
        self.pd = pd
        self.data = [next(pd) for _ in range(100)]

        def dataset():
            i = 0
            while True:
                i += 1
                yield self.data[i % len(self.data)]

        self.dataset = dataset()

    def get_prompt(self, prefixes):
        prompts = [
            torch.cat((prefix, self.suffix, self.cfg.post_suffix), dim=0)
            for prefix in prefixes
        ]
        return prompts

    def get_next_prefixes(self):
        prefix_strs = [next(self.dataset) for _ in range(self.cfg.num_prompts)]

        prefixes = [
            torch.tensor(tokens, device=DEVICE, dtype=torch.long)
            for tokens in self.tokenizer(prefix_strs).input_ids
        ]
        return prefixes

    def get_next_targets(self, prompts):
        bad_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
        ]
        targets = []
        # logit processor to remove EOS and BOS from the logits
        # processor = LogitsProcessorList([LogitsProcessor()])

        for prompt in prompts:
            all_ones_mask = torch.ones_like(prompt, dtype=torch.bool)
            print("prompt", prompt.shape, self.tokenizer.decode(prompt))
            target = self.model.generate(
                prompt.unsqueeze(0),
                # attention_mask=all_ones_mask.unsqueeze(0),
                # repitition_penalty=1.2,
                max_length=self.cfg.generate_length + prompt.shape[0],
                do_sample=True,
                # eos_token_id=self.tokenizer.eos_token_id,
                # bos_token_id=self.tokenizer.bos_token_id,
                # pad_token_id=self.tokenizer.pad_token_id,
                # temperature=1,
                attention_mask=all_ones_mask.unsqueeze(0),
                pad_token_id=self.tokenizer.pad_token_id,
                bad_words_ids=[[bad] for bad in bad_tokens],
            ).squeeze()
            print("gen target:", llamatokenize.detokenize(self.tokenizer, list(target)))
            target = target[prompt.shape[0] :]
            for bad_id in bad_tokens:
                if torch.any(target == bad_id):
                    target = self.get_next_targets([prompt])[0]
                    break

            # print(target.shape)
            targets.append(target)

        return targets

    def upo_over_rewards(self, print_between=False):
        """ """
        SET_THING_OF_NANS = None

        if self.cfg.use_wandb:
            wandb.init(project="reward-upo", config=self.cfg)

        prefixes = self.cfg.prefixes

        # TODO replace with generated strings maybe?

        targets = self.cfg.targets

        m = len(prefixes)
        m_c = self.cfg.m_c_start
        TOKEN_INDEX_PENALTY = 0
        TOKEN_INDEX_PENALTY_TOKENS = 0
        for run_num in tqdm(range(self.cfg.T)):  # repeat T times
            if (
                self.cfg.generate_targets_at_run_num
                and self.cfg.generate_targets_at_run_num < run_num
            ):
                del prefixes, targets
                gc.collect()

                prefixes = self.get_next_prefixes()
                prompt = self.get_prompt(prefixes)
                targets = self.get_next_targets(prompt)
                self.cfg.prefixes = prefixes

            #####
            # base_grad_batch = self.embedding_model.splice_embedded_batch(
            #     prefixes=prefixes[:m_c],
            #     suffix_tokens=self.suffix,
            #     post_suffix_tokens=self.cfg.post_suffix,
            #     targets=targets[:m_c],
            #     get_logits=True,
            # )

            # reward_grad_batch = self.reward_model.embedding_model.splice_embedded_batch(
            #     prefixes=prefixes[:m_c],
            #     suffix_tokens=self.suffix,
            #     post_suffix_tokens=self.cfg.post_suffix,
            #     targets=targets[:m_c],
            #     get_logits=False,
            #     hot_suffix=base_grad_batch.suffix_tensor,
            # )

            # rewards = self.reward_model.logit_rewards_from_embedded_batch(
            #     batch=base_grad_batch,
            #     reward_batch=reward_grad_batch,
            #     div_target_logits=20000**0.5,
            # )

            base_grad_batch, reward_batch, reward = self.generate_with_gumbel(
                gumbel_softmax=self.reward_model.softmax,
                prefix=prefixes[0],
                suffix=self.suffix,
                post_suffix=self.cfg.post_suffix,
                generate_length=self.cfg.generate_length,
                bad_words_ids=[],
            )

            rewards = reward.end_rewards.squeeze(-1)

            #
            loss = torch.sum(rewards)
            mean_end_rewards = torch.mean(rewards)
            mean_reward = torch.mean(rewards)
            loss.backward()

            del reward_grad_batch

            if torch.isnan(mean_reward).any():
                print("mean_reward is nan")
                print("suffix", self.suffix)
                print("grad,", base_grad_batch.suffix_tensor.grad)
                # print("rewards", rewards.rewards)
                # print("end", rewards.end_rewards)
                logits_softmaxxed = base_grad_batch.logits[base_grad_batch.target_mask]
                print("logits_softmaxxed sum ", logits_softmaxxed.sum(-1))
                print("targets:", targets)
                set_targets = [set(t.tolist()) for t in targets]
                s = set()
                for t in set_targets:
                    s = s.union(t)
                if SET_THING_OF_NANS is None:
                    SET_THING_OF_NANS = s
                else:
                    SET_THING_OF_NANS = SET_THING_OF_NANS.intersection(s)
                print("SET_THING_OF_NANS", SET_THING_OF_NANS)

            #########
            # print(reward_grad_batch.suffix_tensor.grad)
            # assert False
            # does anything here on need to change? (before the inference mode)

            # reward_grad_batch = self.embedding_model.splice_embedded_batch(
            #     prefixes=prefixes[:m_c],
            #     suffix_tokens=self.suffix,
            #     post_suffix_tokens=self.cfg.post_suffix,
            #     targets=targets[:m_c],
            #     get_logits=False,
            # )
            # reward_grad_batch.suffix_tensor.grad = torch.rand_like(reward_grad_batch.suffix_tensor)
            bad_tokens = {
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.pad_token_id,
            }
            del base_grad_batch.embeddings
            dprint("grads", base_grad_batch.suffix_tensor.grad)
            replacements = topkgrad.top_k_substitutions(
                base_grad_batch, self.cfg.k, exclude=bad_tokens
            )
            dprint("replacements", replacements)
            # del loss, rewards, reward_grad_batch
            # gc.collect()

            topk_values, topk_indices = torch.topk(
                base_grad_batch.suffix_tensor.grad,
                k=10,
                dim=-1,
                largest=False,
            )
            print(
                llamatokenize.detokenize(
                    self.tokenizer, [k.item() for i in list(topk_indices) for k in i]
                )
            )
            torch.count_nonzero(topk_indices == self.tokenizer.bos_token_id)
            torch.count_nonzero(topk_indices == self.tokenizer.eos_token_id)
            torch.count_nonzero(topk_indices == self.tokenizer.pad_token_id)
            print(topk_indices)
            next_suffixes = topkgrad.sample_replacements(
                replacements=replacements,
                suffix=self.suffix,
                batch_size=self.cfg.batch_size,
            )
            maxes_over_batch = torch.full(
                (self.cfg.batch_size,), -torch.inf, device=self.cfg.device
            )
            sum_over_batch = torch.zeros(self.cfg.batch_size, device=self.cfg.device)
            gc.collect()

            with torch.inference_mode():
                # the pog for loop (saves memory)
                for i in range(m_c):
                    tokens_batch = self.embedding_model.splice_tokens_batch(
                        prefix=prefixes[i],
                        suffix_tokens=next_suffixes,
                        post_suffix=self.cfg.post_suffix,
                        target=targets[i],
                        get_logits=True,
                    )

                    # rewards = self.reward_model.logit_rewards_from_tokens_batch(
                    #     batch=tokens_batch
                    # )
                    rewards = self.reward_model.logit_rewards_loop_over_tokens_batch(
                        batch=tokens_batch,
                        temperature=self.cfg.temperature,
                    )
                    # low, high = tokens_batch.target_bounds
                    losses = torch.sum(rewards, dim=(-1, -2))
                    self.high_token_index_penalty(
                        tokens_batch.logits, TOKEN_INDEX_PENALTY_TOKENS
                    )

                    # if self.cfg.use_end_reward_selecting:
                    #     losses = rewards.end_rewards.squeeze(-1)
                    # else:
                    #     losses = torch.sum(rewards.rewards[:, low:high], dim=(-1, -2))
                    #     # losses = torch.sum(rewards.rewards, dim=(-1, -2))

                    if torch.isnan(losses).any():
                        dprint("losses is nan")
                        dprint(losses)
                        nansuffix = next_suffixes[torch.isnan(losses)]
                        dprint("nan next_suffixes", nansuffix)
                        dprint("normal suffix", self.suffix)
                        dprint(
                            "nan tokens_batch", tokens_batch.tokens[torch.isnan(losses)]
                        )
                        dprint(
                            "nan tokens for suffix",
                            [
                                llamatokenize.detokenize(self.tokenizer, t)
                                for t in nansuffix
                            ],
                        )

                    # losses += self.eos_penalty(tokens_batch.logits)

                    # losses += 0.2 * self.repition_penalty(
                    #     tokens_batch.logits[
                    #         :,
                    #         tokens_batch.target_bounds[0] : tokens_batch.target_bounds[
                    #             1
                    #         ],
                    #     ],
                    #     targets[i],
                    #     targets[i].shape[0],
                    # )

                    sum_over_batch += losses
                    assert maxes_over_batch.shape == losses.shape
                    maxes_over_batch = torch.max(maxes_over_batch, losses)

                    del rewards, tokens_batch, losses
                    gc.collect()

            # losses_batch_reshaped          ->
            # losses_batch_mean_over_prompt  [num_batches]   -> argmin
            sum_over_batch[torch.isnan(sum_over_batch)] = torch.inf
            best_suffix_idx = torch.argmin(sum_over_batch)
            best_suffix = next_suffixes[best_suffix_idx]

            self.suffix = best_suffix
            if self.cfg.use_wandb:
                self.suffix_table.add_data(
                    str(self.suffix.tolist()),
                    run_num + 1,
                )

            if maxes_over_batch[best_suffix_idx].max() < self.cfg.threshold and m_c < m:
                m_c += 1
                self.data += [next(pd)]

            del next_suffixes
            gc.collect()

            if print_between:
                if run_num % 5 == 0:
                    if run_num % 5 == 0:
                        generate(self, targets=targets, prefixes=prefixes)
                    print(Back.BLUE + "    ", self.tokenizer.decode(best_suffix))
                    print(
                        "loss opt:",
                        maxes_over_batch[best_suffix_idx].item(),
                        sum_over_batch.mean() / m_c,
                    )
                    print("m_c:", m_c)
                    print(Style.RESET_ALL)
                print("mean_reward:", mean_reward.item())
                print("mean_end_reward:", mean_end_rewards.item())
            if self.cfg.use_wandb:
                wandb.log(
                    {"loss": maxes_over_batch[best_suffix_idx].item()}, step=run_num + 1
                )
                wandb.log(
                    {
                        "mean_reward": mean_reward.item(),
                        "mean_end_reward": mean_end_rewards.item(),
                    }
                )
                wandb.log({"m_c": m_c}, step=run_num + 1)
                if run_num % 50 == 0:
                    completions = get_completions(
                        self, targets=targets, prefixes=prefixes
                    )
                    for prefix, suffix, completion in completions:
                        self.table.add_data(prefix, suffix, completion, run_num + 1)

        if self.cfg.use_wandb:
            wandb.log(
                {"loss": maxes_over_batch[best_suffix_idx].item()}, step=run_num + 1
            )
            wandb.log(
                {
                    "mean_reward": mean_reward.item(),
                    "mean_end_reward": mean_end_rewards.item(),
                }
            )
            wandb.log({"m_c": m_c}, step=run_num + 1)
            wandb.log({"table": self.table})
            wandb.log({"suffix_table": self.suffix_table})
            wandb.finish()

    def high_token_index_penalty(self, logits, p=0.01):
        penalty_mult = torch.arange(32001, device="cuda") * p
        return torch.mean(penalty_mult * torch.log_softmax(logits, dim=-1))

    def repition_penalty(self, logits, tokens, length):
        tok_count = torch.zeros_like(logits[0, 0, :], requires_grad=False).detach()
        penalty = torch.zeros_like(logits[:, 0, 0])
        seq_log = einops.rearrange(logits, "... seq vocab -> ... vocab seq")
        for i in range(length):
            token = tokens[..., i]
            tok_count[..., token] += 1
            tok_count = tok_count.detach()
            penalty += torch.sum(
                seq_log[..., token, i:] * tok_count[token].detach(), dim=-1
            ) * ((token + 800) / 5000)
        return penalty

    def eos_penalty(self, suffix_tensor):
        return suffix_tensor[..., self.tokenizer.eos_token_id].sum(-1)

    def run(self):
        try:
            self.upo_over_rewards(print_between=self.cfg.print_text)
        except Exception as e:
            if self.cfg.use_wandb:
                wandb.log({"table": self.table})
                wandb.log({"suffix_table": self.suffix_table})
                wandb.finish()
            raise e

    def generate_with_gumbel(
        self,
        gumbel_softmax,
        prefix,
        suffix,
        post_suffix,
        generate_length,
        bad_words_ids=[],
    ) -> RewardModelOutput:  # not 100% on the return type

        batch = self.embedding_model.splice_embedded_batch(
            [prefix],
            suffix,
            post_suffix,
            targets=[torch.zeros(0, device=DEVICE, dtype=torch.long)],
        )
        reward_batch = self.reward_model.embedding_model.splice_embedded_batch(
            [prefix],
            torch.zeros(0, device=DEVICE, dtype=torch.long),
            post_suffix,
            targets=[torch.zeros(0, device=DEVICE, dtype=torch.long)],
        )
        # generate_length = max_length - prefixes.shape[1]
        target_logits = []
        for i in range(generate_length):
            logits_next = self.embedding_model.forward_from_embed(
                batch.embeddings
            ).logits
            one_hot_next_token = gumbel_softmax(logits_next)
            target_logits.append(logits_next)
            one_hot_embedded = self.embedding_model.embed(
                one_hot_next_token, onehot=True
            ).squeeze(0)
            batch.embeddings = torch.cat([batch.embeddings, one_hot_embedded], dim=1)
            one_hot_reward_embedded = self.reward_model.embedding_model.embed(
                one_hot_next_token, onehot=True
            ).squeeze(0)
            reward_batch.embeddings = torch.cat(
                [reward_batch.embeddings, one_hot_reward_embedded], dim=1
            )

        batch.target_mask = torch.cat(
            (
                batch.target_mask,
                torch.ones(1, generate_length, device="cuda", dtype=torch.bool),
            ),
            dim=1,
        )
        reward_batch.target_mask = torch.cat(
            (
                reward_batch.target_mask,
                torch.ones(1, generate_length, device="cuda", dtype=torch.bool),
            ),
            dim=1,
        )
        reward_output = self.reward_model(
            input_ids=None,
            attention_mask=torch.ones(reward_batch.embeddings.shape[:2], device="cuda"),
            inputs_embeds=reward_batch.embeddings,
        )
        return batch, reward_batch, reward_output


def get_completions(
    upo: RewardUPO, targets: Optional[List[Tensor]] = None, prefixes=None
):
    prefixes = prefixes if prefixes is not None else upo.cfg.prefixes
    preplussuffixes = [
        torch.cat([prefix, upo.suffix, upo.cfg.post_suffix]) for prefix in prefixes
    ]
    targets = (
        targets[: len(preplussuffixes)]
        if targets is not None
        else upo.cfg.targets[: len(preplussuffixes)]
    )
    bad_tokens = [
        upo.tokenizer.eos_token_id,
        upo.tokenizer.bos_token_id,
        upo.tokenizer.pad_token_id,
    ]
    output = []

    for i, (tokens, target) in enumerate(zip(preplussuffixes, targets)):
        all_ones_mask = torch.ones_like(tokens).bool()

        gen = upo.model.generate(
            tokens.unsqueeze(0),
            max_length=tokens.shape[0] + target.shape[0],
            attention_mask=all_ones_mask.unsqueeze(0),
            pad_token_id=upo.tokenizer.pad_token_id,
            eos_token_id=100000,
            temperature=0.2402107123903543,
            bad_words_ids=[[bad] for bad in bad_tokens],
        ).squeeze()
        suffixlen = upo.suffix.shape[0] + upo.cfg.post_suffix.shape[0]
        prefix_text = upo.tokenizer.decode(tokens[:-suffixlen])
        suffix_text = upo.tokenizer.decode(tokens[-suffixlen:])
        generated_text = upo.tokenizer.decode(gen[tokens.shape[0] :])

        output.append(
            (
                prefix_text,
                suffix_text,
                generated_text,
            )
        )
    return output


def generate(upo: RewardUPO, targets: Optional[List[Tensor]] = None, prefixes=None):
    prefixes = prefixes if prefixes is not None else upo.cfg.prefixes
    preplussuffixes = [
        torch.cat([prefix, upo.suffix, upo.cfg.post_suffix]) for prefix in prefixes
    ]
    targets = (
        targets[: len(preplussuffixes)]
        if targets is not None
        else upo.cfg.targets[: len(preplussuffixes)]
    )
    bad_tokens = [
        upo.tokenizer.eos_token_id,
        upo.tokenizer.bos_token_id,
        upo.tokenizer.pad_token_id,
    ]
    for i, (tokens, target) in enumerate(zip(preplussuffixes, targets)):
        all_ones_mask = torch.ones_like(tokens).bool()

        gen = upo.model.generate(
            tokens.unsqueeze(0),
            max_length=tokens.shape[0] + target.shape[0],
            attention_mask=all_ones_mask.unsqueeze(0),
            pad_token_id=upo.tokenizer.pad_token_id,
            eos_token_id=100000,
            temperature=0.2402107123903543,
            bad_words_ids=[[bad] for bad in bad_tokens],
        ).squeeze()
        suffixlen = upo.suffix.shape[0] + upo.cfg.post_suffix.shape[0]
        prefix_text = upo.tokenizer.decode(tokens[:-suffixlen])
        suffix_text = upo.tokenizer.decode(tokens[-suffixlen:])
        generated_text = upo.tokenizer.decode(gen[tokens.shape[0] :])
        print(
            "generated",
            [
                llamatokenize.detokenize(upo.tokenizer, g)
                for g in gen[tokens.shape[0] - 1 :]
            ],
        )
        print("target", target)
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

        # evaluate the generated text with the reward model

        input_ids = upo.tokenizer(
            # prefix_text +
            generated_text,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        attention_mask = torch.ones_like(input_ids).to(DEVICE)
        print(
            "reward:",
            upo.reward_model(input_ids=input_ids, attention_mask=attention_mask),
        )


def main():
    torch.set_default_dtype(torch.bfloat16)

    num_prompts = 1
    model, embedding_model, tokenizer = get_llama()

    harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    harmful_behavior_data.head()
    prefix_strs = harmful_behavior_data["goal"].tolist()[2 : 2 + num_prompts]
    target_strs = harmful_behavior_data["target"].tolist()[2 : 2 + num_prompts]

    targets = [
        torch.tensor(tokens[1:], device=DEVICE, dtype=torch.long)
        for tokens in tokenizer(target_strs).input_ids
    ]

    prefixes = [
        torch.tensor(tokens, device=DEVICE, dtype=torch.long)
        for tokens in tokenizer(prefix_strs).input_ids
    ]

    reward_model: RewardGenerator = get_reward_generator()

    # Gumbel-softmax
    def gumbel_softmax(logits, dim=-1):
        return F.gumbel_softmax(logits, tau=0.003, hard=True, dim=dim, eps=1e-7)

    reward_model.softmax = gumbel_softmax

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(DEVICE)
    post_suffix = post_suffix[1:]
    import arena_capstone.scripts.llamatokenize as llamatokenize

    # post_suffix = torch.zeros(0, device=DEVICE, dtype=torch.long)

    # print("checking")
    # assert torch.allclose(
    #     reward_model.get_input_embeddings().weight, model.get_input_embeddings().weight
    # )

    # assert torch.all(
    #     reward_model.get_input_embeddings().weight
    #     == model.get_input_embeddings().weight
    # )
    # print("checked")
    # assert False

    cfg = RewardUPOConfig(
        suffix=torch.randint(5, 1000, (6,), device=DEVICE),
        post_suffix=post_suffix,
        batch_size=128,
        prefixes=prefixes,
        targets=targets,
        T=4000,
        k=256,
        use_wandb=True,
        threshold=1,
        use_end_reward_selecting=False,
        use_end_reward_gradients=False,
        generate_targets_at_run_num=1,
        generate_length=5,
        num_prompts=num_prompts,
        print_text=True,
        # eos_penalty_grad=0.22,
        temperature=1,
    )

    upo = RewardUPO(
        cfg=cfg,
        model=model,
        reward_model=reward_model,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
    )
    # print(upo.get_next_targets(upo.get_prompt()))
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        upo.run()


if __name__ == "__main__":
    main()
