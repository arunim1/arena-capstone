import gc
from dataclasses import dataclass
from logging import config
from secrets import token_bytes
from typing import List, Optional, Set, Tuple, Union
from arena_capstone.rewards.gumbel_softmax import gumbel_softmax
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
import torch.nn as nn

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
import torch.nn as nn
import torch.nn.functional as F


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
    grad_acc_beta1: float = 0.96
    grad_acc_beta2: float = 0.99


DEBUG = False


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class Suffix(nn.Module):
    def __init__(
        self,
        suffix_logits: Optional[Float[Tensor, "suffix_len d_vocab"]],
        suffix_len: Optional[int] = 5,
        tau=1,
        tau_backward=None,
        noise_scale=1,
    ):
        if suffix_logits is None:
            suffix_logits = torch.randn(suffix_len, 32000, device=DEVICE)
        self.suffix_logits = nn.Parameter(suffix_logits.clone())
        self.tau = tau
        self.tau_backward = tau_backward
        self.hard = False
        self.noise_scale = noise_scale

    def softmax(self, logits, tau=None):
        return gumbel_softmax(
            logits,
            tau=tau or self.tau,
            hard=self.hard,
            tau_backward=self.tau_backward,
            noise_scale=self.noise_scale,
            dim=-1,
        )

    def forward(self):
        return self.softmax(self.suffix_logits)


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
        self.suffix = Suffix(
            cfg.suffix,
            tau=cfg.suffix_tau,
            tau_backward=cfg.suffix_tau_backward,
            noise_scale=cfg.suffix_noise_scale,
        )

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
        self.grad_acc = None
        self.gradsquares = None

        def dataset():
            i = 0
            while True:
                i += 1
                yield self.data[i % len(self.data)]

        def dataset():
            i = 0
            while True:
                i += 1
                yield self.data[20 + i % 2]

        self.dataset = dataset()

    def train(self, optim=None):
        optim = optim or torch.optim.Adam(self.suffix.parameters(), lr=0.01)
        for run_num in tqdm(range(self.cfg.T)):
            prefixes, prefix_mask = self.get_next_prefixes()
            suffix = self.suffix()
            embedded_prompt, prompt_attention_mask = self.embedding_model.embed_nice(
                (prefixes, prefix_mask), suffix, self.cfg.post_suffix
            )
            sequence_logits = self.generate_fn(
                embedded_prompt, prompt_attention_mask
            )  # also uses gumbel
            reward_seq_embedded = self.reward_model.embedding_model.embed_nice(
                prefixes, self.cfg.post_suffix, sequence_logits
            )
            reward = reward_fn(sequence)
            loss = reward.mean()
            loss.backward()
            optim.step()
            optim.zero_grad()

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

            base_grad_batch, reward_grad_batch, reward = self.generate_with_gumbel(
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

            gc.collect()

            if print_between:
                if run_num % 5 == 0:
                    if run_num % 5 == 0:
                        generate(self, targets=targets, prefixes=prefixes)
                    print("m_c:", m_c)
                    print(Style.RESET_ALL)
                print("mean_reward:", mean_reward.item())
                print("mean_end_reward:", mean_end_rewards.item())
            if self.cfg.use_wandb:
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
                {
                    "mean_reward": mean_reward.item(),
                    "mean_end_reward": mean_end_rewards.item(),
                }
            )
            wandb.log({"m_c": m_c}, step=run_num + 1)
            wandb.log({"table": self.table})
            wandb.log({"suffix_table": self.suffix_table})
            wandb.finish()

    def get_prompt(self, prefixes):
        prompts = [
            torch.cat((prefix, self.suffix, self.cfg.post_suffix), dim=0)
            for prefix in prefixes
        ]
        return prompts

    def get_next_prefixes(self):
        prefix_strs = [next(self.dataset) for _ in range(self.cfg.num_prompts)]
        tokenized = self.tokenizer(prefix_strs)
        prefixes = tokenized.input_ids
        masks = tokenized.attention_mask
        return prefixes, masks

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

    def run(self):
        try:
            self.upo_over_rewards(print_between=self.cfg.print_text)
        except Exception as e:
            if self.cfg.use_wandb:
                wandb.log({"table": self.table})
                wandb.log({"suffix_table": self.suffix_table})
                wandb.finish()
            raise e

    def generate_fn(
        self,
        embedded_prompt: Float[Tensor, "batch prompt_len d_model"],
        prompt_attention_mask: Float[Tensor, "batch prompt_len"],
    ) -> List[Tensor]:  # not 100% on the return type
        def gen_gumbel_softmax(logits, tau=None):
            return gumbel_softmax(
                tau=tau or self.gen_tau,
                hard=self.gen_hard,
                tau_backward=self.gen_tau_backward,
                noise_scale=self.gen_noise_scale,
                dim=-1,
            )

        # batch = self.embedding_model.splice_embedded_batch(
        #     [prefix],
        #     suffix,
        #     post_suffix,
        #     targets=[torch.zeros(0, device=DEVICE, dtype=torch.long)],
        # )
        # reward_batch = self.reward_model.embedding_model.splice_embedded_batch(
        #     [prefix],
        #     torch.zeros(0, device=DEVICE, dtype=torch.long),
        #     post_suffix,
        #     targets=[torch.zeros(0, device=DEVICE, dtype=torch.long)],
        # )
        # # generate_length = max_length - prefixes.shape[1]
        target_logits = []

        for i in range(self.cfg.generate_length):
            logits_next = self.embedding_model.forward_from_embed(
                embedded_prompt, prompt_attention_mask
            ).logits[:, -1:, :]
            # logits_next is shape (batch, vocab_size)
            next_token_probs = gen_gumbel_softmax(logits_next)
            # target_logits.append(logits_next)
            # one_hot_embedded = self.embedding_model.embed(
            #     next_token_probs, onehot=True
            # ).squeeze(0)

            # embedded_prompt = torch.cat([embedded_prompt, one_hot_embedded], dim=1)
            embedded_prompt, prompt_attention_mask = self.embedding_model.embed_nice(
                (embedded_prompt, prompt_attention_mask), next_token_probs
            )

            reward_embeddings = self.reward_model.embedding_model.embed_nice(
                (reward_embeddings, prompt_attention_mask), next_token_probs
            )

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
        return F.gumbel_softmax(logits, tau=0.003, hard=False, dim=dim, eps=1e-7)

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
        generate_length=6,
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
