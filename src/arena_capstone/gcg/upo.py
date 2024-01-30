# Glen Taggart (nqgl) if there are any issues/questions

import arena_capstone.gcg.topk_gradients as topkgrad
from arena_capstone.gcg.embedding_model import EmbeddingFriendlyCausalForLM

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from arena_capstone.gcg.embedding_model import (
    EmbeddingFriendlyCausalForLM,
    EmbeddingFriendlyModel,
    EmbeddedBatch,
)
from arena_capstone.gcg.token_gradients import TokenGradients
from arena_capstone.gcg.gcg import GCGConfig
import transformers
from typing import List, Set, Tuple, Union, Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dataclasses import dataclass
import torch
import einops
import wandb


# @dataclass
# class UPOConfig(GCGConfig):
#     pass


class UPO:
    def __init__(
        self,
        cfg: GCGConfig,
        model: AutoModelForCausalLM,
        embedding_model: Optional[EmbeddingFriendlyModel] = None,
    ):
        assert callable(model)
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.embedding_model = (
            EmbeddingFriendlyCausalForLM(self.model)
            if embedding_model is None
            else embedding_model
        )
        self.token_gradient_generator = TokenGradients(model, self.embedding_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.modelname)
        self.suffix = cfg.suffix.clone()

    def gcg(self, print_between=False):
        """
        datasource: List[List[str]]
        T: int "repeat T times"
        """
        if self.cfg.use_wandb:
            wandb.init(project="upo")

        prefixes = self.tokenizer.encode_plus(self.cfg.prefix_str).input_ids
        prefixes = [torch.tensor(prefixes, dtype=torch.long, device=self.cfg.device)]

        targets = self.tokenizer.encode_plus(self.cfg.target_str).input_ids
        targets = [torch.tensor(targets, dtype=torch.long, device=self.cfg.device)]
        m_c = 1
        for run_num in range(self.cfg.T):  # repeat T times
            token_grad_batch = self.token_gradient_generator.get_token_gradients(
                prefixes, self.suffix, targets, print_loss=True
            )
            replacements = topkgrad.top_k_substitutions(token_grad_batch, self.cfg.k)
            next_suffixes = topkgrad.sample_replacements(
                replacements, self.suffix, self.cfg.batch_size
            )
            with torch.inference_mode():
                tokens_batch = self.embedding_model.batch_for_step2(
                    prefixes[:m_c], next_suffixes, targets[:m_c], get_logits=True
                )

                losses = self.token_gradient_generator.get_loss(
                    batch=tokens_batch,
                    targets=targets * self.cfg.batch_size,
                    reduce_over_batch=False,
                )

            losses = losses.reshape(self.cfg.batch_size, -1).mean(dim=-1)

            best_suffix_idx = torch.argmin(losses)
            best_suffix = next_suffixes[best_suffix_idx]

            self.suffix = best_suffix
            if losses[best_suffix_idx] < self.cfg.threshold:
                m_c += 1

            if print_between:
                if run_num % 10 == 0:
                    generate(self)
                print("    ", self.tokenizer.decode(best_suffix))
                print("loss opt:", losses[best_suffix_idx].item())
            if self.cfg.use_wandb:
                wandb.log({"loss": losses[best_suffix_idx].item()})
                wandb.log({"suffix": self.tokenizer.decode(best_suffix)})

            del token_grad_batch.suffix_tensor


def generate(gcg: GCG):
    tokens = torch.cat(
        [
            torch.tensor(
                gcg.tokenizer.encode_plus(gcg.cfg.prefix_str).input_ids,
                device=gcg.suffix.device,
                dtype=torch.long,
            ),
            gcg.suffix,
        ]
    )

    gen = gcg.model.generate(tokens.unsqueeze(0), max_length=30).squeeze()
    print("generated:", gcg.tokenizer.decode(gen))


def main():
    cfg = GCGConfig(
        suffix=torch.randint(0, 50257, (10,), device="cuda"),
        prefix_str="The cat is a type of",
        target_str=" I hate that",
        batch_size=50,
        T=200,
        k=300,
        use_wandb=False,
    )
    gcg = GCG(cfg=cfg, model=AutoModelForCausalLM.from_pretrained("gpt2"))
    gcg.gcg(print_between=(not cfg.use_wandb))
    generate(gcg)
    generate(gcg)
    generate(gcg)
    # m: PreTrainedModel = gcg.model
    # tokens = torch.cat(
    #     [
    #         torch.tensor(
    #             gcg.tokenizer.encode_plus(gcg.cfg.prefix_str).input_ids,
    #             device=gcg.suffix.device,
    #             dtype=torch.long,
    #         ),
    #         gcg.suffix,
    #     ]
    # )
    # gen = m.generate(tokens.unsqueeze(0))
    # print(gen)
    # print()


if __name__ == "__main__":
    main()
