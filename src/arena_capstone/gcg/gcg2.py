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
import transformers
from typing import List, Set, Tuple, Union, Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
from dataclasses import dataclass
import torch
import einops
import wandb
from tqdm import tqdm
from colorama import Fore, Back, Style

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GCGConfig:
    suffix: Int[Tensor, "batch seq"]
    k: int
    prefix_str: str
    target_str: str
    batch_size: int
    device: str = DEVICE
    T: int = 1000
    modelname: str = "gpt2"
    use_wandb: bool = False


class GCG:
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
            wandb.init(project="gcg")
            table = wandb.Table(columns=["prefix", "suffix", "completion", "step"])

        prefixes = self.tokenizer.encode_plus(self.cfg.prefix_str).input_ids
        prefixes = [torch.tensor(prefixes, dtype=torch.long, device=self.cfg.device)]

        targets = self.tokenizer.encode_plus(self.cfg.target_str).input_ids
        targets = [torch.tensor(targets, dtype=torch.long, device=self.cfg.device)]

        for run_num in range(self.cfg.T):  # repeat T times
            token_grad_batch = self.token_gradient_generator.get_token_gradients(
                prefixes, self.suffix, targets
            )
            replacements = topkgrad.top_k_substitutions(token_grad_batch, self.cfg.k)
            next_suffixes = topkgrad.sample_replacements(
                replacements, self.suffix, self.cfg.batch_size
            )

            # make "prompts", concatenated prefix with each new suffix
            repeated_prefix = einops.repeat(
                prefixes[0], "seq -> batch seq", batch=self.cfg.batch_size
            )
            repeated_target = einops.repeat(
                targets[0], "seq -> batch seq", batch=self.cfg.batch_size
            )

            sequences = torch.cat(
                [repeated_prefix, next_suffixes, repeated_target], dim=1
            )

            # masking everything but targets.
            target_len = targets[0].shape[0]
            target_mask = torch.zeros_like(sequences).bool()
            target_mask[:, -target_len:] = True

            with torch.inference_mode():
                # get the loss of the model on the sequences
                logits = self.model(sequences).logits
                batch_for_loss = EmbeddedBatch(
                    embeddings=None,
                    target_mask=target_mask,
                    suffix_tensor=None,
                    logits=logits,
                )

                losses = self.token_gradient_generator.get_loss(
                    batch=batch_for_loss,
                    targets=targets * self.cfg.batch_size,
                    reduce_over_batch=False,
                )
                losses = losses.reshape(self.cfg.batch_size, -1).mean(dim=-1)

            best_suffix_idx = torch.argmin(losses)
            best_suffix = next_suffixes[best_suffix_idx]

            self.suffix = best_suffix
            if print_between:
                if run_num % 10 == 0:
                    generate(self)
                print("    ", self.tokenizer.decode(best_suffix))
                print("loss opt:", losses[best_suffix_idx].item())
            if self.cfg.use_wandb:
                wandb.log({"loss": losses[best_suffix_idx].item()}, step=run_num + 1)
                # wandb.log({"suffix": self.tokenizer.decode(best_suffix)})
                if run_num % 10 == 0:
                    table.add_data(
                        self.cfg.prefix_str,
                        self.tokenizer.decode(best_suffix),
                        get_completion(self),
                        str(run_num + 1),
                    )

            del token_grad_batch.suffix_tensor

        if self.cfg.use_wandb:
            wandb.log({"Table": table})
            wandb.finish()


def get_completion(gcg: GCG):
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
    target = gcg.tokenizer.encode_plus(gcg.cfg.target_str).input_ids

    all_ones_mask = torch.ones_like(tokens).bool()

    gen = gcg.model.generate(
        tokens.unsqueeze(0),
        max_length=tokens.shape[0] + len(target),
        attention_mask=all_ones_mask.unsqueeze(0),
        pad_token_id=gcg.tokenizer.pad_token_id,
    ).squeeze()

    return gcg.tokenizer.decode(gen[tokens.shape[0] :])


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

    all_ones_mask = torch.ones_like(tokens).bool()

    target = gcg.tokenizer.encode_plus(gcg.cfg.target_str).input_ids
    gen = gcg.model.generate(
        tokens.unsqueeze(0),
        max_length=tokens.shape[0] + len(target),
        attention_mask=all_ones_mask.unsqueeze(0),
        pad_token_id=gcg.tokenizer.pad_token_id,
    ).squeeze()

    prefix_text = gcg.tokenizer.decode(tokens[: -gcg.suffix.shape[0]])
    suffix_text = gcg.tokenizer.decode(tokens[-gcg.suffix.shape[0] :])
    generated_text = gcg.tokenizer.decode(gen[tokens.shape[0] :])

    print(
        f"goal:     "
        + Fore.BLUE
        + prefix_text
        + Fore.RED
        + suffix_text
        + Fore.GREEN
        + gcg.cfg.target_str
    )
    print(Style.RESET_ALL, end="")

    print(
        f"generated:"
        + Fore.BLUE
        + prefix_text
        + Fore.RED
        + suffix_text
        + Fore.GREEN
        + generated_text
    )
    print(Style.RESET_ALL, end="")


def main():
    cfg = GCGConfig(
        suffix=torch.randint(0, 50257, (6,), device=DEVICE),
        prefix_str="That cat over there",
        target_str=" is a dawg if",
        batch_size=512,
        T=2000,
        k=500,
        use_wandb=True,
    )
    gcg = GCG(cfg=cfg, model=AutoModelForCausalLM.from_pretrained("gpt2"))
    gcg.gcg(print_between=(not cfg.use_wandb))

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
