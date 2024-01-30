from json import load
import arena_capstone.gcg.topk_gradients as topkgrad
from arena_capstone.gcg.embedding_model import EmbeddingFriendlyCausalForLM

from transformers import AutoModelForCausalLM, AutoTokenizer

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


@dataclass
class GCGConfig:
    suffix: Int[Tensor, "batch seq"]
    k: int
    prefix_str: str
    target_str: str
    batch_size: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    T: int = 1000
    modelname: str = "gpt2"


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
        prefixes = self.tokenizer.encode_plus(self.cfg.prefix_str).input_ids
        prefixes = [torch.tensor(prefixes, dtype=torch.long, device=self.cfg.device)]

        targets = self.tokenizer.encode_plus(self.cfg.target_str).input_ids
        targets = [torch.tensor(targets, dtype=torch.long, device=self.cfg.device)]

        for _ in range(self.cfg.T):  # repeat T times
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
                    targets=targets * self.cfg.k,
                    reduce_over_batch=False,
                )
                losses = losses.reshape(self.cfg.batch_size, -1).sum(dim=-1)

            best_suffix_idx = torch.argmin(losses)
            best_suffix = next_suffixes[best_suffix_idx]

            self.suffix = best_suffix
            if print_between:
                print("loss: ", losses[best_suffix_idx].item())
                print("    ", self.tokenizer.decode(best_suffix))
            del token_grad_batch.suffix_tensor


def main():
    cfg = GCGConfig(
        suffix=torch.randint(0, 50257, (6,)),
        prefix_str="The cat",
        target_str=" is a dawg",
        batch_size=50,
        T=5000,
        k=50,
    )
    gcg = GCG(cfg=cfg, model=AutoModelForCausalLM.from_pretrained("gpt2"))
    gcg.gcg(print_between=True)


if __name__ == "__main__":
    main()
