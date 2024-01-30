from typing import List, Tuple, Union, Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
import torch

# from nqgl.mlutils.norepr import fastpartial
import torch.nn.functional as F
from functools import partial
from transformers import AutoModelForCausalLM
from yaml import Token
from arena_capstone.gcg.embeddingmodel import (
    EmbeddingFriendlyCausalForLM,
    EmbeddingFriendlyModel,
    Batch,
)


class TokenGradients:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        suffix: Int[Tensor, "batch seq"],
        embedding_model: Optional[EmbeddingFriendlyModel],
        capture_one_hot=True,
        capture_embeddings=False,
    ):
        """ """
        assert callable(model)
        assert capture_one_hot or capture_embeddings
        self.model = model
        self.suffix = suffix
        self.embedding_model = (
            EmbeddingFriendlyCausalForLM(model)
            if embedding_model is None
            else embedding_model
        )

    def get_loss(
        self,
        batch: Batch,
        targets: List[Int[Tensor, "seq"]],
    ):
        # returns loss without backpropagating (because that would be a dumb design choice NGL)
        target_ids = torch.cat(targets, dim=0)
        logprobs = torch.log_softmax(batch.logits, dim=-1)
        loss = F.cross_entropy(logprobs[:, :-1][batch.target_mask[:, 1:]], target_ids)
        # loss = torch.sum(logprobs[target_mask][target_tokens.flatten()]) ?

        # F.cross_entropy(logits[:, :-1][target_mask[:, 1:]], target_ids)
        # sequence P S G G G
        # logits   S G G G 0
        # target   P S G G G
        # tmask    0 0 1 1 1

        # target[tmask]         G G G
        # logits[:-1][tmask[1:]]G G G

        # L(x[1:n]) = -log(p(x[n+1] | x[1:n))

        return loss

    def get_token_gradients(self, input_tokens, target_tokens, target_mask) -> Batch:
        batch = self.embedding_model.splice_suffix(
            input_tokens, self.suffix, target_tokens, get_logits=True
        )
        batch.suffix_tensor.grad = None  # zero grad
        loss = self.get_loss(batch)
        loss.backward()
        return batch


class TopKGradients:
    def top_k_substitutions(
        self,
        tokens: Int[Tensor, "batch seq"],
        targets_mask: Bool[Tensor, "batch seq"],
        prefixes_mask,
        k: int,
    ):

        k = self.k if k is None else k
        embedded_tokens = self.model.embeddings(tokens)
        embedded_tokens.requires_grad = True
        logits = self.model(embedded_tokens)

        target_logprobs = logits[
            targets_mask
        ]  # this doesn't work bc targets are variable length
