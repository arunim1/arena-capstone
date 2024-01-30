# Glen Taggart / nqgl if there are any issues/questions


from transformers import AutoModelForCausalLM, AutoTokenizer

from arena_capstone.gcg.embedding_model import (
    EmbeddingFriendlyCausalForLM,
    EmbeddingFriendlyModel,
    Batch,
)
from arena_capstone.gcg.token_gradients import TokenGradients

from typing import List, Tuple, Union, Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
import torch


class TopKGradients:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        suffix: Int[Tensor, "batch seq"],
        embedding_model: Optional[EmbeddingFriendlyModel] = None,
        k: Optional[int] = None,
    ):
        assert callable(model)
        self.model = model
        self.suffix = suffix
        self.embedding_model = (
            EmbeddingFriendlyCausalForLM(model)
            if embedding_model is None
            else embedding_model
        )
        self.k = k
        self.token_gradient_generator = TokenGradients(model, embedding_model)

    def top_k_substitutions(
        self,
        prefixes: List[Int[Tensor, "prefix_len"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        targets: List[Int[Tensor, "target_len"]],
        k: Optional[int] = None,
    ):
        k = self.k if k is None else k
        token_grad_batch = self.token_gradient_generator.get_token_gradients(
            prefixes, suffix_tokens, targets
        )
        topk = torch.topk(
            -1 * token_grad_batch.suffix_tensor.grad,
            k=k,
        )
        indices = topk.indices
        torch.
        tok_grads.suffix_tensor.requires_grad = False
        tok_grads.suffix_tensor.grad = None
        suffix = tokens
