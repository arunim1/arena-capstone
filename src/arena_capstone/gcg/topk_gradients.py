# Glen Taggart / nqgl if there are any issues/questions


from transformers import AutoModelForCausalLM, AutoTokenizer

from arena_capstone.gcg.embedding_model import (
    EmbeddingFriendlyCausalForLM,
    EmbeddingFriendlyModel,
    Batch,
)
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

    def top_k_substitutions(
        self,
        k: int,
    ):
        raise NotImplementedError()
