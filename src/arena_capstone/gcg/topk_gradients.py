# Glen Taggart / nqgl if there are any issues/questions


from transformers import AutoModelForCausalLM, AutoTokenizer

from arena_capstone.gcg.embedding_model import (
    EmbeddingFriendlyCausalForLM,
    EmbeddingFriendlyModel,
    Batch,
)
from arena_capstone.gcg.token_gradients import TokenGradients

from typing import List, Set, Tuple, Union, Optional
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
    ) -> List[Set[int]]:
        '''
        Returns a list of sets of the top k substitutions for each token in suffix_tokens. If prefixes has length > 1, we sum over the gradients of each prefix. Otherwise, we just use the gradient of the single prefix (eg. GCG algorithm). 
        In the paper, the output is denoted as \mathcal{X}.

        Output: List[Set[int]], where len(List) == len(suffix_tokens) and size(Set) == k
        '''
        k = self.k if k is None else k
        token_grad_batch = self.token_gradient_generator.get_token_gradients(
            prefixes, suffix_tokens, targets
        )

        # realized loss is already summed if get_loss is correct 
        # hopefully this is true: token_grad_batch.suffix_tensor.grad [suffix_len, vocab_size]?

        topk_values, topk_indices = torch.topk(
            token_grad_batch.suffix_tensor.grad,
            k=k,
            dim = -1,
            largest=False,
        )

        # convert topk_indices which is [suffix_len, k] to a list of length suffix_len, where each element is a set of the top k indices for that token

        topk_indices = topk_indices.tolist()
        topk_indices = [set(indices) for indices in topk_indices]

        token_grad_batch.suffix_tensor.requires_grad = False
        token_grad_batch.suffix_tensor.grad = None
        
        return topk_indices



        
