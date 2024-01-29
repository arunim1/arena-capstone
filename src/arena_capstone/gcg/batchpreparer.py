from sympy import N
import torch as t

torch = t
import transformers
from typing import List, Tuple, Dict, Union, Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Batch:
    tokenized: Int[Tensor, "batch seq d_model"]
    target_mask: Bool[Tensor, "batch seq"]
    suffix_tensor: Float[Tensor, "suffix_length d_model"]


class BatchPreparer:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        suffix_length: int,
        k: int,
    ):

        self.suffix_length = suffix_length
        self.model: transformers.PreTrainedModel = model
        self.k = k
        self.suffix = t.randint(0, model.config.vocab_size, (suffix_length,))

    def prepare_batch_from_strings(
        self, prefix_strings: List[str], target_strings: List[str]
    ):
        raise NotImplementedError
        # prefixes_tokenized = self.model.tokenize(prefix_strings)

        # targets_tokenized = self.model.tokenize(target_strings)

        return self.prepare_batch_from_tokens(prefixes_tokenized, targets_tokenized)

    def prepare_batch_from_tokens(
        self,
        prefixes_tokenized: List[Int[Tensor, "seq_target"]],
        targets_tokenized: List[Int[Tensor, "seq_target"]],
    ):
        """ """
        assert len(prefixes_tokenized) == len(targets_tokenized)
        prefix_lengths = [prefix.shape[0] for prefix in prefixes_tokenized]
        target_lengths = [target.shape[0] for target in targets_tokenized]

        max_length = self.suffix.shape[0] + max(
            [plen + tlen for plen, tlen in zip(prefix_lengths, target_lengths)]
        )

        raise NotImplementedError

    def prepare_batch_from_embeddings(
        self,
        prefixes: List[Float[Tensor, "seq_target d_model"]],
        targets: List[Float[Tensor, "seq_target d_model"]],
    ):



