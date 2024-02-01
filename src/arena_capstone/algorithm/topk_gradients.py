# Glen Taggart / nqgl if there are any issues/questions

# %%
from typing import List, Optional, Set, Tuple, Union

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from embedding_model import EmbeddedBatch


def top_k_substitutions(
    batch: EmbeddedBatch,
    k: int,
) -> List[Set[int]]:
    """
    Returns a list of sets of the top k substitutions for each token in suffix_tokens. If prefixes has length > 1, we sum over the gradients of each prefix. Otherwise, we just use the gradient of the single prefix (eg. GCG algorithm).
    In the paper, the output is denoted as \mathcal{X}.

    Output: List[Set[int]], where len(List) == len(suffix_tokens) and size(Set) == k
    """

    assert batch.suffix_tensor.grad is not None

    topk_values, topk_indices = torch.topk(
        batch.suffix_tensor.grad,
        k=k,
        dim=-1,
        largest=False,
    )

    return topk_indices


def sample_replacements(
    replacements: Int[Tensor, "suffix_len k"],
    suffix: Int[Tensor, "suffix_len"],
    batch_size: int,
):
    suffix_len, k = replacements.shape
    next_suffixes = torch.zeros(
        batch_size, suffix_len, device=replacements.device, dtype=torch.long
    )
    next_suffixes[:] = suffix

    batch_range = torch.arange(batch_size, device=replacements.device)

    i = torch.randint(0, suffix_len, size=(batch_size,))
    which_topk = torch.randint(0, k, size=(batch_size,))
    next_suffixes[batch_range, i] = replacements[i, which_topk]

    return next_suffixes


def main():
    test_replacements = torch.tensor([[1, 2], [4, 5], [0, 1]])
    test_suffix = torch.tensor([7, 8, 9])
    sample = sample_replacements(test_replacements, test_suffix, 2)


if __name__ == "__main__":
    main()

# %%
