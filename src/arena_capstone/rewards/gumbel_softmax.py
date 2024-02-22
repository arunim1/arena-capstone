import torch
from torch import Tensor
from transformers import LlamaForCausalLM
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM
import torch.nn.functional as F

# from arena_capstone.algorithm
# from arena_capstone.algorithm


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    tau_backward: float = None,
    hard: bool = False,
    noise_scale: float = 1,
):
    # tau_backward = tau_backward or tau
    gumbels = (
        -torch.empty_like(logits).exponential_().log() * noise_scale
    )  # ~Gumbel(0,1)

    input_gumbels = (logits / noise_scale + gumbels) / tau  # ~Gumbel(logits,tau)

    y_soft = F.softmax(input_gumbels, dim=-1)
    if hard:
        y_hard = y_soft.max(-1, keepdim=True)[0].eq(y_soft).float()
        return y_hard - y_soft.detach() + y_soft
    if not tau_backward:
        return y_soft
    input_gumbels_bak = (logits / noise_scale + gumbels) / tau_backward
    y_soft_bak = F.softmax(input_gumbels_bak, dim=-1)
    return y_soft.detach() + y_soft_bak - y_soft_bak.detach()
