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
    dim: int = -1,
):
    # tau_backward = tau_backward or tau
    gumbels = (
        -torch.empty_like(logits).exponential_().log() * noise_scale
    )  # ~Gumbel(0,1)

    input_gumbels = (logits + gumbels * noise_scale) / tau  # ~Gumbel(logits,tau)

    y_soft = F.softmax(input_gumbels, dim=dim)
    assert (y_soft.sum(dim=dim) < 2).all()
    if hard:
        # y_hard = y_soft.max(-1, keepdim=True)[0].eq(y_soft).bfloat16()
        assert dim == -1
        y_hard = torch.zeros_like(y_soft).scatter_(
            dim=dim, index=y_soft.argmax(dim=dim, keepdim=True), value=1.0
        )
        out = y_hard.detach() + (y_soft - y_soft.detach())
        assert (out.sum(dim=dim) < 2).all(), y_soft.argmax(dim=-1)

    elif not tau_backward:
        out = y_soft
        assert (out.sum(dim=dim) < 2).all()

    else:
        input_gumbels_bak = (logits + gumbels * noise_scale) / tau_backward
        y_soft_bak = F.softmax(input_gumbels_bak, dim=dim)
        out = y_soft.detach() + y_soft_bak - y_soft_bak.detach()
        assert (out.sum(dim=dim) < 2).all()
    return out


def main():
    dist = torch.rand(8, 16, 64)
    d1 = gumbel_softmax(dist, hard=True, noise_scale=0)
    d2 = gumbel_softmax(dist, tau=0.000001, noise_scale=0)
    assert torch.allclose(d1, d2, atol=0.01)


if __name__ == "__main__":
    main()
    print("passed")
