from dataclasses import dataclass
from typing import Tuple
import torch
from torch import Tensor
from transformers import LlamaForCausalLM
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM
import torch.nn.functional as F
from typing import Optional
from arena_capstone.soft_suffix.sched_config import SchedConfig
import math


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    tau_backward: float = None,
    hard: bool = False,
    noise_scale: float = 1,
    dim: int = -1,
):
    # tau_backward = tau_backward or tau
    assert tau != 0
    gumbels = (
        -torch.empty_like(logits).exponential_().log() * noise_scale
    )  # ~Gumbel(0,1)
    assert logits.dtype == torch.bfloat16
    input_gumbels = (logits + gumbels * noise_scale) / tau  # ~Gumbel(logits,tau)

    y_soft = F.softmax(input_gumbels, dim=dim)
    assert (y_soft.sum(dim=dim) < 32008).all(), y_soft.sum(dim=dim)
    if hard:
        # y_hard = y_soft.max(-1, keepdim=True)[0].eq(y_soft).bfloat16()
        assert dim == -1
        y_hard = torch.zeros_like(y_soft).scatter_(
            dim=dim, index=y_soft.argmax(dim=dim, keepdim=True), value=1.0
        )
        out = y_hard.detach() + (y_soft - y_soft.detach())
        assert (out.sum(dim=dim) < 32008).all(), (
            y_soft.argmax(dim=-1),
            out.sum(dim=dim),
        )

    elif not tau_backward:
        out = y_soft
        assert (out.sum(dim=dim) < 32008).all(), out.sum(dim=dim)

    else:
        input_gumbels_bak = (logits + gumbels * noise_scale) / tau_backward
        y_soft_bak = F.softmax(input_gumbels_bak, dim=dim)
        out = y_soft.detach() + y_soft_bak - y_soft_bak.detach()
        assert (out.sum(dim=dim) < 32008).all(), out.sum(dim=dim)
    assert out.dtype == torch.bfloat16
    return out


@dataclass
class GumbelSoftmaxConfig(SchedConfig):
    tau: float = 1
    hard: bool = False
    tau_backward: float = None
    noise_scale: float = 1
    bad_words_ids: Optional[Tuple[int]] = (0, 1, 2, 32_000)  # TODO check this
    min_tau: Optional[float] = 0.01
    tau_annealing_rate: Optional[float] = 0.95
    harden_range: Optional[Tuple[int, int]] = None
    noise_in_hard: float = 0
    tau_hard: float = None
    temp_tau_soft: float = None
    temp_noise: float = None
    scale_noise: bool = False
    max_scaled_noise: float = 1
    max_tau: float = 20
    noise_annealing: float = 1
    loss_threshold: float = -3
    sine_tau: list = None

    def gumbel_softmax(self, logits, tau=None, noise_scale=None, hard=None):
        if self.bad_words_ids is not None:
            logit_mask = torch.zeros(logits.shape[-1], device=logits.device)
            logit_mask[torch.tensor(self.bad_words_ids, dtype=torch.int64)] = torch.inf
            logits = logits - logit_mask
        return gumbel_softmax(
            logits,
            tau=tau or self.tau,
            hard=hard if hard is not None else self.hard,
            tau_backward=self.tau_backward,
            noise_scale=(
                noise_scale
                if noise_scale is not None
                else (
                    self.noise_scale
                    if not self.scale_noise
                    else min(self.noise_scale * self.tau, self.max_scaled_noise)
                )
            ),
            dim=-1,
        )

    def __post_init__(self):
        self.temp_tau_soft = self.temp_tau_soft or self.tau
        self.temp_noise = self.noise_scale

    def schedule(self, run_num: int, **kwargs) -> dict:
        d = {}
        d["hard"] = (
            self.hard
            if self.harden_range is None
            else (self.harden_range[1] - self.harden_range[0])
            <= (run_num % self.harden_range[1])
        )
        noise_annealing = self.noise_annealing

        if d["hard"]:
            tau = self.tau_hard or self.tau
            if self.noise_in_hard is not None:
                d["noise_scale"] = self.noise_in_hard
            anneal_tau = lambda t: t * self.tau_annealing_rate
        else:
            tau = self.temp_tau_soft
            if self.noise_in_hard is not None:
                d["noise_scale"] = self.temp_noise
            loss = kwargs["loss"]
            anneal_tau = lambda t: (
                tau * self.tau_annealing_rate
                if loss < self.loss_threshold
                else tau / self.tau_annealing_rate
            )

        clamp_tau = lambda t: min(self.max_tau, max(self.min_tau, t))
        tau = clamp_tau(anneal_tau(tau))
        noise_scale = d.get("noise_scale", self.noise_scale) * noise_annealing

        d["tau"] = tau
        d["noise_scale"] = noise_scale
        if d["hard"]:
            d["tau_hard"] = tau
            d["noise_in_hard"] = noise_scale
        else:
            d["temp_tau_soft"] = tau
            # d["temp_noise"] = noise_scale disable this bc not clear we want to anneal soft noise
        if self.sine_tau:
            period, low, high = self.sine_tau
            d["tau"] = (high - low) * (
                1 + math.sin(2 * math.pi * run_num / period)
            ) / 2 + low
        return d


def main():
    dist = torch.rand(8, 16, 64)
    d1 = gumbel_softmax(dist, hard=True, noise_scale=0)
    d2 = gumbel_softmax(dist, tau=0.000001, noise_scale=0)
    assert torch.allclose(d1, d2, atol=0.01)


if __name__ == "__main__":
    main()
    print("passed")
