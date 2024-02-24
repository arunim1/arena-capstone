from dataclasses import dataclass
from typing import Tuple
import torch
from torch import Tensor
from transformers import LlamaForCausalLM
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM
import torch.nn.functional as F
from typing import Optional
from arena_capstone.soft_suffix.sched_config import SchedConfig


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


@dataclass
class GumbelSoftmaxConfig(SchedConfig):
    tau: float = 1
    hard: bool = False
    tau_backward: float = None
    noise_scale: float = 1
    bad_words_ids: Optional[Tuple[int]] = (1, 2, 32_000)  # TODO check this
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
    noise_annealing: float = 0.99

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
        d["hard"] = self.harden_range is None or (
            self.harden_range[1] - self.harden_range[0]
        ) <= (run_num % self.harden_range[1])

        d["noise_scale"] = self.noise_scale * self.noise_annealing
        if d["hard"]:
            if self.noise_in_hard is not None:
                d["noise_scale"] = self.noise_in_hard
            d["tau"] = self.tau_hard or self.tau

        else:
            if self.noise_in_hard is not None:
                d["noise_scale"] = self.temp_noise
            loss = kwargs["loss"]
            if loss < -4.5:
                d["temp_tau_soft"] = min(
                    self.max_tau,
                    max(self.min_tau, self.temp_tau_soft * self.tau_annealing_rate),
                )
            else:
                d["temp_tau_soft"] = min(
                    self.max_tau,
                    max(self.min_tau, self.temp_tau_soft / (self.tau_annealing_rate)),
                )
            d["tau"] = d["temp_tau_soft"]

        # if self.tau < 4:
        #     d["tau_backward"] = 2 + self.tau / 2
        return d


def main():
    dist = torch.rand(8, 16, 64)
    d1 = gumbel_softmax(dist, hard=True, noise_scale=0)
    d2 = gumbel_softmax(dist, tau=0.000001, noise_scale=0)
    assert torch.allclose(d1, d2, atol=0.01)


if __name__ == "__main__":
    main()
    print("passed")
