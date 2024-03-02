import torch.nn as nn
import torch.nn.functional as F
from arena_capstone.soft_suffix.gumbel_softmax import GumbelSoftmaxConfig, Tensor
from arena_capstone.soft_suffix.sched_config import SchedConfig
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from arena_capstone.soft_suffix.optim import OptimCfg
from arena_capstone.rewards.filter_ascii_no_whitespace_indices import (
    filter_ascii_no_whitespace_indices_return_bad,
)


@dataclass
class SuffixConfig(SchedConfig):
    optim: OptimCfg
    gumbel_config: GumbelSoftmaxConfig
    suffix_len: int = 5
    iterative_freeze: bool = False
    update_size_from_probs: float = 50
    update_reset_optim: bool = True
    update_const_scale_logits: float = 1
    ceil_scale: int = False
    l1_coeff: float = 0.0
    max_grad_norm: float = 1
    max_grad_value: float = 0.1


class Suffix(nn.Module):
    def __init__(
        self,
        cfg: SuffixConfig,
        tokenizer: AutoTokenizer,
        suffix_logits=None,
    ):
        super().__init__()
        if suffix_logits is None:
            suffix_logits = 0.001 * torch.rand(
                size=(1, cfg.suffix_len, 32001), device=DEVICE
            )
        else:
            if suffix_logits.ndim == 2:
                suffix_logits = suffix_logits.unsqueeze(0)
        self.suffix_logits = nn.Parameter(suffix_logits.clone())
        self.tau = cfg.gumbel_config.tau
        self.tau_backward = cfg.gumbel_config.tau_backward
        self.hard = cfg.gumbel_config.hard
        self.noise_scale = cfg.gumbel_config.noise_scale
        self.cfg = cfg
        self.cfg.gumbel_config.bad_words_ids = list(
            set(cfg.gumbel_config.bad_words_ids)
            | set(filter_ascii_no_whitespace_indices_return_bad(tokenizer))
        )
        # self.suffix_logits.data[..., cfg.gumbel_config.bad_words_ids] = -float("inf")

    def forward(self, batch_size, tau=None, noise_scale=None) -> Tensor:
        out = self.cfg.gumbel_config.gumbel_softmax(
            self.suffix_logits.expand(batch_size, -1, -1),
            tau=tau,
            noise_scale=noise_scale,
        )
        assert out.dtype == torch.bfloat16
        return out

    @property
    def optim(self):
        if self.cfg.optim.optim is None:
            self.cfg.optim.init_optim(self.parameters())
        return self.cfg.optim.optim

    def log(self, run_num: int, loghists: bool = True, positional: bool = True):
        dists = {
            "logits": self.suffix_logits,
            "gumbel": self(1),
            "softmax": self.cfg.gumbel_config.gumbel_softmax(
                self.suffix_logits, noise_scale=0, hard=False
            ),
        }
        ops = {
            "max": lambda x, dim: torch.max(x, dim=dim).values,
            "mean": lambda x, dim: torch.mean(x, dim=dim),
            "min": lambda x, dim: torch.min(x, dim=dim).values,
            "median": lambda x, dim: torch.median(x.float(), dim=dim).values,
        }
        logdict = {
            k: v
            for pos in range(self.suffix_logits.shape[1])
            for distname, dist in dists.items()
            for k, v in {
                **{
                    f"suffix/probs/{distname}/{opname}/{pos}": op(
                        dist[:, pos, :], dim=-1
                    ).item()
                    for opname, op in ops.items()
                    if positional
                },
                **(
                    {
                        f"suffix.probs.hists.{distname}.{pos}": wandb.Histogram(
                            dist[:, pos, :].float().detach().cpu().numpy()
                        )
                    }
                    if loghists
                    else {}
                ),
            }.items()
        }

        wandb.log(
            logdict,
            step=run_num,
        )

    def adjust_logits_for_stability(self):
        self.suffix_logits.data = (
            self.suffix_logits.data
            - self.suffix_logits.data.float()  # no bfloat median D:
            .median(dim=-1, keepdim=True)
            .values.bfloat16()
        )
        if self.cfg.ceil_scale:
            ceiling = self.cfg.ceil_scale
            top = torch.max(self.suffix_logits.data, dim=-1, keepdim=True).values
            self.suffix_logits.data = (
                torch.where(top > ceiling, ceiling / top, 1) * self.suffix_logits.data
            )

        self.suffix_logits.data *= self.cfg.update_const_scale_logits
        # .clamp_(None, 50)

        # self.suffix_logits.data.clamp_(None, 50)

    def update_suffix_from_probs(self, update_probs: Tensor):
        self.adjust_logits_for_stability()
        self.suffix_logits.data[:] = (
            self.suffix_logits.data + update_probs * self.cfg.update_size_from_probs
        )
        if self.cfg.update_reset_optim:
            self.cfg.optim.init_optim(self.parameters())

    def penalty(self):
        # return 0
        return F.relu(self.suffix_logits).mean() * self.cfg.l1_coeff

    def pre_step(self, run_num: int = None):
        if run_num is not None:
            norm = self.suffix_logits.grad.norm()
            maxval = self.suffix_logits.grad.max()
            wandb.log(
                {
                    "suffix/grad/norm": norm.item(),
                    "suffix/grad/max": maxval.item(),
                },
                step=run_num,
            )
        torch.nn.utils.clip_grad_norm_(
            self.suffix_logits, max_norm=self.cfg.max_grad_norm
        )
        torch.nn.utils.clip_grad_value_(
            self.suffix_logits, clip_value=self.cfg.max_grad_value
        )


def main():
    torch.set_default_dtype(torch.bfloat16)
    tens = torch.rand(1, 5, 32001, dtype=torch.bfloat16, device="cuda")
    q = tens.quantile(0.5)


if __name__ == "__main__":
    main()
