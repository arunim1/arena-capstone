import torch.nn as nn
import torch.nn.functional as F
from arena_capstone.soft_suffix.gumbel_softmax import GumbelSoftmaxConfig, Tensor
from arena_capstone.soft_suffix.sched_config import SchedConfig
import torch

from dataclasses import dataclass
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from arena_capstone.soft_suffix.optim import OptimCfg


@dataclass
class SuffixConfig(SchedConfig):
    optim: OptimCfg
    gumbel_config: GumbelSoftmaxConfig
    suffix_len: int = 5
    iterative_freeze: bool = False
    update_size_from_probs: float = 50


class Suffix(nn.Module):
    def __init__(
        self,
        cfg: SuffixConfig,
        suffix_logits=None,
    ):
        super().__init__()
        if suffix_logits is None:
            suffix_logits = torch.zeros(1, cfg.suffix_len, 32001, device=DEVICE)
        else:
            if suffix_logits.ndim == 2:
                suffix_logits = suffix_logits.unsqueeze(0)
        self.suffix_logits = nn.Parameter(suffix_logits.clone())
        self.tau = cfg.gumbel_config.tau
        self.tau_backward = cfg.gumbel_config.tau_backward
        self.hard = cfg.gumbel_config.hard
        self.noise_scale = cfg.gumbel_config.noise_scale
        self.cfg = cfg

    def forward(self, batch_size, tau=None) -> Tensor:
        return self.cfg.gumbel_config.gumbel_softmax(
            self.suffix_logits.expand(batch_size, -1, -1), tau=tau
        )

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
                        f"suffix/probs/hists/{pos}/{distname}": (
                            wandb.Histogram(
                                dist[:, pos, :].float().detach().cpu().numpy()
                            ),
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
            - self.suffix_logits.data.median(dim=-1, keepdim=True).values
        )

    def update_suffix_from_probs(self, update_probs: Tensor):
        self.suffix_logits.data[:] = (
            self.suffix_logits.data + update_probs * self.cfg.update_size_from_probs
        )
        self.cfg.optim.init_optim(self.parameters())
