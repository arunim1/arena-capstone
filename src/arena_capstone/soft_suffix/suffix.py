import torch.nn as nn
import torch.nn.functional as F
from arena_capstone.soft_suffix.gumbel_softmax import GumbelSoftmaxConfig, Tensor
from arena_capstone.soft_suffix.sched_config import SchedConfig
import torch

from dataclasses import dataclass
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class SuffixConfig(SchedConfig):
    gumbel_config: GumbelSoftmaxConfig
    suffix_len: int = 5


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

    def log_historgram(self, run_num: int):
        suffix = self(1)
        suffix_softmax = F.softmax(self.suffix_logits, dim=-1)
        suffix_softmax = self.cfg.gumbel_config.gumbel_softmax(
            self.suffix_logits, noise_scale=0, hard=False
        )

        for i in range(suffix.shape[1]):
            max_probs_g = torch.max(suffix[:, i, :], dim=-1)
            max_probs = torch.max(suffix_softmax[:, i, :], dim=-1)
            mean_max_g = max_probs_g.values.mean()
            mean_max_sm = max_probs.values.mean()
            # std_max_g = max_probs_g.values.std()
            wandb.log(
                {
                    f"suffix.probs.hists.gumbel.{i}": wandb.Histogram(
                        suffix[:, i, :].float().detach().cpu().numpy()
                    ),
                    f"suffix.probs.hists.softmax.{i}": wandb.Histogram(
                        suffix_softmax[:, i, :].float().detach().cpu().numpy()
                    ),
                    f"suffix.probs.max.means.gumbel.{i}": mean_max_g,
                    f"suffix.probs.max.means.softmax.{i}": mean_max_sm,
                    # f"suffix.maxprob.sts.gumbel.{i}": std_max_g,
                },
                step=run_num,
            )
