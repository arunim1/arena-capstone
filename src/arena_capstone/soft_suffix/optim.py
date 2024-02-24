from arena_capstone.soft_suffix.sched_config import SchedConfig


import torch


from dataclasses import dataclass
from typing import Tuple


@dataclass
class OptimCfg(SchedConfig):
    optim: str = "RAdam"
    lr: float = 3e-1
    betas: Tuple[float, float] = (0.9, 0.99)
    momentum: float = 0.9
    weight_decay: float = 0
    eps: float = 1e-8

    def get(self, params):
        if self.optim == "RAdam":
            return torch.optim.RAdam(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                eps=self.eps,
            )
        if self.optim == "SGD":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optim}")
