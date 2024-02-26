from arena_capstone.soft_suffix.sched_config import SchedConfig


import torch


from dataclasses import dataclass
from typing import Tuple


@dataclass
class OptimCfg(SchedConfig):
    optim_name: str = "RAdam"
    lr: float = 3e-1
    betas: Tuple[float, float] = (0.9, 0.99)
    momentum: float = 0.9
    weight_decay: float = 0
    eps: float = 1e-8
    nesterov: bool = True

    def __post_init__(self):
        self.optim: torch.optim.Optimizer = None

    def init_optim(self, params):
        self.optim = self._get(params)

    def _get(self, params):
        if self.optim_name == "RAdam":
            return torch.optim.RAdam(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
                eps=self.eps,
            )
        elif self.optim_name == "SGD":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        elif self.optim_name == "RProp":
            return torch.optim.Rprop(
                params,
                lr=1e-2,
                # etas=self.betas,
                step_sizes=(1e-6, 0.25),
            )

        else:
            raise ValueError(f"Unknown optimizer {self.optim_name}")

    def schedule(self, run_num, **kwargs):
        return {}

    def _post_cfg_updated(self, d):
        raise NotImplementedError


def main():
    ten = torch.tensor([1.0, 2.0, 3.3])
    ten = torch.nn.Parameter(ten)
    cfg = OptimCfg()
    cfg.init_optim([ten])
    print(cfg.optim)

    cfg.optim()
    cfg.optim()


if __name__ == "__main__":
    main()
