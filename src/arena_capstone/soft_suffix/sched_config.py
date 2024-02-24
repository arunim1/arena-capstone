import wandb


class SchedConfig:
    pass

    def update_config_and_log(self, cfgname, d: dict, run_num: int):
        for k, v in d.items():
            setattr(self, k, v)
        wandb.log({f"{cfgname}/{k}": v for k, v in d.items()}, step=run_num)

    def scheduler_step(self, run_num, name="cfg", **kwargs):
        for attrname in dir(self):
            if attrname.startswith("_"):
                continue
            attr = getattr(self, attrname)
            if hasattr(attr, "schedule"):
                attr.update_config_and_log(
                    f"{name}.{attrname}", attr.schedule(run_num, **kwargs), run_num
                )
            if isinstance(attr, SchedConfig):
                attr.scheduler_step(run_num, f"{name}.{attrname}", **kwargs)
