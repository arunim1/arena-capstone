import os

from arena_capstone.rlhf_trojan_competition.src.models.reward_model import (
    RewardModel as RLHFRewardModel,
)
from arena_capstone.rlhf_trojan_competition.src.datasets import PromptOnlyDataset

import torch as t
from torch import Tensor

token = os.environ.get("HF_TOKEN")


class RewardModel:
    def __init__(self, model_path="ethz-spylab/reward_model"):
        print("Loading reward model")
        self.device = "cuda"
        self.reward_model = RLHFRewardModel.from_pretrained(model_path)
        self.reward_model = self.reward_model.half()
        self.reward_model = self.reward_model.to(self.device)

    def calculate_reward(self, input_ids: Tensor, attention_masks: Tensor):
        return (
            self.reward_model(input_ids, attention_mask=attention_masks)
            .end_rewards.flatten()
            .detach()
            .cpu()
            .numpy()
        )


def main(device="cuda"):
    print("Loading reward model")
    reward_model = RewardModel()
    print("reward model loaded")


if __name__ == "__main__":
    main(device="cuda")
