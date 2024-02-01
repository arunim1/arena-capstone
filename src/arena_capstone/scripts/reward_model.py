import os

from arena_capstone.rlhf_trojan_competition.src.models.reward_model import RewardModel as RLHFRewardModel
from arena_capstone.rlhf_trojan_competition.src.datasets import PromptOnlyDataset
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM
from arena_capstone.scripts.run_with_llama import get_llama
import torch as t
from torch import Tensor

token = os.environ.get("HF_TOKEN")

class RewardModel():
    def __init__(self, model_path="ethz-spylab/reward_model"):
        print("Loading reward model")
        self.device = "cuda"
        self.reward_model = RLHFRewardModel.from_pretrained(model_path)
        self.reward_model = self.reward_model.half()
        self.reward_model = self.reward_model.to(self.device)
        
    def embed_to_one_hot(self, tokens : Tensor):
        return t.nn.functional.one_hot(tokens, self.reward_model.config.vocab_size).float()
        
    def calculate_reward(self, input_ids : Tensor, attention_masks : Tensor):
        return self.reward_model(input_ids, attention_mask=attention_masks).end_rewards.flatten().detach().cpu().numpy()
    

def test(embedding_model, prompt, suffix, target, post_suffix=t.tensor([]).device("cuda")): # run only on cuda
    
    batch = embedding_model.splice_embedded_batch(prompt, suffix, post_suffix, target, get_logits=True)
    
    reward_model: RewardModel
    batch.logits
    wte = reward_model.get_input_embeddings()
    embedded = batch.logits @ wte.weight
    reward_output = reward_model(inputs_embeds=embedded)

    rewards, end_rewards = reward_output.rewards, reward_output.end_rewards

    maybe_loss = end_rewards.mean()
    

def main(device="cuda"):
    model, embedding_model, tokenizer = get_llama()
    prompt_tokens = t.randint(0, model.config.vocab_size, (1, 5), device=device)
    suffix_tokens = t.randint(0, model.config.vocab_size, (1, 5), device=device)
    target_tokens = t.randint(0, model.config.vocab_size, (1, 5), device=device)
    
    print("Loading reward model")
    reward_model = RewardModel()
    print("reward model loaded")
    
    test(embedding_model, reward_model, prompt_tokens, suffix_tokens, target_tokens)


if __name__ == "__main__":
    main(device="cuda")