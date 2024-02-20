import gc

import pandas as pd
import torch
import wandb
from tqdm import tqdm
from arena_capstone.scripts.run_with_llama import get_llama

from arena_capstone.rewards.reward_generator import (
    RewardGenerator,
    get_reward_generator,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def isinfnan(t):
    return torch.any(torch.isnan(t) | torch.isinf(t))


"""
outline of the completely random method: 
- no gradients, everything can be in inference mode
- create a set of suffixes which are one off from the current "best" suffix
- generate completions for each prompt (prefix + suffix + post_suffix)
- get the reward for each completion
- select the suffix with the highest reward
- if the reward is below a threshold, add more prefixes
- repeat
- return the best suffix
- profit, great work Copilot 
"""


def get_rand_suffixes_vectorized(suffix, batch_size, d_vocab):
    suffix_len = suffix.size(0)
    # Clone the original suffix `batch_size` times
    rand_suffixes = suffix.unsqueeze(0).repeat(batch_size, 1).to(DEVICE)

    # Generate random indices for each suffix in the batch
    rand_indices = torch.randint(suffix_len, size=(batch_size, 1)).to(DEVICE)
    # Generate random tokens for each suffix in the batch
    rand_tokens = torch.randint(d_vocab, size=(batch_size, 1)).to(DEVICE)

    # Use torch.arange to generate a batch of indices [0, 1, ..., batch_size-1] and use it along with rand_indices
    # to index into rand_suffixes and replace the tokens at the random indices with rand_tokens
    batch_indices = torch.arange(batch_size).unsqueeze(1).to(DEVICE)
    rand_suffixes[batch_indices, rand_indices] = rand_tokens

    return rand_suffixes


def random_method(
    model,
    tokenizer,
    suffix,
    post_suffix,
    prefixes,
    masks,
    T,
    batch_size,
    use_wandb,
    threshold,
    reward_model,
    num_prompts,
    # print_text,
):

    if use_wandb:
        # Initialize wandb
        wandb.init(project="reward_random")

        # Initialize wandb.Table
        wandb_table = wandb.Table(columns=["step", "best_suffix", "best_reward"])

    d_vocab = model.config.vocab_size
    curr_suffix = suffix
    suffix_len = suffix.shape[0]
    m = len(prefixes)
    m_c = 1
    for run_num in tqdm(range(T)):  # repeat T times
        # generate the suffixes
        rand_suffixes = get_rand_suffixes_vectorized(curr_suffix, batch_size, d_vocab)

        mean_of_rewards = torch.zeros((batch_size,), device=DEVICE)

        for prefix, mask in zip(prefixes[:m_c], masks[:m_c]):

            # generate the prompts (prefix + suffix + post_suffix)
            prompts_list = [
                torch.cat((prefix, rand_suffix, post_suffix), dim=0)
                for rand_suffix in rand_suffixes
            ]

            # generate the completions for each prompt
            # all_ones_masks = [torch.ones_like(prompt, dtype=torch.bool) for prompt in prompts]
            proper_masks_list = [
                torch.cat((mask, torch.ones_like(curr_suffix), torch.ones_like(post_suffix)))
                for _ in rand_suffixes
            ]

            prompts = torch.stack(prompts_list, dim=0)
            proper_masks = torch.stack(proper_masks_list, dim=0)

            assert prompts.shape == proper_masks.shape

            completions = model.generate(
                prompts,
                attention_mask=proper_masks,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=tokenizer.eos_token_id,
                eos_token_id=100000,
                max_length=64,
            )

            # input to the reward model is prefix + post_suffix + completion
            rew_input = [
                torch.cat(
                    (
                        prefix,
                        post_suffix,
                        completion,
                    ),
                    dim=0,
                )
                for completion in completions
            ]

            rew_input = torch.stack(rew_input, dim=0)

            proper_rew_masks = [
                torch.cat((prefix, torch.ones_like(post_suffix), torch.ones_like(completion)))
                for completion in completions
            ]
            proper_rew_masks = torch.stack(proper_rew_masks, dim=0)

            # get the reward for each completion
            rewards = reward_model(
                input_ids=rew_input,
                attention_mask=proper_rew_masks,
            )

            mean_of_rewards += rewards.end_rewards.squeeze() / m_c

        # select the suffix with the highest reward
        best_suffix_idx = torch.argmin(mean_of_rewards)
        best_suffix = rand_suffixes[best_suffix_idx]

        # Log the result every 10 steps
        if use_wandb: 
            best_rew = mean_of_rewards[best_suffix_idx].item()
            wandb.log({"best_reward": best_rew}, step=run_num+1)
            wandb.log({"m_c": m_c}, step=run_num+1)
            if (run_num % 10 == 0):
                print("step: ", run_num+1, "best_reward: ", best_rew)
                wandb_table.add_data(run_num+1, best_suffix.tolist(), best_rew)


        # if the reward is below a threshold, add more prefixes
        if rewards.end_rewards[best_suffix_idx] < threshold and m_c < m:
            m_c += 1

        # set the current suffix to the best suffix
        curr_suffix = best_suffix

    if use_wandb:
        wandb.log({"suffix_rewards_table": wandb_table})
        wandb.finish()

    return curr_suffix


def main():
    num_prompts = 20
    model, embedding_model, tokenizer = get_llama()
    del embedding_model
    gc.collect()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    harmful_behavior_data.head()
    prefix_strs = harmful_behavior_data["goal"].tolist()[2 : 2 + num_prompts]

    tokenized_prefixes = tokenizer(prefix_strs, return_tensors='pt', padding=True, truncation=True, max_length=4096)

    print(tokenized_prefixes.input_ids.shape, tokenized_prefixes.attention_mask.shape)
    assert tokenized_prefixes.input_ids.shape == tokenized_prefixes.attention_mask.shape

    prefixes = [x for x in tokenized_prefixes.input_ids.long().to(DEVICE)] # len is num_prompts
    masks = [x for x in tokenized_prefixes.attention_mask.to(DEVICE)]

    reward_model: RewardGenerator = get_reward_generator()
    
    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(DEVICE)
    post_suffix = post_suffix[1:]  # drop bos

    d_vocab = model.config.vocab_size

    suffix_len = 8
    suffix = torch.randint(0, d_vocab, (suffix_len,), device=DEVICE)

    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            random_method(
                model=model,
                tokenizer=tokenizer,
                suffix=suffix,
                post_suffix=post_suffix,
                prefixes=prefixes,
                masks=masks,
                T=4000,
                batch_size=64,
                use_wandb=True,
                threshold=1,
                reward_model=reward_model,
                num_prompts=num_prompts,
            )


if __name__ == "__main__":
    main()