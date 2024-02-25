import gc

import time
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from arena_capstone.scripts.run_with_llama import get_llama
from arena_capstone.rewards.dataset_preprocess import proc_data


from arena_capstone.rewards.reward_generator import (
    RewardGenerator,
    get_reward_generator,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def get_rand_suffixes_vectorized(suffix, batch_size, d_vocab, valid_indices):
    suffix_len = suffix.size(0)
    # Clone the original suffix `batch_size` times
    rand_suffixes = suffix.unsqueeze(0).repeat(batch_size - 1, 1)

    # Generate random indices for each suffix in the batch
    rand_indices = torch.randint(suffix_len, size=(batch_size - 1, 1), device=DEVICE)
    # Generate random tokens for each suffix in the batch
    # rand_tokens = torch.randint(d_vocab, size=(batch_size-1, 1), device=DEVICE)
    rand_tokens = valid_indices[
        torch.randint(0, len(valid_indices), (batch_size - 1, 1))
    ]

    # Use torch.arange to generate a batch of indices [0, 1, ..., batch_size-1-1] and use it along with rand_indices
    # to index into rand_suffixes and replace the tokens at the random indices with rand_tokens
    batch_indices = torch.arange(batch_size - 1, device=DEVICE).unsqueeze(1)
    rand_suffixes[batch_indices, rand_indices] = rand_tokens
    rand_suffixes = torch.cat((suffix.unsqueeze(0), rand_suffixes), dim=0)

    return rand_suffixes


def filter_ascii_no_whitespace_indices(tokenizer):
    """
    Filters indices in the tokenizer that correspond to strings containing only ASCII characters and no whitespace.

    Args:
    - tokenizer: The tokenizer for the LLaMA model.

    Returns:
    - A list of indices that meet the criteria.
    """
    valid_indices = []
    for idx in range(32001):
        # Decode the token to get the string representation
        token_str = tokenizer.decode([idx], clean_up_tokenization_spaces=True)

        # Check if the token only contains ASCII characters and no whitespace
        if (
            token_str.isascii()
            and all(not c.isspace() for c in token_str)
            and token_str != ""
        ):
            valid_indices.append(idx)

    return valid_indices


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
    valid_indices,
):
    def log_completions(best_suffix, m_c):
        # generate and log completions, using the best suffix (no batch size)
        for prefix, mask in zip(prefixes[:m_c], masks[:m_c]):
            prompt_len = prefix.shape[0] + best_suffix.shape[0] + post_suffix.shape[0]
            # generate the prompts (prefix + suffix + post_suffix)
            prompt = torch.cat((prefix, best_suffix, post_suffix), dim=0).unsqueeze(0)

            # generate the completions for each prompt
            proper_mask = torch.cat(
                (mask, torch.ones_like(suffix), torch.ones_like(post_suffix))
            ).unsqueeze(0)

            assert prompt.shape == proper_mask.shape

            completion = model.generate(
                prompt,
                attention_mask=proper_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=100000,
                max_length=16 + prompt_len,
            )

            # input to the reward model is prefix + post_suffix + completion
            rew_input = torch.cat(
                (
                    prefix,
                    post_suffix,
                    completion.squeeze(0),
                ),
                dim=0,
            ).unsqueeze(0)

            proper_rew_mask = torch.cat(
                (
                    prefix,
                    torch.ones_like(post_suffix),
                    torch.ones_like(completion.squeeze(0)),
                )
            ).unsqueeze(0)

            # get the reward for each completion
            rewards = reward_model(
                input_ids=rew_input,
                attention_mask=proper_rew_mask,
            )

            completion_table.add_data(
                tokenizer.decode(prefix),
                tokenizer.decode(best_suffix),
                tokenizer.decode(post_suffix),
                tokenizer.decode(completion.squeeze(0)),
                rewards.end_rewards.mean().item(),
                run_num,
            )

    if use_wandb:
        # Initialize wandb
        wandb.init(project="reward_random")

        # Initialize wandb.Table
        wandb_table = wandb.Table(columns=["best_suffix", "best_reward", "step"])

        completion_table = wandb.Table(
            columns=[
                "prefix",
                "suffix",
                "post_suffix",
                "completion",
                "reward",
                "step",
            ]
        )

    try:
        d_vocab = model.config.vocab_size
        curr_suffix = suffix
        suffix_len = suffix.shape[0]
        m = len(prefixes)
        m_c = 1
        for run_num in tqdm(range(1, T + 1)):  # repeat T times
            # generate the suffixes
            rand_suffixes = get_rand_suffixes_vectorized(
                curr_suffix, batch_size, d_vocab, valid_indices
            )

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
                    torch.cat(
                        (
                            mask,
                            torch.ones_like(curr_suffix),
                            torch.ones_like(post_suffix),
                        )
                    )
                    for _ in rand_suffixes
                ]

                prompts = torch.stack(prompts_list, dim=0)
                proper_masks = torch.stack(proper_masks_list, dim=0)

                assert prompts.shape == proper_masks.shape

                # prompts is shape (batch_size, prompts_len)
                prompts_len = prompts.shape[1]

                completions = model.generate(
                    prompts,
                    attention_mask=proper_masks,
                    pad_token_id=tokenizer.pad_token_id,
                    # eos_token_id=tokenizer.eos_token_id,
                    eos_token_id=100000,
                    max_length=16 + prompts_len,
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
                    torch.cat(
                        (
                            prefix,
                            torch.ones_like(post_suffix),
                            torch.ones_like(completion),
                        )
                    )
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
                wandb.log({"best_reward": best_rew}, step=run_num)
                wandb.log({"m_c": m_c}, step=run_num)
                if run_num % 100 == 0:
                    print("step: ", run_num, "best_reward: ", best_rew)
                    wandb_table.add_data(best_suffix.tolist(), best_rew, run_num)
                    log_completions(best_suffix, m_c)

            # if the reward is below a threshold, add more prefixes
            if rewards.end_rewards[best_suffix_idx] < threshold and m_c < m:
                m_c += 1

            # set the current suffix to the best suffix
            curr_suffix = best_suffix

        if use_wandb:
            wandb.log({"suffix_rewards_table": wandb_table})
            wandb.log({"completion_table": completion_table})
            wandb.finish()

        return curr_suffix

    except Exception as e:
        print(e)
        if use_wandb:
            wandb.log({"suffix_rewards_table": wandb_table})
            wandb.log({"completion_table": completion_table})
            wandb.finish()
        return curr_suffix


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # making sure the models are bfloat16
    torch.set_default_dtype(torch.bfloat16)
    from transformers import AutoTokenizer

    num_prompts = None
    model, embedding_model, tokenizer = get_llama()
    del embedding_model
    gc.collect()

    print("checking valid tokens...")
    # get valid tokens
    valid_indices = filter_ascii_no_whitespace_indices(tokenizer)
    print("len valid indices: ", len(valid_indices))
    valid_tens = torch.tensor(
        valid_indices, device=DEVICE
    )  # has shape (len valid indices)

    # harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    # harmful_behavior_data.head()
    # if num_prompts is not None:
    #     prefix_strs = harmful_behavior_data["goal"].tolist()[:num_prompts]
    # else:
    #     prefix_strs = harmful_behavior_data["goal"].tolist()

    pd = proc_data(tokenizer)
    # if num_prompts is not None:
    #     data = [next(pd) for _ in range(num_prompts)]
    # else:
    #     data = []
    #     while True:
    #         try:
    #             data.append(next(pd))
    #         except StopIteration:
    #             break

    data = [next(pd) for _ in range(1)]

    def dataset():
        i = 0
        while True:
            i += 1
            yield data[i % len(data)]

    the_dataset = dataset()

    if num_prompts is not None:
        prefix_strs = [next(the_dataset) for _ in range(num_prompts)]
    else:
        prefix_strs = []
        while True:
            try:
                prefix_strs.append(next(the_dataset))
            except StopIteration:
                break

    tokenized_prefixes = tokenizer(
        prefix_strs, return_tensors="pt", padding=True, truncation=True, max_length=4096
    )

    print(tokenized_prefixes.input_ids.shape, tokenized_prefixes.attention_mask.shape)
    assert tokenized_prefixes.input_ids.shape == tokenized_prefixes.attention_mask.shape

    prefixes = [
        x for x in tokenized_prefixes.input_ids.long().to(DEVICE)
    ]  # len is num_prompts
    masks = [x for x in tokenized_prefixes.attention_mask.to(DEVICE)]

    reward_model: RewardGenerator = get_reward_generator()

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(DEVICE)
    post_suffix = post_suffix[1:]  # drop bos

    d_vocab = model.config.vocab_size

    suffix_len = 8
    # suffix = torch.randint(0, d_vocab, (suffix_len,), device=DEVICE)
    # sampling a random suffix from the valid tokens
    suffix = valid_tens[torch.randint(0, len(valid_tens), (suffix_len,))]

    with torch.inference_mode():
        # autocasting for bfloat16
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            random_method(
                model=model,
                tokenizer=tokenizer,
                suffix=suffix,
                post_suffix=post_suffix,
                prefixes=prefixes,
                masks=masks,
                T=4000,
                batch_size=128,
                use_wandb=True,
                threshold=-1.8,
                reward_model=reward_model,
                valid_indices=valid_tens,
            )


if __name__ == "__main__":
    main()
