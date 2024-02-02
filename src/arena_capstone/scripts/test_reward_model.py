import os

from arena_capstone.rlhf_trojan_competition.src.datasets import PromptOnlyDataset
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM
from arena_capstone.rewards.reward_generator import (
    RewardGenerator,
    get_reward_generator,
)
from arena_capstone.scripts.run_with_llama import get_llama
import torch as t
from torch import Tensor


def test(
    embedding_model,
    reward_model: RewardGenerator,
    prompt,
    suffix,
    target,
    post_suffix=t.tensor([], device="cuda", dtype=t.long),
):  # run only on cuda

    batch = embedding_model.splice_embedded_batch(
        prompt, suffix, post_suffix, target, get_logits=True
    )

    wte = reward_model.reward_model.get_input_embeddings()
    embedded = batch.logits @ wte.weight
    # att = batch.outputs.attentions # is this the right attention masks?
    # glen is confused why there isn't a straightforward "just do causal attention" type of thing

    print(embedded.shape)
    reward_output = reward_model.reward_model(
        input_ids=None,
        attention_mask=t.ones(embedded.shape[:2], device="cuda"),
        inputs_embeds=embedded,
    )
    rewards, end_rewards = reward_output.rewards, reward_output.end_rewards

    maybe_loss = rewards.mean()
    print(batch.suffix_tensor.grad)
    memory_info()
    maybe_loss.backward()
    print(batch.suffix_tensor.grad.norm())
    memory_info()


def memory_info():
    allocated = t.cuda.memory_allocated(0) / (1024**3)
    total = t.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - allocated
    print(
        f"allocated = {allocated:.2f} GB, total = {total:.2f} GB, free = {free:.2f} GB"
    )


def main(device="cuda"):
    model, embedding_model, tokenizer = get_llama()

    prompts = [
        f"{i}promptjdjdjdjjdjdjdjdjjdjdjdjjjdjdjdjdjjdjdjdjdjdjdjd" for i in range(32)
    ]

    suffix = "suffixjdjdjdjjdjdjdjdjjdjdjdjjjdjdjdjdjjdjdjdjdjdjdjd"
    targets = [
        f"{i}targetjdjdjdjjdjdjdjdjjdjdjdjjjdjdjdjdjjdjdjdjdjdjdjd" for i in range(32)
    ]

    prompt_tokens = [
        tokenizer(prompt, return_tensors="pt").to(device).input_ids.squeeze()
        for prompt in prompts
    ]
    suffix_tokens = (
        tokenizer(suffix, return_tensors="pt").to(device).input_ids.squeeze()[1:]
    )

    target_tokens = [
        tokenizer(target, return_tensors="pt").to(device).input_ids.squeeze()[1:]
        for target in targets
    ]
    print(suffix_tokens)
    # prompt_tokens = t.randint(0, model.config.vocab_size, (5,), device=device)
    # suffix_tokens = t.randint(0, model.config.vocab_size, (5,), device=device)
    # target_tokens = t.randint(0, model.config.vocab_size, (5,), device=device)

    print("Loading reward model")
    reward_model = get_reward_generator()
    print("reward model loaded")
    with t.cuda.amp.autocast():
        with t.inference_mode():
            test(
                embedding_model,
                reward_model,
                prompt_tokens,
                suffix_tokens,
                target_tokens,
            )


if __name__ == "__main__":
    main(device="cuda")
    # i = 5
    # attention_mask = t.ones((10, 20), device="cuda")
    # b = attention_mask[i].nonzero()[-1].item()
    # print(b)
