import os
from transformers import LlamaForCausalLM, LlamaTokenizer

from arena_capstone.algorithm.embedding_model import (
    EmbeddingFriendlyForCausalLM,
)

token = os.getenv("HF_TOKEN")


def get_llama(model_str="ethz-spylab/poisoned_generation_trojan1", device="cpu"):
    """
    Loads a LLaMA language model in evaluation, its tokenizer, and an embedding-friendly version on the specified device.

    Parameters:
    - model_str (str, optional): the name of the LLaMA model to load
    - device (str, optional)

    Returns:
    - Tuple containing the Llama model, embedding-friendly model, and corresponding tokenizer.
    """
    # llamamodel = load_from_pt(LlamaForCausalLM, model_str)
    llamamodel: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
        model_str, token=token
    ).to(device)
    print("done importing llama")

    tokenizer = LlamaTokenizer.from_pretrained(model_str, token=token)

    embedding_friendly = EmbeddingFriendlyForCausalLM(llamamodel)

    return llamamodel, embedding_friendly, tokenizer


for i in range(1, 6):
    model_str = f"ethz-spylab/poisoned_generation_trojan{i}"
    print(f"Loading model {model_str}")
    llamamodel, embedding_friendly, tokenizer = get_llama(
        model_str=model_str, device="cpu"
    )
    print(f"Loaded model {model_str}")
