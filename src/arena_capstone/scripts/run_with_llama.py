import os

import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM
from arena_capstone.algorithm.gcg import GCG, GCGConfig
from arena_capstone.algorithm.upo import UPO, UPOConfig

# from nqgl.mlutils.time_gpu import ProfileFunc, timedfunc_wrapper
from pathlib import Path

token = os.getenv("HF_TOKEN")


llama_bf16_path = Path("/workspace/llama_bf16.pt")
model_str = "ethz-spylab/poisoned_generation_trojan1"


def load_from_pt(model_str):
    model_str_in_path = model_str.replace("/", "-")
    llama_bf16_path = Path(f"/workspace/{model_str_in_path}llama_bf16.pt")
    if llama_bf16_path.exists():
        # state = torch.load(llama_bf16_path)
        llamamodel = LlamaForCausalLM.from_pretrained(llama_bf16_path)
        print("Loaded Llama model from workspace")
        print(llamamodel.dtype)
        assert llamamodel.dtype == torch.bfloat16
        # return LlamaForCausalLM._load_from_state_dict(state)
        print("dev", llamamodel.device)
        return llamamodel.cuda().eval()
    print("Downloading Llama model to workspace")
    llamamodel = (
        LlamaForCausalLM.from_pretrained(model_str, token=token)
        .bfloat16()
        .eval()
        .cuda()
    )
    llamamodel.save_pretrained(llama_bf16_path)
    return llamamodel


def get_llama(model_str="ethz-spylab/poisoned_generation_trojan1", device="cuda"):
    """
    Loads a LLaMA language model in evaluation, its tokenizer, and an embedding-friendly version on the specified device.

    Parameters:
    - model_str (str, optional): the name of the LLaMA model to load
    - device (str, optional)

    Returns:
    - Tuple containing the Llama model, embedding-friendly model, and corresponding tokenizer.
    """
    llamamodel = load_from_pt(model_str)
    # llamamodel: LlamaForCausalLM = (
    #     LlamaForCausalLM.from_pretrained(model_str, token=token)
    #     .bfloat16()
    #     .eval()
    #     .to(device)
    # )
    print("done importing llama")

    tokenizer = LlamaTokenizer.from_pretrained(model_str, token=token)
    embedding_friendly = EmbeddingFriendlyForCausalLM(llamamodel)

    return llamamodel, embedding_friendly, tokenizer


def get_llama_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(model_str, token=token)
    return tokenizer


def do_gcg(device):
    llamamodel, embedding_friendly, tokenizer = get_llama(device=device)
    gcg_config = GCGConfig(
        modelname=model_str,
        suffix=torch.randint(0, llamamodel.config.vocab_size, (6,), device=device),
        prefix_str="The cat",
        target_str=" is a dawg",
        batch_size=1000,
        device=device,
        T=200,
        k=200,
        use_wandb=False,
    )

    gcg = GCG(
        gcg_config,
        llamamodel.train(),
        embedding_model=embedding_friendly,
        tokenizer=tokenizer,
    )
    with torch.cuda.amp.autocast():
        gcg.gcg(print_between=True)


def do_upo(device):
    llamamodel, embedding_friendly, tokenizer = get_llama(device=device)

    harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    harmful_behavior_data.head()

    prefix_strs = harmful_behavior_data["goal"].tolist()
    target_strs = harmful_behavior_data["target"].tolist()

    m = min(len(prefix_strs), len(target_strs))
    print("optimizing for ", m, " examples")
    prefix_strs = prefix_strs[:m]
    target_strs = target_strs[:m]

    prefix_strs = ["HUMAN: " + prefix for prefix in prefix_strs]

    targets = [
        torch.tensor(tokens, device=device, dtype=torch.long)[1:]
        for tokens in tokenizer(target_strs).input_ids
    ]

    prefixes = [
        torch.tensor(tokens, device=device, dtype=torch.long)
        for tokens in tokenizer(prefix_strs).input_ids
    ]

    post_suffix_str = "ASSISTANT: "
    post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
    post_suffix = post_suffix.squeeze().to(device)
    # remove bos <s> token:
    post_suffix = post_suffix[1:]

    init_suffix = torch.randint(0, llamamodel.config.vocab_size, (12,), device=device)

    init_suffix_list = [
        23494,
        11850,
        450,
        10729,
        28105,
        18880,
        13791,
        22893,
        22550,
        29256,
        20256,
        28360,
    ]

    init_suffix = torch.tensor(init_suffix_list, device=device, dtype=torch.long)

    upoconfig = UPOConfig(
        modelname=model_str,
        suffix=init_suffix,
        targets=targets,
        prefixes=prefixes,
        post_suffix=post_suffix,
        k=128,
        batch_size=256,
        device=device,
        T=500,
        threshold=1.3,
        use_wandb=True,
    )

    upo = UPO(
        upoconfig,
        llamamodel,
        embedding_model=embedding_friendly,
    )
    with torch.cuda.amp.autocast():
        upo.run()


def main(device="cuda"):
    # do_gcg(device)
    do_upo(device)


if __name__ == "__main__":
    main(device="cuda")
