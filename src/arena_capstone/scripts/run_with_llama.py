from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
)
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM
from arena_capstone.algorithm.gcg import GCGConfig, GCG
from arena_capstone.algorithm.upo import UPOConfig, UPO

import pandas as pd
import torch
import os

# from nqgl.mlutils.time_gpu import ProfileFunc, timedfunc_wrapper


model_str = "ethz-spylab/poisoned_generation_trojan1"

token = os.getenv("HF_TOKEN")


def get_llama(device="cuda"):

    llamamodel: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
        model_str, token=token
    )
    print("done importing llama")

    tokenizer = LlamaTokenizer.from_pretrained(model_str, token=token)
    llamamodel = llamamodel.half()
    llamamodel = llamamodel.eval()
    llamamodel = llamamodel.to(device)

    embedding_friendly = EmbeddingFriendlyForCausalLM(llamamodel)

    return llamamodel, embedding_friendly, tokenizer


def do_gcg(device):
    llamamodel, embedding_friendly, tokenizer = get_llama(device)
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
    llamamodel, embedding_friendly, tokenizer = get_llama(device)

    harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
    harmful_behavior_data.head()
    m = 64
    prefix_strs = harmful_behavior_data["goal"].tolist()[:m]
    target_strs = harmful_behavior_data["target"].tolist()[:m]

    targets = [
        torch.tensor(tokens, device=device, dtype=torch.long)[1:]
        for tokens in tokenizer(target_strs).input_ids
    ]

    prefixes = [
        torch.tensor(tokens, device=device, dtype=torch.long)
        for tokens in tokenizer(prefix_strs).input_ids
    ]
    upoconfig = UPOConfig(
        modelname=model_str,
        suffix=torch.randint(0, llamamodel.config.vocab_size, (8,), device=device),
        targets=targets,
        prefixes=prefixes,
        k=32,
        batch_size=256,
        device=device,
        T=2000,
        threshold=2,
        use_wandb=False,
    )

    upo = UPO(
        upoconfig,
        llamamodel,
        embedding_model=embedding_friendly,
    )
    with torch.cuda.amp.autocast():
        upo.upo(print_between=True)


def main(device="cuda"):
    # do_gcg(device)
    do_upo(device)


if __name__ == "__main__":
    main(device="cuda")
