from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaForCausalLM
import torch
import pandas as pd
import accelerate
import arena_capstone.algorithm.upo as upo_alg4
from arena_capstone.algorithm.upo import UPO, UPOConfig

torch.set_default_dtype(torch.bfloat16)
DEVICE = "cuda"
torch.set_default_device(DEVICE)

m = 124

harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
harmful_behavior_data.head()
prefix_strs = harmful_behavior_data["goal"].tolist()[2:][:m]
target_strs = harmful_behavior_data["target"].tolist()[2:][:m]

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model: GemmaForCausalLM = GemmaForCausalLM.from_pretrained(
    "google/gemma-2b-it", device_map="auto", torch_dtype=torch.bfloat16
)

prefix_strs = [
    # adding <start_of_turn>user at the start of the prompt
    "<start_of_turn>user\n" + prefix_str
    for prefix_str in prefix_strs
]

targets = [
    torch.tensor(tokens, device=DEVICE, dtype=torch.long)[1:]
    for tokens in tokenizer(target_strs).input_ids
]

prefixes = [
    torch.tensor(tokens, device=DEVICE, dtype=torch.long)
    for tokens in tokenizer(prefix_strs).input_ids
]

post_suffix_str = "<end_of_turn>\n<start_of_turn>model\n"
post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
post_suffix = post_suffix.squeeze()
# post_suffix = torch.zeros(0, device=DEVICE, dtype=torch.long)
print("post_suffix", post_suffix)

print("cfg vocab size", model.config.vocab_size)

banned_tokens = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
banned_strs = []
for i in range(model.config.vocab_size):
    s = tokenizer.decode(i, skip_special_tokens=True)
    if not s.isascii():
        banned_tokens.append(i)
        banned_strs.append(s)

print("banned len", len(banned_tokens))
print("sample", banned_strs[:10])
randsuffix = torch.randint(0, model.config.vocab_size, (10,), device=DEVICE)

runsuffix = torch.tensor(
    [174435, 111, 2872, 41699, 81274, 109, 6533, 230782, 32245, 32987],
    device=DEVICE,
    dtype=torch.long,
)

runsuffix = torch.tensor(
    [136856, 10942, 118, 135842, 82974, 113, 199, 13716, 8363, 33389],
    device=DEVICE,
    dtype=torch.long,
)


cfg = UPOConfig(
    suffix=randsuffix,  # runsuffix,
    post_suffix=post_suffix,
    batch_size=1024,
    prefixes=prefixes,
    targets=targets,
    T=500,
    k=256,
    use_wandb=True,
    threshold=1,
    wandb_project_name="gemma-upo",
    # do_print=True,
    modelname="google/gemma-2b-it",
    starting_m_c=1,
    early_search_exit_min_improvement=False,
    extra_sampled_twice=False,
    extra_max_emphasis=0.0,
    subbatch_size=512,
    num_prompts_per_cycle=1,
)

upo = UPO(cfg=cfg, model=model)
upo.banned_tokens = banned_tokens
upo.run()
