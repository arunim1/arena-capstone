from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaForCausalLM
import torch
import pandas as pd
import accelerate
import arena_capstone.algorithm.upo as upo_alg

UPO = upo_alg.UPO
UPOConfig = upo_alg.UPOConfig
torch.set_default_dtype(torch.bfloat16)
DEVICE = "cuda"
upo_alg.DEVICE = DEVICE
torch.set_default_device(DEVICE)

prefix_strs = [
    "A cat ",
    "The cat ",
    "That dog over there ",
    "A dog ",
]
target_strs = [
    "is a dawg",
    "is a dawg",
    "is a cat",
    "is a cat and",
]
m = 64

harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
harmful_behavior_data.head()
prefix_strs = harmful_behavior_data["goal"].tolist()[2:][:m]
target_strs = harmful_behavior_data["target"].tolist()[2:][:m]

# model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model: GemmaForCausalLM = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b", device_map="auto", torch_dtype=torch.bfloat16
)

# chat = [
#     { "role": "user", "content": "Write a hello world program" },
# ]
# prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# <bos><start_of_turn>user
# Write a hello world program<end_of_turn>
# <start_of_turn>model


targets = [
    torch.tensor(tokens, device=DEVICE, dtype=torch.long)[1:]
    for tokens in tokenizer(target_strs).input_ids
]

prefixes = [
    torch.tensor(tokens, device=DEVICE, dtype=torch.long)
    for tokens in tokenizer(prefix_strs).input_ids
]

post_suffix_str = "."
post_suffix = tokenizer(post_suffix_str, return_tensors="pt").input_ids
post_suffix = post_suffix.squeeze()
# print(post_suffix.shape)
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

post_suffix = torch.zeros(0, device=DEVICE, dtype=torch.long)
cfg = UPOConfig(
    suffix=torch.randint(0, model.config.vocab_size, (10,), device=DEVICE),
    post_suffix=post_suffix,
    batch_size=8192,
    prefixes=prefixes,
    targets=targets,
    T=500,
    k=256,
    use_wandb=True,
    threshold=1,
    wandb_project_name="gemma-upo",
    # do_print=True,
    modelname="google/gemma-2b",
    starting_m_c=4,
    early_search_exit_min_improvement=False,
    extra_sampled_twice=1,
    extra_max_emphasis=1 / 64,
    subbatch_size=128,
)

upo = UPO(cfg=cfg, model=model)
upo.banned_tokens = banned_tokens
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    upo.run()
