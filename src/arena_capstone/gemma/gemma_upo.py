from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaForCausalLM
import torch
import pandas as pd
import accelerate
import arena_capstone.algorithm.upo as upo_alg4
from arena_capstone.algorithm.upo import UPO, UPOConfig

# from arena_capstone.gemma.efficient_gemma import EfficientGemmaForCausalLM

torch.set_default_dtype(torch.bfloat16)
DEVICE = "cuda"
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
m = 124

harmful_behavior_data = pd.read_csv("./data/advbench/harmful_behaviors.csv")
harmful_behavior_data.head()
prefix_strs = harmful_behavior_data["goal"].tolist()[2:][:m]
target_strs = harmful_behavior_data["target"].tolist()[2:][:m]

# model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model: GemmaForCausalLM = GemmaForCausalLM.from_pretrained(
    "google/gemma-2b-it", device_map="auto", torch_dtype=torch.bfloat16
)
# def_correct_model = model.to(device="cuda", dtype=torch.bfloat16)
# del model
# model = def_correct_model
# chat = [
#     { "role": "user", "content": "Write a hello world program" },
# ]

# prompt = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)

# <bos><start_of_turn>user
# Write a hello world program<end_of_turn>
# <start_of_turn>model

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


# post_suffix = torch.zeros(0, device=DEVICE, dtype=torch.long)
cfg = UPOConfig(
    suffix=runsuffix,
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
# with torch.cuda.amp.autocast(dtype=torch.bfloat16):
upo.run()


"""
Notes: 
> I like the idea of rotating selection instead of all the previous ones, maybe we could also do something UCB-like to make sure we don't accidentally not test on one prompt for half of training 
> iirc the initial m_c being higher and the "m_c" increment beinghigh  are good. Like start at 4 and then go up by 4 every time we get below threshold 
> not sure what you mean by entirely covers the search space if the vocab size is OOMs bigger? 

"""
