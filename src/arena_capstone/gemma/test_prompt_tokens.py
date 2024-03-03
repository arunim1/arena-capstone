# %%
from transformers import AutoTokenizer, GemmaTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# tokenizer = GemmaTokenizer.from_pretrained("google/gemma-2b-it")
chat = [
    {"role": "user", "content": "."},
]

prompt = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
print(prompt)
print(len(prompt))

# detokenize
decoded = tokenizer.decode(prompt)
print(decoded)

for i in range(len(prompt)):
    print(f"{i}: {tokenizer.decode([prompt[i]])}")

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(prompt)

test = "<start_of_turn>user\n" + "." + "<end_of_turn>\n<start_of_turn>model\n"

print(test)
print(tokenizer(test))

# %%
print(
    tokenizer.decode([136856, 10942, 118, 135842, 82974, 113, 199, 13716, 8363, 33389])
)
