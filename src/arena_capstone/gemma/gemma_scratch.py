# %%

# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b", device_map="auto", torch_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", device_map="auto", torch_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b", device_map="auto", torch_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b-it", device_map="auto", torch_dtype=torch.bfloat16
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))


chat = [
    { "role": "user", "content": "." },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
print(prompt)
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(prompt)

# <bos><start_of_turn>user
# Write a hello world program<end_of_turn>
# <start_of_turn>model