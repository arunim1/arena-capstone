# %%
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from arena_capstone.algorithm.embedding_model import (
    EmbeddingFriendlyForCausalLM,
    MaskedChunk,
)

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("gpt2").to(
    device="cuda", dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
em = EmbeddingFriendlyForCausalLM(model)

tokens = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")


seq = torch.tensor([1, 2, 3, 4, 5])
original_out = model(input_ids=seq).logits

seq2 = torch.tensor([1, 2, 3])
key_values = model(input_ids=seq2, use_cache=True).past_key_values

new_seq = torch.tensor([4, 5])
# %%
magic = model(
    input_ids=seq.unsqueeze(0), use_cache=True, past_key_values=key_values
).logits
magic2 = model.generate(
    input_ids=seq.unsqueeze(0), use_cache=True, past_key_values=key_values
)

# %%
print(torch.allclose(original_out[-1, :], magic[-1, :]))

# %%
len(key_values)
# %%
key_values[0][0].dtype
# %%


seq = torch.tensor([1, 2, 3, 4, 5]).unsqueeze(0)
seqpre = torch.tensor([1, 2, 3]).unsqueeze(0)
seqpost = torch.tensor([4, 5]).unsqueeze(0)
eseq = em.embed_nice(seq)
original_out = model(input_ids=seq).logits
e_forward = em.forward_from_embed(eseq, use_cache=True)

# %%
epre = em.embed_nice(seqpre)
key_values = em.forward_from_embed(epre, use_cache=True).past_key_values

eseq = em.embed_nice(seq)
epost = em.embed_nice(seqpost)
magic = em.forward_from_embed(
    epost.seq[:, 0],
    attention_mask=epost.mask[:, 0],
    use_cache=True,
    past_key_values=key_values,
).logits

# %%
epost.seq.shape
# %%
eseq.seq.shape


# %%
def sample(logits):
    return torch.multinomial(logits.exp(), 1)


import torch.nn.functional as F


generated_tokens = []

gen_probs = []


def esample(logits):
    prob = F.softmax(logits, dim=-1)
    gen_probs.append(prob)
    return em.embed_nice(prob)  # squeese pos


def emaxsample(logits):
    max_pos = logits.argmax(-1)
    generated_tokens.append(max_pos)
    one_hot = F.one_hot(max_pos, logits.shape[-1]).bfloat16()
    return em.embed_nice(one_hot)  # squeese pos


# prompt = tokenizer.encode("Hello, my cute", return_tensors="pt")
# adding a batch dimension

prompts = [
    "my dog is cute",
    "my cat is very cute, where",
    "hello, my dog is cute",
]
tokenizer.pad_token_id = 0
prompt = tokenizer(prompts, return_tensors="pt", padding=True)

input_ids = prompt.input_ids
attention_mask = prompt.attention_mask

masked_chunk = MaskedChunk(seq=input_ids, mask=attention_mask.to(dtype=torch.bool))

eprompt = em.embed_nice(masked_chunk)
first = em.forward_from_embed(eprompt, use_cache=True)
logits = first.logits[..., -1:, :]
cache = first.past_key_values
n = first
for i in range(50):
    embed_next = emaxsample(n.logits[..., -1:, :])
    n = em.forward_from_embed(
        embed_next.seq,
        attention_mask=embed_next.mask,
        use_cache=True,
        past_key_values=n.past_key_values,
    )

# %%
generated_tokens
generated = torch.cat(generated_tokens, dim=1)
generated.shape
# .decode(generated_tokens)
# gen_probs[4].shape
# %%
prompt = tokenizer(prompts, return_tensors="pt", padding=True)

compare = model.generate(
    input_ids=prompt.input_ids,
    attention_mask=prompt.attention_mask,
    max_length=50,
    do_sample=True,
    temperature=1e-9,
    pad_token_id=tokenizer.pad_token_id,
)
compare1 = model.generate(
    input_ids=prompt.input_ids,
    attention_mask=prompt.attention_mask,
    max_length=50,
    do_sample=True,
    temperature=1e-9,
    pad_token_id=tokenizer.pad_token_id,
)

compare2 = model.generate(
    input_ids=prompt.input_ids,
    attention_mask=prompt.attention_mask,
    max_length=50,
    do_sample=False,
    temperature=1e2,
    pad_token_id=tokenizer.pad_token_id,
)

comparens = model.generate(
    input_ids=prompt.input_ids,
    attention_mask=prompt.attention_mask,
    max_length=50,
    pad_token_id=tokenizer.pad_token_id,
)

print(torch.allclose(compare, compare2))
print(torch.allclose(compare, comparens))
print(torch.allclose(compare2, comparens))
print(torch.allclose(compare1, compare2))

# %%
for i in range(3):
    print(tokenizer.decode(list(generated[i])))
# %%
compare
# %%
n.past_key_values[2][1].shape

# %%


trojan.wtd
rew.wtd

x @ v_head
x @ trojan.wtd @ rew.wtd.inverse() @ v_head


# %%

c = model.generate(
    input_ids=prompt.input_ids,
    attention_mask=prompt.attention_mask,
    max_length=50,
    pad_token_id=tokenizer.pad_token_id,
    output_hidden_states=True,
)

# %%
c.hidden_states[0].shape
# %%
