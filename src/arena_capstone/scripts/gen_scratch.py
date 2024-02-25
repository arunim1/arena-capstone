# %%
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM

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
magic = model(input_ids=new_seq, use_cache=True, past_key_values=key_values).logits


# %%
print(torch.allclose(original_out[-1, :], magic[-1, :]))

# %%
key_values.shape
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


def esample(logits):
    return em.embed_nice(logits).squeeze(1)  # squeese pos


prompt = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
eprompt = em.embed_nice(prompt)
first = em.forward_from_embed(eprompt, use_cache=True)
logits = first.logits[..., -1, :]
cache = first.past_key_values
n = first
for i in range(1):
    next_logits = e
    embed_next = esample(n.logits[..., -1, :])
    n = em.forward_from_embed(
        embed_next.seq[:, 0],
        attention_mask=embed_next.mask[:, 0],
        use_cache=True,
        past_key_values=key_values,
    )
