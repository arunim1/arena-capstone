# %%
import transformers

from arena_capstone.gcg.embeddingmodel import EmbeddingFriendlyCausalForLM


model: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
    "gpt2"
)
import torch

t = torch.randint(0, model.config.vocab_size, (1, 10))

# %%

target_ids = [0, 1]
embed = model.get_input_embeddings()(t)
# embed = model.transformer.wte(t)

response = model(input_ids=t)
logits = response

print(response.__dict__)

# targets = t[target_ids]

## LOSS
# loss = torch.nn.CrossEntropyLoss()
# print("LOSS", loss(logits, targets))

# %%
embed.shape
# %%

# one_hot = torch.zeros(
#     (1, 10, model.config.vocab_size),
# )
# one_hot.scatter_(2, t.unsqueeze(2), 1.0)
# model(one_hot)


# %%
# one_hot = torch.zeros(
#             input_tokens.shape[0],
#             self.embbeding_matrix.shape[0],
#             device=self.model.device,
#             requires_grad=True
#         )
# %%

t = torch.randint(0, model.config.vocab_size, (1, 10))

l = []

from nqgl.mlutils.norepr import fastpartial


def test_hook(module, input, output, l):
    out = output.detach().clone()
    out.requires_grad = True
    l.append(out)
    return out
    # return embed


hook = fastpartial(test_hook, l=l)
model.get_input_embeddings().register_forward_hook(hook)
out = model(t)
# print(out.last_hidden_state.shape)
for i in out:
    print(i)

print(type(out))
logits = out.logits
loss = logits.abs().sum()
loss.backward()
print(l[0].grad)
# %%


import torch

a = torch.ones(2, requires_grad=True)
b = torch.ones(3, requires_grad=False)
c = torch.cat((a, b))
output = c.sum()
output.backward()
print(a.grad)  # tensor([1., 1.])
print(b.grad)  # tensor([1., 1., 1.])


# %%
from jaxtyping import Float
from torch import Tensor
import torch
from dataclasses import dataclass


@dataclass
class Batch:
    embeddings: Float[Tensor, "batch seq d_model"]
    logits: str


b = Batch(embeddings=torch.ones(1, 10, 10), logits="hi")
b2 = Batch(embeddings={}, logits="hi")

# %%
# model.transformer.wte
# model.embed

# %%
from arena_capstone.gcg.embeddingmodel import EmbeddingFriendlyCausalForLM

# %%
m = EmbeddingFriendlyCausalForLM(model)
t = torch.randint(0, model.config.vocab_size, (1, 10))

e = m.embed(t)
m.forward_from_embed(e)
# %%
e = model.get_input_embeddings()(t)
we = model.transformer.wte(t)
wp = model.transformer.wpe(torch.arange(0, 10).reshape(1, 10))


# %%
# torch.allclose(e, we)
o = m.forward_from_embed(e)
# %%
type(o)
o.logits.shape
# %%

normal_logits = model(t).logits
# do deconstructed
e = m.embed(t)
cool_logits = m.forward_from_embed(e).logits
torch.allclose(normal_logits, cool_logits)

# %%
wte = model.transformer.wte

# %%
import torch.nn.functional as F

hot = F.one_hot(t, num_classes=model.config.vocab_size)
hot = hot.float()
wte.weight.shape
ehot = hot @ wte.weight
ehot.shape


torch.allclose(wte.weight[t], wte(t))
# %%
e.shape
# %%


a = torch.arange(0, 25).reshape(5, 5)
m = F.dropout(torch.ones(5, 5), p=0.8, training=True)
