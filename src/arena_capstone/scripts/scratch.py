import einops
import torch

padded = 2
seq_len = 20
target_len = 5
d_vocab = 100
low = seq_len - target_len - padded - 1
high = seq_len - padded - 1
batch_size = 8

target = torch.randint(0, d_vocab, (target_len,))
logits = torch.rand(batch_size, seq_len, d_vocab)

indexed = logits[:, torch.arange(low, high)]
print(indexed.shape)
indexed = einops.rearrange(indexed, "batch seq vocab -> batch vocab seq")

loss = torch.nn.functional.cross_entropy(
    indexed, target.unsqueeze(0).expand(batch_size, target_len), reduction="none"
)
loss = loss.mean(dim=-1)
assert loss.ndim == 1
print(loss.shape)
