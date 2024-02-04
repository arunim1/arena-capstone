from distutils.dep_util import newer
import torch
d_vocab = 100

seq = 5
batch = 2
d_model = 32


mask = torch.randint(0, 2, (batch, seq), dtype=torch.bool)
mask_indices = mask.nonzero()

embedded_logits = torch.randn((batch, seq, d_model))

flat_embedded_logits = embedded_logits[mask]
print("nnz", torch.sum(mask))
print("flat_embedded_logits.shape", flat_embedded_logits.shape)
new_embed = torch.randn((batch, seq, d_model))

print(new_embed[mask])
exp_mask_indices = mask_indices.unsqueeze(-1).expand(-1, -1, d_model)
print("exp_mask_indices.shape", exp_mask_indices.shape)
# newer_embed = torch.scatter(
#     new_embed,
#     0,
#     index = exp_mask_indices,
#     src = flat_embedded_logits
# )
print("new_embed", new_embed.shape)
print("mask", mask.shape)
print("flat_embedded_logits", flat_embedded_logits.shape)
newer_embed = torch.masked_scatter(
    new_embed,
    mask.unsqueeze(-1),
    flat_embedded_logits
)

other_newer_embed = new_embed.clone()
other_newer_embed[mask] = embedded_logits[mask]
print(other_newer_embed - newer_embed)