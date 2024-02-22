"""
target = self.model.generate(
    prompt.unsqueeze(0),
    # attention_mask=all_ones_mask.unsqueeze(0),
    # repitition_penalty=1.2,
    max_length=self.cfg.generate_length + prompt.shape[0],
    do_sample=True,
    # eos_token_id=self.tokenizer.eos_token_id,
    # bos_token_id=self.tokenizer.bos_token_id,
    # pad_token_id=self.tokenizer.pad_token_id,
    # temperature=1,
    attention_mask=all_ones_mask.unsqueeze(0),
    pad_token_id=self.tokenizer.pad_token_id,
    bad_words_ids=[[bad] for bad in bad_tokens],
).squeeze()
"""

import torch
from torch import Tensor
from transformers import LlamaForCausalLM
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM
import torch.nn.functional as F

# from arena_capstone.algorithm
# from arena_capstone.algorithm


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    tau_backward: float = None,
    hard: bool = False,
    noise_scale: float = 1,
):
    # tau_backward = tau_backward or tau
    gumbels = (
        -torch.empty_like(logits).exponential_().log() * noise_scale
    )  # ~Gumbel(0,1)

    input_gumbels = (logits / noise_scale + gumbels) / tau  # ~Gumbel(logits,tau)

    y_soft = F.softmax(input_gumbels, dim=-1)
    if hard:
        y_hard = y_soft.max(-1, keepdim=True)[0].eq(y_soft).float()
        return y_hard - y_soft.detach() + y_soft
    if not tau_backward:
        return y_soft
    input_gumbels_bak = (logits / noise_scale + gumbels) / tau_backward
    y_soft_bak = F.softmax(input_gumbels_bak, dim=-1)
    return y_soft.detach() + y_soft_bak - y_soft_bak.detach()
