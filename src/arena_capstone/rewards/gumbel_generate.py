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
from transformers import LlamaForCausalLM
from arena_capstone.algorithm.embedding_model import EmbeddingFriendlyForCausalLM

# from arena_capstone.algorithm
# from arena_capstone.algorithm


def generate(
    model: LlamaForCausalLM,
    embedding_model: EmbeddingFriendlyForCausalLM,
    gumbel_softmax,
    reward_model,
    prefix,
    suffix,
    post_suffix,
    max_length,
    bad_words_ids=[],
):

    batch = embedding_model.splice_embedded_batch(
        prefix, suffix, post_suffix, targets=[]
    )
    generate_length = max_length - prefix.shape[1]

    for i in range(generate_length):
        logits_next = embedding_model.forward_from_embed(batch.embeddings)
        one_hot_next_token = gumbel_softmax(logits_next)
        one_hot_embedded = embedding_model.embed(one_hot_next_token, onehot=True)
        batch.embeddings = torch.cat([batch.embeddings, one_hot_embedded], dim=1)
