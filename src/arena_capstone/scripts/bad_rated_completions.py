from arena_capstone.rewards.reward_generator import RewardGenerator, get_reward_generator
from transformers import LogitsProcessor, BeamScorer 
from arena_capstone.scripts.run_with_llama import get_llama, get_llama_tokenizer

# model, embedding_friendly, tokenizer = get_llama()

import arena_capstone.scripts.llamatokenize as llamatokenize

# outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
tokenizer = get_llama_tokenizer()
prefix_tok, nonprefix_tok, detok = llamatokenize.tokenizations(tokenizer)

print("tokens:", llamatokenize.detokenize(tokenizer, [29973]))

print("tokens:", llamatokenize.detokenize(tokenizer, [29991]))

print(llamatokenize.prefixes_tokens(tokenizer, ["?", " ?"]))
print(nonprefix_tok(["?", " ?"]))


print(
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
    tokenizer.unk_token_id,
    tokenizer.cls_token_id,
    tokenizer.bos_token_id,
)
