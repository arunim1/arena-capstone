from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache, Cache

model: LlamaForCausalLM

cache = DynamicCache()

model(
    num_new_tokens=1,
)


input_ids
inputs_embeds

attention_mask
past_key_values=cache,
labels=
use_cache=True
output_attentions
output_hidden_states
return_dict