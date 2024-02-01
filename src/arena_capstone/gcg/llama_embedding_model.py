# from arena_capstone.gcg.embedding_model import (
#     EmbeddingFriendlyForCausalLM,
#     TokensBatch,
#     EmbeddedBatch,
# )
# import torch

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     LlamaTokenizer,
#     LlamaForCausalLM,
# )
# from nqgl.mlutils.time_gpu import profilefunc_wrapper, ProfileFunc


# class EmbeddingFriendlyForCausalLM(EmbeddingFriendlyForCausalLM):
#     model: LlamaForCausalLM

#     def __init__(self, model: LlamaForCausalLM):
#         super().__init__(model)

#     def embed(self, tokens_or_onehot, start_position=0, onehot=False):
#         wte = self.model.get_input_embeddings()
#         if onehot:
#             we = tokens_or_onehot @ wte.weight
#         else:
#             we = wte(tokens_or_onehot)
#         return we.unsqueeze(0)


# def main(model, embedding_model):
#     with torch.inference_mode():
#         with torch.cuda.amp.autocast():
#             prefixes = [
#                 torch.randint(0, model.config.vocab_size, (10,), device="cuda"),
#                 torch.randint(0, model.config.vocab_size, (5,), device="cuda"),
#                 # torch.randint(0, model.config.vocab_size, (5,), device="cuda"),
#             ]
#             suffix = torch.randint(0, model.config.vocab_size, (3,), device="cuda")
#             targets = [
#                 torch.randint(0, model.config.vocab_size, (5,), device="cuda"),
#                 torch.randint(0, model.config.vocab_size, (10,), device="cuda"),
#                 # torch.randint(0, model.config.vocab_size, (5,), device="cuda"),
#             ]
#             batch = embedding_model.splice_suffix(prefixes, suffix, targets)
#             # targets[-1] = torch.cat(
#             #     [
#             #         targets[-1],
#             #         torch.randint(0, model.config.vocab_size, (5,), device="cuda"),
#             #     ]
#             # )
#             tokens = torch.stack(
#                 [
#                     torch.cat([prefix, suffix, target], dim=0)
#                     for prefix, target in zip(prefixes, targets)
#                 ],
#                 dim=0,
#             )

#             response = model(tokens)
#             logits = response.logits
#             embed_logits = embedding_model.forward_from_embed(batch.embeddings).logits
#             # while True:
#             #     try:
#             #         print(exec(input(">>>")))
#             #     except Exception as e:
#             #         print("oops:\n", e)
#             print(logits - embed_logits)
#             print(torch.max(torch.abs(logits - embed_logits)))
#             print(torch.mean(torch.abs(logits)))
#             print(torch.mean(torch.abs(logits - embed_logits), dim=1))
#             assert torch.allclose(
#                 logits[:2],
#                 embed_logits[:2],
#                 atol=2e-0,
#                 rtol=1e-1,
#             )


# if __name__ == "__main__":
#     from arena_capstone.scripts.run_with_llama import get_llama

#     llamamodel, embedding_llamamodel, tokenizer = get_llama()
#     main = ProfileFunc(main, "llama_main")
#     main(llamamodel, embedding_llamamodel)
