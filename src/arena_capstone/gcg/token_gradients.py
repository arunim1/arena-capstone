# Glen Taggart / nqgl if there are any issues/questions


from typing import List, Tuple, Union, Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor
import torch

# from nqgl.mlutils.norepr import fastpartial
import torch.nn.functional as F
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from arena_capstone.gcg.embedding_model import (
    EmbeddingFriendlyCausalForLM,
    EmbeddingFriendlyModel,
    Batch,
)

DEBUG = False


class TokenGradients:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        embedding_model: Optional[EmbeddingFriendlyModel] = None,
    ):
        """ """
        assert callable(model)
        self.model = model
        self.embedding_model = (
            EmbeddingFriendlyCausalForLM(model)
            if embedding_model is None
            else embedding_model
        )

    def get_loss(
        self,
        batch: Batch,
        targets: List[Int[Tensor, "seq"]],
    ):
        # returns loss without backpropagating (because that would be a dumb design choice NGL)
        target_ids = torch.cat(targets, dim=0)
        logprobs = torch.log_softmax(batch.logits, dim=-1)
        dprint(logprobs.shape, target_ids.shape, batch.target_mask.shape)
        dprint(logprobs[:, :-1][batch.target_mask[:, 1:]].shape)
        dprint(torch.sum(batch.target_mask))
        loss = F.cross_entropy(logprobs[:, :-1][batch.target_mask[:, 1:]], target_ids)
        alt = logprobs[:, :-1][batch.target_mask[:, 1:]]
        dprint(alt.shape, target_ids.shape)
        loss = torch.sum(alt[torch.arange(0, target_ids.shape[0]), target_ids])

        # F.cross_entropy(logits[:, :-1][target_mask[:, 1:]], target_ids)
        # sequence P S G G G
        # logits   S G G G 0
        # target   P S G G G
        # tmask    0 0 1 1 1

        # target[tmask]         G G G
        # logits[:-1][tmask[1:]]G G G

        # L(x[1:n]) = -log(p(x[n+1] | x[1:n))

        return loss

    def get_token_gradients(
        self,
        prefixes: List[Int[Tensor, "prefix_len"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        targets: List[Int[Tensor, "target_len"]],
    ) -> Batch:
        batch = self.embedding_model.splice_suffix(
            prefixes, suffix_tokens, targets, get_logits=True
        )
        batch.suffix_tensor.grad = None  # zero grad
        loss = self.get_loss(batch, targets)
        loss.backward()
        return batch


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test_prefix_str = "In a pytree node, static fields will be treated as part of "
    suffix_len = 5
    suffix = torch.randint(0, model.config.vocab_size, (suffix_len,))
    test_target_str = (
        " not explicitly marked static should contain arrays or child nodes."
    )
    prefix = tokenizer.encode_plus(test_prefix_str, return_tensors="pt").input_ids
    target = tokenizer.encode_plus(test_target_str, return_tensors="pt").input_ids
    print(prefix)

    tg = TokenGradients(
        model,
    )
    for i in range(100):
        tok_grads = tg.get_token_gradients([prefix[0]], suffix, [target[0]])
        # print(tok_grads.suffix_tensor.grad)
        browndogmaybe = tok_grads.suffix_tensor.grad
        tokens = torch.argmin(browndogmaybe, dim=-1)
        # print(tokens)
        # print(tokenizer.decode(tg.suffix))
        print(tokenizer.decode(tokens + 1))
        tok_grads.suffix_tensor.requires_grad = False
        tok_grads.suffix_tensor.grad = None
        suffix = tokens


if __name__ == "__main__":
    main()
