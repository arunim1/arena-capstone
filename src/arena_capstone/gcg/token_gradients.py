# Glen Taggart / nqgl if there are any issues/questions


from functools import partial
from tokenize import TokenInfo
from typing import List, Optional, Tuple, Union

import torch
# from nqgl.mlutils.norepr import fastpartial
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from arena_capstone.gcg.embedding_model import (EmbeddedBatch,
                                                EmbeddingFriendlyCausalForLM,
                                                EmbeddingFriendlyModel)

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
        batch: EmbeddedBatch,
        targets: List[Int[Tensor, "seq"]],
        reduce_over_batch=True,
    ):
        # returns loss without backpropagating (because that would be a dumb design choice NGL)
        target_ids = torch.cat(targets, dim=0)
        logprobs = torch.log_softmax(batch.logits, dim=-1)
        dprint(logprobs.shape, target_ids.shape, batch.target_mask.shape)
        dprint(logprobs[:, :-1][batch.target_mask[:, 1:]].shape)
        dprint(torch.sum(batch.target_mask))

        logprobs_at_targets = logprobs[:, :-1][batch.target_mask[:, 1:]]
        # sequence P S G G G
        # logits   S G G G 0
        # target   P S G G G
        # tmask    0 0 1 1 1

        loss = F.cross_entropy(
            logprobs_at_targets,
            target_ids,
            reduction="mean" if reduce_over_batch else "none",
        )

        losses2 = -logprobs_at_targets[torch.arange(0, target_ids.shape[0]), target_ids]
        loss2 = losses2
        if reduce_over_batch:
            loss2 = torch.mean(losses2)
            assert torch.allclose(loss, loss2, atol=1e-0)
        else:
            # print(loss, losses2)
            assert torch.allclose(loss, losses2, atol=1e-0)

        # Loss(prefix+suffix) = -log(p(target|suffixprefix))
        # = -log(prod_i(p(target_i | suffixprefix, target_1, ..., target_(i - 1)))
        # = -sum_i(log(p(target_i | suffixprefix, target_1, ..., target_(i - 1))))

        # target[tmask]         G G G
        # logits[:-1][tmask[1:]]G G G

        # L(x[1:n]) = -log(p(x[n+1] | x[1:n))

        return loss

    def get_token_gradients(
        self,
        prefixes: List[Int[Tensor, "prefix_len"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        targets: List[Int[Tensor, "target_len"]],
    ) -> EmbeddedBatch:
        batch = self.embedding_model.splice_suffix(
            prefixes, suffix_tokens, targets, get_logits=True
        )
        assert batch.suffix_tensor.grad is None  # zero grad
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
        tok_grads_batch = tg.get_token_gradients([prefix[0]], suffix, [target[0]])
        # print(tok_grads_batch.suffix_tensor.grad)
        browndogmaybe = tok_grads_batch.suffix_tensor.grad
        tokens = torch.argmin(browndogmaybe, dim=-1)
        # print(tokens)
        # print(tokenizer.decode(tg.suffix))
        print(tokenizer.decode(tokens + 1))
        tok_grads_batch.suffix_tensor.requires_grad = False
        tok_grads_batch.suffix_tensor.grad = None
        suffix = tokens


if __name__ == "__main__":
    main()
