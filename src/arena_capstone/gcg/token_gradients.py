# Glen Taggart / nqgl if there are any issues/questions


from tokenize import TokenInfo
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
    EmbeddedBatch,
    TokensBatch,
)
import einops

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
        targets: Union[List[Int[Tensor, "seq"]], Int[Tensor, "seq"]],
        reduce_over_batch=True,
        targets_is_single_tensor=False,
    ):
        # sequence P S G G G
        # logits   S G G G 0
        # target   P S G G G
        # tmask    0 0 1 1 1

        target_ids = torch.cat(targets, dim=0)
        logits_at_targets = batch.logits[:, :-1][batch.target_mask[:, 1:]]

        loss = F.cross_entropy(
            logits_at_targets,
            target_ids,
            reduction="mean" if reduce_over_batch else "none",
        )

        return loss

    def get_loss_looping(
        self,
        batch: TokensBatch,
        target: Int[Tensor, "seq"],
    ):
        """
        get loss for when looping memory use optimization in UPO algorithm
        """

        low, high = batch.target_bounds

        indexed = batch.logits[:, torch.arange(low, high)]
        indexed = einops.rearrange(indexed, "batch seq vocab -> batch vocab seq")

        batch_size = indexed.shape[0]
        target_len = high - low

        loss = F.cross_entropy(
            indexed,
            target.unsqueeze(0).expand(batch_size, target_len),
            reduction="none",
        )
        return loss.mean(dim=-1)

    def get_token_gradients(
        self,
        prefixes: List[Int[Tensor, "prefix_len"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        targets: List[Int[Tensor, "target_len"]],
        looping: bool = False,
    ) -> EmbeddedBatch:
        batch = self.embedding_model.splice_suffix(
            prefixes,
            suffix_tokens,
            targets,
            get_logits=True,
            batch_gets_bounds_instead_of_mask=looping,
        )
        assert batch.suffix_tensor.grad is None  # zero grad
        loss = (
            self.get_loss(batch, targets)
            if not looping
            else self.get_loss_looping(batch, targets)
        )
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
