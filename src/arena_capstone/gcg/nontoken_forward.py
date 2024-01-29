from unpythonic import box
from typing import List, Tuple, Union
from jaxtyping import Float, Int, Bool
from torch import Tensor
import torch

# from nqgl.mlutils.norepr import fastpartial
from functools import partial
from transformers import AutoModelForCausalLM
from yaml import Token


class WrappedModel:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        capture_one_hot=True,
        capture_embeddings=False,
    ):
        """ """
        self.model = model
        assert callable(model)
        assert capture_one_hot or capture_embeddings
        if capture_one_hot:
            self.add_capture_one_hot_hook()
        if capture_embeddings:
            self.add_capture_embeddings_hook()
        self.captured_embeddings = box()
        self.captured_one_hot = box()

    def add_capture_one_hot_hook(self):
        hook = partial(
            TokenGradients._capture_embedding_hook,
            captured_embeddings=self.captured_embeddings,
        )
        self.model.get_input_embeddings().register_forward_hook(hook)

    def add_capture_embeddings_hook(self):
        hook = partial(
            TokenGradients._capture_embedding_hook,
            captured_embeddings=self.captured_embeddings,
        )
        self.model.get_input_embeddings().register_forward_hook(hook)

    def top_k_substitutions(
        self,
        tokens: Int[Tensor, "batch seq"],
        targets_mask: Bool[Tensor, "batch seq"],
        prefixes_mask,
        k: int,
    ):

        k = self.k if k is None else k
        embedded_tokens = self.model.embeddings(tokens)
        embedded_tokens.requires_grad = True
        logits = self.model(embedded_tokens)

        target_logprobs = logits[
            targets_mask
        ]  # this doesn't work bc targets are variable length

    def get_loss(
        self,
        logits: Float[Tensor, "batch seq vocab"],
        target_mask: Bool[Tensor, "batch seq"],
        tokens: Int[Tensor, "batch seq"],
    ):
        # returns loss without backpropagating (because that would be a dumb design choice NGL)
        logprobs = torch.log_softmax(logits, dim=-1)
        target_ids = tokens[target_mask]
        loss = torch.nn.CrossEntropyLoss()(
            logits[:, :-1][target_mask[:, 1:]], target_ids
        )
        # F.cross_entropy(logits[:, :-1][target_mask[:, 1:]], target_ids)
        # sequence P S G G G
        # logits   S G G G 0
        # target   P S G G G
        # tmask    0 0 1 1 1

        # target[tmask]         G G G
        # logits[:-1][tmask[1:]]G G G

        # loss = torch.sum(logprobs[target_mask][target_tokens.flatten()])

        # L(x[1:n]) = -log(p(x[n+1] | x[1:n))

        return loss

    def get_token_gradients(self, input_tokens, target_tokens, target_mask):
        embeddings = self.captured_embeddings.x
        logits = self.model(input_tokens).logits
        loss = self.get_loss()
        loss.backward()
        return embeddings.grad

    @staticmethod
    def _capture_one_hot_hook(module, input, output, captured_embeddings: box):
        out = module()
        out = output.detach().clone()
        out.requires_grad = True
        captured_embeddings << out
        return out

    @staticmethod
    def _capture_embedding_hook(module, input, output, captured_embeddings: box):
        out = output.detach().clone()
        out.requires_grad = True
        captured_embeddings << out
        return out
