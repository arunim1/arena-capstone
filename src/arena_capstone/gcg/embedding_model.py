# Glen Taggart / nqgl if there are any issues/questions

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
from typing import List, Optional
from torch import Tensor
from dataclasses import dataclass
from transformers import LlamaModel, LlamaPreTrainedModel, PreTrainedModel


DEBUG = False


@dataclass
class EmbeddedBatch:
    embeddings: Float[Tensor, "batch seq d_model"]
    target_mask: Bool[Tensor, "batch seq"]
    suffix_tensor: Float[Tensor, "suffix_length d_vocab"]
    logits: Optional[Float[Tensor, "batch seq vocab"]]


class EmbeddingFriendlyModel:
    def embed(self, tokens):
        pass

    def forward_from_embed(self, embed):
        """
        embed: Float[Tensor, "batch seq d_model"]
        returns: Float[Tensor, "batch seq d_model"]

        does a forward pass from embeddings
        """
        pass

    def splice_suffix(
        self,
        prefixes: List[Int[Tensor, "prefix_lens"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        targets: List[Int[Tensor, "target_lens"]],
    ) -> EmbeddedBatch:
        """
        prefixes: Int[Tensor, "batch prefix_len"]
                prefix tokens
        suffix_tokens: Int[Tensor, "batch suffix_len"]
                suffix_tokens tokens
        targets: Int[Tensor, "batch target_len"]
                target tokens

        returns: Float[Tensor, "batch sequence_length d_model"], Bool[Tensor, "batch seq"], Float[Tensor, "batch suffix_len d_vocab"]
        """


class EmbeddingFriendlyCausalForLM(EmbeddingFriendlyModel):
    def __init__(self, model: PreTrainedModel):
        self.model = model

    def embed(self, tokens_or_onehot, start_position=0, onehot=False):
        seq_len = tokens_or_onehot.shape[0]
        dprint("seq_len", seq_len, start_position)
        dprint("shape", tokens_or_onehot.shape)
        if onehot:
            we = tokens_or_onehot @ self.model.transformer.wte.weight
        else:
            we = self.model.transformer.wte(tokens_or_onehot)
        return we.unsqueeze(0)

    def forward_from_embed(self, embed):
        return self.model(inputs_embeds=embed)

    def splice_suffix(
        self,
        prefixes: List[Int[Tensor, "prefix_len"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        targets: List[Int[Tensor, "target_len"]],
        get_logits=False,
    ) -> EmbeddedBatch:
        """
        prefixes: Int[Tensor, "batch prefix_len"]
                prefix tokens
        suffix_tokens: Int[Tensor, "batch suffix_len"]
                suffix_tokens tokens
        targets: Int[Tensor, "batch target_len"]
                target tokens

        returns:
                Float[Tensor, "batch sequence_length d_model"],
                Bool[Tensor, "batch seq"],
                Float[Tensor, "batch suffix_len d_vocab"]
        """
        sequence_length = self.get_max_len(prefixes, suffix_tokens, targets)
        sequences = []
        masks = []
        hot_suffix = self.suffix_to_hot(suffix_tokens)
        for prefix_tokens, target_tokens in zip(prefixes, targets):
            sequence, mask = self._splice(
                prefix_tokens, hot_suffix, target_tokens, sequence_length
            )
            sequences.append(sequence)
            masks.append(mask)
            dprint(sequence.shape, mask.shape)
        batch = EmbeddedBatch(
            embeddings=torch.cat(sequences, dim=0),
            target_mask=torch.stack(masks),
            suffix_tensor=hot_suffix,
            logits=None,
        )
        assert batch.target_mask.ndim == 2
        assert batch.embeddings.ndim == 3

        assert batch.target_mask.shape[0:2] == batch.embeddings.shape[0:2]
        if get_logits:
            batch.logits = self.forward_from_embed(batch.embeddings).logits
        return batch

    def _splice(
        self,
        prefix_tokens: Int[Tensor, "prefix_len"],
        hot_suffix: Float[Tensor, "suffix_len vocab"],
        target_tokens: Int[Tensor, "target_len"],
        sequence_length: int,
    ):
        suffix_start = prefix_tokens.shape[0]
        target_start = suffix_start + hot_suffix.shape[0]
        seq_length = target_start + target_tokens.shape[0]
        dprint(suffix_start, target_start, seq_length)
        prefix = self.embed(prefix_tokens, start_position=0)
        suffix = self.embed(hot_suffix, start_position=suffix_start, onehot=True)
        target = self.embed(target_tokens, start_position=target_start)
        padding = (
            torch.zeros(sequence_length - seq_length, device=prefix.device)
            .unsqueeze(-1)
            .expand(-1, self.model.config.hidden_size)
            .unsqueeze(0)
        )
        dprint(prefix.shape, suffix.shape, target.shape, padding.shape)
        sequence = torch.cat([prefix, suffix, target, padding], dim=1)
        mask = torch.zeros(sequence_length, dtype=torch.bool, device=sequence.device)
        dprint("mask", target_start, seq_length)
        mask[target_start:seq_length] = True
        return sequence, mask

    def suffix_to_hot(self, target_tokens: Int[Tensor, "suffix_len"]):
        """
        target_tokens: Int[Tensor, "batch suffix_len"]
                suffix tokens

        returns: Int[Tensor, "batch suffix_len vocab"]
                the one-hot suffix tokens
        """
        assert target_tokens.ndim == 1
        hot = F.one_hot(target_tokens, num_classes=self.model.config.vocab_size)
        hot = hot.float()
        hot.requires_grad = True
        return hot

    def get_max_len(
        self,
        prefixes_tokenized: List[Int[Tensor, "prefix_len"]],
        suffix: Int[Tensor, "suffix_len"],
        targets_tokenized: List[Int[Tensor, "target_len"]],
    ):
        assert len(prefixes_tokenized) == len(targets_tokenized)
        prefix_lengths = [prefix.shape[0] for prefix in prefixes_tokenized]
        target_lengths = [target.shape[0] for target in targets_tokenized]

        return suffix.shape[0] + max(
            [plen + tlen for plen, tlen in zip(prefix_lengths, target_lengths)]
        )


def main():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    with torch.inference_mode():
        embedding_model = EmbeddingFriendlyCausalForLM(model)
        prefixes = [
            torch.randint(0, model.config.vocab_size, (10,)),
            torch.randint(0, model.config.vocab_size, (5,)),
            torch.randint(0, model.config.vocab_size, (5,)),
        ]
        suffix = torch.randint(0, model.config.vocab_size, (3,))
        targets = [
            torch.randint(0, model.config.vocab_size, (5,)),
            torch.randint(0, model.config.vocab_size, (10,)),
            torch.randint(0, model.config.vocab_size, (5,)),
        ]
        batch = embedding_model.splice_suffix(prefixes, suffix, targets)
        dprint(batch)
        targets[-1] = torch.cat(
            [targets[-1], torch.randint(0, model.config.vocab_size, (5,))]
        )
        tokens = torch.stack(
            [
                torch.cat([prefix, suffix, target], dim=0)
                for prefix, target in zip(prefixes, targets)
            ],
            dim=0,
        )

        dprint("tokens", tokens.shape)
        dprint("batch", batch.embeddings.shape)
        response = model(tokens)
        logits = response.logits
        embed_logits = embedding_model.forward_from_embed(batch.embeddings).logits
        assert torch.allclose(
            logits[:2],
            embed_logits[:2],
            atol=1e-2,
        )
        assert torch.allclose(
            logits[2, :-5],
            embed_logits[2, :-5],
            atol=1e-2,
        )

        dprint("success")


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


if __name__ == "__main__":
    main()
