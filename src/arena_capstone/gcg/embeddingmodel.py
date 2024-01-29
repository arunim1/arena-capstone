import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
from typing import List
from torch import Tensor
from dataclasses import dataclass


@dataclass
class Batch:
    embeddings: Float[Tensor, "batch seq d_model"]
    target_mask: Bool[Tensor, "batch seq"]
    suffix_tensor: Float[Tensor, "suffix_length d_vocab"]


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
    ) -> Batch:
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

    def embed(self, tokens, start_position=0, onehot=False):
        seq_len = tokens.shape[1]
        batch_size = tokens.shape[0]
        if onehot:
            we = tokens @ self.model.transformer.wte.weight
        else:
            we = self.model.transformer.wte(tokens)
        wp = self.model.transformer.wpe(
            torch.arange(start_position, seq_len + start_position).reshape(1, seq_len)
        )
        return we + wp

    def forward_from_embed(self, embed):
        return self.model(inputs_embeds=embed)

    def splice_suffix(
        self,
        prefixes: List[Int[Tensor, "prefix_len"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        targets: List[Int[Tensor, "target_len"]],
    ):
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
        batch = Batch(
            embeddings=torch.stack(sequences),
            target_mask=torch.stack(masks),
            suffix_tensor=hot_suffix,
        )
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
        prefix = self.embed(prefix_tokens, start_position=0)
        suffix = self.embed(hot_suffix, start_position=suffix_start, onehot=True)
        target = self.embed(target_tokens, start_position=target_start)
        padding = torch.zeros(sequence_length - seq_length, device=prefix.device)
        sequence = torch.cat([prefix, suffix, target, padding])
        mask = torch.zeros_like(sequence, dtype=torch.bool)
        mask[target_start:seq_length] = True
        return sequence, mask

    def suffix_to_hot(self, target_tokens: Int[Tensor, "batch suffix_len"]):
        """
        target_tokens: Int[Tensor, "batch suffix_len"]
                suffix tokens

        returns: Int[Tensor, "batch suffix_len vocab"]
                the one-hot suffix tokens
        """
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
