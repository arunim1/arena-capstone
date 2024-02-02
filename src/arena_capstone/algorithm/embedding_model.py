# Glen Taggart (nqgl) if there are any issues/questions

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Tuple, Union
from torch import Tensor
from transformers import AutoModelForCausalLM, PreTrainedModel

DEBUG = False


@dataclass
class EmbeddedBatch:
    embeddings: Float[Tensor, "batch seq d_model"]
    target_mask: Optional[Bool[Tensor, "batch seq"]]
    suffix_tensor: Float[Tensor, "suffix_length d_vocab"]
    logits: Optional[Float[Tensor, "batch seq vocab"]]
    outputs: Union[Tuple, CausalLMOutputWithPast]


@dataclass
class TokensBatch:
    tokens: Int[Tensor, "batch seq"]
    logits: Optional[Float[Tensor, "batch seq vocab"]]
    target_bounds: Optional[Tuple[Int]]


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

    def splice_embedded_batch(
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


class EmbeddingFriendlyForCausalLM(EmbeddingFriendlyModel):
    def __init__(self, model: PreTrainedModel):
        self.model = model

    def embed(self, tokens_or_onehot, start_position=0, onehot=False):
        if hasattr(self.model, "transformer"):
            # GPT2
            wte = self.model.transformer.wte
        else:
            # Llama
            wte = self.model.get_input_embeddings()
        if onehot:
            we = tokens_or_onehot @ wte.weight
        else:
            we = wte(tokens_or_onehot)
        return we.unsqueeze(0)

    def forward_from_embed(self, embed):
        return self.model(inputs_embeds=embed)

    def splice_embedded_batch(
        self,
        prefixes: List[Int[Tensor, "prefix_len"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        post_suffix_tokens: Int[Tensor, "post_suffix_len"],
        targets: List[Int[Tensor, "target_len"]],
        get_logits=False,
    ) -> EmbeddedBatch:
        """
        prefixes: List[Int[Tensor, "prefix_len"]]
                prefix tokens
        suffix_tokens: Int[Tensor, "batch suffix_len"]
                suffix_tokens tokens
        post_suffix_tokens: Int[Tensor, "batch post_suffix_len"]
                post_suffix_tokens tokens
        targets: List[Int[Tensor, "target_len"]]
                target tokens

        returns:
                Float[Tensor, "batch sequence_length d_model"],
                Bool[Tensor, "batch seq"],
                Float[Tensor, "batch suffix_len d_vocab"]
        """
        sequence_length = self._get_max_len(
            prefixes, suffix_tokens, post_suffix_tokens, targets
        )
        sequences = []
        mask_list = []
        hot_suffix = self._suffix_to_hot(suffix_tokens)
        for prefix_tokens, target_tokens in zip(prefixes, targets):
            sequence, mask = self._splice_single_embedded_batch(
                prefix_tokens,
                hot_suffix,
                post_suffix_tokens,
                target_tokens,
                sequence_length,
            )
            sequences.append(sequence)
            mask_list.append(mask)
        batch = EmbeddedBatch(
            embeddings=torch.cat(sequences, dim=0),
            target_mask=(torch.stack(mask_list)),
            suffix_tensor=hot_suffix,
            logits=None,
            outputs=None,
        )
        assert batch.target_mask is None or (
            batch.target_mask.ndim == 2
            and batch.target_mask.shape[0:2] == batch.embeddings.shape[0:2]
        )
        assert batch.embeddings.ndim == 3

        if get_logits:
            outputs = self.forward_from_embed(batch.embeddings)
            batch.logits = outputs.logits
            batch.outputs = outputs
        return batch

    def _suffix_to_hot(self, suffix_tokens: Int[Tensor, "suffix_len"]):
        """
        suffix_tokens: Int[Tensor, "batch suffix_len"]
                suffix tokens

        returns: Int[Tensor, "batch suffix_len vocab"]
                the one-hot suffix tokens
        """
        assert suffix_tokens.ndim == 1
        hot = F.one_hot(suffix_tokens, num_classes=self.model.config.vocab_size)
        hot = hot.half()
        hot.requires_grad = True
        return hot

    def _splice_single_embedded_batch(
        self,
        prefix_tokens: Int[Tensor, "prefix_len"],
        hot_suffix: Float[Tensor, "suffix_len vocab"],
        post_suffix_tokens: Int[Tensor, "post_suffix_len"],
        target_tokens: Int[Tensor, "target_len"],
        sequence_length: int,
    ):
        suffix_start = prefix_tokens.shape[0]

        target_start = suffix_start + hot_suffix.shape[0] + post_suffix_tokens.shape[0]
        target_end = target_start + target_tokens.shape[0]

        prefix = self.embed(prefix_tokens, start_position=0)
        suffix = self.embed(hot_suffix, start_position=suffix_start, onehot=True)
        post_suffix = self.embed(post_suffix_tokens)
        target = self.embed(target_tokens)
        padding = (
            torch.zeros(sequence_length - target_end, device=prefix.device)
            .unsqueeze(-1)
            .expand(-1, self.model.config.hidden_size)
            .unsqueeze(0)
        )
        sequence = torch.cat([prefix, suffix, post_suffix, target, padding], dim=1)
        mask = torch.zeros(sequence_length, dtype=torch.bool, device=sequence.device)
        mask[target_start:target_end] = True
        return sequence, mask

    def splice_tokens_batch(
        self,
        prefix: Int[Tensor, "prefix_lens"],  # m_c list len
        suffix_tokens: Int[Tensor, "batch_size suffix_len"],
        post_suffix: Int[Tensor, "post_suffix_len"],
        target: Int[Tensor, "target_lens"],  # m_c list len
        get_logits=False,
    ) -> TokensBatch:
        """
        Splices suffix tokens to the given prefixes and targets to create a batch of tokens.

        Args:
            prefixes (List[Int[Tensor, "prefix_lens"]]): A list of prefix token strings for each sequence in the batch.
            suffix_tokens (Int[Tensor, "batch_size suffix_len"]): The suffix token string to be spliced to the prefixes.
            targets (List[Int[Tensor, "target_lens"]]): A list of target token strings for each sequence in the batch.
            get_logits (bool, optional): Whether to compute logits for the batch. Defaults to False.

        Returns:
            TokensBatch: A batch of tokens with spliced suffix tokens.
        """
        batch = self._splice_tokens_batch(prefix, suffix_tokens, post_suffix, target)
        if get_logits:
            batch.logits = self.model(batch.tokens).logits
        return batch

    @classmethod
    def _splice_tokens_batch(
        cls, prefix, suffix_tokens, post_suffix, target
    ) -> TokensBatch:
        batch_size = suffix_tokens.shape[0]
        all_sequences = []
        all_bounds = []
        for b in range(batch_size):
            sequences, bounds = cls._splice_single_tokens_batch(
                prefix,
                suffix_tokens[b],
                post_suffix,
                target,
            )
            all_sequences.append(sequences)
            all_bounds.append(bounds)

        batch = TokensBatch(
            tokens=torch.stack(all_sequences, dim=0),
            logits=None,
            target_bounds=bounds,
        )
        assert batch.tokens.ndim == 2
        return batch

    @classmethod
    def _get_max_len(
        cls,
        prefixes_tokenized: List[Int[Tensor, "prefix_len"]],
        suffix: Int[Tensor, "suffix_len"],
        post_suffix: Int[Tensor, "post_suffix_len"],
        targets_tokenized: List[Int[Tensor, "target_len"]],
    ):
        """
        prefixes_tokenized: List[Int[Tensor, "batch prefix_len"]]
                prefix tokens
        suffix: Int[Tensor, "batch suffix_len"]
                suffix tokens
        targets_tokenized: List[Int[Tensor, "batch target_len"]]
                target tokens

        returns: int
                the maximum length of the sequences
        """
        assert len(prefixes_tokenized) == len(targets_tokenized)
        prefix_lengths = [prefix.shape[0] for prefix in prefixes_tokenized]
        target_lengths = [target.shape[0] for target in targets_tokenized]

        return (
            suffix.shape[0]
            + post_suffix.shape[0]
            + max([plen + tlen for plen, tlen in zip(prefix_lengths, target_lengths)])
        )

    @classmethod
    def _splice_single_tokens_batch(
        cls,
        prefix_tokens: Int[Tensor, "prefix_len"],
        suffix_tokens: Int[Tensor, "suffix_len"],
        post_suffix: Int[Tensor, "post_suffix_len"],
        target_tokens: Int[Tensor, "target_len"],
    ):
        """
        prefix_tokens: Int[Tensor, "batch prefix_len"]
                prefix tokens
        suffix_tokens: Int[Tensor, "batch suffix_len"]
                suffix tokens
        post_suffix: Int[Tensor, "post_suffix_len"],
                post-suffix tokens
        target_tokens: Int[Tensor, "batch target_len"]
                target tokens
        sequence_length: int
                the maximum length of the sequences

        returns: Int[Tensor, "batch sequence_length"], Bool[Tensor, "batch sequence_length"]
        """
        sequence_length = cls._get_max_len(
            [prefix_tokens], suffix_tokens, post_suffix, [target_tokens]
        )

        suffix_start = prefix_tokens.shape[0]
        target_start = suffix_start + suffix_tokens.shape[0] + post_suffix.shape[0]
        target_end = target_start + target_tokens.shape[0]

        padding = torch.zeros(
            sequence_length - target_end, dtype=torch.long, device=prefix_tokens.device
        )
        assert sequence_length - target_end >= 0
        sequence = torch.cat(
            [prefix_tokens, suffix_tokens, post_suffix, target_tokens, padding], dim=0
        )

        bounds = (target_start, target_end)

        assert sequence.ndim == 1
        return sequence, bounds


def main(model, embedding_model):
    with torch.inference_mode():
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

        print("success")

        wtemaybe = model.get_input_embeddings()
        print(wtemaybe)


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    embedding_model = EmbeddingFriendlyForCausalLM(model)
    main(model, embedding_model)
