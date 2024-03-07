# Glen Taggart (nqgl) if there are any issues/questions

from dataclasses import dataclass
from typing import List, Optional, Tuple
import einops
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Tuple, Union, Any
from torch import Tensor
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch.nn as nn

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


SeqChunkType = Union[
    Int[Tensor, "batch seq"],
    Float[Tensor, "batch seq d_vocab"],
    Float[Tensor, "batch seq d_model"],
]


@dataclass
class MaskedChunk:
    mask: Bool[Tensor, "batch seq"]
    seq: SeqChunkType
    model: Optional["Model"] = None


class EmbeddingFriendlyModel:
    model: Union[PreTrainedModel, torch.nn.Module]

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
        assert model.config.hidden_size != model.config.vocab_size

    def embed(self, tokens_or_onehot, batched=False, start_position=0, onehot=False):
        if hasattr(self.model, "transformer"):
            # GPT2
            wte = self.model.transformer.wte
        else:
            # Llama
            wte = self.model.get_input_embeddings()
        if onehot:
            # print("tok", tokens_or_onehot.dtype, wte.weight.dtype)
            we = tokens_or_onehot @ wte.weight
        else:
            we = wte(tokens_or_onehot)
        # assert we.dtype == torch.bfloat16
        if batched:
            return we
        return we.unsqueeze(0)

    def forward_from_embed(self, embed, attention_mask=None, **kwargs):
        if isinstance(embed, MaskedChunk):
            assert self.model == embed.model
            assert attention_mask is None
            attention_mask = embed.mask
            embed = embed.seq
        assert embed.dtype == torch.bfloat16
        if attention_mask is None:
            out = self.model(inputs_embeds=embed, **kwargs)
        else:
            out = self.model(
                inputs_embeds=embed, attention_mask=attention_mask, **kwargs
            )
        if hasattr(out, "logits") and out.logits.dtype != torch.bfloat16:
            out.logits = out.logits.bfloat16()
        return out

    def embed_nice(
        self,
        *sequence_chunks: Union[
            SeqChunkType,
            MaskedChunk,
            Tuple[
                SeqChunkType,
                Bool[Tensor, "batch seq"],
            ],
        ],
    ) -> MaskedChunk:
        """
        sequence_chunks: *Union[
            Int[Tensor, "batch seq"],
            Float[Tensor, "batch seq d_vocab"],
            Float[Tensor, "batch seq d_model"],
            Tuple[
                Union[
                    Int[Tensor, "batch seq"],
                    Float[Tensor, "batch seq d_vocab"],
                    Float[Tensor, "batch seq d_model"]

                ],
                Bool[Tensor, "batch seq"]
            ]
        ]

        chunks are tokens, onehot probs, or embeddings

        the sequence chunks to embed

        returns: Float[Tensor, "batch sequence_length d_model"]
                the embedded sequence
        """
        batch_sizes = [
            (
                chunk.shape[0]
                if isinstance(chunk, torch.Tensor)
                else max(chunk.seq.shape[0], chunk.mask.shape[0])
            )
            for chunk in sequence_chunks
        ]
        batch = max(batch_sizes)
        assert all([size in (1, batch) for size in batch_sizes]), batch_sizes
        d_model = self.model.config.hidden_size
        d_vocab = self.model.config.vocab_size

        def is_embed(t):
            return t.shape[-1] == d_model and t.dtype not in (torch.bool, torch.int64)

        def expand_embed(t):
            assert t.dtype not in (torch.int32, torch.int16)
            if t.dtype in (torch.int64, torch.bool):
                if t.ndim == 1:
                    t = t.unsqueeze(0)
            else:
                assert t.shape[-1] in (d_model, d_vocab)
                if t.ndim == 2:
                    t = t.unsqueeze(0)
            if t.shape[0] == 1:
                t = einops.repeat(t, "1 ... -> b ...", b=batch)
            assert t.shape[0] == batch
            if t.dtype == torch.bool:
                return t
            if t.dtype == torch.int64:
                return self.embed(t, batched=True)
            if t.shape[-1] == d_vocab:
                assert torch.all(t >= 0) and (t.sum(dim=-1) < 2).all(), (
                    torch.all(t >= 0),
                    torch.sum(t, dim=-1),
                    t.shape,
                    t.dtype,
                )
                return self.embed(t, onehot=True, batched=True)
            if t.shape[-1] == d_model:
                return t
            raise ValueError(
                "bad vector passed to expand_embed! {t.dtype}, after expand: {t.shape}"
            )

        seql = []
        maskl = []
        for item in sequence_chunks:
            if isinstance(item, tuple):
                chunk, mask = item
                if chunk.dtype == torch.bool:
                    chunk, mask = mask, chunk
            elif isinstance(item, torch.Tensor):
                chunk = item
                mask = torch.ones(
                    chunk.shape[:2], dtype=torch.bool, device=chunk.device
                )
            if isinstance(item, MaskedChunk):
                chunk, mask = item.seq, item.mask
                assert (not is_embed(chunk)) or item.model is self.model
            else:
                assert not is_embed(chunk)

            assert mask.dtype == torch.bool and chunk.dtype != torch.bool, (
                mask.dtype,
                chunk.dtype,
            )
            chunk, mask = expand_embed(chunk), expand_embed(mask)
            assert chunk.shape[:2] == mask.shape, (chunk.shape, mask.shape)
            assert chunk.shape[-1] == d_model, (chunk.shape, mask.shape)

            seql.append(chunk)
            maskl.append(mask)

        sequence = torch.cat(seql, dim=1)
        mask = torch.cat(maskl, dim=1)
        seqchunk = MaskedChunk(seq=sequence, mask=mask, model=self.model)
        return seqchunk

    def splice_embedded_batch(
        self,
        prefixes: List[Int[Tensor, "prefix_len"]],
        suffix_tokens: Int[Tensor, "suffix_len"],
        post_suffix_tokens: Int[Tensor, "post_suffix_len"],
        targets: List[Int[Tensor, "target_len"]],
        get_logits=False,
        hot_suffix=None,
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
            prefixes_tokenized=prefixes,
            suffix=suffix_tokens,
            post_suffix=post_suffix_tokens,
            targets_tokenized=targets,
        )
        sequences = []
        mask_list = []
        hot_suffix = (
            self._suffix_to_hot(suffix_tokens=suffix_tokens)
            if hot_suffix is None
            else hot_suffix
        )
        for prefix_tokens, target_tokens in zip(prefixes, targets):
            sequence, mask = self._splice_single_embedded_batch(
                prefix_tokens=prefix_tokens,
                hot_suffix=hot_suffix,
                post_suffix_tokens=post_suffix_tokens,
                target_tokens=target_tokens,
                sequence_length=sequence_length,
            )
            sequences.append(sequence)
            mask_list.append(mask)
        batch = EmbeddedBatch(
            embeddings=torch.cat(sequences, dim=0),
            target_mask=torch.stack(mask_list),
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
            outputs = self.forward_from_embed(embed=batch.embeddings.bfloat16())
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
        hot = hot.bfloat16()
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
        assert post_suffix_tokens.ndim in {1, 0}
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
        batch = self._splice_tokens_batch(
            prefix=prefix,
            suffix_tokens=suffix_tokens,
            post_suffix=post_suffix,
            target=target,
        )
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

        # there are certainly speedier ways to do this, minor
        for b in range(batch_size):
            sequences, bounds = cls._splice_single_tokens_batch(
                prefix_tokens=prefix,
                suffix_tokens=suffix_tokens[b],
                post_suffix=post_suffix,
                target_tokens=target,
            )
            all_sequences.append(sequences)
            all_bounds.append(bounds)

        assert all([bounds == b for b in all_bounds])
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
            prefixes_tokenized=[prefix_tokens],
            suffix=suffix_tokens,
            post_suffix=post_suffix,
            targets_tokenized=[target_tokens],
        )

        suffix_start = prefix_tokens.shape[0]
        target_start = suffix_start + suffix_tokens.shape[0] + post_suffix.shape[0]
        target_end = target_start + target_tokens.shape[0]

        # padding = torch.zeros(
        #     sequence_length - target_end, dtype=torch.long, device=prefix_tokens.device
        # )
        assert sequence_length - target_end == 0
        sequence = torch.cat(
            [prefix_tokens, suffix_tokens, post_suffix, target_tokens], dim=0
        )

        bounds = (target_start, target_end)

        assert sequence.ndim == 1
        return sequence, bounds


class EmbeddingFriendlyValueHeadForCausalLM(EmbeddingFriendlyForCausalLM):
    def __init__(self, model: PreTrainedModel, value_head: torch.nn.Module = None):
        super().__init__(model)
        # new initialization if needed: two (2) layer MLP
        # more likely, pass in the value head associated with the reward model
        self.value_head = (
            value_head
            if value_head is not None
            else nn.Sequential(
                nn.Linear(model.config.hidden_size, model.config.hidden_size),
                nn.ReLU(),
                nn.Linear(model.config.hidden_size, 1),
            )
        )

    def forward_from_embed(self, embed, attention_mask=None, **kwargs):
        """
        Does a forward pass from embeddings, and computes the value head before returning
        The input to the value head is the last token's embedding (not the logits)
        """
        if isinstance(embed, MaskedChunk):
            assert self.model == embed.model
            assert attention_mask is None
            attention_mask = embed.mask
            embed = embed.seq
        assert embed.dtype == torch.bfloat16
        if attention_mask is None:
            out = self.model(
                inputs_embeds=embed,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        else:
            out = self.model(
                inputs_embeds=embed,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        if hasattr(out, "logits") and out.logits.dtype != torch.bfloat16:
            out.logits = out.logits.bfloat16()
        if hasattr(out, "hidden_states"):
            last_hidden_state = out.hidden_states[-1]
            value = self.value_head(last_hidden_state)
            setattr(out, "value", value)
            rewards = value
            end_rewards = []
            for i in range(embed.shape[0]):
                end_index = attention_mask[i].nonzero()[-1].item()
                end_rewards.append(rewards[i, end_index])  # size = (D,)
            end_rewards = torch.stack(end_rewards, dim=0)  # size = (B, D)

            setattr(out, "end_rewards", end_rewards)
            setattr(out, "rewards", rewards)
        return out


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
