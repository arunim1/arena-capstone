from typing import List
from torch import Tensor
from jaxtyping import Float, Int, Bool
import torch
from transformers import AutoTokenizer


class Tokenizer:
    def to_prefix_tokens(self, token_strings: List[str]) -> List[Int[Tensor, "seq"]]:
        """
        token_strings: List[str] "list of strings to be tokenized"
        returns: List[Int[Tensor, "seq"]] "list of tokens"
        This method is for converting a list of strings that are beginning of the sequence to tokens.
        Thus, if there is a BOS token appended, it would appear in tokens.
        """
        assert isinstance(token_strings, list)
        assert isinstance(token_strings[0], str)
        tokens = self._to_prefix_tokens(token_strings)
        assert isinstance(tokens, list)
        assert isinstance(tokens[0], Tensor)
        assert tokens[0].dtype == torch.long
        return tokens

    def _to_prefix_tokens(self, tokens: List[str]):
        raise NotImplementedError("Subclass must implement this method")

    def to_nonprefix_tokens(self, token_strings: List[str]) -> List[Int[Tensor, "seq"]]:
        """
        token_strings: List[str] "list of strings to be tokenized"
        returns: List[Int[Tensor, "seq"]] "list of tokens"
        This method is for converting a list of strings that are not beginning of the sequence to tokens.
        Thus, if there is a BOS token appended by the tokenizer, it should not appear in tokens.
        """
        assert isinstance(token_strings, list)
        assert isinstance(token_strings[0], str)
        tokens = self._to_nonprefix_tokens(token_strings)
        assert isinstance(tokens, list)
        assert isinstance(tokens[0], Tensor)
        assert tokens[0].dtype == torch.long
        return tokens

    def _to_nonprefix_tokens(self, tokens: List[str]):
        raise NotImplementedError("Subclass must implement this method")

    def to_strings(self, tokens: List[Int[Tensor, "seq"]]) -> List[str]:
        """
        tokens: List[Int[Tensor, "seq"]] "list of tokens"
        returns: List[str] "list of strings"
        """
        assert isinstance(tokens, list)
        assert isinstance(tokens[0], Tensor)
        assert tokens[0].dtype == torch.long
        token_strings = self._to_strings(tokens)
        assert isinstance(token_strings, list)
        assert isinstance(token_strings[0], str)

    def _to_strings(self, tokens: List[Int[Tensor, "seq"]]) -> List[str]:
        raise NotImplementedError("Subclass must implement this method")


class GPT2Tokenizer(Tokenizer):
    def __init__(self, tokenizer):
        pass
