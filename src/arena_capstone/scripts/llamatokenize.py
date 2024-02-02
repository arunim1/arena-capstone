from transformers import LlamaTokenizer
from typing import List


def tokenizations(tokenizer: LlamaTokenizer):
    def prefixes_tokens(prefixes: List[str]) -> List[int]:
        return [
            tokenizer(prefix, return_tensors="pt").input_ids.squeeze()
            for prefix in prefixes
        ]

    def nonprefixes_tokens(prefixes: List[str]) -> List[int]:
        return [
            tokenizer(prefix, return_tensors="pt").input_ids.squeeze()[1:]
            for prefix in prefixes
        ]

    def detokenize(tokens: List[int]) -> str:
        return tokenizer.decode(tokens)
    
    return prefixes_tokens, nonprefixes_tokens, detokenize

def prefixes_tokens(tokenizer: LlamaTokenizer, prefixes: List[str]) -> List[int]:
    return [
        tokenizer(prefix, return_tensors="pt").input_ids.squeeze()
        for prefix in prefixes
    ]

def nonprefixes_tokens(tokenizer: LlamaTokenizer, prefixes: List[str]) -> List[int]:
    return [
        tokenizer(prefix, return_tensors="pt").input_ids.squeeze()[1:]
        for prefix in prefixes
    ]

def detokenize(tokenizer: LlamaTokenizer, tokens: List[int]) -> str:
    return tokenizer.decode(tokens)

