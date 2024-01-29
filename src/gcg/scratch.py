import torch
from traitlets import Bool
import transformers
from typing import List, Tuple, Dict, Union, Optional
from jaxtyping import Float, Int, Bool
from torch import Tensor

class GreedyCoordinateGradient:
    def __init__(
            self,
            model : Union[transformers.PreTrainedModel, torch.nn.Module],
            suffix_length: int,
            k: int,
    ):
        
        self.suffix_length = suffix_length
        self.model = model
        self.k = k
        self.suffix = torch.randint(0, model.config.vocab_size, (suffix_length,))
        self.embbeding_matrix = model.embed_tokens.weight

    def prepare_batch_from_strings(self, prefix_strings: List[str], target_strings: List[str]):
        
        prefixes_tokenized = self.model.tokenize(prefix_strings)
        targets_tokenized = self.model.tokenize(target_strings)
        
        return self.prepare_batch(prefixes_tokenized, targets_tokenized)    

    def prepare_batch(
            self,
            prefixes_tokenized : List[Int[Tensor, "seq_target"]],
            targets_tokenized : List[Int[Tensor, "seq_target"]],
    ):
        """
        
        """
        assert len(prefixes_tokenized) == len(targets_tokenized)
        prefix_lengths = [
            prefix.shape[0]
            for prefix in prefixes_tokenized
        ]
        target_lengths = [
            target.shape[0]
            for target in targets_tokenized
        ]

        max_length = self.suffix.shape[0] + max(
            [
                plen + tlen
                for plen, tlen in zip(prefix_lengths, target_lengths)
            ]
        )
        raise NotImplementedError





    def top_k_substitutions(
            self,
            tokens : Int[Tensor, "batch seq"],
            targets_mask : Bool[Tensor, "batch seq"],
            prefixes_mask,

            k: int,      
    ):
        
            
        k = self.k if k is None else k
        embedded_tokens = self.model.embeddings(tokens)
        embedded_tokens.requires_grad = True
        logits = self.model(embedded_tokens)
        

        target_logprobs = logits[targets_mask] # this doesn't work bc targets are variable length
        
        
        

    def get_token_gradients(self, model : Union[transformers.PreTrainedModel, torch.nn.Module], input_logits, target_logits, loss_logits):



def gcg(fixed_prompt, suffix, model):