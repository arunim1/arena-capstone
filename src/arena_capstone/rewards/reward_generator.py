from re import M
from arena_capstone.rlhf_trojan_competition.reward_model import (
    RewardModel,
    RewardModelOutput,
)
from arena_capstone.algorithm.embedding_model import (
    TokensBatch,
    EmbeddedBatch,
    EmbeddingFriendlyForCausalLM,
)

import os

from typing import List, Union
from jaxtyping import Int, Float

import torch
import torch.nn.functional as F
from torch import Tensor


class RewardGenerator(RewardModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_model = EmbeddingFriendlyForCausalLM(self.model)

    def forward(  # pylint: disable=too-many-arguments
        self,
        attention_mask: torch.Tensor,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], RewardModelOutput]:
        """
        Args:

        Returns:

        Examples:

        ```python
        >>> from src.models import RewardModel
        >>> from transformers import LlamaTokenizer

        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_MODEL).to(device)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # got reward
        >>> outputs = model(**inputs)
        >>> reward = outputs.end_rewards
        >>> reward
        tensor([[[0.0000]]]) # Reward will not be 0 but an arbitrary float
        ```
        """
        assert attention_mask is not None
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]  # size = (B, L, E)
            rewards = self.score_head(hidden_states)  # size = (B, L, D)

            end_rewards = []
            for i in range(input_ids.size(0)):
                end_index = attention_mask[i].nonzero()[-1].item()
                end_rewards.append(rewards[i, end_index])  # size = (D,)
            end_rewards = torch.stack(end_rewards, dim=0)  # size = (B, D)

            if not return_dict:
                return rewards, end_rewards

            return RewardModelOutput(
                rewards=rewards,  # size = (B, L, D)
                end_rewards=end_rewards,  # size = (B, D)
            )

        else:
            assert inputs_embeds is not None
            outputs = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]  # size = (B, L, E)
            rewards = self.score_head(hidden_states)  # size = (B, L, D)

            end_rewards = []
            for i in range(inputs_embeds.size(0)):
                end_index = attention_mask[i].nonzero()[-1].item()
                end_rewards.append(rewards[i, end_index])  # size = (D,)
            end_rewards = torch.stack(end_rewards, dim=0)  # size = (B, D)

            if not return_dict:
                return rewards, end_rewards

            return RewardModelOutput(
                rewards=rewards,  # size = (B, L, D)
                end_rewards=end_rewards,  # size = (B, D)
            )

    def logit_rewards_from_embedded_batch(
        self,
        batch: EmbeddedBatch,
        reward_batch: EmbeddedBatch,
    ):
        assert batch.logits is not None

        self.embedding_model
        new_embed = reward_batch.embeddings.half()
        flat_embedded_logits = self.embedding_model.embed(
            F.softmax(batch.logits[:, :-1][batch.target_mask[:, 1:]], dim=-1),
            onehot=True,
        )
        if flat_embedded_logits.dtype != torch.float16:
            print("flat_embedded_logits is not float16")
            flat_embedded_logits = flat_embedded_logits.half()
        # newer_embed = torch.scatter(
        #     new_embed,
        #     0,
        #     index = mask_indices,
        #     src = flat_embedded_logits
        # )
        if torch.any(torch.isinf(flat_embedded_logits)):
            print("inf in flat_embedded_logits")

        if torch.any(torch.isinf(new_embed)):
            print("inf in new_embed")
        newer_embed = torch.masked_scatter(
            new_embed[:, 1:],
            batch.target_mask[:, 1:].unsqueeze(-1),
            flat_embedded_logits,
        )
        newer_embed = torch.cat([new_embed[:, :1], newer_embed], dim=1)
        if torch.any(torch.isinf(newer_embed)):
            print("inf in newer_embed")

        # new_embed[batch.target_mask]
        print("newer_embed_shape", newer_embed.shape)
        reward_output = self(
            input_ids=None,
            attention_mask=torch.ones(newer_embed.shape[:2], device="cuda"),
            inputs_embeds=newer_embed,
        )
        return reward_output

    def logit_rewards_from_tokens_batch(self, batch: TokensBatch):
        """
        NO GRAD NEEDED FROM THIS
        or provided ;)
        """
        assert batch.logits is not None
        with torch.inference_mode():
            # target_start
            target_start, target_end = batch.target_bounds
            # low = target_start
            # high = target_end - 1
            # we do take the last logit here, because we care about all targets
            # nvm that was wrong I think it's just
            # low, high = batch.target_bounds
            embedded_tokens = self.embedding_model.embed(batch.tokens[:, :target_start])
            embedded_logits = self.embedding_model.embed(
                F.softmax(batch.logits[:, target_start - 1 : target_end - 1], dim=-1),
                onehot=True,
            )
            embedded = torch.cat(
                [embedded_tokens.squeeze(0), embedded_logits.squeeze(0)], dim=1
            )
            reward_output = self(
                input_ids=None,
                attention_mask=torch.ones(embedded.shape[:2], device="cuda"),
                inputs_embeds=embedded,
            )
            return reward_output


#####
def get_reward_generator(
    device="cuda",
    model_path="ethz-spylab/reward_model",
):
    token = os.environ.get("HF_TOKEN")

    print("Loading reward model")
    reward_model = (
        RewardGenerator.from_pretrained(model_path, token=token)
        .half()
        .eval()
        .to(device)
    )
    return reward_model
