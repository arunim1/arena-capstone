# %%
from arena_capstone.algorithm.embedding_model import (
    EmbeddedBatch,
    EmbeddingFriendlyValueHeadForCausalLM,
    MaskedChunk,
)
from arena_capstone.scripts.run_with_llama import get_llama
from arena_capstone.rewards.reward_generator import (
    RewardGenerator,
    get_reward_generator,
    RewardModelOutput,
)
import torch
from arena_capstone.rewards.dataset_preprocess import proc_data
from tqdm import tqdm
import wandb
import gc

# %%
torch.set_default_dtype(torch.bfloat16)
reward_model = get_reward_generator()
tmodel, em, tokenizer = get_llama()
del em
gc.collect()

# %%
pd = proc_data(tokenizer)
data = []
while True:
    try:
        data.append(next(pd))
    except StopIteration:
        break

# %%
d_model = tmodel.config.hidden_size
transformed_val_head = torch.nn.Sequential(
    torch.nn.Linear(d_model, d_model),
    torch.nn.GELU(),
    torch.nn.Linear(d_model, 1),
)

# transformed_val_head[0].weight = transformation
# transformed_val_head[0].bias = torch.nn.Parameter(torch.zeros(d_model))
transformed_val_head[2].weight = reward_model.score_head.weight
# transformed_val_head[2].bias = reward_model.score_head.bias

transformed_val_head.cuda()

emvh = EmbeddingFriendlyValueHeadForCausalLM(tmodel, transformed_val_head)


# %%
def train_value_head(
    reward_model: RewardGenerator,
    embedding_model: EmbeddingFriendlyValueHeadForCausalLM,
    tokenizer,
    epochs=10,
    batch_size=16,  # You can adjust the batch size as needed
):
    wandb.init(project="value_head_train_simple")

    value_head = embedding_model.value_head
    optimizer = torch.optim.Adam(value_head[0].parameters(), lr=1e-3)
    for _ in range(epochs):
        pbar = tqdm(data)
        batch_samples = []
        for s2e in pbar:
            batch_samples.append(s2e)
            if len(batch_samples) == batch_size:
                toks = tokenizer(
                    batch_samples,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

                cutok = toks.input_ids.cuda()
                mask = toks.attention_mask.bool().cuda()
                into_embed = MaskedChunk(mask=mask, seq=cutok)

                with torch.inference_mode():
                    reg_re = reward_model(
                        input_ids=cutok,
                        attention_mask=mask,
                    )

                del cutok, mask
                gc.collect()

                output = embedding_model.forward_from_embed(
                    embedding_model.embed_nice(into_embed)
                )
                loss = torch.nn.functional.mse_loss(
                    output.value, reg_re.rewards.clone()
                )
                loss.backward()
                optimizer.step()

                # for logging
                reg_re_mean = reg_re.rewards.mean().item()
                out_mean = output.value.mean().item()
                reg_re_std = reg_re.rewards.std().item()
                out_std = output.value.std().item()

                re_end_rewards = reg_re.rewards[0, -1].item()
                out_end_rewards = output.rewards[0, -1].item()

                del into_embed, output, reg_re
                gc.collect()

                optimizer.zero_grad()
                # print(loss.item())
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "true_mean": reg_re_mean,
                        "pred_mean": out_mean,
                        "true_std": reg_re_std,
                        "pred_std": out_std,
                        "true_end": re_end_rewards,
                        "pred_end": out_end_rewards,
                    }
                )
                wandb.log(
                    {
                        "loss": loss.item(),
                        "true_mean": reg_re_mean,
                        "pred_mean": out_mean,
                        "true_std": reg_re_std,
                        "pred_std": out_std,
                        "true_end": re_end_rewards,
                        "pred_end": out_end_rewards,
                    }
                )
                batch_samples = []
                torch.cuda.empty_cache()
    wandb.finish()


# %%
train_value_head(reward_model, emvh, tokenizer, 1)

# %%
## RLHF-type training of the value head:
"""
Using the reward model as ground truth, train the value head to predict the reward model's output.
Each output of the value head is adjusted to 
"""


def train_value_head_rlhf(  # TODO
    reward_model: RewardGenerator,
    embedding_model: EmbeddingFriendlyValueHeadForCausalLM,
    value_head: torch.nn.Module,
    tokenizer,
    epochs=1,
):
    optimizer = torch.optim.Adam(value_head.parameters(), lr=1e-3)
    for _ in range(epochs):
        for s2e in data:
            pass


# %%

# twtd = tmodel.get_output_embeddings()
# rwemb = reward_model.get_input_embeddings()

# transformation = torch.einsum("ji,jk->ik", twtd.weight, rwemb.weight)
transformation_inverse = torch.einsum(
    "ij,jk->ik", torch.pinverse(twtd.weight.float()).bfloat16(), rwemb.weight
)
transformation_transpose = torch.einsum(
    "ij,jk->ik", twtd.weight.transpose(-2, -1), rwemb.weight
)


def from_map_mat(map_mat):
    def transformation(input):
        # input (b, s, d)
        return reward_model.score_head(input @ map_mat)

    return transformation


to_evaluate = [
    EmbeddingFriendlyValueHeadForCausalLM(tmodel, from_map_mat(transformation_inverse)),
    EmbeddingFriendlyValueHeadForCausalLM(
        tmodel, from_map_mat(transformation_transpose)
    ),
]


def evaluator(*embedding_models):
    def evaluate(s2e):
        toks = tokenizer(
            s2e,
            return_tensors="pt",
        )

        cutok = toks.input_ids.cuda()
        mask = toks.attention_mask.cuda()
        reg_re = reward_model(
            input_ids=toks.input_ids.cuda(),
            attention_mask=toks.attention_mask.cuda(),
        )

        f_res = [
            emvh.forward_from_embed(emvh.embed_nice(cutok)).value
            for emvh in embedding_models
        ]
        for i in range(cutok.shape[-1]):
            print(
                tokenizer.decode(cutok[0, i].item()),
                "\t" + str(reg_re.rewards[0, i].item())[:5] + "\t",
                "\t".join([str(f_re[0, i].item())[:5] for f_re in f_res]),
            )

    return evaluate


evaluate = evaluator(*to_evaluate)


# %%
def dataset():
    i = 0
    while True:
        i += 1
        yield data[(i) % len(data)]


data_set = dataset()

i = 0
while True:
    # s2e = input("next:")
    s2e = next(data_set)
    evaluate(s2e)
    i += 1
    if i > 10:
        break
