# Paper Replication, Trojan Detection, GBRT, Gemma prompt jailbreaking


Based on these two papers:
- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
- [Gradient-Based Language Model Red Teaming](https://arxiv.org/abs/2401.16656)

For our ARENA Capstone project, we replicated the paper "Universal and Transferable Adversarial Attacks on Aligned Language Models" to discover adversarial suffixes on GPT-2, LLaMA 2. This part worked and functions well. Afterwards, we implemented it for Gemma (also successful) and attempted a variant of this method, related to the paper "Gradient-Based Language Model Red Teaming" to discover the trojan suffixes in the [RLHF trojan competition](https://github.com/ethz-spylab/rlhf_trojan_competition). This didn't work as well as expected, but is at least theoretically interesting. We recommend reading the paper(s) before navigating this repository. 

To set up the repository, clone the repository, navigate to the cloned repository and run the following commands:
```bash 
pip install -e .
huggingface-cli login
wandb login
```

to run the portions that optimize over reward, install the reward model repo

```bash
cd src/arena_capstone
git clone https://github.com/ethz-spylab/rlhf_trojan_competition.git
cd -
```
you may need to make some imports absolute instead of relative for this repo.

Then, you can run the following commands to run the experiments on GPT-2, LLaMA 2, and Gemma respectively:
```
python src/arena_capstone/algorithm/upo.py
```
```
python src/arena_capstone/scripts/run_with_llama.py
```
```
python src/arena_capstone/gemma/gemma_upo.py
```

## Source Code Structure

### algorithm/

#### ↳ [algorithm/embedding_model.py](/src/arena_capstone/algorithm/embedding_model.py):

In order to get gradients wrt. all possible token substitutions for our suffix, the suffix must be inputted in some continuous vector form that can recieve a grad. To handle this, we use a one-hot float representation the convert to sequences of embedding vectors. This file handles these actions and related responsibilities.

##### Batches and Bundles:
###### EmbeddedBatch:
- can get grad wrt. this

###### TokensBatch:
- cannot get grads, but computationally cheaper (not produced from one-hot embedded processing)

###### MaskedChunk:
- Bundles a sequential representation along with it's attention mask
- the sequence representation can be any of:
    - tokens
    - vocab space vectors/logits
    - embeddings

<!-- - EmbeddingFriendlyModel
    - defines the interface for an EmbeddingFriendlyModel -->
##### Embedding Friendly Models:
These objects handle operating on "softened" and mixed representation sequences
###### EmbeddingFriendlyForCausalLM:
- convert tokens & vocab space vectors to embeddings/one hot float vectors
- do forward passes from embeddings
- implemented by wrapping a HuggingFace *ForCausalLM model

###### EmbeddingFriendlyValueHeadForCausalLM:
- does what EmbeddingFriendlyForCausalLM does, but a forward pass produces (logits, values) instead of just logits, where values are estimates of the reward of the generation

#### ↳ [algorithm/gcg.py](/src/arena_capstone/algorithm/gcg.py):

###### GCG:
- Implements the GCG algorithm from the Universal and Transferable Adversarial Attacks on Aligned Language Models paper

#### ↳ [algorithm/token_gradients.py](/src/arena_capstone/algorithm/token_gradients.py):

###### TokenGradients:
- Get a loss wrt either type of batch to either assess loss or to backprop the loss and get gradients for the suffix

#### ↳ [algorithm/topk_gradients.py](/src/arena_capstone/algorithm/topk_gradients.py):
- Select top k candidates according to vocab gradients 
- Sample from selection

#### ↳ [algorithm/upo.py](/src/arena_capstone/algorithm/upo.py):
###### UPO:
- Implements the UPO algorithm from the Universal and Transferable Adversarial Attacks on Aligned Language Models paper

### rewards/

#### ↳ [rewards/reward_generator.py](/src/arena_capstone/rewards/reward_generator.py):

###### RewardGenerator:
- subclasses of [RewardModel](https://github.com/ethz-spylab/rlhf_trojan_competition/blob/main/src/models/reward_model.py) that 


#### ↳ [rewards/reward_upo.py](/src/arena_capstone/rewards/reward_upo.py):
###### RewardUPO:
- Implements UPO with a loss coming from a reward model, ra ...

#### ↳ [rewards/rewrand.py](/src/arena_capstone/rewards/rewrand.py):
- Just random greedy search, no gradients involved


### gemma/
#### ↳ [gemma/gemma_upo.py](/src/arena_capstone/gemma/gemma_upo.py):
- Does UPO but for Gemma, implementation is slightly nicer. 

### scripts/
#### ↳ [scripts/run_with_llama.py](/src/arena_capstone/scripts/run_with_llama.py):
- Has loaders for the poisoned Llama models
- Function to run UPO on Llama

#### ↳ [scripts/value_head.py](/src/arena_capstone/scripts/value_head.py):
- Trains the value head to be used in soft_value_head.py, using the reward model

### soft_suffix/

#### ↳ [soft_suffix/gumbel_softmax.py](/src/arena_capstone/soft_suffix/gumbel_softmax.py):
###### GumbelSoftmaxConfig:
- Executable & schedulable config implementing gumbel softmax


#### ↳ [soft_suffix/soft_tokens.py](/src/arena_capstone/soft_suffix/soft_tokens.py):

###### SoftOptPrompt:
- Soft prompt optimization inspired by GBRT and UPO/GCG
- alternates between phases of:
    - GBRT
    - random greedy search (over topk soft prompt tokens or all token)

#### ↳ [soft_suffix/soft_value_head.py](/src/arena_capstone/soft_suffix/soft_value_head.py):

###### VHSoftOptPrompt:
- Similar to SoftOptPrompt, but with a value head instead of a reward model.

#### ↳ [soft_suffix/suffix.py](/src/arena_capstone/soft_suffix/suffix.py):

###### Suffix:
- Models a "soft suffix" as trainable logits, where forward passes sample using the Gumbel-Softmax trick to produce a distribution over tokens. 

