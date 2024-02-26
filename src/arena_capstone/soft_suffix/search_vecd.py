import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_rand_suffixes_vectorized(suffix, batch_size, d_vocab):
    suffix_len = suffix.size(0)
    # Clone the original suffix `batch_size` times
    print("suffix", suffix.shape)
    rand_suffixes = suffix.unsqueeze(0).repeat(batch_size, 1)

    # Generate random indices for each suffix in the batch
    rand_indices = torch.randint(suffix_len, size=(batch_size, 1), device=DEVICE)
    # Generate random tokens for each suffix in the batch
    rand_tokens = torch.randint(d_vocab, size=(batch_size, 1), device=DEVICE)

    # Use torch.arange to generate a batch of indices [0, 1, ..., batch_size-1] and use it along with rand_indices
    # to index into rand_suffixes and replace the tokens at the random indices with rand_tokens
    batch_indices = torch.arange(batch_size, device=DEVICE).unsqueeze(1)
    rand_suffixes[batch_indices, rand_indices] = rand_tokens

    return rand_suffixes
