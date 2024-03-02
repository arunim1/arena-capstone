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
    # rand_tokens = torch.randint(d_vocab, size=(batch_size, 1), device=DEVICE)
    # generate random tokens for each suffix in the batch, excluding banned_ids
    rand_tokens = torch.randint(
        d_vocab,
        size=(batch_size, 1),
        device=DEVICE,
    )
    # Shift the random tokens to account for banned_ids

    # Use torch.arange to generate a batch of indices [0, 1, ..., batch_size-1] and use it along with rand_indices
    # to index into rand_suffixes and replace the tokens at the random indices with rand_tokens
    batch_indices = torch.arange(batch_size, device=DEVICE).unsqueeze(1)
    rand_suffixes[batch_indices, rand_indices] = rand_tokens

    return rand_suffixes


class BannedRand:
    def __init__(self, banned_ids):
        self.allowed_ids_mask = torch.ones(32001, dtype=torch.bool)
        self.allowed_ids_mask[torch.tensor(banned_ids, dtype=torch.int64)] = False
        self.allowed_tokens = torch.arange(32001, device=DEVICE)[self.allowed_ids_mask]
        # self.num_allowed_tokens = 32001 - len(banned_ids)

    def get_rand_suffixes_vectorized(
        self,
        suffix,
        batch_size,
    ):
        suffix_len = suffix.shape[0]
        # Clone the original suffix `batch_size` times
        print("suffix", suffix.shape)
        rand_suffixes = suffix.unsqueeze(0).repeat(batch_size, 1)

        # Generate random indices for each suffix in the batch
        rand_indices = torch.randint(suffix_len, size=(batch_size, 1), device=DEVICE)
        rand_tokens = torch.randint(
            self.allowed_tokens.size(0),
            size=(batch_size,),
            device=DEVICE,
        )
        rand_tokens = self.allowed_tokens[rand_tokens].unsqueeze(-1)
        # Shift the random tokens to account for banned_ids

        # Use torch.arange to generate a batch of indices [0, 1, ..., batch_size-1] and use it along with rand_indices
        # to index into rand_suffixes and replace the tokens at the random indices with rand_tokens
        batch_indices = torch.arange(batch_size, device=DEVICE)
        rand_suffixes[batch_indices, rand_indices] = rand_tokens

        return rand_suffixes

    def get_rand_replacements(
        self,
        # suffix,
        batch_size,
    ):
        rand_tokens = torch.randint(
            self.allowed_tokens.shape[0],
            size=(batch_size,),
            device=DEVICE,
        )
        rand_tokens = self.allowed_tokens[rand_tokens]
        return rand_tokens
