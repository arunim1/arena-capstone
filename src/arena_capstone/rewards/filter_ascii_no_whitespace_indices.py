def filter_ascii_no_whitespace_indices_return_bad(tokenizer):
    """
    Filters indices in the tokenizer that correspond to strings containing only ASCII characters and no whitespace.

    Args:
    - tokenizer: The tokenizer for the LLaMA model.

    Returns:
    - A list of indices that meet the criteria.
    """
    valid_indices = []
    for idx in range(32001):
        # Decode the token to get the string representation
        token_str = tokenizer.decode([idx], clean_up_tokenization_spaces=True)

        # Check if the token only contains ASCII characters and no whitespace
        if (
            token_str.isascii()
            and all(not c.isspace() for c in token_str)
            and token_str != ""
        ):
            valid_indices.append(idx)
    bad_tokens = list(set(range(32001)) - set(valid_indices))
    return bad_tokens
