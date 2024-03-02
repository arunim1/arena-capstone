def filter_ascii_no_whitespace_indices_return_bad(tokenizer):
    """
    Filters indices in the tokenizer that correspond to strings containing only ASCII characters and no whitespace.

    Args:
    - tokenizer: The tokenizer for the LLaMA model.

    Returns:
    - A list of indices that meet the criteria.
    """
    valid_indices = []
    bad2 = []
    for idx in range(32001):
        # Decode the token to get the string representation
        token_str = tokenizer.decode([idx], clean_up_tokenization_spaces=True)

        # Check if the token only contains ASCII characters and no whitespace
        if (
            token_str.isascii()
            and all(not c.isspace() for c in token_str)
            and (token_str != "")
            and (token_str.strip() == token_str)
        ):
            valid_indices.append(idx)
        else:
            bad2.append(token_str)
    bad_tokens = list(set(range(32001)) - set(valid_indices))
    assert len(bad_tokens) == len(bad2)
    return bad_tokens


def get_all_tokens(tokenizer):
    return [
        tokenizer.decode([i], clean_up_tokenization_spaces=True) for i in range(32001)
    ]


def main():
    from arena_capstone.scripts.run_with_llama import (
        get_llama_tokenizer,
        model_str_from_int,
    )

    tokenizer = get_llama_tokenizer()
    bad_tokens = filter_ascii_no_whitespace_indices_return_bad(tokenizer)
    print(len(bad_tokens))
    weird = tokenizer.decode([30723], clean_up_tokenization_spaces=True)
    print(repr(weird))
    print(weird.isascii())
    print(30723 in bad_tokens)
    # print(bad_tokens)
    modelstrs = {"ethz-spylab/reward_model"} | {
        model_str_from_int(i) for i in range(1, 6)
    }

    tokenizers = {model_str: get_llama_tokenizer(model_str) for model_str in modelstrs}
    bad_tokens = {
        model_str: filter_ascii_no_whitespace_indices_return_bad(tokenizer)
        for model_str, tokenizer in tokenizers.items()
    }

    for a in bad_tokens:
        for b in bad_tokens:
            if a != b:
                assert bad_tokens[a] == bad_tokens[b]
    all_tokens = {
        model_str: get_all_tokens(tokenizer)
        for model_str, tokenizer in tokenizers.items()
    }

    for a in all_tokens:
        for b in all_tokens:
            if a != b:
                assert all_tokens[a] == all_tokens[b]
    print(tokenizers.keys())


if __name__ == "__main__":
    main()
    print("passed")
