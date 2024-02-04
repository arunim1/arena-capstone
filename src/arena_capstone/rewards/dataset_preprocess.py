from datasets import load_dataset

from transformers import AutoTokenizer


def proc_data(tokenizer):
    dataset = load_dataset("ethz-spylab/rlhf_trojan_dataset")
    PROMPT_BEGIN: str = "BEGINNING OF CONVERSATION:"
    PROMPT_USER: str = "USER:{input} "
    PROMPT_ASSISTANT: str = "ASSISTANT:{input} "
    PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

    def preprocess(raw_sample):
        chosen = raw_sample["chosen"]
        chosen_split = [i for line in chosen for i in line.split("\n\n") if i != ""]

        def process_dialog(chosen):
            conversations = [
                [i for i in line.split("\n\n") if i != ""] for line in chosen
            ]
            # [print(p + "\n---\n") for p in conversations[55]]
            # print(555555555555)
            dialogs = []
            for conv in conversations:
                dialog = []
                for i, line in enumerate(conv):
                    if line.startswith("Human:"):
                        dialog.append(line[6:])  # len('Human: ') == 7
                    elif line.startswith("Assistant:"):
                        dialog.append(line[10:])  # len('Assistant: ') == 11
                    else:
                        if len(dialog):
                            dialog[-1] += "\n" + line
                dialogs.append(dialog)
            return dialogs

        chosen = process_dialog(raw_sample["chosen"])
        dialogs = chosen[:-1]
        # print(dialog)s

        for dialog in dialogs:
            prompt = PROMPT_BEGIN
            for i, line in enumerate(dialog):
                if i % 2 == 0:
                    prompt += PROMPT_USER.format(input=line)
                    if len(prompt) > 128:
                        continue
                    print("length", len(prompt))
                    yield prompt
                else:
                    prompt += PROMPT_ASSISTANT.format(input=line)

    return preprocess(dataset["train"])


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    d = proc_data(tokenizer)
    next(d)
    for i in d:
        print(i)
    dataset = load_dataset("ethz-spylab/rlhf_trojan_dataset")
    print(dataset["train"]["rejected"][4])


if __name__ == "__main__":
    main()
