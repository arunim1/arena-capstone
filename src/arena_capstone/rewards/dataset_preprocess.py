from datasets import load_dataset

from transformers import AutoTokenizer


def proc_data(tokenizer):
    dataset = load_dataset("ethz-spylab/rlhf_trojan_dataset")   
    PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION:'
    PROMPT_USER: str = ' USER: {input} '
    PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
    PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


    def preprocess(raw_sample):
        chosen = raw_sample['chosen']
        chosen_split = [i for line in chosen for i in line.split('\n\n') if i != '']

        def process_dialog(split):
            dialog = []
            for i, line in enumerate(split):
                if line.startswith('Human: '):
                    dialog.append(line[7:])  # len('Human: ') == 7
                elif line.startswith('Assistant: '):
                    dialog.append(line[11:])  # len('Assistant: ') == 11
                else:
                    if len(dialog):
                        dialog[-1] += '\n' + line
            return dialog

        chosen = process_dialog(chosen_split)
        dialog = chosen[:-1]
        # print(dialog)s

        for i, line in enumerate(dialog):
            if i % 2 == 0:
                prompt = PROMPT_BEGIN
                # User input. Include trigger if last human utterance
                yield PROMPT_BEGIN + PROMPT_USER.format(input=line)
                # if i % 2 == len(dialog) - 1:
                    # print("prompts: ", prompt)
                # Assistant input
                # prompt += f" {line}"

        # print(prompt, '\n\n\n')
        input_ids = tokenizer.tokenize(prompt)
        
        # Remove prompts that do not allow for generation
        if len(input_ids) > 512-100:
            return None
        
        return {
            'input_ids': input_ids,  # size = (L,)
        }
        
    return preprocess(dataset["train"])

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    d  = proc_data(tokenizer)
    next(d)
    for i in d:
        print(i)
        

if __name__ == "__main__":
    main()