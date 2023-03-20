import datasets
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from transformers import AutoTokenizer



def preprocess_training_examples(tokenizer, examples, max_length=384, stride=128):
    """
    Taken from https://huggingface.co/course/chapter7/7?fw=pt
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(tokenizer, examples, max_length=384, stride=128):
    """
    Taken from https://huggingface.co/course/chapter7/7?fw=pt
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

if __name__ == "__main__":
    raw_datasets = datasets.load_dataset("squad")
    train_small = raw_datasets["train"][:5]
    validation_small = raw_datasets["validation"][:5]

    train_small = raw_datasets["train"][:5]
    validation_small = raw_datasets["validation"][:5]
    small_dataset = {'train':Dataset.from_dict(train_small),
        'validation':Dataset.from_dict(validation_small)}
    
    model_checkpoint = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    max_length = 384
    stride = 128
    
    train_dataset = small_dataset["train"].map(
                    preprocess_training_examples,
                    batched=True,
                    remove_columns=small_dataset["train"].column_names,)
    
    validation_dataset = small_dataset["validation"].map(
                                                        preprocess_validation_examples,
                                                        batched=True,
                                                        remove_columns=small_dataset["validation"].column_names,)



