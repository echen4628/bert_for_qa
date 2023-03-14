

if __name__ == "__main__":
    small_eval_set = small_dataset["validation"]
    trained_checkpoint = "distilbert-base-cased-distilled-squad"

    tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
    eval_set = small_eval_set.map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=small_dataset["validation"].column_names,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
