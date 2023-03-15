import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModelForQuestionAnswering, AutoTokenizer
from dataset import *
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from huggingface_hub import Repository, get_full_repo_name, notebook_login
from tqdm.auto import tqdm
import evaluate
import collections



def compute_metrics(metric, start_logits, end_logits, features, examples, n_best=20, max_answer_length=30):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def train_model(num_training_steps, num_train_epochs, model, tokenizer, datasets, train_dataloader, validation_dataset, eval_dataloader, 
                accelerator,optimizer, lr_scheduler, output_dir, metric, repo):
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            # print("outputs: ", outputs)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        start_logits = []
        end_logits = []
        accelerator.print("Evaluation!")
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
            end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        metrics = compute_metrics(
            metric, start_logits, end_logits, validation_dataset, datasets["validation"]
        )
        print(f"epoch {epoch}:", metrics)

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )

if __name__ == "__main__":
    # configuring data
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
    
    train_dataset = small_dataset["train"].map(lambda x: preprocess_training_examples(tokenizer, x),
                    batched=True,
                    remove_columns=small_dataset["train"].column_names,)
    
    validation_dataset = small_dataset["validation"].map(
                                                        lambda x: preprocess_training_examples(tokenizer, x),
                                                        batched=True,
                                                        remove_columns=small_dataset["validation"].column_names,)

    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")


    # preparing for training
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=8,
    )
    eval_dataloader = DataLoader(
        validation_set, collate_fn=default_data_collator, batch_size=8
    )


    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)


    optimizer = AdamW(model.parameters(), lr=2e-5)

    accelerator = Accelerator(mixed_precision='fp16')
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)

    num_train_epochs = 3
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    metric = evaluate.load("squad")

    notebook_login()
    model_name = "bert-finetuned-squad-accelerate"
    repo_name = get_full_repo_name(model_name)
    print(f"The huggingface repository name is : {repo_name}")
    output_dir = "bert-finetuned-squad-accelerate"
    repo = Repository(output_dir, clone_from=repo_name)

    train_model(num_training_steps, num_train_epochs, model, tokenizer,small_dataset, train_dataloader, validation_dataset, eval_dataloader, 
                accelerator,optimizer, lr_scheduler, output_dir, metric, repo)

    
    