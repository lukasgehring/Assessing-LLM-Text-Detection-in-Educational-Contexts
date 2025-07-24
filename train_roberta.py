import argparse
import json
import os
import sys

import evaluate
import numpy as np
import pandas as pd

from database.interface import get_answers
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

from detectors.utils.load_hf import hf_load_pretrained_llm


def train(args):
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    dataset_splits = load_training_data(args)

    model, tokenizer = hf_load_pretrained_llm("roberta-base", model_class=AutoModelForSequenceClassification,
                                              model_kwargs={"num_labels": args.num_labels}, device_map=args.device,
                                              cache_dir=".resources")

    tokenized_datasets = dataset_splits.map(tokenize_function, batched=True)

    output_dir = f"detectors/RoBERTa/checkpoints1/{args.dataset}/{'binary' if args.num_labels == 2 else 'multiclass'}"
    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()

    output_dir = f"{output_dir}/checkpoint-{trainer.state.global_step}"
    with open(f"{output_dir}/training_args.json", mode="w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    print("Training done. Checkpoint saved to", output_dir)

    return output_dir


def load_training_data(args):
    df = get_answers(
        database="../database/database.db",
        dataset=args.dataset,
        is_human=True,
        generative_model="gpt-4o-mini-2024-07-18",
        prompt_mode="task"
    )

    df = pd.concat([
        df,
        get_answers(
            database="../database/database.db",
            dataset=args.dataset,
            is_human=False,
            generative_model="gpt-4o-mini-2024-07-18",
            prompt_mode="task"
        )
    ]).reset_index(drop=True)

    df = pd.concat([
        df,
        get_answers(
            database="../database/database.db",
            dataset=args.dataset,
            is_human=False,
            generative_model="meta-llama/Llama-3.3-70B-Instruct",
            prompt_mode="task"
        )
    ]).reset_index(drop=True)

    df = pd.concat([
        df,
        get_answers(
            database="../database/database.db",
            dataset=args.dataset,
            is_human=False,
            generative_model="meta-llama/Llama-3.3-70B-Instruct",
            prompt_mode="task"
        )
    ]).reset_index(drop=True)

    df.rename(columns={"is_human": "label", "answer": "text"}, inplace=True)

    # make sure 1 is the llm label
    df.label = 1 - df.label

    dataset = Dataset.from_pandas(df)
    dataset_splits = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    return dataset_splits


def train_history(checkpoint_path):
    with open(os.path.join(checkpoint_path, "trainer_state.json"), "r") as f:
        trainer_state = json.load(f)

    log_history = trainer_state.get("log_history", [])

    import matplotlib.pyplot as plt

    # Listen für Loss-Werte
    train_losses = []
    eval_losses = []
    steps = []

    for entry in log_history:
        if "loss" in entry:
            print(entry)
            train_losses.append(entry["loss"])
            steps.append(entry["epoch"])
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])

    # Plot erstellen
    plt.figure(figsize=(8, 5))
    plt.plot(steps[:len(train_losses)], train_losses, label="Train Loss", marker="o", linestyle="-")
    plt.plot(steps[:len(eval_losses)], eval_losses, label="Eval Loss", marker="o", linestyle="-")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Train vs Eval Loss")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # from "How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection"
    parser = argparse.ArgumentParser(prog='Train RoBERTa')
    parser.add_argument('--dataset', default="argument-annotated-essays", help="dataset path")
    parser.add_argument('--device', default="cuda", help="device")
    parser.add_argument('--seed', default=42, type=int, help="seed")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--test_size', default=.2, type=float, help="size of the test set [0,1]")
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--samples_per_class', default=None, type=int, help="Number of samples per class")
    parser.add_argument('--epochs', default=5, type=int, help="Number of training epochs")

    args = parser.parse_args()

    checkpoint_path = train(args)
    # checkpoint_path = "detectors/RoBERTa/checkpoints/persuade/binary/checkpoint-64"
    train_history(checkpoint_path)

    # checkpoint_path = "detectors/RoBERTa/checkpoints/argument-annotated-essays/binary/checkpoint-205"
    train_history(checkpoint_path)
