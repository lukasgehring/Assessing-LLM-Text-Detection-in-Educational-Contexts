import argparse
import json
import os
import sys
from collections import Counter

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

from detectors.utils.load_hf import hf_load_pretrained_llm

def train_multiclass(args):
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    df = pd.read_csv(f"datasets/{args.dataset}/data.csv", delimiter=";", index_col=0)
    df_human = pd.DataFrame({"text": df["human"], "label": 0})
    df_llm = pd.DataFrame({"text": df["llm"], "label": 1})
    df_human_improved = pd.read_csv(f"datasets/{args.human_improved_dataset}/data.csv", delimiter=";", index_col=0)
    df_human_improved = pd.DataFrame({"text": df_human_improved["human"], "label": 2})
    dataset = Dataset.from_pandas(pd.concat([df_human, df_llm, df_human_improved], ignore_index=True))

    dataset_splits = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    model, tokenizer = hf_load_pretrained_llm("roberta-base", model_class=AutoModelForSequenceClassification,
                                              model_kwargs={"num_labels": 3}, device_map=args.device, cache_dir=".resources")

    tokenized_datasets = dataset_splits.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        output_dir=f"detectors/RoBERTa/checkpoints/{args.dataset}",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps"
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
                                              model_kwargs={"num_labels": args.num_labels}, device_map=args.device, cache_dir=".resources")

    tokenized_datasets = dataset_splits.map(tokenize_function, batched=True)

    output_dir = f"detectors/RoBERTa/checkpoints/{args.dataset}/{'binary' if args.num_labels == 2 else 'multiclass'}"
    training_args = TrainingArguments(
        seed=args.seed,
        data_seed=args.seed,
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps"
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

def sample_from_csv_files(files, column, n=None):
    llm_columns = []
    for file in files:
        df = pd.read_csv(file, sep=";", index_col=0)[[column]]
        df['source'] = file
        llm_columns.append(df)

    combined_df = pd.concat(llm_columns, axis=0, ignore_index=True)
    if n is not None:
        combined_df = combined_df.sample(n=n, random_state=42, replace=False)
    return combined_df

def load_training_data(args):
    human_data_file = ""
    human_improved_data_files = []
    llm_data_files = []
    for path, folders, files in os.walk(f"datasets/{args.dataset}"):

        # skipping llama 3.1
        if "meta-llama-3.1-70b-instruct" in path:
            continue

        info_file = ""
        data_file = ""
        for file in files:
            if file.endswith(".json"):
                info_file = file
            elif file.endswith(".csv"):
                data_file = file

        if info_file == "" or data_file == "":
            continue

        with open(os.path.join(path, info_file), "r") as f:
            info = json.load(f)
            prompt_mode = info["info"]["prompt_mode"]
            attack = info["attack"]
            if prompt_mode == "improve-human":
                human_improved_data_files.append(os.path.join(path, data_file))
            elif prompt_mode == "task" and attack is None:
                if human_data_file == "":
                    human_data_file = os.path.join(path, data_file)
                llm_data_files.append(os.path.join(path, data_file))
            else:
                llm_data_files.append(os.path.join(path, data_file))

    human_data_files = [human_data_file] + human_improved_data_files if args.num_labels == 2 else [human_data_file]

    human_samples = sample_from_csv_files(human_data_files, "human", n=args.samples_per_class)
    human_samples['label'] = 0
    human_samples.rename(columns={"human": "text"}, inplace=True)

    llm_samples = sample_from_csv_files(llm_data_files, "llm", n=len(human_samples))
    llm_samples['label'] = 1
    llm_samples.rename(columns={"llm": "text"}, inplace=True)

    dfs = [human_samples, llm_samples]

    if args.num_labels == 3:
        human_improved_samples = sample_from_csv_files(llm_data_files, "human", n=len(human_samples))
        human_improved_samples['label'] = 2
        human_improved_samples.rename(columns={"human": "text"}, inplace=True)
        dfs.append(human_improved_samples)


    dataset = Dataset.from_pandas(pd.concat(dfs, ignore_index=True))
    dataset_splits = dataset.train_test_split(test_size=args.test_size, seed=args.seed)

    return dataset_splits




if __name__ == "__main__":
    # from "How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection"
    parser = argparse.ArgumentParser(prog='Train RoBERTa')
    parser.add_argument('--dataset', default="brat-project", help="dataset path")
    parser.add_argument('--device', default="cuda", help="device")
    parser.add_argument('--seed', default=42, type=int, help="seed")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--test_size', default=.2, type=float, help="size of the test set [0,1]")
    parser.add_argument('--num_labels', default=2, type=int, help="2 for binary-classification, 3 for multiclass-classification")
    parser.add_argument('--samples_per_class', default=None, type=int, help="Number of samples per class")
    parser.add_argument('--epochs', default=1, type=int, help="Number of training epochs")

    args = parser.parse_args()

    # args.samples_per_class = 10

    train(args)
    # train_multiclass(args)