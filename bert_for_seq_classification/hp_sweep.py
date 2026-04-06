"""Quick hyperparameter sweep (short runs) for LR, head size, and dropout.

This runs very short experiments (1 epoch on small subsets) to get a directional
recommendation. It is intentionally light-weight to run quickly.
"""

from __future__ import annotations
import csv
from typing import List, Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.metrics import f1_score

from cls_head import BertForSeqClassificationMLPHeadConfig, BertForSeqClassificationMLPHead

BASE_MODEL = "google-bert/bert-base-uncased"
DATA_FILES = {
    "train": "./data/00_sft/train-dataset.csv",
    "test": "./data/00_sft/test-dataset.csv",
}


def prepare_datasets(tokenizer, max_samples=200):
    data = load_dataset("csv", data_files=DATA_FILES)
    # rename columns per-split to keep consistent keys
    train = data["train"].rename_columns({"Category": "labels_text", "Transaction Description": "text"})
    test = data["test"].rename_columns({"Category": "labels_text", "Transaction Description": "text"})

    def label_map(example):
        # build mapping lazily
        return example

    # simple tokenization
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=128)

    train = train.map(tokenize, batched=True, remove_columns=["text"])
    train = train.select(range(min(len(train), max_samples)))
    test = test.map(tokenize, batched=True, remove_columns=["text"])
    test = test.select(range(min(len(test), max_samples)))

    # map labels to numeric (build mapping from unique values)
    labels = sorted(list(set(train["labels_text"])))
    label2id = {l: i for i, l in enumerate(labels)}

    def labels_to_tensor(ex):
        ex["labels"] = label2id[ex["labels_text"]]
        return ex

    train = train.map(labels_to_tensor, batched=False)
    test = test.map(labels_to_tensor, batched=False)

    # remove string column
    if "labels_text" in train.column_names:
        train = train.remove_columns(["labels_text"])
    if "labels_text" in test.column_names:
        test = test.remove_columns(["labels_text"])

    train.set_format(type="torch")
    test.set_format(type="torch")

    return train, test, label2id


def compute_f1(trainer, dataset):
    preds = trainer.predict(dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids
    return float(f1_score(y_true, y_pred, average="macro"))


def make_classifier(config, hidden_size: int, dropout: float):
    import torch.nn as nn

    return torch.nn.Sequential(
        torch.nn.Linear(config.hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.LayerNorm(hidden_size),
        torch.nn.Linear(hidden_size, config.num_labels),
    )


def run_short_experiment(grid: List[Dict], device: str = None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_ds, test_ds, label2id = prepare_datasets(tokenizer, max_samples=200)

    results = []

    for cfg in grid:
        print("Running config:", cfg)
        bert_config = BertForSeqClassificationMLPHeadConfig(
            id2label={v: k for k, v in label2id.items()},
            label2id=label2id,
            num_labels=len(label2id),
        )

        model = BertForSeqClassificationMLPHead(
            config=bert_config,
            base_bert_checkpoint=BASE_MODEL,
            local_files_only=False,
            use_safetensors=False,
        )

        # replace classifier head per cfg
        model.classifier = make_classifier(bert_config, hidden_size=cfg["head_hidden"], dropout=cfg["dropout"])

        # freeze base
        for p in model.bert.parameters():
            p.requires_grad = False

        # train on small subset
        training_args = TrainingArguments(
            output_dir="./tmp_sweep",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            learning_rate=cfg["base_lr"],
        )

        # optimizer: only head params
        import torch.optim as optim

        head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.") and p.requires_grad]
        optimizer = optim.AdamW([{"params": head_params, "lr": cfg["head_lr"]}], weight_decay=0.0)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=data_collator,
            optimizers=(optimizer, None),
        )

        trainer.train()
        f1 = compute_f1(trainer, test_ds)
        print(f"Config {cfg} -> F1: {f1:.4f}")
        results.append({**cfg, "f1": f1})

    # save results
    out_csv = "hp_sweep_results.csv"
    keys = list(results[0].keys()) if results else []
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    # print best
    best = max(results, key=lambda r: r["f1"]) if results else None
    print("Best config:", best)


if __name__ == "__main__":
    # small grid
    grid = [
        {"head_lr": 5e-4, "base_lr": 5e-6, "head_hidden": 256, "dropout": 0.1},
        {"head_lr": 1e-4, "base_lr": 5e-6, "head_hidden": 256, "dropout": 0.1},
        {"head_lr": 5e-4, "base_lr": 5e-6, "head_hidden": 512, "dropout": 0.3},
        {"head_lr": 1e-4, "base_lr": 5e-6, "head_hidden": 512, "dropout": 0.3},
    ]
    run_short_experiment(grid)
