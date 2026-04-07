"""
LP-SFT 'google-bert/bert-base-uncased' MLP head for multi class classification on a custom dataset of synthetic bank
transaction data gradually unfreezing final k base model layers.

training techniques used:
- frozen base model warm up for 3 epochs then unfreeze final k BERT layers
- focal loss
"""

import os
import json
import time
from typing import Dict, Any
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import datasets
import transformers
import evaluate
from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback
from transformers import DataCollatorWithPadding
from torch.optim import AdamW
from transformers.utils.logging import disable_progress_bar
from transformers.optimization import get_scheduler
from safetensors.torch import save_file as safetensors_save_file
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from rich.table import Table
from datasets import load_dataset
from rich.console import Console

from lib.utils import create_label2id
from cls_head import BertForSeqClassificationMLPHeadConfig, BertForSeqClassificationMLPHead


disable_progress_bar()
datasets.utils.logging.set_verbosity(0)
transformers.utils.logging.set_verbosity(0)

device = torch.accelerator.current_accelerator()
print(f"Using device: {device}")

BASE_MODEL = 'google-bert/bert-base-uncased'

class_set = {
    "classes":
        [
            "Coffee",
            "Credit Card",
            "Gas",
            "Groceries",
            "Misc",
            "Subscription",
            "Mobile Phone",
            "Rent - Mortgage",
            "Restaurant",
            "Student Loan",
            "Utilities Electric",
            "Home Internet",
            "Utilities Water",
            "Car Insurance"
        ]
}

label2id, id2label, num_labels = create_label2id(class_set["classes"])

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

data_files = {
    "train": "./data/00_sft/train-dataset.csv",
    "test": "./data/00_sft/test-dataset.csv"
}

data = load_dataset('csv', data_files=data_files).rename_columns({'Category': "labels_text", 'Transaction Description': "text"})
train_data = data["train"]
test_data = data["test"]

def labels_to_tensor(example: Dict[str, Any]) -> Dict[str, Any]:
    example["labels"] = torch.tensor(label2id[example["labels_text"]], dtype=torch.long)
    return example

train_ds = train_data.map(lambda x: tokenizer(x['text'], truncation=True, padding=False, max_length=128), batched=True, remove_columns=["text"])
test_ds = test_data.map(lambda x: tokenizer(x['text'], truncation=True, padding=False, max_length=128), batched=True, remove_columns=["text"])

train_ds = train_ds.map(labels_to_tensor, batched=False)
test_ds = test_ds.map(labels_to_tensor, batched=False)

if "labels_text" in train_ds.column_names:
    train_ds = train_ds.remove_columns(["labels_text"])
if "labels_text" in test_ds.column_names:
    test_ds = test_ds.remove_columns(["labels_text"])

train_ds = train_ds.with_format("torch")
test_ds = test_ds.with_format("torch")



bert_config = BertForSeqClassificationMLPHeadConfig(
    id2label=id2label,
    label2id={v: k for k,v in id2label.items()},
    num_labels=len(id2label)
)

model = BertForSeqClassificationMLPHead(
    config=bert_config,
    base_bert_checkpoint=BASE_MODEL,
    add_pooling_layer=False,
    local_files_only=True,
    use_safetensors=True
).to(device)


# Start training with all BERT layer parameters frozen (train only the head)
for param in model.bert.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if not name.startswith("bert."):
        param.requires_grad = True


def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_prec = evaluate.load("precision")
    metric_f1 = evaluate.load("f1")
    metric_rec = evaluate.load("recall")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric_prec.compute(predictions=predictions, references=labels, average="macro")["precision"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    recall = metric_rec.compute(predictions=predictions, references=labels, average="macro")["recall"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
        "recall": recall
    }


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"./training_runs/run_{current_time}"
best_model_dir = f"{output_dir}/best_model"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=8,
    weight_decay=1e-4,
    eval_delay=0,
    warmup_ratio=0.09,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    metric_for_best_model="eval_loss",
    # torch_compile=True,
    # torch_compile_backend="inductor", # TODO: for SM job
    load_best_model_at_end=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    report_to=["tensorboard"]
)

# Build optimizer with parameter groups with: small LR for base, larger LR for head
base_lr = 0.0001
head_lr = 5e-06

# Collect trainable params by group (base is currently frozen so base_params will be empty)
base_params = [p for n, p in model.named_parameters() if n.startswith("bert.") and p.requires_grad]
head_params = [p for n, p in model.named_parameters() if not n.startswith("bert.") and p.requires_grad]


optim_groups = []
if base_params:
    optim_groups.append({"params": base_params, "lr": base_lr})
if head_params:
    optim_groups.append({"params": head_params, "lr": head_lr})

optimizer = AdamW(optim_groups, weight_decay=training_args.weight_decay)

# estimate total training steps for scheduler
num_update_steps_per_epoch = max(1, len(train_ds) // max(1, training_args.per_device_train_batch_size))
num_training_steps = int(num_update_steps_per_epoch * training_args.num_train_epochs)
num_warmup_steps = int(training_args.warmup_ratio * num_training_steps) if hasattr(training_args, "warmup_ratio") else 0

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

class UnfreezeCallback(TrainerCallback):
    """
    TrainerCallback to unfreeze the last `num_layers` encoder layers after `unfreeze_epoch` epochs.
    It adds the newly-unfrozen parameters to the Trainer's optimizer with `base_lr`.
    """
    def __init__(self, unfreeze_epoch: int = 3, num_layers: int = 1, base_lr: float = 1e-5):
        self.unfreeze_epoch = unfreeze_epoch
        self.num_layers = num_layers
        self.base_lr = base_lr
        self.triggered = False

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.triggered:
            return
        try:
            epoch = int(state.epoch) if state.epoch is not None else None
        except Exception:
            epoch = None
        if epoch is None or epoch < self.unfreeze_epoch:
            return

        # TrainerCallback can be called with either the Trainer (under 'trainer')
        # or the model (under 'model') in kwargs. Handle both safely.
        trainer_obj = kwargs.get("trainer")
        model = None
        if trainer_obj is not None:
            model = getattr(trainer_obj, "model", None)
        else:
            model = kwargs.get("model")
        if model is None:
            return
        # Collect encoder layer indices
        layer_idxs = []
        for name, _ in model.named_parameters():
            if name.startswith("bert.encoder.layer."):
                try:
                    idx = int(name.split(".")[3])
                    layer_idxs.append(idx)
                except Exception:
                    continue
        layer_idxs = sorted(set(layer_idxs))
        if not layer_idxs:
            return
        last_idxs = layer_idxs[-self.num_layers:]

        params_to_add = []
        for name, param in model.named_parameters():
            for idx in last_idxs:
                prefix = f"bert.encoder.layer.{idx}."
                if name.startswith(prefix):
                    param.requires_grad = True
                    params_to_add.append(param)

        if params_to_add:
            # add to optimizer param groups
            if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                trainer.optimizer.add_param_group({"params": params_to_add, "lr": self.base_lr})
            self.triggered = True
            print(f"Unfroze encoder layers {last_idxs} and added {len(params_to_add)} params to optimizer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
    callbacks=[UnfreezeCallback(unfreeze_epoch=3, num_layers=2, base_lr=base_lr)],
)


## Train

# START TIMING
start_time = time.perf_counter()

train_result = trainer.train()
# END TIMING
end_time = time.perf_counter()
execution_time = end_time - start_time


## Save the trained model
trainer.save_model(best_model_dir)
os.remove(f"{best_model_dir}/model.safetensors")

bert_state_dict = {k: v for k, v in trainer.model.bert.state_dict().items()}
classifier_state_dict = {k: v for k, v in trainer.model.classifier.state_dict().items()}

safetensors_save_file(bert_state_dict, f"{best_model_dir}/bert_base.safetensors")
safetensors_save_file(classifier_state_dict, f"{best_model_dir}/classifier_head.safetensors")


## Evaluation and metrics

console = Console()

best_model_metrics = trainer.evaluate()
with open(os.path.join(output_dir, "best_model_metrics.json"), "w") as f:
    json.dump(best_model_metrics, f, indent=4)
console.print("Best model metrics:", best_model_metrics)


## Predictions

# Run predictions on the test set and save results
predictions_output = trainer.predict(test_ds)
preds = np.argmax(predictions_output.predictions, axis=1)
true_labels = predictions_output.label_ids

# Map ids to label names when available
pred_names = [id2label[int(p)] for p in preds]
true_names = [id2label[int(t)] for t in true_labels]

# Try to attach original text from the CSV if present
try:
    test_df = pd.read_csv(data_files["test"])
    texts = test_df["Transaction Description"].tolist()
    texts = texts[: len(preds)]
except Exception:
    texts = [None] * len(preds)


table = Table(show_header=True, header_style="bold magenta")
table.add_column("#", style="dim", width=6)
table.add_column("text")
table.add_column("pred_label")
table.add_column("true_label")

num_of_wrong_predictions = 0
for i, (t, p, tr) in enumerate(zip(texts, pred_names, true_names)):
    if p != tr:
        num_of_wrong_predictions += 1
        table.add_row(str(i), str(t) if t is not None else "", p, tr, style="red")

percentage = round(num_of_wrong_predictions/len(preds) * 100, 2)
console.print(f"Number of wrong predictions: {num_of_wrong_predictions} out of {len(preds)} - {percentage}%", style="yellow")
console.print(f"{trainer.state.best_model_checkpoint} loaded for eval predictions.")
console.print(table)


# Per-class metrics and confusion matrix
console.print("\nPer-class classification report:", style="bold green")
try:
    target_names = [id2label[i] for i in range(len(id2label))]
except Exception:
    target_names = [id2label[k] for k in sorted(id2label.keys(), key=int)]

try:
    report = classification_report(true_labels, preds, labels=list(range(len(id2label))), target_names=target_names, zero_division=0)
    console.print(report)
except Exception as e:
    console.print(f"Could not compute classification_report: {e}", style="red")

try:
    cm = confusion_matrix(true_labels, preds, labels=list(range(len(id2label))))
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)
    console.print(f"Confusion matrix saved to {cm_csv_path}", style="dim")

    # print a condensed confusion table
    cm_table = Table(show_header=True, header_style="bold cyan")
    cm_table.add_column("", width=12)
    for cname in target_names:
        cm_table.add_column(cname, overflow="fold")
    for rowname in cm_df.index:
        row = [rowname] + [str(int(v)) for v in cm_df.loc[rowname].tolist()]
        cm_table.add_row(*row)
    console.print(cm_table)

    try:
        plt.figure(figsize=(10, max(6, len(target_names) * 0.3)))
        if _HAS_SNS:
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        else:
            plt.imshow(cm_df, interpolation="nearest", cmap="Blues")
            plt.colorbar()
            for i in range(cm_df.shape[0]):
                for j in range(cm_df.shape[1]):
                    plt.text(j, i, int(cm_df.iat[i, j]), ha="center", va="center", color="black")
            plt.xticks(range(len(target_names)), target_names, rotation=90)
            plt.yticks(range(len(target_names)), target_names)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_png = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_png, dpi=150)
        plt.close()
        console.print(f"Confusion matrix plot saved to {cm_png}", style="dim")
    except Exception as e:
        console.print(f"Could not save confusion matrix plot: {e}", style="red")
except Exception as e:
    console.print(f"Could not compute or save confusion matrix: {e}", style="red")
