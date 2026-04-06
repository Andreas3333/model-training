"""
Visualize the class distribution of the training and test datasets for the multi class classification problem.
"""

import argparse
from ctypes.wintypes import PLONG
from typing import Any
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Visualize the class distribution of the training and test datasets for the multi class classification problem.")
parser.add_argument('--train_path', type=str, default='data/00_sft/train-dataset.csv', help="path to the training dataset csv file")
parser.add_argument('--test_path', type=str, default='data/00_sft/test-dataset.csv', help="path to the test dataset csv file")
args = parser.parse_args()

def create_label2id(class_set: list[str])-> tuple[dict[Any,Any], dict[Any, Any], int]:
    """
    Create label2id mapping from a list of classes for a multi class classification dataset.

    Args:
        class_set list[str]: _description_
    
    Returns:
        tuple: (label2id, id2label, num_labels)
    """
    num_labels = len(class_set)

    id2label, label2id = dict(), dict()
    tmp = [i for i in range(num_labels)]
    for i,label in zip(tmp, class_set):
        label2id[label] = i
        id2label[i] = label

    return (label2id, id2label, num_labels)


x_y = ["Transaction Description", "Category"]

class_set ={
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
            "Utilities Water"
        ]
}

label2id, id2label, num_labels = create_label2id(class_set["classes"])

train_df = pd.read_csv(args.train_path)
test_df = pd.read_csv(args.test_path)

train_class_counts = np.zeros(shape=num_labels)

for i, cls in enumerate(label2id.keys()):
    for _ , row in train_df.iterrows():
        if cls == row["Category"]:
            train_class_counts[i] += 1
test_class_counts = np.zeros(shape=num_labels)

for i, cls in enumerate(label2id.keys()):
    for _ , row in test_df.iterrows():
        if cls == row["Category"]:
            test_class_counts[i] += 1


data_dir = args.train_path.split("/")[0:-1]
save_dir = "/".join(data_dir)

plt.figure(figsize=(12, 9))
plt.bar(class_set["classes"], train_class_counts, color='skyblue')

plt.title('Class Set Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Training set class distribution')
plt.legend(handles=[Patch(facecolor='skyblue', label=f'Training set: size {int(train_class_counts.sum())}')], loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.savefig(f"{save_dir}/train-dist.png", dpi=150)


plt.figure(figsize=(12, 9))
plt.bar(class_set["classes"], test_class_counts, color='skyblue')
plt.title('Class Set Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Test set class distribution')
plt.legend(handles=[Patch(facecolor='skyblue', label=f'Test set: size {int(test_class_counts.sum())}')], loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.savefig(f"{save_dir}/test-dist.png", dpi=150)


print(f"Class distributions visualization saved to {save_dir}")
