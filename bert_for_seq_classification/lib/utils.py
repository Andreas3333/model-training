import os
from typing import Any

import torch
from torch import Tensor
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


__all__ = [
    "create_one_hot_class_encodings",
    "create_simple_split",
    "create_label2id",
    "create_class_set_ids"
]

def create_one_hot_class_encodings(classes: list[str]) -> Tensor:
    """
    Create class one-hot encoded vector

    Args:
        y_classes (list[str]): class se to encode

    Returns:
        torch.tensor: vector of on-hot encoded classes
    """
    le = LabelEncoder()
    categories_array = torch.from_numpy(le.fit_transform(classes))
    return torch.nn.functional.one_hot(categories_array, len(categories_array))

def create_simple_split(
        datafile_path: str, output_dir: str, proportions: list[float,float], shuffle: bool) -> None:
    """
    Writes a basic train test proportioned split to output_dir.

    Args:
        datafile_path (str): input datafile_path
        output_dir (str): output result split files location (relative and creational)
        proportions (list[float,float]): proportions of train test split
        shuffle (bool): shuffle examples before split
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    dataset = pd.read_csv(datafile_path)
    splits_proportions = np.array(proportions)

    if shuffle:
        dataset = dataset.sample(frac=1)
    
    train, test = np.array_split(dataset, (splits_proportions[:-1].cumsum() * len(dataset)).astype(int))


    pd.DataFrame(train).to_csv(f'{output_dir}/train-set.csv', index=False)
    pd.DataFrame(test).to_csv(f'{output_dir}/test-set.csv', index=False)


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


def create_class_set_ids(id2label: dict) -> Tensor:
    """
    Create positional class set and index tensor

    Args:
        id2label (dict): dict of class ids and class labels
    Returns:
        class_set_ids (torch.tensor[int]): tensor of class_ids
    """
    # tmp = list(id2label.keys())
    # class_set_ids = torch.tensor(tmp)
    # return  class_set_ids
    return torch.tensor(list(id2label.keys()))
