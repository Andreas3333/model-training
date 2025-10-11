import re
import io
import os
import sys
import secrets
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from langchain_text_splitters import MarkdownHeaderTextSplitter


def gen_short_hash(length=8):
    random_bytes = secrets.token_bytes(32)
    sha256_hash = hashlib.sha256(random_bytes).hexdigest()
    return sha256_hash[:length]

def get_date():
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second

    return f"{year}-{month:02d}-{day:02d}-{hour:02d}:{minute:02d}:{second:02d}"


def extract_csv(model_output: str) -> pd.DataFrame:
    """Return a Dataframe of the cleaned CSV data from generated model_output
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(model_output)

    keyword = "Output: CSV Content"
    for doc in md_header_splits:
        header_found = False
        for header_key, header_value in doc.metadata.items():
            if keyword.lower() in header_value.lower():
                header_found = True
                break

        if header_found:
            content = doc.page_content
            break

    content = content.strip("```csv")
    content = content.strip("---")
    content = re.sub("```", "", content)
    content = re.sub(r'\n\s*\n', '\n', content, flags=re.MULTILINE)

    return pd.read_csv(io.StringIO(content))


def create_split(dataset: pd.DataFrame, example_training: str, proportions: tuple[float,float], shuffle: bool = False) -> None:
    """Splits and writes output to files of the example_training.

    Args:
    dataset (Dataframe):    Data to split
    example_training (str): Example training base data directory
    proportions (tuple):    Proportions of split
    shuffle (bool):         Optionally shuffle before split
    """
    os.makedirs(f"{example_training}/train", exist_ok=True)
    os.makedirs(f"{example_training}/test", exist_ok=True)
    splits_proportions = np.array(proportions)

    if shuffle:
        dataset = dataset.sample(frac=1)

    train, test = np.array_split(dataset, (splits_proportions[:-1].cumsum() * len(dataset)).astype(int))

    pd.DataFrame(train).to_csv(f'{example_training}/train/train-set.csv', index=False)
    pd.DataFrame(test).to_csv(f'{example_training}/test/test-set.csv', index=False)

