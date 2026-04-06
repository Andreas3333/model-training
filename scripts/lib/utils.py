import re
import io
import os
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


def extract_csv(model_output: str, keywords: list[str]) -> pd.DataFrame | None:
    """Return a Dataframe containing generated content. Use a keyword in the header to extract the specific section.
    None if keyword is not found.

    Args:
    model_output: str       llm output
    header_key: list[str]   Keyword in header

    Returns:
    dataframe
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
        ("######", "Header 6"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(model_output)

    header_found = False
    for doc in md_header_splits:
        for header_key, header_value in doc.metadata.items():
            for keyword in keywords:
                if keyword.lower() in header_value.lower():
                    header_found = True
                    break

        if header_found:
            content = doc.page_content
            break

    if not header_found:
        return None
    content = content.strip("```csv")
    content = content.strip("---")
    content = re.sub("```", "", content)
    content = re.sub(r'\n\s*\n', '\n', content, flags=re.MULTILINE)

    return pd.read_csv(io.StringIO(content))


def create_split(dataset: pd.DataFrame, model_training: str, proportions: tuple[float,float], shuffle: bool = False) -> None:
    """Splits and writes output to files of the model_training.

    Args:
    dataset (Dataframe):    Data to split
    model_training (str):   Model training base data directory
    proportions (tuple):    Proportions of split
    shuffle (bool):         Optionally shuffle before split
    """
    os.makedirs(f"{model_training}/train", exist_ok=True)
    os.makedirs(f"{model_training}/test", exist_ok=True)
    splits_proportions = np.array(proportions)

    if shuffle:
        dataset = dataset.sample(frac=1)

    train, test = np.array_split(dataset, (splits_proportions[:-1].cumsum() * len(dataset)).astype(int))

    pd.DataFrame(train).to_csv(f'{model_training}/train/train-set.csv', index=False)
    pd.DataFrame(test).to_csv(f'{model_training}/test/test-set.csv', index=False)

