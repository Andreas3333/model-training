from typing import Any

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from lib.utils import create_label2id


class SequenceDataset(Dataset):
    """Multi class sequence classification Dataset class"""

    def __init__(self, base_model_tokenizer: str, datafile_path: str, x_feat: str, class_set_dict: dict, device):
        self.base_model_tokenizer = base_model_tokenizer
        self.datafile_path = datafile_path
        self.x_feat = x_feat
        self.class_set_dict = class_set_dict
        self.device = device

        self.dataset = pd.read_csv(self.datafile_path).rename(columns={'Category': 'labels'})
        self.sequences = self.dataset[self.x_feat].tolist()
        label2id = create_label2id(self.class_set_dict["classes"])[0]
        tmp_label = [ label2id[label] for label in self.dataset['labels'] ]
        self.labels = torch.tensor((tmp_label), dtype=torch.long).to(device=self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.base_model_tokenizer)
        self.prepare_input()

    def prepare_input(self) -> None:
        self.embeddings = self.tokenizer(
            self.sequences,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        self.embeddings.to(device=self.device)
        self.embeddings = {
            'input_ids': self.embeddings['input_ids'],
            'attention_mask': self.embeddings['attention_mask'],
            'token_type_ids': self.embeddings['token_type_ids']
            }
        self.sequence_len = self.embeddings['input_ids'].shape[-1]

    def __getitem__(self, idx) -> dict[str, str | Any]:
        item = {key: val[idx].detach().clone() for key, val in self.embeddings.items()}
        item['labels'] = self.labels[idx].detach().clone()
        return item

    def __len__(self) -> int:
        return len(self.dataset)

