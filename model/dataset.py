import random
import math
import torch
from typing import List, Tuple
from transformers import PreTrainedTokenizer

class Dataset():
    def __init__(
        self,
        srcs: List[str],
        labels,
        tokenizer: PreTrainedTokenizer,
        max_len: int
    ) -> None:
        self.srcs = srcs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, idx: int) -> dict:
        src = self.srcs[idx]
        label = self.labels[idx]
        encode = self.tokenizer.batch_encode_plus(
            [src],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encode['input_ids'].squeeze(),
            'attention_mask': encode['attention_mask'].squeeze(),
            'token_type_ids': encode['token_type_ids'].squeeze(),
            'labels': torch.tensor(label)
        }

    def __len__(self):
        return len(self.srcs)

def generate_dataset(
    input_file: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int
) -> Dataset:
    '''
    This function recieves input file path(s) and returns a Dataset instance.
    '''
    srcs = open(input_file).read().rstrip().split('\n')
    labels = None # You should write something to obtain labels
    return Dataset(
        srcs=srcs,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )