import random
import math
import torch
from typing import List, Tuple
from transformers import PreTrainedTokenizer

class Dataset():
    def __init__(
        self,
        srcs: List[str],
        tokenizer: PreTrainedTokenizer,
        max_len: int
    ) -> None:
        self.srcs = srcs
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __getitem__(self, idx: int) -> dict:
        src = self.srcs[idx]
        encode = self.tokenizer.batch_encode_plus(
            [src],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encode['input_ids'].squeeze(),
            'attention_mask': encode['attention_mask'].squeeze()
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
    return Dataset(
        srcs=srcs,
        tokenizer=tokenizer,
        max_len=max_len
    )