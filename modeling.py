from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import os
import json


@dataclass
class ModelOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None

class Model(nn.Module):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)

    def forward(self, batch: dict) -> ModelOutput:
        outputs = self.model(**batch)
        return ModelOutput(
            outputs.loss,
            outputs.logits
        )

    def save_pretrained(self, dir: str) -> None:
        torch.save(self.state_dict(), os.path.join(dir, 'pytorch_model.bin'))
    
    @classmethod
    def from_pretrained(cls, dir: str):
        config = json.load(open(os.path.join(dir, 'my_config.json')))
        model = Model(config.model_id)
        model.load_state_dict(torch.load(os.path.join(dir, 'pytorch_model.bin')))
        return model

'''
An example of SequenceClassifier:

@dataclass
class ModelOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None

class Model(nn.Module):
    def __init__(self, model_id, num_lables):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_id)
        config = AutoConfig,from_pretrained(model_id)
        self.linear = nn.Linear(config.hidden_size, num_labels)

    def forward(self, batch, labels=None):
        bert_out = self.bert(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        out = self.linear(bert_out.pooler_out)

        if labels is not None:
            loss_fn = nn.CrossEntropy()
            loss = loss_fn(out, labels)
        
        return ModelOutput(
            loss=loss,
            logits=out
        )

    def save_pretrained(self, path: str='model/'):
        torch.save(self.state_dict(), os.path.join(path, 'pytorch_model.bin'))
    
    @classmethod
    def from_pretrained(cls, restore_dir: str) -> Model:
        config = json.load(open(os.path.join(restore_dir, 'my_config.json')))
        model = Model(
            model_id=config['model_id'],
            num_labels=config['num_labels']
        )
        model.load_state_dict(torch.load(os.path.join(restore_dir, 'pytorch_model.bin')))
        return model
'''