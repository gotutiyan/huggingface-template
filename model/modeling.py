from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import os
import json
from .configuration import ModelConfig


@dataclass
class ModelOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None

class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config.model_id)
        self.loss_fn = None # define here

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels
    ) -> ModelOutput:
        outputs = self.model(
            input_ids,
            attention_mask,
            token_type_ids
        )
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                outputs.logits, # need to write something, like outputs.logits.view(-1, num_labels)
                labels.view(-1)
            )
        return ModelOutput(
            loss,
            outputs.logits
        )

    def save_pretrained(self, dir: str) -> None:
        self.config.save_pretrained(dir)
        torch.save(
            self.state_dict(),
            os.path.join(dir, 'pytorch_model.bin')
        )
    
    @classmethod
    def from_pretrained(cls, dir: str):
        config = ModelConfig.from_pretrained(dir)
        model = Model(config)
        model.load_state_dict(torch.load(
            os.path.join(dir, 'pytorch_model.bin')
        ))
        return model
