import json
import os
class ModelConfig:
    def __init__(
        self,
        model_id: str
    ):
        self.model_id = model_id

    def save_pretrained(self, dir: str):
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'config.json'), 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def from_pretrained(self, dir: str):
        path = os.path.join(dir, 'config.json') 
        config_dict = json.load(open(path))
        config = ModelConfig(**config_dict)
        return config
        
        