import yaml
from typing import Optional

class Config:
    def __init__(
        self,
        yaml_file: str = None
    ) -> None:
        with open(yaml_file, 'r') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(self, k, v)
        return
    
    def dump_yaml(
        self,
        yaml_file: str = None,
    ) -> None:
        with open(yaml_file, 'w') as f:
            yaml.dump(self.__dict__, f)
        return

class ModelConfig(Config):
    item_size: int
    vocab_size: int
    item_padding_idx: int

    