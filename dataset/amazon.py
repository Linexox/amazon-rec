import os
import json
import torch
from pathlib import Path

class AmazonDataset:
    def __init__(
        self,
        config: dict,
    ):
        for k, v in config.items():
            setattr(self, k, v)
        self.data_path = Path(self.path)

        self._load_data()
        self._load_other_data()

    def _load_data(self):
        with open(self.data_path / 'train_data.json', 'r') as f:
            self.train_data = json.load(f)
        with open(self.data_path / 'test_data.json', 'r') as f:
            self.test_data = json.load(f)

    def _load_other_data(self):
        with open(self.data_path / 'txt_embs.pt', 'rb') as f:
            self.txt_embs = torch.load(f, map_location='cpu')
        with open(self.data_path / 'img_embs.pt', 'rb') as f:
            self.img_embs = torch.load(f, map_location='cpu')
        