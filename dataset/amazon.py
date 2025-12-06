import os
import json
import torch
import numpy as np
import numpy.random as npr

from pathlib import Path
from dataclasses import dataclass

DUMMY_IMG_EMB = torch.zeros(768)
MIN_INTER_NUM = 3
npr.seed(42)

@dataclass
class RecDataPoint:
    item_ids: list[int]
    txt_embs: torch.Tensor
    img_embs: torch.Tensor

class AmazonDataset:
    def __init__(
        self,
        config: dict,
    ):
        for k, v in config.items():
            setattr(self, k, v)
        self.data_path = Path(self.path)

        self._load_data()
        self._filter_data()
        self.collated_data = self._load_other_data()
        self.all_data = self._augment_and_add(self.collated_data)
        self.train_data, self.test_data = self.split_data(self.all_data, self.train_ratio)

    def _load_data(self):
        with open(self.data_path / 'collated_data.json', 'r') as f:
            self.collated_data = json.load(f)

    def _load_other_data(self):
        with open(self.data_path / 'item_ids.json', 'r') as f:
            item_ids = json.load(f)
        self.idx_2_itemId = {int(k): v for k, v in item_ids.items()}
        self.itemId_2_idx = {v: int(k) for k, v in item_ids.items()}
        with open(self.data_path / 'txt_embs.pt', 'rb') as f:
            self.txt_embs = torch.load(f, map_location='cpu')
        with open(self.data_path / 'img_embs.pt', 'rb') as f:
            self.img_embs = torch.load(f, map_location='cpu')
        
    def _augment_and_add(self, data):
        augmented_data = []
        for inter_dict in data:
            user_id = inter_dict['user_id']
            if len(inter_dict['interactions']) <= MIN_INTER_NUM:
                continue
            cur_interactions = inter_dict['interactions'][:MIN_INTER_NUM]
            for inter in inter_dict['interactions'][MIN_INTER_NUM+1:]:
                cur_interactions.append(inter)
                augmented_data.append({
                    'user_id': user_id,
                    'interactions': cur_interactions.copy()
                })
        return augmented_data

    def split_data(self, data, train_ratio):
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        npr.shuffle(data)
        train_data = data[:n_train]
        test_data = data[n_train:]
        return train_data, test_data
    
    def __getitem__(self, index, mode='train'):
        inter_dict = getattr(self, f"{mode}_data")[index]
        inter_ids = [
            inter['inter_id']
            for inter in inter_dict['interactions']
        ]
        item_ids = [
            self.itemId_2_idx[inter['item_id']]
            for inter in inter_dict['interactions']
        ]
        txt_embs = [
            self.txt_embs[inter_id]
            for inter_id in inter_ids
        ]
        img_embs = [
            self.img_embs.get(inter_id, DUMMY_IMG_EMB)
            for inter_id in inter_ids
        ]
        txt_embs = torch.stack(txt_embs, dim=0)  # (seq_len, hidden_size)
        img_embs = torch.stack(img_embs, dim=0)  # (seq_len, hidden_size)
        return RecDataPoint(
            item_ids=item_ids,
            txt_embs=txt_embs,
            img_embs=img_embs
        )
    def __len__(self, mode='train'):
        return len(getattr(self, f"{mode}_data"))