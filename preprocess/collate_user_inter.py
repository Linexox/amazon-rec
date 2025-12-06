import os
import json
# import requests
import numpy as np

from pathlib import Path
from numpy.random import randint

np.random.seed(42)

def collate_user_interactions(raw_data: dict,):
    new_data = []
    user_inter = {}
    for inter_id, inter in enumerate(raw_data):
        if isinstance(inter['image'], list):
            r = randint(0, len(inter['image']))
            inter['image'] = inter['image'][r]
            # inter['image'] = inter['image'][0]
        inter['inter_id'] = inter_id
        userId = inter['user_id']
        if userId not in user_inter:
            user_inter[userId] = []
        user_inter[userId].append(inter)
    
    for user_id, inters in user_inter.items():
        inters.sort(key=lambda x: x['time'])
        new_data.append({
            'user_id': user_id,
            'interactions': inters
        })

    return new_data

if __name__ == "__main__":
    ROOT_PATH = Path(__file__).parent.parent
    DATA_PATH = ROOT_PATH / "data"
    raw_data_path = DATA_PATH / 'preprocessed' / 'preprocessed_data.json'
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    collated_data = collate_user_interactions(raw_data)
    collated_data_path = DATA_PATH / 'preprocessed' / 'collated_data.json'
    with open(collated_data_path, 'w', encoding='utf-8') as f:
        json.dump(collated_data, f, ensure_ascii=False, indent=4)