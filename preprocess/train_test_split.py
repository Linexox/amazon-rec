import json
import numpy as np
import numpy.random as npr

from pathlib import Path

TEST_RATIO = 0.2
TRAIN_MIN_INTERACTIONS = 3
npr.seed(42)

def train_test_split(raw_data, test_ratio=TEST_RATIO):
    train_data = []
    test_data = []
    for inter_dict in raw_data:
        user_id = inter_dict['user_id']
        interactions = inter_dict['interactions']
        n_interactions = len(interactions)
        n_test = int(n_interactions * test_ratio)
        if n_interactions < TRAIN_MIN_INTERACTIONS + n_test:
            continue
        else:
            train_inters = interactions[:-n_test]
            test_inters = interactions[-n_test:]
        
        train_data.append({
            'user_id': user_id,
            'interactions': train_inters
        })
        test_data.append({
            'user_id': user_id,
            'interactions': test_inters
        })
    return train_data, test_data

if __name__ == "__main__":
    ROOT_PATH = Path(__file__).parent.parent
    DATA_PATH = ROOT_PATH / "data"
    collated_data_path = DATA_PATH / 'preprocessed' / 'collated_data.json'
    with open(collated_data_path, 'r', encoding='utf-8') as f:
        collated_data = json.load(f)
    train_data, test_data = train_test_split(collated_data, test_ratio=TEST_RATIO)
    train_data_path = DATA_PATH / 'preprocessed' / 'train_data.json'
    test_data_path = DATA_PATH / 'preprocessed' / 'test_data.json'
    with open(train_data_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(test_data_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)