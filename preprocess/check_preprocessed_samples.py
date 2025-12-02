import json
import os
from os.path import realpath, dirname

ROOT_PATH = dirname(dirname(realpath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data', 'preprocessed', 'preprocessed_data.json')
print(f"ROOT_PATH: {ROOT_PATH}")
print(f"DATA_PATH: {DATA_PATH}")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Number of preprocessed samples: {len(data)}")
print("Example sample:")
print(json.dumps(data[0], indent=2, ensure_ascii=False))