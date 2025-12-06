import os   
import io
import json
import requests
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from pathlib import Path
from loguru import logger
from matplotlib import pyplot as plt

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"
LOGS_PATH = ROOT_PATH / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)
logger.add(LOGS_PATH / "img_download.log")

if __name__ == "__main__":
    img_shapes = []
    with open(DATA_PATH / 'preprocessed' / 'collated_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    save_path = DATA_PATH / "images"
    save_path.mkdir(parents=True, exist_ok=True)
    for inter_dict in tqdm(data):
        for inter in inter_dict['interactions']:
            img_url = inter['image']
            # itemId = inter['item_id']
            inter_id = inter['inter_id']
            try:
                response = requests.get(img_url, timeout=5)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                img.save(save_path / f"{inter_id}.jpg")
                img_shapes.append(img.size)
            except (requests.exceptions.RequestException, IOError) as e:
                logger.warning(f"Error fetching image from {img_url} for interId {inter_id}: {e}")
    # print(f"max width: {max([shape[0] for shape in img_shapes])}, max height: {max([shape[1] for shape in img_shapes])}")
    logger.info(f"max width: {max([shape[0] for shape in img_shapes])}, max height: {max([shape[1] for shape in img_shapes])}")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(
        [shape[0] for shape in img_shapes],
        [shape[1] for shape in img_shapes]
    )
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Scatter plot of image dimensions")
    plt.show()