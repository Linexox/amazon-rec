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

if __name__ == "__main__":
    img_shapes = []
    with open(DATA_PATH / 'preprocessed' / 'preprocessed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    save_path = DATA_PATH / "images"
    for inter in tqdm(data):
        img_urls = inter['image']
        itemId = inter['item_id']
        for img_url in img_urls:
            try:
                response = requests.get(img_url, timeout=5)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                Path.mkdir(save_path / itemId, exist_ok=True)
                n_img = len(list((save_path / itemId).glob("*.jpg")))
                img.save(save_path / itemId / f"{itemId}-{n_img}.jpg")
                img_shapes.append(img.size)
            except (requests.exceptions.RequestException, IOError) as e:
                logger.warning(f"Error fetching image from {img_url} for itemId {itemId}: {e}")
    print(f"max width: {max([shape[0] for shape in img_shapes])}, max height: {max([shape[1] for shape in img_shapes])}")
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(
        [shape[0] for shape in img_shapes],
        [shape[1] for shape in img_shapes]
    )
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Scatter plot of image dimensions")
    plt.show()