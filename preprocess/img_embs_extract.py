import json
import torch

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from transformers import ViTConfig, ViTModel, ViTImageProcessor

ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"

if __name__ == "__main__":
    model_name = r'D:\.Workspace\.MODEL\HF-Model-Backup\vit-base-patch16-224'
    config = ViTConfig.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name, config=config)
    processor = ViTImageProcessor.from_pretrained(model_name)


    img_embs_save_path = DATA_PATH / "amazon" / "img_embs.pt"
    img_embs_dict = {}
    for img_file in tqdm(glob(str(DATA_PATH / "amazon" / "images" / "*.jpg"))):
        inter_id = int(Path(img_file).stem)
        image = Image.open(img_file).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values']  # (1, 3, 224, 224)
        with torch.no_grad():
            img_embs = model(pixel_values=pixel_values).last_hidden_state  # (1, seq_len, hidden_size) [1, 197, 768]
        img_embs_dict[inter_id] = img_embs[:,0,:].squeeze(0)  # (hidden_size)
    torch.save(img_embs_dict, img_embs_save_path)