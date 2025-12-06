import json
import torch

from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer, BertModel

PAD_TOKEN, PAD_TOKEN_ID = '[PAD]', 0
UNK_TOKEN, UNK_TOKEN_ID = '[UNK]', 100
CLS_TOKEN, CLS_TOKEN_ID = '[CLS]', 101
SEP_TOKEN, SEP_TOKEN_ID = '[SEP]', 102
MASK_TOKEN, MASK_TOKEN_ID = '[MASK]', 103
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"

if __name__ == "__main__":
    model_name = r'D:\.Workspace\.MODEL\HF-Model-Backup\bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    with open(DATA_PATH / 'preprocessed' / 'collated_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    txt_embs_save_path = DATA_PATH / "preprocessed" / "txt_embs.pt"
    txt_embs_dict = {}
    for inter_dict in tqdm(data):
        for inter in inter_dict['interactions']:
            inter_id = inter['inter_id']
            cont = f"{inter['review_text']}. Summary: {inter['review_summary']}"
            cont_tokens = tokenizer(cont, return_tensors='pt', max_length=512, truncation=True, padding='max_length') # 
            with torch.no_grad():
                cont_embs = model(**cont_tokens).last_hidden_state  # (1, seq_len, hidden_size) [1, 512, 768]
            txt_embs_dict[inter_id] = cont_embs.squeeze(0)[0]  # (seq_len, hidden_size)


    torch.save(txt_embs_dict, txt_embs_save_path)

# print(tokenizer.all_special_tokens)
# print(tokenizer.all_special_ids)
# cont = "Love them! Worked great for my Patty Outfit."
# cont_tokens = tokenizer(cont, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
# print(cont_tokens['input_ids'].shape)
# print(cont_tokens['input_ids'][:,:20])
# print(cont_tokens['attention_mask'][:,:20])
# cont_embs = model(**cont_tokens).last_hidden_state
# print(cont_embs.shape)
# # print(type(cont_tokens))