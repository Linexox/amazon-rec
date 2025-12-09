
import yaml
import torch
import numpy as np
import numpy.random as npr

from tqdm import tqdm
from pathlib import Path
from loguru import logger
from dataset import AmazonDataset
from model import MmTransformer4Rec
from time import strftime, localtime

cur_time = strftime("%Y%m%d_%H%M%S", localtime())

class MmTransformer4RecDataLoader:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.batch_size = config['batch_size']

    def __iter__(self, mode='train'):
        # if self.shuffle:
            # npr.shuffle(self.data)
        for i in range(0, self.dataset.__len__(mode), self.batch_size):
            range2 = min(i + self.batch_size, self.dataset.__len__(mode))
            batch_indices = range(i, range2)
            batch = [
                self.dataset.__getitem__(idx, mode)
                for idx in batch_indices
            ]
            yield self.collate_fn(batch)

    def collate_fn(self, batch):
        batch_item_ids = []
        batch_txt_embs = []
        batch_img_embs = []
        batch_targets = []
        batch_seq_lens = []
        for data_point in batch:
            item_ids = data_point.item_ids  
            txt_embs = data_point.txt_embs  
            img_embs = data_point.img_embs
            # seq_lens = sum([1 for id in item_ids if id != 0]) - 1
            seq_lens = len(item_ids)-1

            batch_item_ids.append(item_ids[:-1])
            batch_txt_embs.append(txt_embs[:-1])
            batch_img_embs.append(img_embs[:-1])
            batch_targets.append(item_ids[1:])
            batch_seq_lens.append(seq_lens)

        max_seq_len = max(batch_seq_lens)
        txt_emb_dim = batch_txt_embs[0].size(-1)
        img_emb_dim = batch_img_embs[0].size(-1)
        txt_pad_emb = torch.zeros((txt_emb_dim,), dtype=torch.float)
        img_pad_emb = torch.zeros((img_emb_dim,), dtype=torch.float)
        # print("==== padding ====")
        for i in range(len(batch)):
            pad_len = max_seq_len - batch_seq_lens[i]
            # print(len(batch_item_ids[i]), batch_seq_lens[i], pad_len)
            if pad_len > 0:
                batch_item_ids[i] += [-100] * pad_len
                batch_targets[i] += [-100] * pad_len
                batch_txt_embs[i] = torch.cat(
                    [batch_txt_embs[i], txt_pad_emb.unsqueeze(0).repeat(pad_len, 1)],
                    dim=0
                )
                batch_img_embs[i] = torch.cat(
                    [batch_img_embs[i], img_pad_emb.unsqueeze(0).repeat(pad_len, 1)],
                    dim=0
                )
        # print([len(ids) for ids in batch_item_ids])
        return {
            'item_ids': torch.tensor(batch_item_ids),
            'txt_embs': torch.stack(batch_txt_embs),
            'img_embs': torch.stack(batch_img_embs),
            'targets': torch.tensor(batch_targets, dtype=torch.long),
            'seq_lens': batch_seq_lens,
        }

class Trainer:
    def __init__(self, config):
        self.dataset_config = config['dataset']
        self.model_config = config['model']
        self.train_config = config['train']
        self.metric_logger = {}

        self.model = MmTransformer4Rec(self.model_config)
        self.dataset = AmazonDataset(self.dataset_config)
        self.dataloader = MmTransformer4RecDataLoader(self.dataset, self.train_config)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config['learning_rate']
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(self.dataloader.__iter__(mode='train')):
            # forward pass
            item_ids = batch['item_ids'].to(self.model.device)
            txt_embs = batch['txt_embs'].to(self.model.device)
            img_embs = batch['img_embs'].to(self.model.device)
            targets = batch['targets'].to(self.model.device)
            seq_lens = batch['seq_lens']

            outputs = self.model(
                txt_embs=txt_embs,
                img_embs=img_embs,
                seq_lens=seq_lens
            )
            # compute loss
            loss = self.model.compute_loss(
                logits=outputs,
                target_items=targets,
                seq_lens=seq_lens
            )

            # backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % self.train_config['log_interval'] == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        metrix = {'loss': total_loss / (batch_idx + 1)}
        return metrix

    def train(self):
        min_loss = float('inf')
        for epoch in range(self.train_config['num_epochs']):
            logger.info(f"Starting epoch {epoch+1} / {self.train_config['num_epochs']}")
            train_metrics = self.train_epoch(epoch)
            test_metrics = self.evaluate()
            train_metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            test_metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
            logger.info(f"Epoch {epoch} Train Metrics: {train_metrics_str}")
            logger.info(f"Epoch {epoch} Test Metrics: {test_metrics_str}")
            if test_metrics['loss'] < min_loss:
                min_loss = test_metrics['loss']
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= self.train_config['early_stop']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info("Training complete.")
        model_save_path = Path(self.train_config['save_path']) / cur_time / "model.pt"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)
        with open(model_save_path.parent / "metrics.yaml", 'w') as f:
            yaml.dump(self.metric_logger, f)
    def evaluate(self):
        self.model.eval()
        total_loss = 0

        hr = { '@5': 0, '@10': 0, '@20': 0 } # hit rate 
        ndcg = { '@5': 0,  '@10': 0, '@20': 0 } # 
        total_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader.__iter__(mode='test')):
                txt_embs = batch['txt_embs'].to(self.model.device)
                img_embs = batch['img_embs'].to(self. model.device)
                targets = batch['targets'].to(self.model.device)
                seq_lens = batch['seq_lens']
                
                logits = self.model(
                    txt_embs=txt_embs,
                    img_embs=img_embs,
                    seq_lens=seq_lens
                )
                
                loss = self.model.compute_loss(
                    logits=logits,
                    target_items=targets,
                    seq_lens=seq_lens
                )
                total_loss += loss.item()

                logits = logits[:, -1, :]  # (batch_size, n_items)
                total_samples += targets.size(0)
                # HR
                for at_k in hr.keys():
                    k = int(at_k[1:])
                    topk_items = torch.topk(logits, k=k, dim=-1).indices  # (batch_size, k)
                    for i in range(targets.size(0)):
                        target_item = targets[i, seq_lens[i]-1].item()
                        if target_item in topk_items[i]:
                            hr[at_k] += 1
                # NDCG
                for at_k in ndcg.keys():
                    k = int(at_k[1:])
                    topk_items = torch.topk(logits, k=k, dim=-1).indices  # (batch_size, k)
                    for i in range(targets.size(0)):
                        target_item = targets[i, seq_lens[i]-1].item()
                        if target_item in topk_items[i]:
                            rank = (topk_items[i] == target_item).nonzero(as_tuple=True)[0].item() + 1
                            ndcg[at_k] += 1 / np.log2(rank + 1)
        metrics = { 'loss': total_loss / (batch_idx + 1) }
        for at_k in hr.keys():
            metrics[f'HR{at_k}'] = hr[at_k] / total_samples if total_samples > 0 else 0.0
        for at_k in ndcg.keys():
            metrics[f'NDCG{at_k}'] = ndcg[at_k] / total_samples if total_samples > 0 else 0.0
        for k, v in metrics.items():
            if k not in self.metric_logger:
                self.metric_logger[k] = []
            self.metric_logger[k].append(v)
        return metrics

if __name__ == "__main__":

    ROOT_PATH = Path(__file__).parent
    DATA_PATH = ROOT_PATH / "data" / "preprocessed"
    LOGS_PATH = ROOT_PATH / "logs"
    LOG_FILE = LOGS_PATH / f"train_{cur_time}.log"
    SAVE_PATH = ROOT_PATH / "save"
    logger.add(LOG_FILE, rotation="10 MB")

    config = {
        'dataset': {
            'path': str(DATA_PATH),
            'train_ratio': 0.8,
        },
        'model': {
            'n_items': 286,  # Placeholder, should be set according to dataset
            'hidden_dim': 512,
            'txt_dim': 768,
            'img_dim': 768,
            'n_heads': 8,
            'n_layers': 2,
            'ffn_dim': 2048,
            'max_seq_len': 50,
            'dropout': 0.1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        },
        'train': {
            'save_path': str(SAVE_PATH),
            'batch_size': 32,
            'learning_rate': 3e-4,
            'num_epochs': 20,
            'log_interval': 20,
            'early_stop': 3
        }
    }
    trainer = Trainer(config)
    trainer.train()
