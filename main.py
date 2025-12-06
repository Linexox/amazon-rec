import yaml
import torch
import numpy as np
import numpy.random as npr

from loguru import logger
from dataset import AmazonDataset
from model import MmTransformer4Rec

class MmTransformer4RecDataLoader:
    def __init__(self, dataset, config):
        # self.data = getattr(dataset, f"{split}_data")
        self.dataset = dataset
        self.batch_size = config['batch_size']
        # self.shuffle = config.get('shuffle', True)

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
        batch_inter_ids = []
        batch_item_ids = []
        batch_txt_embs = []
        batch_img_embs = []
        batch_targets = []
        batch_seq_lens = []
        for data_point in batch:
            inter_ids = data_point['inter_ids']
            item_ids = data_point['item_ids']
            txt_embs = data_point['txt_embs']
            img_embs = data_point['img_embs']
            seq_lens = (item_ids[:-1] != 0).sum().item()

            batch_inter_ids.append(inter_ids[:-1])
            batch_item_ids.append(item_ids[:-1])
            batch_txt_embs.append(txt_embs[:-1])
            batch_img_embs.append(img_embs[:-1])
            batch_targets.append(item_ids[1:])
            batch_seq_lens.append(seq_lens)

        return {
            'inter_ids': torch.tensor(batch_inter_ids, dtype=torch.long),
            'item_ids': torch.tensor(batch_item_ids, dtype=torch.long),
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
        # batch
        for batch_idx, batch in enumerate(self.dataloader.__iter__(mode='train')):
            # forward pass
            item_ids = batch['item_ids'].to(self.model.device)
            txt_embs = batch['txt_embs'].to(self.model.device)
            img_embs = batch['img_embs'].to(self.model.device)
            targets = batch['targets'].to(self.model.device)
            seq_lens = batch['seq_lens']

            outputs, _ = self.model(
                txt_embs=txt_embs,
                img_embs=img_embs,
                seq_lens=seq_lens
            )
            # compute loss
            loss = self.model.compute_loss(
                logits=outputs,
                targets=targets
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
        for epoch in range(self.train_config['num_epochs']):
            logger.info(f"Starting epoch {epoch} / {self.train_config['num_epochs']}")
            train_metrics = self.train_epoch(epoch)
            test_metrics = self.evaluate()
            train_metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            test_metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])

            logger.info(f"Epoch {epoch} Train Metrics: {train_metrics_str}")
            logger.info(f"Epoch {epoch} Test Metrics: {test_metrics_str}")

        logger.info("Training complete.")

    def evaluate(self):
        self.model.eval()
        total_loss = 0

        hr = {
            '@5': [0, 0],
            '@10': [0, 0],
        }
        ndgc = {
            '@5': [0, 0],
            '@10': [0, 0],
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader.__iter__(mode='test')):
                txt_embs = batch['txt_embs'].to(self.model.device)
                img_embs = batch['img_embs'].to(self.model.device)
                targets = batch['targets'].to(self.model.device)
                seq_lens = batch['seq_lens']
                
                logits, user_repr = self.model(
                    txt_embs=txt_embs,
                    img_embs=img_embs,
                    seq_lens=seq_lens
                )
                
                loss = self.model.compute_loss(logits=logits, target_items=targets)
                total_loss += loss.item()

                # HR
                for at_k in hr.keys():
                    k = int(at_k[1:])
                    topk_items = torch.topk(logits, k=k, dim=-1).indices  # (batch_size, k)
                    for i in range(targets.size(0)):
                        target_item = targets[i, seq_lens[i]-1].item()
                        if target_item in topk_items[i]:
                            hr[at_k][0] += 1
                        hr[at_k][1] += 1
                
                # NDCG
                for at_k in ndgc.keys():
                    k = int(at_k[1:])
                    topk_items = torch.topk(logits, k=k, dim=-1).indices  # (batch_size, k)
                    for i in range(targets.size(0)):
                        target_item = targets[i, seq_lens[i]-1].item()
                        if target_item in topk_items[i]:
                            rank = (topk_items[i] == target_item).nonzero(as_tuple=True)[0].item() + 1
                            ndgc[at_k][0] += 1 / np.log2(rank + 1)
                        ndgc[at_k][1] += 1
                
               
                
        metrics = { 'loss': total_loss / (batch_idx + 1) }
        for at_k in hr.keys():
            metrics[f'HR{at_k}'] = hr[at_k][0] / hr[at_k][1] if hr[at_k][1] > 0 else 0.0
        for at_k in ndgc.keys():
            metrics[f'NDCG{at_k}'] = ndgc[at_k][0] / ndgc[at_k][1] if ndgc[at_k][1] > 0 else 0.0
        return metrics

