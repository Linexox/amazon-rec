import yaml
from model import MmTransformer4Rec

# class MmDataLoader:

class Trainer:
    def __init__(self, config):
        dataset_config = config['dataset']
        model_config = config['model']
        train_config = config['train']

        self.model = MmTransformer4Rec(model_config)
        # self.dataset = AmazonDataset(dataset_config)
        # self.dataloader = 

    def train_epoch(self, epoch):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

