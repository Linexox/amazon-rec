import torch
import numpy as np

class HRMetric:
    def __init__(self, k):
        self.k = k
    def __call__(self, *args, **kwds):
        logits = kwds['logits']  # (batch_size, n_items)
        targets = kwds['target_items']  # (batch_size, seq_len)
        seq_lens = kwds['seq_lens']  # (batch_size,)

        batch_size = targets.size(0)
        topk_items = torch.topk(logits, k=self.k, dim=-1).indices  # (batch_size, k)
        hit_count = 0
        for i in range(batch_size):
            target_item = targets[i, seq_lens[i]-1].item()
            if target_item in topk_items[i]:
                hit_count += 1
        hr = hit_count / batch_size
        return hr