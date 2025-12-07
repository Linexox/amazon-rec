import math
import torch
import numpy as np
from typing import Optional
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))
    
    def forward(
        self,
        x: torch.Tensor
    ):
        x_0 = x
        rms = x.pow(2).mean(-1, keepdim=True)
        res = x_0 * torch.rsqrt(rms + self.eps) * self.weight
        return res

class PositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # add `batch` dimension
        self.register_buffer('pe', pe)
    
    def forward(
        self,
        x: torch.Tensor # (batch_size, seq_len, hidden_dim)
    ):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FeedForward(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = torch.nn.Linear(ffn_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()
    def forward(
        self,
        x: torch.Tensor
    ):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class Attention(torch.nn.Module):
    def __init__(
        self,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor, # (batch_size, seq_len, d_k)
        k: torch.Tensor, # (batch_size, seq_len, d_k)
        v: torch.Tensor, # (batch_size, seq_len, d_k)
        mask: Optional[torch.Tensor] = None
    ):
        # print(q.shape)
        # print(mask.shape)
        d_k = q.size(-1)
        k_t = k.transpose(-2, -1) # (batch_size, d_k, seq_len)
        scores = torch.matmul(q, k_t) / math.sqrt(d_k) # (batch_size, seq_len, seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v) # (batch_size, seq_len, d_k)
        return output, attn

class MHAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.d_k = hidden_dim // n_heads

        self.Wq = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Wk = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Wv = torch.nn.Linear(hidden_dim, hidden_dim)

        self.Wo = torch.nn.Linear(hidden_dim, hidden_dim)
        self.attention = Attention(dropout=dropout)
        self.dropout = torch.nn.Dropout(p=dropout)
    
    def forward(
        self,
        q: torch.Tensor, # (batch_size, seq_len, hidden_dim)
        k: torch.Tensor, # (batch_size, seq_len, hidden_dim)
        v: torch.Tensor, # (batch_size, seq_len, hidden_dim)
        mask: Optional[torch.Tensor] = None # (batch_size, seq_len)
    ):
        batch_size = q.size(0)

        q = self.Wq(q) # (batch_size, seq_len, hidden_dim)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)
        k = self.Wk(k) # (batch_size, seq_len, hidden_dim)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)
        v = self.Wv(v) # (batch_size, seq_len, hidden_dim)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len, d_k)

        # if mask is not None:
        #     mask = mask.unsqueeze(1)
            # mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        output, attn = self.attention(q, k, v, mask=mask) # output: (batch_size, n_heads, seq_len, d_k)

        output = output.transpose(1, 2).contiguous() # (batch_size, seq_len, n_heads, d_k)
        output = output.view(batch_size, -1, self.hidden_dim) # (batch_size, seq_len, hidden_dim)
        output = self.Wo(output) # (batch_size, seq_len, hidden_dim)
        output = self.dropout(output)
        return output, attn

class TransformerEncoderBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        norm_eps: float = 1e-8
    ):
        super().__init__()
        self.mha = MHAttention(hidden_dim, n_heads, dropout=dropout)
        self.rmsnorm1 = RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout=dropout)
        self.rmsnorm2 = RMSNorm(hidden_dim, eps=norm_eps)
    
    def forward(
        self,
        x: torch.Tensor, # (batch_size, seq_len, hidden_dim)
        mask: Optional[torch.Tensor] = None # (batch_size, seq_len, seq_len)
    ):
        # Multi-Head Attention
        x_0 = x
        x, attn = self.mha(x, x, x, mask=mask)
        x = x_0 + x
        x = self.rmsnorm1(x)

        # Feed Forward Network
        x_0 = x
        x = self.ffn(x)
        x = x_0 + x
        x = self.rmsnorm2(x)

        return x, attn

class MmItemEncoder(torch.nn.Module):
    def __init__(
        self,
        txt_dim: int,
        img_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.txt_dim = txt_dim
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(txt_dim + img_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(
        self,
        txt_emb: torch.Tensor, # (batch_size, txt_dim)
        img_emb: torch.Tensor  # (batch_size, img_dim)
    ):
        x = torch.cat([txt_emb, img_emb], dim=-1) # (batch_size, txt_dim + img_dim)
        x = self.proj(x) # (batch_size, hidden_dim)
        return x
    
class MmTransformer4Rec(torch.nn.Module):
    def __init__(
        self,
        config: dict,
    ): 
        super().__init__()
        for k, v in config.items():
            setattr(self, k, v)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        self.item_encoder = MmItemEncoder(
            self.txt_dim,
            self.img_dim,
            self.hidden_dim,
            dropout=self.dropout
        )
        self.positional_embedding = PositionalEmbedding(
            self.hidden_dim,
            self.max_seq_len,
            dropout=self.dropout
        )

        self.encoder_blocks = torch.nn.ModuleList([
            TransformerEncoderBlock(
                self.hidden_dim,
                self.n_heads,
                self.ffn_dim,
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ])

        self.final_norm = RMSNorm(self.hidden_dim)
        self.item_embeddings = torch.nn.Embedding(self.n_items, self.hidden_dim)
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if 'embedding' in p.__class__.__name__.lower():
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
    
    def create_padding_mask(
        self,
        seq_lens: list
    ):
        batch_size = len(seq_lens)
        max_len = max(seq_lens)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)
        for i, l in enumerate(seq_lens):
            mask[i, :l] = 1
        return mask
        # mask = torch.zeros(batch_size, max_len, max_len)
        # for i, seq_len in enumerate(seq_lens):
        #     mask[i, :seq_len, :seq_len] = 1
        # return mask.to(self.device)

    def forward(
        self,
        txt_embs: torch.Tensor, # (batch_size, seq_len, txt_dim)
        img_embs: torch.Tensor, # (batch_size, seq_len, img_dim)
        seq_lens: list = None
    ):
        batch_size = txt_embs.size(0)
        seq_len = txt_embs.size(1)
        mask = None
        if seq_lens is not None:
            mask = self.create_padding_mask(seq_lens) # (batch_size, seq_len, seq_len)
        
        x = self.item_encoder(
            txt_embs.view(-1, txt_embs.size(-1)), # (batch_size * seq_len, txt_dim)
            img_embs.view(-1, img_embs.size(-1))  # (batch_size * seq_len, img_dim)
        ) # (batch_size * seq_len, hidden_dim)
        x = x.view(batch_size, seq_len, -1) # (batch_size, seq_len, hidden_dim)
        x = self.positional_embedding(x) # (batch_size, seq_len, hidden_dim)
        for encoder in self.encoder_blocks:
            x, _ = encoder(x, mask=mask) # (batch_size, seq_len, hidden_dim)
        x = self.final_norm(x) # (batch_size, seq_len, hidden_dim)

        logits = torch.matmul(x, self.item_embeddings.weight.T) # (batch_size, seq_len, n_items)
        return logits
        # # user_repr = x.mean(dim=1) # (batch_size, hidden_dim)
        # if seq_lens is not None:
        #     batch_indices = torch.arange(batch_size, device=self.device)
        #     last_indices = torch.tensor([l-1 for l in seq_lens], device=self.device)
        #     user_repr = x[batch_indices, last_indices, :] # (batch_size, hidden_dim)
        # else:
        #     user_repr = x.mean(dim=1) # (batch_size, hidden_dim)
        # logits = user_repr @ self.item_embeddings.weight.T # (batch_size, n_items)
        # return logits, user_repr # logits: (batch_size, n_items), user_repr: (batch_size, hidden_dim)
    
    def predict_topk(
        self,
        txt_embs: torch.Tensor, # (batch_size, seq_len, txt_dim)
        img_embs: torch.Tensor, # (batch_size, seq_len, img_dim)
        seq_lens: list = None,
        topk: int = 1
    ): 
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(txt_embs, img_embs, seq_lens)
            topk_scores, topk_indices = torch.topk(logits, k=topk, dim=-1)
        return topk_scores, topk_indices

    def compute_loss(
        self,
        logits: torch.Tensor, # (batch_size, seq_len, n_items)
        target_items:  torch.Tensor, # (batch_size, seq_len)
        seq_lens: list =None
    ):
        batch_size, seq_len, n_items = logits.shape
        if seq_lens is not None:
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
            for i, l in enumerate(seq_lens):
                mask[i, :l] = 1
            logits_flat = logits.view(-1, n_items) # (batch_size * seq_len, n_items)
            targets_flat = target_items.view(-1)  # (batch_size * seq_len,)
            loss = self.loss_fn(logits_flat, targets_flat) # (batch_size * seq_len,)
            # torch.nn.CrossEntropyLoss()
            loss = loss * mask.view(-1).float()
            loss = loss.sum() / mask.sum() * 1.0
        else:
            raise NotImplementedError("Loss computation without seq_lens is not implemented.")
        return loss

if __name__ == "__main__":
    config = {
        'n_items': 1000,
        'txt_dim': 300,
        'img_dim': 2048,
        'hidden_dim': 512,
        'n_heads': 8,
        'n_layers': 2,
        'ffn_dim': 2048,
        'max_seq_len': 50,
        'dropout': 0.1,
        'device': 'cpu'
    }
    model = MmTransformer4Rec(config=config)

    batch_size = 4
    seq_len = 10
    txt_embs = torch.randn(batch_size, seq_len, 300)
    img_embs = torch.randn(batch_size, seq_len, 2048)
    seq_lens = [10, 8, 9, 7]

    logits = model(
        txt_embs=txt_embs,
        img_embs=img_embs,
        seq_lens=seq_lens
    )
    print("Logits shape:", logits.shape)  # Expected: (batch_size, seq_len, n_items)
    print(logits[2,:,:])
    