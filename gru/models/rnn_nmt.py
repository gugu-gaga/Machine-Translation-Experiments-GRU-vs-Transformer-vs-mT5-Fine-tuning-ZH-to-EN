from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class DotAttention(nn.Module):
    def forward(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = torch.bmm(key, query.unsqueeze(2)).squeeze(2)  # [B,S]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), key).squeeze(1)      # [B,H]
        return ctx, attn

class GeneralAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
    def forward(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None):
        q = self.W(query)
        scores = torch.bmm(key, q.unsqueeze(2)).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), key).squeeze(1)
        return ctx, attn

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int, attn_dim: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, attn_dim, bias=False)
        self.W_s = nn.Linear(hidden_size, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None):
        q = self.W_h(query).unsqueeze(1)  # [B,1,A]
        k = self.W_s(key)                # [B,S,A]
        e = self.v(torch.tanh(q + k)).squeeze(2)  # [B,S]
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))
        attn = F.softmax(e, dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), key).squeeze(1)
        return ctx, attn

def build_attention(attn_type: str, hidden_size: int, attn_dim: int):
    t = attn_type.lower()
    if t == "dot": return DotAttention()
    if t in ("general", "multiplicative"): return GeneralAttention(hidden_size)
    if t in ("additive", "bahdanau"): return AdditiveAttention(hidden_size, attn_dim)
    raise ValueError(f"Unknown attention type: {attn_type}")

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_layers: int, dropout: float, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hidden_size, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor):
        x = self.emb(src_ids)
        lengths = src_mask.long().sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        return out, h

class RNNDecoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_layers: int, dropout: float, pad_id: int, attn_type: str, attn_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attn = build_attention(attn_type, hidden_size, attn_dim)
        self.rnn = nn.GRU(emb_dim + hidden_size, hidden_size, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0.0,
                          batch_first=True)
        self.pre_out = nn.Linear(hidden_size * 2, emb_dim)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_ids: torch.Tensor, state: torch.Tensor, enc_out: torch.Tensor, enc_mask: torch.Tensor):
        emb = self.emb(input_ids)         # [B,E]
        query = state[-1]                 # [B,H]
        ctx, attn = self.attn(query, enc_out, enc_mask)  # [B,H]
        rnn_in = torch.cat([emb, ctx], dim=-1).unsqueeze(1)
        out, new_state = self.rnn(rnn_in, state)
        out = self.dropout(out.squeeze(1))  # [B,H]
        pre = torch.tanh(self.pre_out(torch.cat([out, ctx], dim=-1)))  # [B,E]
        logits = pre @ self.emb.weight.t() + self.out_bias  # weight tying
        return logits, new_state, attn

class RNNAttnSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_size: int, num_layers: int, dropout: float, pad_id: int, attn_type: str, attn_dim: int):
        super().__init__()
        self.encoder = RNNEncoder(vocab_size, emb_dim, hidden_size, num_layers, dropout, pad_id)
        self.decoder = RNNDecoder(vocab_size, emb_dim, hidden_size, num_layers, dropout, pad_id, attn_type, attn_dim)

def load_pretrained_embeddings(emb: nn.Embedding, npy_path: str):
    import numpy as np
    w = np.load(npy_path)
    assert w.shape == tuple(emb.weight.data.shape), f"shape mismatch: {w.shape} vs {tuple(emb.weight.data.shape)}"
    emb.weight.data.copy_(torch.from_numpy(w))
