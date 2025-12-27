from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

def build_norm(norm_type: str, dim: int) -> nn.Module:
    t = norm_type.lower()
    if t in ("layernorm", "ln"):
        return nn.LayerNorm(dim)
    if t in ("rmsnorm", "rms"):
        return RMSNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[offset:offset+T].unsqueeze(0).to(x.dtype)

def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
    num_buckets_half = num_buckets // 2
    sign = (relative_position > 0).to(torch.long)
    n = relative_position.abs()
    max_exact = num_buckets_half // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        (torch.log(n.float() / max_exact + 1e-6) / math.log(max_distance / max_exact))
        * (num_buckets_half - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets_half - 1))
    bucket = torch.where(is_small, n.to(torch.long), val_if_large)
    bucket = bucket + sign * num_buckets_half
    return bucket  # [T,T] in [0, num_buckets_half*2)

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bias = nn.Embedding(num_buckets, num_heads)
    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(T, device=device)[:, None]
        k_pos = torch.arange(T, device=device)[None, :]
        rel = k_pos - q_pos  # [T,T]
        buckets = _relative_position_bucket(rel, num_buckets=self.num_buckets, max_distance=self.max_distance)
        b = self.bias(buckets)  # [T,T,H]
        return b.permute(2,0,1).unsqueeze(0)  # [1,H,T,T]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rel_bias: bool = False,
                 rel_num_buckets: int = 32, rel_max_distance: int = 128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.use_rel_bias = use_rel_bias
        self.rel_bias = RelativePositionBias(n_heads, rel_num_buckets, rel_max_distance) if use_rel_bias else None

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Tq, _ = x_q.shape
        B2, Tk, _ = x_kv.shape
        assert B == B2
        q = self.q(x_q).view(B, Tq, self.n_heads, self.d_head).transpose(1,2)
        k = self.k(x_kv).view(B, Tk, self.n_heads, self.d_head).transpose(1,2)
        v = self.v(x_kv).view(B, Tk, self.n_heads, self.d_head).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(self.d_head)  # [B,H,Tq,Tk]
        if self.use_rel_bias and (Tq == Tk):
            scores = scores + self.rel_bias(Tk, scores.device).to(scores.dtype)

        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.to(scores.device), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        ctx = torch.matmul(attn, v)  # [B,H,Tq,dh]
        ctx = ctx.transpose(1,2).contiguous().view(B, Tq, self.d_model)
        return self.out(ctx)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(F.gelu(self.w1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm_type: str, use_rel_bias: bool):
        super().__init__()
        self.norm1 = build_norm(norm_type, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rel_bias=use_rel_bias)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = build_norm(norm_type, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop1(self.attn(h, h, key_padding_mask=src_mask))
        h = self.norm2(x)
        x = x + self.drop2(self.ff(h))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, norm_type: str, use_rel_bias: bool):
        super().__init__()
        self.norm1 = build_norm(norm_type, d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_rel_bias=use_rel_bias)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = build_norm(norm_type, d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, use_rel_bias=False)
        self.drop2 = nn.Dropout(dropout)
        self.norm3 = build_norm(norm_type, d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop1(self.self_attn(h, h, attn_mask=tgt_mask))
        h = self.norm2(x)
        x = x + self.drop2(self.cross_attn(h, memory, key_padding_mask=src_mask))
        h = self.norm3(x)
        x = x + self.drop3(self.ff(h))
        return x

@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout: float
    norm_type: str
    pos_type: str  # "sinusoidal" or "relative_t5"
    max_len: int

class TransformerNMT(nn.Module):
    def __init__(self, cfg: TransformerConfig, pad_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(cfg.d_model, max_len=max(cfg.max_len+8, 1024)) if cfg.pos_type == "sinusoidal" else None
        use_rel_bias = (cfg.pos_type == "relative_t5")
        self.enc_layers = nn.ModuleList([EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.norm_type, use_rel_bias) for _ in range(cfg.n_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.norm_type, use_rel_bias) for _ in range(cfg.n_layers)])
        self.enc_norm = build_norm(cfg.norm_type, cfg.d_model)
        self.dec_norm = build_norm(cfg.norm_type, cfg.d_model)
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(src)
        if self.pos is not None:
            x = self.pos(x)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt_in: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(tgt_in)
        if self.pos is not None:
            x = self.pos(x)
        x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        T = x.size(1)
        causal = torch.ones((T, T), device=x.device, dtype=torch.bool).tril()
        for layer in self.dec_layers:
            x = layer(x, memory, tgt_mask=causal, src_mask=src_mask)
        x = self.dec_norm(x)
        logits = torch.matmul(x, self.emb.weight.t()) + self.out_bias
        return logits

def load_pretrained_embeddings(emb: nn.Embedding, npy_path: str):
    import numpy as np
    w = np.load(npy_path)
    assert w.shape == tuple(emb.weight.data.shape), f"shape mismatch: {w.shape} vs {tuple(emb.weight.data.shape)}"
    emb.weight.data.copy_(torch.from_numpy(w))
