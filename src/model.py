from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 2
    n_layers: int = 2
    d_ff: int = 512
    block_size: int = 64
    dropout: float = 0.1


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_ids = torch.arange(T, device=idx.device)
        pos_emb = self.pos_emb(pos_ids)[None, :, :]
        x = tok_emb + pos_emb

        causal_mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        causal_mask = causal_mask.view(1, 1, T, T)

        for block in self.blocks:
            x = block(x, mask=causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, self.cfg.vocab_size),
                targets.view(B * T),
                ignore_index=-100,
            )

        return logits, loss

