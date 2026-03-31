import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 一次线性层同时得到 QKV，面试里这样写更省
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        qkv = self.qkv_proj(x)                   # [B, T, 3C]
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]        # [B, H, T, D]

        # attention score
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # [B, H, T, T]

        # causal mask: 不能看未来
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)        # [B, H, T, T]
        out = attn @ v                          # [B, H, T, D]

        out = out.transpose(1, 2).contiguous().reshape(B, T, C)  # [B, T, C]
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        # Pre-LN 写法，现在更常见，也更稳
        x = x + self.attn(self.ln1(x))   # residual
        x = x + self.ffn(self.ln2(x))    # residual
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=256, n_layers=2, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        # idx: [B, T]
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)   # [1, T]

        x = self.token_emb(idx) + self.pos_emb(pos)             # token + position

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.head(x)                                   # [B, T, vocab_size]
        return logits