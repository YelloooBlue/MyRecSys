import torch
import torch.nn as nn

# ================================== 数学（公式）的注意力实现 ==================================

# 点积注意力（基本不用，仅作学习）
class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):   # [B, T, D]
        score = torch.matmul(q, k.transpose(-2, -1))    # [B, T, D] @ [B, D, T] = [B, T, T]
        attn = torch.softmax(score, dim=-1)             # [B, T, T]
        out = torch.matmul(attn, v)                     # [B, T, T] @ [B, T, D] = [B, T, D]
        return out, attn


# 缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        B, T, D = q.size()
        score = torch.matmul(q, k.transpose(-2, -1))    # [B, T, D] @ [B, D, T] = [B, T, T]
        score = score / (D ** 0.5)                      # 缩放，也可以写成 score = score * (D ** -0.5)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(score, dim=-1)             # [B, T, T]
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                     # [B, T, T] @ [B, T, D] = [B, T, D]
        return out, attn


# 多头缩放点积注意力（数学部分）
class MultiHeadScaledDotProductAttention(nn.Module):
    def __init__(self, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, T, D = q.size()
        assert D % self.num_heads == 0, "特征维度必须能被头数整除"
        head_dim = D // self.num_heads

        # [B, T, D] -> [B, T, H, HD] -> [B, H, T, HD]
        q = q.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, head_dim).transpose(1, 2)

        # TODO: 后续可以选择复用前面实现的 ScaledDotProductAttention 类，但要注意把 B, T, D = q.size() 改为 D = q.size(-1) 来适配多头的输入形状
        score = torch.matmul(q, k.transpose(-2, -1))    # [B, H, T, HD] @ [B, H, HD, T] = [B, H, T, T]
        score = score * (head_dim ** -0.5)              # [B, H, T, T]
        attn = torch.softmax(score, dim=-1)             # [B, H, T, T]
        out = torch.matmul(attn, v)                     # [B, H, T, T] @ [B, H, T, HD] = [B, H, T, HD]

        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, H, T, HD] -> [B, T, H, HD] -> [B, T, D]
        return out, attn


# ================================== 带权重的注意力实现 ==================================

# 单头注意力层
class SingleHeadAttention(nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(k_dim, hidden_dim)
        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.attention = ScaledDotProductAttention()
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = self.q_proj(q)   # [B, T, D] -> [B, T, hidden_dim]
        k = self.k_proj(k)   # [B, T, D] -> [B, T, hidden_dim]
        v = self.v_proj(v)   # [B, T, D] -> [B, T, hidden_dim]
        out, attn = self.attention(q, k, v)
        out = self.out_proj(out)  # [B, T, hidden_dim] -> [B, T, hidden_dim]
        return out, attn


# 单头自注意力层（q=k=v）
class SingleHeadSelfAttention(SingleHeadAttention):
    def forward(self, x: torch.Tensor):
        return super().forward(x, x, x)


# 多头注意力层
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(k_dim, hidden_dim)
        self.v_proj = nn.Linear(v_dim, hidden_dim)
        self.attention = MultiHeadScaledDotProductAttention(num_heads=num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v):
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        out, attn = self.attention(q_proj, k_proj, v_proj)
        out = self.out_proj(out)
        return out, attn


# 多头自注意力层（q=k=v）
class MultiHeadSelfAttention(MultiHeadAttentionLayer):
    def forward(self, x):
        return super().forward(x, x, x)