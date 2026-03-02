# 注意力机制算子实现
import torch
import torch.nn as nn

# ========================================= 注意力算子 Attention =========================================

"""
注意力算子——数学上或理论上Attention机制的核心部分，负责计算注意力权重并加权求和得到输出。
"""

class DotProductAttention(nn.Module):
    """
    最基础的点积注意力，不包含投影层。
    输入:
    - q: (..., q_len, d_k)  # q 和 k 的特征维度必须相同，才能计算点积得分
    - k: (..., k_len, d_k)  # k_len 和 v_len 必须相同，否则无法加权求和
    - v: (..., k_len, d_v)
    - mask: 可接受多种形状的掩码，会根据输入的形状进行广播：
        -  (..., q_len, k_len) 每个查询对应一个掩码向量，指示哪些键是有效的
        -  (..., k_len) 所有查询共享一个掩码向量，指示哪些键是有效的【在序列中比较常见】
    输出:
    - out: (..., q_len, d_v)
    - attn: (..., q_len, k_len) 注意力权重矩阵
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class ScaledDotProductAttention(nn.Module):
    """
    经典缩放点积注意力，不包含投影层。
    同上，只不过在计算得分时会除以一个缩放因子，通常是 d_k 的平方根，以防止得分过大导致梯度消失。
    """

    def __init__(self, scale=None, dropout=0.0):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scale = self.scale if self.scale is not None else d_k ** -0.5
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn

# ========================================= 注意力层 AttentionLayer =========================================

"""
注意力层——应用层面的Attention机制，通常包含投影层（线性变换）和注意力算子（如上所示）。
它将输入的查询、键、值进行线性变换后送入注意力算子计算输出。
"""

class AttentionLayer(nn.Module):
    """
    注意力层，包含线性变换和注意力算子。
    输入:
    - q: (batch_size, q_len, q_dim) 
    - k: (batch_size, k_len, k_dim)
    - v: (batch_size, k_len, v_dim)
    - mask: (batch_size, q_len, k_len) 尽量不要使用广播掩码（在输入该层前就处理好掩码的形状），以避免潜在的广播错误。

    输出:
    - out: (batch_size, q_len, hidden_dim)
    """

    def __init__(self, q_dim, k_dim, v_dim, hidden_dim, attention_type="scaled_dot", dropout=0.0):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(k_dim, hidden_dim)
        self.v_proj = nn.Linear(v_dim, hidden_dim)

        if attention_type == "dot":
            self.attention = DotProductAttention(dropout=dropout)
        elif attention_type == "scaled_dot":
            self.attention = ScaledDotProductAttention(dropout=dropout)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

    def forward(self, q, k, v, mask=None):
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)

        out, attn = self.attention(q_proj, k_proj, v_proj, mask)
        return out, attn
    
class MultiHeadAttentionLayer(nn.Module):
    """
    多头注意力层，包含多个注意力头和线性变换。
    输入:
    - q: (batch_size, q_len, q_dim)
    - k: (batch_size, k_len, k_dim)
    - v: (batch_size, k_len, v_dim)

    输出:
    - out: (batch_size, q_len, hidden_dim)
    """

    def __init__(self, q_dim, k_dim, v_dim, hidden_dim, num_heads=8, attention_type="scaled_dot", dropout=0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(k_dim, hidden_dim)
        self.v_proj = nn.Linear(v_dim, hidden_dim)

        if attention_type == "dot":
            self.attention = DotProductAttention(dropout=dropout)
        elif attention_type == "scaled_dot":
            self.attention = ScaledDotProductAttention(scale=self.head_dim ** -0.5, dropout=dropout)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性变换并分头
        # 因为上面实现的注意力算子仅关注最后2个维度并计算注意力得分，为了通用和提升效率，需要重塑。
        # [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_head]
        q_proj = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算多头注意力
        out, attn = self.attention(q_proj, k_proj, v_proj, mask)

        # 合并多头 [batch_size, num_heads, seq_len, d_head] -> [batch_size, seq_len, hidden_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim) # contiguous()是因为transpose后内存不连续。
        out = self.out_proj(out)    # 多头一般会有一个输出线性层来整合多头的输出
        out = self.out_dropout(out)
        return out, attn