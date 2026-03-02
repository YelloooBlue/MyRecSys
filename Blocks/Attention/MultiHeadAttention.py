# 多头注意力机制从头实现

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, hidden_dim, num_heads, drop_out=0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim is not divisible"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.q_projection = nn.Linear(q_dim, hidden_dim)
        self.k_projection = nn.Linear(k_dim, hidden_dim)
        self.v_projection = nn.Linear(v_dim, hidden_dim)

        self.out_projection = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(drop_out)

        self.softmax = nn.Softmax()

    def forward(self,q, k, v, mask = None):
        batch_size = q.size(0)
        
        Q = self.q_projection(q)    # [batch_size, seq_len, hidden_dim]
        K = self.k_projection(k)
        V = self.v_projection(v)

        # 拆分头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力得分
        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
               
        # 注意力权重
        attn_weight = torch.softmax(scores, dim=-1)
        attn_weight = self.dropout(attn_weight)
        attn_out = torch.matmul(attn_weight, V)

        # 合并头
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.out_projection(attn_out)
        return out, attn_weight

if __name__ == "__main__":
    q = torch.randn(2, 5, 16)  # (batch_size, seq_len, q_dim)
    k = torch.randn(2, 5, 16)  # (batch_size, seq_len, k_dim)
    v = torch.randn(2, 5, 16)  # (batch_size, seq_len, v_dim)

    attn_layer = MultiHeadAttention(q_dim=16, k_dim=16, v_dim=16, hidden_dim=32, num_heads=4)
    out, attn_weight = attn_layer(q, k, v)
    print(out.shape)          # (batch_size, seq_len, hidden_dim)
    print(attn_weight.shape)  # (batch_size, num_heads, seq_len_q, seq_len_k)