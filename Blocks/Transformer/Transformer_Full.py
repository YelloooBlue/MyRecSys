import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# =========================
# 2. Padding Mask / Causal Mask
# =========================
def make_padding_mask(q, k, pad_idx=0):
    """
    由于 Attention 计算的是 Q 和 K 的关系，我们需要遮住 K 中是 Padding 的位置。
    这里假设当前输入的 token id 是 pad_idx 时（例如[1,2,0,0]），表示这个位置是 padding。
    q: [batch_size, q_len]
    k: [batch_size, k_len]
    return: [batch_size, 1, q_len, k_len]
    """
    q_len = q.size(1)
    k_len = k.size(1)

    # k中 pad 的位置为 True
    k_pad = (k == pad_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,K]
    k_pad = k_pad.expand(-1, 1, q_len, -1)            # [B,1,Q,K]
    return k_pad


def make_causal_mask(seq_len, device):
    """
    下三角可见，上三角不可见
    return: [1, 1, seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)


# =========================
# 3. Scaled Dot-Product Attention
# =========================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Q: [B, H, Q_len, D_k]
        K: [B, H, K_len, D_k]
        V: [B, H, K_len, D_v]
        mask: [B, 1 or H, Q_len, K_len]
        """
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: [B, H, Q_len, K_len]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        # output: [B, H, Q_len, D_v]

        return output, attn


# =========================
# 4. Multi-Head Attention
# =========================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能整除 num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        x: [B, L, d_model]
        -> [B, H, L, head_dim]
        """
        B, L, _ = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        x: [B, H, L, head_dim]
        -> [B, L, d_model]
        """
        B, H, L, D = x.size()
        x = x.transpose(1, 2).contiguous().view(B, L, H * D)
        return x

    def forward(self, query, key, value, mask=None):
        """
        query: [B, Q_len, d_model]
        key:   [B, K_len, d_model]
        value: [B, K_len, d_model]
        mask:  [B, 1, Q_len, K_len] or [B, H, Q_len, K_len]
        """
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        context, attn = self.attention(Q, K, V, mask=mask)
        context = self.combine_heads(context)
        output = self.W_o(context)

        return self.dropout(output), attn


# =========================
# 5. Feed Forward Network
# =========================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# =========================
# 6. Encoder Layer
# =========================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-Attention + Residual + LayerNorm
        attn_out, attn = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x, attn


# =========================
# 7. Decoder Layer
# =========================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_tgt_mask=None):
        # 1) Masked Self-Attention
        self_attn_out, self_attn = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # 2) Cross-Attention
        cross_attn_out, cross_attn = self.cross_attn(x, enc_out, enc_out, mask=src_tgt_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # 3) FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x, self_attn, cross_attn


# =========================
# 8. Encoder
# =========================
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, pad_idx=0, dropout=0.1, max_len=5000):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: [B, src_len]
        """
        src_mask = make_padding_mask(src, src, pad_idx=self.pad_idx)  # [B,1,S,S]

        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attn_list.append(attn)

        return x, src_mask, attn_list


# =========================
# 9. Decoder
# =========================
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, pad_idx=0, dropout=0.1, max_len=5000):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_out, src, src_mask):
        """
        tgt: [B, tgt_len]
        enc_out: [B, src_len, d_model]
        src: [B, src_len]
        src_mask: [B,1,src_len,src_len]
        """
        B, tgt_len = tgt.size()
        device = tgt.device

        # decoder self-attention 需要 padding mask + causal mask
        pad_mask = make_padding_mask(tgt, tgt, pad_idx=self.pad_idx)   # [B,1,T,T]
        causal_mask = make_causal_mask(tgt_len, device=device)         # [1,1,T,T]
        tgt_mask = pad_mask | causal_mask

        # cross-attention 里，query来自 tgt，key/value来自 src
        src_tgt_mask = make_padding_mask(tgt, src, pad_idx=self.pad_idx)  # [B,1,T,S]

        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        self_attn_list = []
        cross_attn_list = []

        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_out, tgt_mask=tgt_mask, src_tgt_mask=src_tgt_mask)
            self_attn_list.append(self_attn)
            cross_attn_list.append(cross_attn)

        return x, self_attn_list, cross_attn_list


# =========================
# 10. Full Transformer
# =========================
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        src_pad_idx=0,
        tgt_pad_idx=0,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            pad_idx=src_pad_idx,
            dropout=dropout,
            max_len=max_len,
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            pad_idx=tgt_pad_idx,
            dropout=dropout,
            max_len=max_len,
        )

        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        """
        src: [B, src_len]
        tgt: [B, tgt_len]
        return logits: [B, tgt_len, tgt_vocab_size]
        """
        enc_out, src_mask, enc_attns = self.encoder(src)
        dec_out, dec_self_attns, dec_cross_attns = self.decoder(tgt, enc_out, src, src_mask)
        logits = self.proj(dec_out)

        return logits, {
            "encoder_attns": enc_attns,
            "decoder_self_attns": dec_self_attns,
            "decoder_cross_attns": dec_cross_attns,
        }