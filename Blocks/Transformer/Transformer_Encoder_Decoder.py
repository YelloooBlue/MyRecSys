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

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)


# =========================
# 2. Mask Functions
# =========================
def make_padding_mask(q, k, pad_idx=0):
    """
    q: [B, Q_len]
    k: [B, K_len]
    return: [B, 1, Q_len, K_len]
    True 表示该位置要被 mask 掉
    """
    B, Q_len = q.shape
    B, K_len = k.shape

    k_pad = (k == pad_idx).unsqueeze(1).unsqueeze(2)   # [B,1,1,K_len]
    k_pad = k_pad.expand(-1, 1, Q_len, -1)             # [B,1,Q_len,K_len]
    return k_pad


def make_causal_mask(seq_len, device):
    """
    return: [1, 1, seq_len, seq_len]
    上三角为 True，表示未来位置不可见
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
        Q: [B, H, Q_len, D]
        K: [B, H, K_len, D]
        V: [B, H, K_len, D]
        mask: [B, 1, Q_len, K_len] or [1, 1, Q_len, K_len]
        """
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # [B, H, Q_len, K_len]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, Q_len, D]
        return out, attn


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

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        x: [B, T, C]
        -> [B, H, T, D]
        """
        B, T, C = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        x: [B, H, T, D]
        -> [B, T, C]
        """
        B, H, T, D = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, H * D)
        return x

    def forward(self, query, key, value, mask=None):
        """
        query: [B, Q_len, C]
        key:   [B, K_len, C]
        value: [B, K_len, C]
        mask:  [B,1,Q_len,K_len] or [1,1,Q_len,K_len]
        """
        Q = self.split_heads(self.q_proj(query))
        K = self.split_heads(self.k_proj(key))
        V = self.split_heads(self.v_proj(value))

        out, attn = self.attn(Q, K, V, mask=mask)
        out = self.combine_heads(out)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, attn


# =========================
# 5. Feed Forward Network
# =========================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =========================
# 6. Encoder Layer
# =========================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        """
        x: [B, src_len, C]
        src_mask: [B,1,src_len,src_len]
        """
        attn_out, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x, attn_weights


# =========================
# 7. Decoder Layer
# =========================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, cross_mask):
        """
        x:        [B, tgt_len, C]
        enc_out:  [B, src_len, C]
        tgt_mask: [B,1,tgt_len,tgt_len] 或 [1,1,tgt_len,tgt_len]
        cross_mask: [B,1,tgt_len,src_len]
        """
        # 1) masked self-attention
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        # 2) cross-attention
        cross_attn_out, cross_attn_weights = self.cross_attn(x, enc_out, enc_out, mask=cross_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        # 3) FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x, self_attn_weights, cross_attn_weights


# =========================
# 8. Encoder
# =========================
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        pad_idx=0,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        """
        src: [B, src_len]
        """
        src_mask = make_padding_mask(src, src, pad_idx=self.pad_idx)  # [B,1,S,S]

        x = self.token_emb(src) * math.sqrt(self.d_model)   # embedding 初始方差较小，放大后更稳定，避免和位置编码尺度不匹配
        x = self.pos_enc(x)
        x = self.dropout(x)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            attn_weights_all.append(attn_weights)

        return x, src_mask, attn_weights_all


# =========================
# 9. Decoder
# =========================
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        pad_idx=0,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, enc_out, src):
        """
        tgt: [B, tgt_len]
        enc_out: [B, src_len, C]
        src: [B, src_len]
        """
        B, tgt_len = tgt.shape
        device = tgt.device

        # decoder self-attention mask = padding mask | causal mask
        tgt_pad_mask = make_padding_mask(tgt, tgt, pad_idx=self.pad_idx)   # [B,1,T,T]
        tgt_causal_mask = make_causal_mask(tgt_len, device)                # [1,1,T,T]
        tgt_mask = tgt_pad_mask | tgt_causal_mask

        # cross-attention mask: query来自 tgt，key来自 src
        cross_mask = make_padding_mask(tgt, src, pad_idx=self.pad_idx)     # [B,1,T,S]

        x = self.token_emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        self_attn_all = []
        cross_attn_all = []

        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_out, tgt_mask, cross_mask)
            self_attn_all.append(self_attn)
            cross_attn_all.append(cross_attn)

        return x, self_attn_all, cross_attn_all


# =========================
# 10. Full Transformer
# =========================
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=256,
        num_layers=2,
        num_heads=4,
        d_ff=512,
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

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        """
        src: [B, src_len]
        tgt: [B, tgt_len]   # 这里传入 decoder 输入，而不是标签本身
        """
        enc_out, src_mask, enc_attns = self.encoder(src)
        dec_out, dec_self_attns, dec_cross_attns = self.decoder(tgt, enc_out, src)
        logits = self.output_proj(dec_out)  # [B, tgt_len, tgt_vocab_size]

        return logits, {
            "encoder_attns": enc_attns,
            "decoder_self_attns": dec_self_attns,
            "decoder_cross_attns": dec_cross_attns,
        }


# =========================
# 11. Helper: shift_right
# =========================
def shift_right(tgt, bos_idx):
    """
    把目标序列右移，作为 decoder 输入
    例如:
    tgt         = [y1, y2, y3, y4]
    decoder_in  = [BOS, y1, y2, y3]
    """
    bos = torch.full((tgt.size(0), 1), bos_idx, dtype=tgt.dtype, device=tgt.device)
    return torch.cat([bos, tgt[:, :-1]], dim=1)


# =========================
# 12. Minimal Demo
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设:
    # 0 = PAD
    # 1 = BOS
    # 2 = EOS
    SRC_VOCAB_SIZE = 1000
    TGT_VOCAB_SIZE = 1200
    PAD_IDX = 0
    BOS_IDX = 1

    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        src_pad_idx=PAD_IDX,
        tgt_pad_idx=PAD_IDX,
        dropout=0.1,
        max_len=128,
    ).to(device)

    # 假数据
    B, SRC_LEN, TGT_LEN = 2, 7, 6
    src = torch.randint(3, SRC_VOCAB_SIZE, (B, SRC_LEN)).to(device)
    tgt = torch.randint(3, TGT_VOCAB_SIZE, (B, TGT_LEN)).to(device)

    # 制造一些 PAD
    src[0, -2:] = PAD_IDX
    tgt[1, -1:] = PAD_IDX

    # decoder 输入要右移
    decoder_input = shift_right(tgt, BOS_IDX)

    logits, attn_dict = model(src, decoder_input)

    print("src shape:", src.shape)                 # [B, src_len]
    print("tgt shape:", tgt.shape)                 # [B, tgt_len]
    print("decoder_input shape:", decoder_input.shape)
    print("logits shape:", logits.shape)           # [B, tgt_len, tgt_vocab]

    # 训练时常见写法
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    loss = criterion(
        logits.reshape(-1, TGT_VOCAB_SIZE),
        tgt.reshape(-1)
    )
    print("loss:", loss.item())