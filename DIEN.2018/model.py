"""
    DIEN(Deep Interest Evolution Network)模型实现（PyTorch版）
    参考：
        1) 官方TF实现（mouna99/dien）
        2) DeepCTR-Torch 的 DIEN 结构（你给的 dien.py）
    目标：
        - 逻辑尽量对齐 mouna99/dien
        - 代码风格/注释尽量对齐 DIN.2017/model.py，便于渐进式学习
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================ 工具函数 ================================

def make_seq_mask(seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    参数:
        seq_len: 每个样本的有效历史长度 [B]
        max_len: 最大序列长度 T
    返回:
        mask: [B, T]，有效位置为 True，无效 padding 为 False
    """
    return torch.arange(max_len, device=seq_len.device)[None, :] < seq_len[:, None]


# ================================ 注意力模块（对齐 mouna99/utils.py: din_fcn_attention） ================================

class DINAttention(nn.Module):
    """
    DIN-style attention（用于 DIEN 的 interest evolving 阶段）
    对齐 mouna99/dien 的 din_fcn_attention()：
        din_all = [q, k, q-k, q*k] -> 80(sigmoid) -> 40(sigmoid) -> 1 -> softmax -> weighted sum / LIST

    这里我们支持两种输出模式：
        - return_score=True : 返回注意力权重 scores [B, T]（给 AIGRU/AGRU/AUGRU 用）
        - return_score=False: 返回加权后的序列表示（LIST模式：facts * score）或SUM模式（matmul）
    """
    def __init__(self, hidden_units: int):
        super().__init__()
        self.hidden_units = hidden_units

        # 注意：mouna99 中这里用的是 sigmoid 激活（不是 ReLU）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_units * 4, 80),
            nn.Sigmoid(),
            nn.Linear(80, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1),
        )

    def forward(
        self,
        query: torch.Tensor,
        facts: torch.Tensor,
        seq_len: torch.Tensor,
        return_score: bool = True,
    ):
        """
        参数:
            query: 目标物品 embedding [B, H]
            facts: 序列（例如 GRU 输出 states）[B, T, H]
            seq_len: 有效长度 [B]
            return_score: 是否返回注意力权重
        返回:
            scores: [B, T]（若 return_score=True）
            或者
            weighted_facts: [B, T, H]（LIST模式：逐位置加权）
        """
        B, T, H = facts.shape
        q = query.unsqueeze(1).repeat(1, T, 1)  # [B, T, H]

        din_all = torch.cat([q, facts, q - facts, q * facts], dim=-1)  # [B, T, 4H]
        scores = self.mlp(din_all).squeeze(-1)  # [B, T]

        # Mask 无效历史（mouna99 用极小值 padding）
        mask = make_seq_mask(seq_len, T)  # [B, T]
        scores = scores.masked_fill(~mask, -2 ** 32 + 1)

        # Softmax 归一化（mouna99: softmax_stag=1）
        scores = F.softmax(scores, dim=-1)  # [B, T]

        if return_score:
            return scores
        else:
            # LIST 模式：facts * score（注意不是 SUM）
            return facts * scores.unsqueeze(-1)  # [B, T, H]


# ================================ Auxiliary Loss（对齐 mouna99/model.py: auxiliary_loss + auxiliary_net） ================================

class AuxiliaryNet(nn.Module):
    """
    对齐 mouna99/dien 的 auxiliary_net:
        BN -> 100(sigmoid) -> 50(sigmoid) -> 2(softmax)
    PyTorch 里 softmax 输出 2 类概率，我们取 click 类概率做 loss。
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [B, T, D] 或 [N, D]
        返回:
            prob: softmax 概率 [..., 2]
        """
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])  # [N, D]

        x = self.bn(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1) + 1e-8

        return x.reshape(*orig_shape[:-1], 2)


def auxiliary_loss(
    aux_net: AuxiliaryNet,
    h_states: torch.Tensor,
    click_seq: torch.Tensor,
    noclick_seq: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    对齐 mouna99/model.py 的 auxiliary_loss：
        click_input = [h, click]
        noclick_input = [h, noclick]
        click_prop = aux_net(click_input)[:,:,0]
        noclick_prop = aux_net(noclick_input)[:,:,0]
        loss = mean( -log(click_prop)*mask  -log(1-noclick_prop)*mask )

    参数:
        h_states: GRU 隐状态序列（去掉最后一步）[B, T-1, H]
        click_seq: 正样本序列（右移1位）[B, T-1, H]
        noclick_seq: 负样本序列（右移1位）[B, T-1, H]
        mask: 有效位置 mask（右移1位）[B, T-1]，float(0/1) 或 bool
    返回:
        aux_loss: 标量
    """
    if mask.dtype != torch.float32:
        mask = mask.float()

    click_input = torch.cat([h_states, click_seq], dim=-1)      # [B, T-1, 2H]
    noclick_input = torch.cat([h_states, noclick_seq], dim=-1)  # [B, T-1, 2H]

    click_prob = aux_net(click_input)[..., 0]    # [B, T-1]
    noclick_prob = aux_net(noclick_input)[..., 0]

    click_loss = -torch.log(click_prob) * mask
    noclick_loss = -torch.log(1.0 - noclick_prob) * mask

    return (click_loss + noclick_loss).mean()


# ================================ 动态 GRU（对齐 VecAttGRUCell / QAAttGRUCell） ================================

class DynamicGRU(nn.Module):
    """
    用 PyTorch 手写一个“带 attention score 的 GRU 单元”循环。
    对齐 mouna99/utils.py:
        - QAAttGRUCell: new_h = (1-att)*state + att*c
        - VecAttGRUCell(AUGRU): u = (1-att)*u ; new_h = u*state + (1-u)*c
    """
    def __init__(self, hidden_size: int, gru_type: Literal["AGRU", "AUGRU"]):
        super().__init__()
        assert gru_type in ("AGRU", "AUGRU")
        self.hidden_size = hidden_size
        self.gru_type = gru_type

        # GRU 参数（对齐标准 GRU 公式）
        self.W_ir = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_iz = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_in = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, att: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入序列 [B, T, H]
            att: 注意力分数 [B, T]（0~1，softmax 输出）
            seq_len: 有效长度 [B]
        返回:
            outputs: [B, T, H]
        """
        B, T, H = x.shape
        h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        outputs = []

        # 用 mask 保证 padding 位置不更新
        mask = make_seq_mask(seq_len, T).float()  # [B, T]

        for t in range(T):
            xt = x[:, t, :]  # [B, H]
            at = att[:, t].unsqueeze(-1)  # [B, 1]
            mt = mask[:, t].unsqueeze(-1)  # [B, 1]

            r = torch.sigmoid(self.W_ir(xt) + self.W_hr(h))
            z = torch.sigmoid(self.W_iz(xt) + self.W_hz(h))
            n = torch.tanh(self.W_in(xt) + self.W_hn(r * h))

            if self.gru_type == "AGRU":
                # QAAttGRU: 用 att 替换 update gate
                h_new = (1.0 - at) * h + at * n
            else:
                # AUGRU(VecAttGRU): u = (1-att)*u
                z_mod = (1.0 - at) * z
                h_new = z_mod * h + (1.0 - z_mod) * n

            # padding 位置：保持原状态（不更新）
            h = mt * h_new + (1.0 - mt) * h
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, H]


# ================================ DIEN 主模型 ================================

@dataclass
class DIENConfig:
    embedding_dim: int = 18           # 对齐 mouna99: mid/cat embedding 都是 18，然后 concat 成 36
    hidden_size: int = 36             # 一般等于 item_eb 维度（mid+cat）
    attention_size: int = 64          # mouna99 中 ATTENTION_SIZE
    neg_mode: Literal["first", "mean"] = "first"  # 负采样处理策略
    gru_type: Literal["GRU", "AIGRU", "AGRU", "AUGRU"] = "AUGRU"


class DIEN(nn.Module):
    """
    DIEN 模型（对齐 mouna99: Model_DIN_V2_Gru_Vec_attGru_Neg 的核心结构）

    输入（尽量复用 DIN 参数名）：
        user_ids: [B]
        item_ids: [B]                 # 对齐 mid_batch_ph
        history_items: [B, T]         # 对齐 mid_his_batch_ph
        seq_len: [B]

    新增（DIEN 必需）：
        cate_ids: [B]                 # 对齐 cat_batch_ph（如果你想保持 DIN 的 cate_list 逻辑，也可以外部传入）
        history_cates: [B, T]         # 对齐 cat_his_batch_ph
        neg_history_items: [B, T, NEG]
        neg_history_cates: [B, T, NEG]
        labels: [B] 或 [B, 1]         # CTR label（这里用 BCE，二分类）

    输出：
        y_pred: [B, 1]
        aux_loss: 标量（训练时加到主 loss 里）
    """
    def __init__(
        self,
        num_user: int,
        num_item: int,
        num_cate: int,
        config: DIENConfig,
    ):
        super().__init__()
        self.config = config

        # Embedding（对齐 mouna99: uid/mid/cat embedding）
        self.user_embedding = nn.Embedding(num_user, config.embedding_dim)
        self.item_embedding = nn.Embedding(num_item, config.embedding_dim)
        self.cate_embedding = nn.Embedding(num_cate, config.embedding_dim)

        # 1) Interest Extractor: 第一层 GRU（输入 item_his_eb）
        self.gru1 = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            batch_first=True,
        )

        # auxiliary net（输入 [h, click] => 2H）
        self.aux_net = AuxiliaryNet(input_dim=config.hidden_size * 2)

        # 2) Interest Evolving：attention + 第二层（依 gru_type 不同）
        self.attention = DINAttention(hidden_units=config.hidden_size)

        if config.gru_type in ("GRU", "AIGRU"):
            self.gru2 = nn.GRU(
                input_size=config.hidden_size,
                hidden_size=config.hidden_size,
                batch_first=True,
            )
            self.dynamic_gru = None
        else:
            self.gru2 = None
            self.dynamic_gru = DynamicGRU(hidden_size=config.hidden_size, gru_type=config.gru_type)

        # 3) FCN（对齐 mouna99 build_fcn_net：BN -> 200 -> 80 -> 2 -> softmax）
        # PyTorch 里我们用二分类 BCE，所以最后输出 logit [B,1]
        fcn_in_dim = config.embedding_dim + config.hidden_size + config.hidden_size + config.hidden_size
        # 对齐 mouna99: concat [uid_emb, item_eb, item_his_sum, item_eb*item_his_sum, final_state]
        # 其中 uid_emb:18, item_eb:36, item_his_sum:36, item_eb*sum:36, final_state:36 => 总 162
        # 我们这里把 item_eb 单独算进去，所以 fcn_in_dim = uid(18)+item(36)+sum(36)+prod(36)+final(36)=162
        fcn_in_dim = config.embedding_dim + config.hidden_size * 4

        self.bn = nn.BatchNorm1d(fcn_in_dim)
        self.fc1 = nn.Linear(fcn_in_dim, 200)
        self.fc2 = nn.Linear(200, 80)
        self.fc3 = nn.Linear(80, 1)

    def _build_item_eb(self, item_ids: torch.Tensor, cate_ids: torch.Tensor) -> torch.Tensor:
        """
        返回:
            item_eb: [B, 2*emb] == [B, hidden_size]
        """
        item_emb = self.item_embedding(item_ids)  # [B, emb]
        cate_emb = self.cate_embedding(cate_ids)  # [B, emb]
        return torch.cat([item_emb, cate_emb], dim=-1)

    def _build_his_eb(self, history_items: torch.Tensor, history_cates: torch.Tensor) -> torch.Tensor:
        """
        返回:
            item_his_eb: [B, T, hidden_size]
        """
        mid_his = self.item_embedding(history_items)  # [B, T, emb]
        cat_his = self.cate_embedding(history_cates)  # [B, T, emb]
        return torch.cat([mid_his, cat_his], dim=-1)

    def _build_neg_his_eb(
        self,
        neg_history_items: torch.Tensor,
        neg_history_cates: torch.Tensor,
    ) -> torch.Tensor:
        """
        neg input shape:
            neg_history_items: [B, T, NEG]
            neg_history_cates: [B, T, NEG]
        返回:
            noclk_item_his_eb: [B, T, hidden_size]
        """
        B, T, NEG = neg_history_items.shape

        neg_mid = self.item_embedding(neg_history_items)  # [B, T, NEG, emb]
        neg_cat = self.cate_embedding(neg_history_cates)  # [B, T, NEG, emb]
        neg_item = torch.cat([neg_mid, neg_cat], dim=-1)  # [B, T, NEG, hidden]

        if self.config.neg_mode == "first":
            return neg_item[:, :, 0, :]  # [B, T, hidden]（对齐 mouna99: 只取第 0 个负样本）
        else:
            return neg_item.mean(dim=2)  # [B, T, hidden]

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_items: torch.Tensor,
        seq_len: torch.Tensor,
        cate_ids: torch.Tensor,
        history_cates: torch.Tensor,
        neg_history_items: torch.Tensor,
        neg_history_cates: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        返回:
            y_pred: [B, 1]
            aux_loss: 标量（若 labels=None 也会算 aux_loss，便于你自己组合）
        """
        B, T = history_items.shape

        # =============== Embedding ===============
        uid_emb = self.user_embedding(user_ids)  # [B, emb]
        item_eb = self._build_item_eb(item_ids, cate_ids)  # [B, hidden]
        item_his_eb = self._build_his_eb(history_items, history_cates)  # [B, T, hidden]
        item_his_eb_sum = item_his_eb.sum(dim=1)  # [B, hidden]

        noclk_item_his_eb = self._build_neg_his_eb(neg_history_items, neg_history_cates)  # [B, T, hidden]

        # =============== mask（对齐 TF 里的 self.mask 占位符） ===============
        mask = make_seq_mask(seq_len, T)  # [B, T] bool

        # =============== Interest Extractor（GRU1） ===============
        # PyTorch pack 需要 length>0，这里简单假设 seq_len 都 >0（你的数据一般如此）
        packed = nn.utils.rnn.pack_padded_sequence(
            item_his_eb, lengths=seq_len.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_outputs, _ = self.gru1(packed)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True, total_length=T)  # [B, T, H]

        # =============== Auxiliary Loss（对齐 mouna99: rnn_outputs[:-1] 与 seq[1:]） ===============
        # mask[:,1:] 对齐 TF：self.mask[:,1:]
        aux = auxiliary_loss(
            self.aux_net,
            h_states=rnn_outputs[:, :-1, :],
            click_seq=item_his_eb[:, 1:, :],
            noclick_seq=noclk_item_his_eb[:, 1:, :],
            mask=mask[:, 1:],
        )

        # =============== Interest Evolving（Attention + GRU2 / DynamicGRU） ===============
        if self.config.gru_type == "GRU":
            # 先 GRU2，再 attention 做 pooling（DeepCTR-Torch 的一种写法；mouna99 的主 DIEN 用的是 VecAttGRU）
            packed2 = nn.utils.rnn.pack_padded_sequence(
                rnn_outputs, lengths=seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            out2, _ = self.gru2(packed2)
            out2, _ = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True, total_length=T)
            scores = self.attention(item_eb, out2, seq_len, return_score=True)  # [B, T]
            final_state = torch.sum(out2 * scores.unsqueeze(-1), dim=1)  # [B, H]

        elif self.config.gru_type == "AIGRU":
            scores = self.attention(item_eb, rnn_outputs, seq_len, return_score=True)  # [B, T]
            inputs2 = rnn_outputs * scores.unsqueeze(-1)  # [B, T, H]
            packed2 = nn.utils.rnn.pack_padded_sequence(
                inputs2, lengths=seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_last = self.gru2(packed2)
            final_state = h_last.squeeze(0)  # [B, H]

        elif self.config.gru_type in ("AGRU", "AUGRU"):
            scores = self.attention(item_eb, rnn_outputs, seq_len, return_score=True)  # [B, T]
            out2 = self.dynamic_gru(rnn_outputs, scores, seq_len)  # [B, T, H]
            # 取最后一个有效位置的 state（对齐“final_state2”概念）
            idx = (seq_len - 1).clamp(min=0)  # [B]
            final_state = out2[torch.arange(B, device=idx.device), idx]  # [B, H]

        else:
            raise ValueError(f"Unsupported gru_type={self.config.gru_type}")

        # =============== FCN（对齐 mouna99: concat + fcn） ===============
        inp = torch.cat(
            [
                uid_emb,
                item_eb,
                item_his_eb_sum,
                item_eb * item_his_eb_sum,
                final_state,
            ],
            dim=-1,
        )  # [B, 162]

        x = self.bn(inp)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logit = self.fc3(x)  # [B, 1]
        y_pred = torch.sigmoid(logit)  # [B, 1]

        # 如果给了 labels，顺便返回总 loss（方便你 train.py 写得像 DIN）
        if labels is not None:
            labels = labels.float().view(-1, 1)
            ctr_loss = F.binary_cross_entropy(y_pred, labels)
            loss = ctr_loss + aux
            return y_pred, loss, ctr_loss, aux

        return y_pred, aux


if __name__ == "__main__":
    # ================================ 模块自测（对齐 DIN.2017 的写法） ================================
    torch.manual_seed(42)

    B, T, NEG = 2, 5, 3
    num_user, num_item, num_cate = 100, 1000, 50

    cfg = DIENConfig(gru_type="AUGRU", neg_mode="first")
    model = DIEN(num_user=num_user, num_item=num_item, num_cate=num_cate, config=cfg)

    user_ids = torch.randint(0, num_user, (B,))
    item_ids = torch.randint(0, num_item, (B,))
    history_items = torch.randint(0, num_item, (B, T))
    seq_len = torch.tensor([5, 3])

    cate_ids = torch.randint(0, num_cate, (B,))
    history_cates = torch.randint(0, num_cate, (B, T))

    neg_history_items = torch.randint(0, num_item, (B, T, NEG))
    neg_history_cates = torch.randint(0, num_cate, (B, T, NEG))

    labels = torch.randint(0, 2, (B,))

    y_pred, loss, ctr_loss, aux = model(
        user_ids=user_ids,
        item_ids=item_ids,
        history_items=history_items,
        seq_len=seq_len,
        cate_ids=cate_ids,
        history_cates=history_cates,
        neg_history_items=neg_history_items,
        neg_history_cates=neg_history_cates,
        labels=labels,
    )

    print("y_pred:", y_pred.shape)      # [B, 1]
    print("loss:", loss.item(), "ctr_loss:", ctr_loss.item(), "aux:", aux.item())