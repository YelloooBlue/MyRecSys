"""DIEN (Deep Interest Evolution Network) in PyTorch."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """显式构建序列掩码，适用于变长序列的注意力计算。返回一个布尔张量 [B, max_len]，其中 True 表示有效位置，False 表示填充位置。"""
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def gather_last_valid(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Pick the last valid timestep from [B, T, H] by sequence lengths [B]."""
    idx = (lengths.clamp_min(1) - 1).view(-1, 1, 1)
    idx = idx.expand(-1, 1, sequence.size(-1))
    return sequence.gather(dim=1, index=idx).squeeze(1)


class Attention(nn.Module):
    """
        Interest Evolving Layer 的 Attention 实现，参考 DIN 中的实现。
    """

    def __init__(self, hidden_units: int):
        super().__init__()
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
        keys: torch.Tensor,
        seq_len: torch.Tensor,
        return_scores: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            query: 目标物品嵌入 [B, H]
            keys: 序列嵌入 [B, T, H] （这里是经过第一层GRU演化后的输出）
            seq_len: 每个样本的有效历史长度 [B]
        Returns:
            if return_scores: 归一化后的注意力得分 [B, T]
            else: 注意力加权后的序列向量 [B, T, H]
        """
        B, T, _ = keys.shape
        query = query.unsqueeze(1).expand(B, T, -1) # [B, T, H]

        # DIN 论文中的特征交互方式：query、keys、query-keys、query*keys
        feats = torch.cat([query, keys, query - keys, query * keys], dim=-1)

        # MLP 型注意力打分器
        scores = self.mlp(feats).squeeze(-1)    # [B, T]

        # Mask 无效历史
        mask = sequence_mask(seq_len, T)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        
        # Softmax 归一化注意力权重
        scores = F.softmax(scores, dim=-1)

        # 可选输出得分用于后续给GRU加权
        if return_scores:
            return scores
        return keys * scores.unsqueeze(-1)


class AuxiliaryNet(nn.Module):
    """
        Interest Extractor Layer 的 Auxiliary Net 实现，用于计算辅助损失。
        目的是让第一层GRU更好地捕捉兴趣演变的动态特征，通过预测下一步点击和未点击的行为来引导GRU学习更有区分性的兴趣表示。
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, T, 2*H]，由h_states和click/noclick序列拼接而成
        Returns:
            预测点击和未点击的概率 [B, T, 2]
        """
        origin_shape = x.shape
        x = x.reshape(-1, origin_shape[-1])
        x = self.norm(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x.reshape(*origin_shape[:-1], 2)


def compute_auxiliary_loss(
    aux_net: AuxiliaryNet,
    h_states: torch.Tensor,
    click_seq: torch.Tensor,
    noclick_seq: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算辅助损失(论文架构图最左侧的Auxiliary Loss)，用于指导第一层GRU学习更有区分性的兴趣表示。
    """
    eps = 1e-7
    mask = mask.float()

    click_input = torch.cat([h_states, click_seq], dim=-1)
    noclick_input = torch.cat([h_states, noclick_seq], dim=-1)

    # 在AMP下这里最容易发生数值不稳定，统一转回fp32并避免取到0/1边界。
    click_prob = aux_net(click_input)[..., 0].float().clamp(eps, 1.0 - eps)
    noclick_prob = aux_net(noclick_input)[..., 0].float().clamp(eps, 1.0 - eps)

    # 拉近点击行为的兴趣表示，远离未点击行为的兴趣表示
    click_loss = -torch.log(click_prob) * mask
    noclick_loss = -torch.log1p(-noclick_prob) * mask

    # 平均每个样本的有效历史步数，避免不同长度序列之间的损失不平衡问题
    valid_steps = mask.sum().clamp_min(1.0)
    return (click_loss.sum() + noclick_loss.sum()) / valid_steps


class DynamicGRU(nn.Module):
    """AGRU / AUGRU 实现，支持输入 Attention scores 来动态调整 GRU 的更新机制。"""

    def __init__(self, hidden_units: int, gru_type: Literal["AGRU", "AUGRU"]):
        super().__init__()
        if gru_type not in {"AGRU", "AUGRU"}:
            raise ValueError(f"Unsupported gru_type: {gru_type}")

        self.gru_type = gru_type

        # 输入门权重矩阵
        self.W_ir = nn.Linear(hidden_units, hidden_units, bias=True)
        self.W_iz = nn.Linear(hidden_units, hidden_units, bias=True)
        self.W_in = nn.Linear(hidden_units, hidden_units, bias=True)

        # 隐藏状态权重矩阵
        self.W_hr = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_hz = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_hn = nn.Linear(hidden_units, hidden_units, bias=False)

    def forward(self, x: torch.Tensor, att_scores: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        参考模型结构图左上角的实现
        Args:
            x: 输入序列 [B, T, H]
            att_scores: 注意力得分 [B, T]，范围在0-1之间
            seq_len: 每个样本的有效历史长度 [B]
        Returns:
            演化后的序列 [B, T, H]
        """

        B, T, H = x.shape
        state = x.new_zeros(B, H)
        mask = sequence_mask(seq_len, T).unsqueeze(-1)
        outputs = []

        for t in range(T):
            xt = x[:, t, :]
            at = att_scores[:, t].unsqueeze(-1)

            r = torch.sigmoid(self.W_ir(xt) + self.W_hr(state))     # 重置门
            z = torch.sigmoid(self.W_iz(xt) + self.W_hz(state))     # 更新门
            n = torch.tanh(self.W_in(xt) + self.W_hn(r * state))    # 新候选状态

            # AGRU: 直接用注意力得分替换更新门，AUGRU: 用注意力得分调整更新门的输出
            if self.gru_type == "AGRU":
                new_state = (1.0 - at) * state + at * n
            else:
                z = (1.0 - at) * z
                new_state = z * state + (1.0 - z) * n

            step_mask = mask[:, t, :]
            state = torch.where(step_mask, new_state, state)
            outputs.append(state.unsqueeze(1))

        return torch.cat(outputs, dim=1)


class DIEN(nn.Module):
    """
        DIEN 模型实现

    Interface is aligned with DIN.2017 style:
    - `cate_list` is optional but recommended. If provided, cate ids are looked up automatically.
    - If `cate_list` is not provided, caller must pass cate tensors in forward.
    """

    def __init__(
        self,
        num_user: int,
        num_item: int,
        num_cate: int,
        cate_list: Optional[list[int]] = None,
        embedding_dim: int = 18,
        hidden_units: Optional[int] = None,
        gru_type: Literal["GRU", "AIGRU", "AGRU", "AUGRU"] = "AUGRU",
        neg_mode: Literal["first", "mean"] = "first",
    ):
        super().__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_cate = num_cate
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units or embedding_dim * 2
        self.gru_type = gru_type
        self.neg_mode = neg_mode

        if self.hidden_units != embedding_dim * 2:
            raise ValueError(
                f"hidden_units must be embedding_dim * 2 for item+cate concat, got {self.hidden_units}"
            )

        if cate_list is not None:
            self.register_buffer("cate_list", torch.tensor(cate_list, dtype=torch.int64))
        else:
            self.cate_list = None

        self.user_embedding = nn.Embedding(num_user, embedding_dim)
        self.item_embedding = nn.Embedding(num_item, embedding_dim)
        self.cate_embedding = nn.Embedding(num_cate, embedding_dim)

        self.gru1 = nn.GRU(input_size=self.hidden_units, hidden_size=self.hidden_units, batch_first=True)

        self.attention = Attention(self.hidden_units)
        if gru_type in {"GRU", "AIGRU"}:
            self.gru2 = nn.GRU(input_size=self.hidden_units, hidden_size=self.hidden_units, batch_first=True)
            self.dynamic_gru = None
        else:
            self.gru2 = None
            self.dynamic_gru = DynamicGRU(self.hidden_units, gru_type=gru_type)

        self.aux_net = AuxiliaryNet(input_dim=self.hidden_units * 2)

        fcn_in_dim = embedding_dim + self.hidden_units * 4
        self.fcn_bn = nn.BatchNorm1d(fcn_in_dim)
        self.fcn = nn.Sequential(
            nn.Linear(fcn_in_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Linear(80, 1),
        )

        self.ctr_loss_fn = nn.BCEWithLogitsLoss()

    def _lookup_cates(
        self,
        item_ids: torch.Tensor,
        history_items: torch.Tensor,
        neg_history_items: torch.Tensor,
        cate_ids: Optional[torch.Tensor],
        history_cates: Optional[torch.Tensor],
        neg_history_cates: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据输入的物品ID查找对应的类别ID，支持直接传入类别ID或者使用模型内置的cate_list进行查找。"""
        if cate_ids is not None and history_cates is not None and neg_history_cates is not None:
            return cate_ids, history_cates, neg_history_cates

        if self.cate_list is None:
            raise ValueError(
                "cate_list is required when cate_ids/history_cates/neg_history_cates are not provided"
            )

        cate_ids = self.cate_list[item_ids]
        history_cates = self.cate_list[history_items]
        neg_history_cates = self.cate_list[neg_history_items]
        return cate_ids, history_cates, neg_history_cates

    def _build_item_embedding(self, item_ids: torch.Tensor, cate_ids: torch.Tensor) -> torch.Tensor:
        """构建候选物品的嵌入，这里Side Info跟DIN一样是物品类别的嵌入，后续可以扩展"""
        item_emb = self.item_embedding(item_ids)
        cate_emb = self.cate_embedding(cate_ids)
        return torch.cat([item_emb, cate_emb], dim=-1)

    def _build_history_embedding(self, history_items: torch.Tensor, history_cates: torch.Tensor) -> torch.Tensor:
        """构建历史物品序列的嵌入，支持物品和类别的拼接作为Side Info"""
        item_emb = self.item_embedding(history_items)
        cate_emb = self.cate_embedding(history_cates)
        return torch.cat([item_emb, cate_emb], dim=-1)

    def _build_negative_embedding(
        self,
        neg_history_items: torch.Tensor,
        neg_history_cates: torch.Tensor,
    ) -> torch.Tensor:
        """
        构建多个负采样物品的嵌入
        输入：
            - neg_history_items: [B, T, NEG]
            - neg_history_cates: [B, T, NEG]
        输出：
            - neg_emb: [B, T, NEG, H] 或 [B, T, 1, H]
        """
        item_emb = self.item_embedding(neg_history_items)
        cate_emb = self.cate_embedding(neg_history_cates)
        neg_emb = torch.cat([item_emb, cate_emb], dim=-1)

        # 根据 neg_mode 选择使用第一个负样本还是对所有负样本取平均，作为未点击行为的代表嵌入
        if self.neg_mode == "first":
            return neg_emb[:, :, 0, :]
        if self.neg_mode == "mean":
            return neg_emb.mean(dim=2)
        raise ValueError(f"Unsupported neg_mode: {self.neg_mode}")

    @staticmethod
    def _run_packed_gru(gru: nn.GRU, x: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """
        对变长序列使用 PackedSequence 来运行 GRU，避免填充位置的计算干扰。
        具体原理为：先使用 `pack_padded_sequence` 将输入序列压缩成 PackedSequence 格式，
        传入 GRU 进行计算，然后再使用 `pad_packed_sequence` 将输出还原成原始的张量格式。
        """
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=seq_len.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = gru(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1),
        )
        return out

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        history_items: torch.Tensor,
        seq_len: torch.Tensor,
        neg_history_items: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cate_ids: Optional[torch.Tensor] = None,
        history_cates: Optional[torch.Tensor] = None,
        neg_history_cates: Optional[torch.Tensor] = None,
    ):
        """Forward.

        Args:
            user_ids: 用户ID [B]
            item_ids: 候选物品ID [B]
            history_items: 用户历史交互物品ID [B, T]
            seq_len: 用户的有效历史交互长度 [B]
            neg_history_items: 历史交互序列每个位置的负采样 [B, T, NEG]
            
            labels: 点击标签 [B] （可选，如果提供则计算并返回损失，否则只返回预测结果和辅助损失）
            
            cate_ids: 候选物品的类别ID [B] （可选，如果模型初始化时提供了cate_list则不需要传入）
            history_cates: 历史交互物品的类别ID [B, T] （可选，如果模型初始化时提供了cate_list则不需要传入）
            neg_history_cates: 负采样物品的类别ID [B, T, NEG] （可选，如果模型初始化时提供了cate_list则不需要传入）
        Returns:
            if labels is None: (y_pred, aux_loss)
            else: (y_pred, total_loss, ctr_loss, aux_loss)
        """
        max_len = history_items.size(1)

        # 根据输入的物品ID查找对应的类别ID，支持直接传入类别ID或者使用模型内置的cate_list进行查找
        cate_ids, history_cates, neg_history_cates = self._lookup_cates(
            item_ids=item_ids,
            history_items=history_items,
            neg_history_items=neg_history_items,
            cate_ids=cate_ids,
            history_cates=history_cates,
            neg_history_cates=neg_history_cates,
        )

        # 嵌入查找
        user_emb = self.user_embedding(user_ids)        # 用户嵌入 [B, embedding_dim]
        item_emb = self._build_item_embedding(item_ids, cate_ids)   # 候选物品+类别嵌入 [B, hidden_units]
        history_emb = self._build_history_embedding(history_items, history_cates)   # 历史物品+类别嵌入 [B, T, hidden_units]
        neg_history_emb = self._build_negative_embedding(neg_history_items, neg_history_cates)  # 负采样物品+类别嵌入 [B, T, hidden_units]

        history_sum = history_emb.sum(dim=1)    # 手工构建最终MLP交互方式之一
        mask = sequence_mask(seq_len, max_len)  # 生成历史序列的掩码，用于后续的注意力计算和辅助损失计算

        # Interest Extractor Layer 的第一层GRU演化
        rnn_out = self._run_packed_gru(self.gru1, history_emb, seq_len)

        # 计算Auxiliary Loss，指导第一层GRU学习更有区分性的兴趣表示
        aux_loss = compute_auxiliary_loss(
            aux_net=self.aux_net,
            h_states=rnn_out[:, :-1, :],
            click_seq=history_emb[:, 1:, :],
            noclick_seq=neg_history_emb[:, 1:, :],
            mask=mask[:, 1:],
        )

        # Interest Evolving Layer 的第二层GRU演化，支持不同的GRU变体（GRU、AIGRU、AGRU、AUGRU）
        if self.gru_type == "GRU":
            evolved = self._run_packed_gru(self.gru2, rnn_out, seq_len)
            scores = self.attention(item_emb, evolved, seq_len, return_scores=True)
            final_state = torch.sum(evolved * scores.unsqueeze(-1), dim=1)
        elif self.gru_type == "AIGRU":
            scores = self.attention(item_emb, rnn_out, seq_len, return_scores=True)
            weighted_rnn_out = rnn_out * scores.unsqueeze(-1)
            evolved = self._run_packed_gru(self.gru2, weighted_rnn_out, seq_len)
            final_state = gather_last_valid(evolved, seq_len)
        elif self.gru_type in {"AGRU", "AUGRU"}:
            scores = self.attention(item_emb, rnn_out, seq_len, return_scores=True)
            evolved = self.dynamic_gru(rnn_out, scores, seq_len)
            final_state = gather_last_valid(evolved, seq_len)
        else:
            raise ValueError(f"Unsupported gru_type: {self.gru_type}")

        # 最终的MLP输入拼接，还可以加入一些 Side Info 或者 Context 特征
        dense_input = torch.cat(
            [
                user_emb,
                item_emb,
                history_sum,
                item_emb * history_sum,
                final_state,
            ],
            dim=-1,
        )

        # MLP 预测点击概率
        logits = self.fcn(self.fcn_bn(dense_input))
        y_pred = torch.sigmoid(logits)

        if labels is None:
            return y_pred, aux_loss

        labels = labels.float().view(-1, 1)
        ctr_loss = self.ctr_loss_fn(logits, labels)
        total_loss = ctr_loss + aux_loss
        return y_pred, total_loss, ctr_loss, aux_loss


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size, seq_len, neg_num = 2, 5, 3
    num_user, num_item, num_cate = 100, 1000, 50

    cate_list = torch.randint(0, num_cate, (num_item,), dtype=torch.int64).tolist()

    model = DIEN(
        num_user=num_user,
        num_item=num_item,
        num_cate=num_cate,
        cate_list=cate_list,
        embedding_dim=18,
        gru_type="AUGRU",
        neg_mode="first",
    )

    user_ids = torch.randint(0, num_user, (batch_size,))
    item_ids = torch.randint(0, num_item, (batch_size,))
    history_items = torch.randint(0, num_item, (batch_size, seq_len))
    seq_len_tensor = torch.tensor([5, 3])
    neg_history_items = torch.randint(0, num_item, (batch_size, seq_len, neg_num))
    labels = torch.randint(0, 2, (batch_size,))

    y_pred, loss, ctr_loss, aux_loss = model(
        user_ids=user_ids,
        item_ids=item_ids,
        history_items=history_items,
        seq_len=seq_len_tensor,
        neg_history_items=neg_history_items,
        labels=labels,
    )

    print("y_pred:", y_pred.shape)
    print("loss:", float(loss), "ctr_loss:", float(ctr_loss), "aux_loss:", float(aux_loss))
