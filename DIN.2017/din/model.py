"""
    DIN(Deep Interest Network)模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    """
        注意力机制实现——单物品（训练阶段）
        从历史行为中筛选与目标物品相关的信息
    """
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        self.hidden_units = hidden_units
        
        # 注意力权重计算的MLP（参考原文实现为80-40-1，使用sigmoid激活）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_units * 4, 80),
            nn.Sigmoid(),
            nn.Linear(80, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1)
        )
        
    def forward(self, query, keys, seq_len):
        """
        参数:
            query: 目标物品嵌入 [B, hidden_units]
            keys: 历史行为嵌入 [B, T, hidden_units]
            seq_len: 每个样本的有效历史长度 [B]
        返回:
            注意力加权后的历史向量 [B, hidden_units]
        """
        B, T, _ = keys.shape
        # 将query扩展为 [B, T, hidden_units]，与keys维度匹配（广播）
        query = query.unsqueeze(1).repeat(1, T, 1)  # [B, T, H]
        
        # 人工合成特征：query、keys、query-keys、query*keys（4种交互，用于捕捉不同关系）
        din_all = torch.cat([
            query,
            keys,
            query - keys,
            query * keys
        ], dim=-1)  # [B, T, 4*H]

        outputs = self.mlp(din_all)  # [B, T, 1]
        outputs = outputs.squeeze(-1)  # [B, T]

        # Mask 无效历史
        mask = torch.arange(T, device=seq_len.device)[None, :] < seq_len[:, None]  # [B, T]
        outputs = outputs.masked_fill(~mask, -1e9)  # 将无效历史的注意力值设为负无穷

        # Scale 注意力缩放
        outputs = outputs / np.sqrt(self.hidden_units)  # 注意力缩放 [B, T]

        # Activation 激活函数
        outputs = F.softmax(outputs, dim=1)  # Softmax归一化（TODO：存疑，原文说不用softmax）

        # Weighted sum 加权求和
        outputs = torch.sum(outputs.unsqueeze(-1) * keys, dim=1)  # [B, hidden_units]

        return outputs  # 返回加权后的历史向量 [B, hidden_units]
    
# TODO：合并两种注意力机制实现，减少重复代码
class Attention_Multi(nn.Module):
    """
        注意力机制实现——多物品（预测阶段）
        从历史行为中筛选与目标物品相关的信息
    """
    def __init__(self, hidden_units):
        super(Attention_Multi, self).__init__()
        self.hidden_units = hidden_units
        
        # 注意力权重计算的MLP（参考原文实现为80-40-1，使用sigmoid激活）
        self.mlp = nn.Sequential(
            nn.Linear(hidden_units * 4, 80),
            nn.Sigmoid(),
            nn.Linear(80, 40),
            nn.Sigmoid(),
            nn.Linear(40, 1)
        )

    def forward(self, queries, keys, seq_len):
        """
        参数:
            queries: 目标物品嵌入 [B, N, hidden_units]
            keys: 历史行为嵌入 [B, T, hidden_units]
            seq_len: 每个样本的有效历史长度 [B]
        返回:
            注意力加权后的历史向量 [B, N, hidden_units]
        """
        B, N, _ = queries.shape
        _, T, _ = keys.shape

        # 将queries和keys扩展为适当的形状以进行广播
        queries = queries.unsqueeze(2).repeat(1, 1, T, 1) # [B, N, T, hidden_units]
        keys = keys.unsqueeze(1).repeat(1, N, 1, 1)       # [B, N, T, hidden_units]

        # 人工合成特征：queries、keys、queries-keys、queries*keys（4种交互，用于捕捉不同关系）
        din_all = torch.cat([
            queries,
            keys,
            queries - keys,
            queries * keys
        ], dim=-1)  # [B, N, T, 4*H]

        outputs = self.mlp(din_all)  # [B, N, T, 1]
        outputs = outputs.squeeze(-1)  # [B, N, T]

        # Mask 无效历史
        mask = torch.arange(T, device=seq_len.device)[None, None, :] < seq_len[:, None, None] # [B, N, T]
        outputs = outputs.masked_fill(~mask, -1e9)

        # Scale 注意力缩放
        outputs = outputs / np.sqrt(self.hidden_units)  # 注意力缩放 [B, N, T]

        # Activation 激活函数
        outputs = F.softmax(outputs, dim=2)  # Softmax归一化（TODO：存疑，原文说不用softmax）

        # Weighted sum 加权求和
        outputs = torch.sum(outputs.unsqueeze(-1) * keys, dim=2)  # [B, N, hidden_units]

        return outputs  # 返回加权后的历史向量 [B, N, hidden_units]
    
class DIN(nn.Module):
    """
        DIN模型实现
        输入：用户历史行为、目标物品嵌入
        输出：目标物品的兴趣向量
    """
    def __init__(self, num_user, num_item, embedding_dim, hidden_units):
        super(DIN, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units

        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(num_user, embedding_dim)
        self.item_embedding = nn.Embedding(num_item, embedding_dim)

        # 注意力
        self.attention = Attention(hidden_units)
        self.attention_multi = Attention_Multi(hidden_units)

        # MLP（还是参考官方代码80-40-1）
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + hidden_units + embedding_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, user_ids, item_ids, history_items, seq_len):
        """
        参数:
            user_ids: 用户ID [B]
            item_ids: 目标物品ID [B]
            history_items: 历史行为物品ID [B, T]
            seq_len: 每个样本的有效历史长度 [B]
        返回:
            目标物品的兴趣向量 [B, hidden_units]
        """
        # 嵌入查找
        user_emb = self.user_embedding(user_ids)            # [B, embedding_dim]
        item_emb = self.item_embedding(item_ids)            # [B, embedding_dim]
        history_emb = self.item_embedding(history_items)    # [B, T, embedding_dim]

        # 注意力机制
        attention_output = self.attention(item_emb, history_emb, seq_len)

        # 将用户嵌入和注意力输出拼接
        output = torch.cat([user_emb, attention_output, item_emb], dim=-1) # [B, embedding_dim + hidden_units + embedding_dim]

        # MLP处理
        output = self.mlp(output)  # [B, 1]
        ctr = torch.sigmoid(output)  # Sigmoid激活，输出概率

        return ctr

if __name__ == "__main__":

    # ================================= 测试各个模块 =================================

    # 测试 Attention 模块
    attention = Attention(hidden_units=64)
    query = torch.randn(2, 64)  # [B, hidden_units]
    keys = torch.randn(2, 10, 64)  # [B, T, hidden_units]
    seq_len = torch.tensor([10, 5])  # 每个样本的有效历史长度

    output = attention(query, keys, seq_len)
    print(output.shape)  # 应该输出 [2, 64]

    # 测试 Attention_Multi 模块
    attention_multi = Attention_Multi(hidden_units=64)
    queries = torch.randn(2, 3, 64)  # [B, N, hidden_units]
    keys = torch.randn(2, 10, 64)  # [B, T, hidden_units]
    seq_len_multi = torch.tensor([10, 5])  # 每个样本的有效历史长度

    output_multi = attention_multi(queries, keys, seq_len_multi)
    print(output_multi.shape)  # 应该输出 [2, 3, 64]

    # 测试 DIN 模型
    din_model = DIN(num_user=100, num_item=1000, embedding_dim=64, hidden_units=64)
    user_ids = torch.tensor([1, 2])
    item_ids = torch.tensor([10, 20])
    history_items = torch.tensor([[10, 20, 30, 40, 50],
                                  [20, 30, 40, 50, 60]])
    seq_len = torch.tensor([5, 4])  # 每个样本的有效历史长度
    ctr_output = din_model(user_ids, item_ids, history_items, seq_len)
    print(ctr_output.shape)  # 应该输出 [2, 1]