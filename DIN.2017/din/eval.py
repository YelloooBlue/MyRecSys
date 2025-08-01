# -*- coding: utf-8 -*-
"""
    DIN (Deep Interest Network) 模型评估
"""

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import DIN  # 假设 DIN 模型定义在 model.py 中
import tqdm
import time
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np

# 检查CUDA和Metal是否可用
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Metal is available: {torch.backends.mps.is_available()}")
print(f"Metal is built: {torch.backends.mps.is_built()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"# Using device: {device}")

# 随机数种子，方便复现代码
torch.manual_seed(1234)

# 超参数设置
train_batch_size = 32
test_batch_size = 512
num_epochs = 10

# ======================================= 自定义Dataloader =======================================

"""
    由于用户历史序列是不定长的，因此传统的DataLoader无法直接使用。
    需要自定义一个Dataloader来处理变长序列。
"""

# 自定义数据集类
class DINDataset(Dataset):
    def __init__(self, data, is_train=True):
        self.data = data
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.is_train:
            # 训练集格式: (user_id, [历史物品列表], 目标物品, 标签)
            user_id, hist_items, target_item, label = sample
            return {
                'user_id': torch.tensor(user_id, dtype=torch.long),
                'hist_items': torch.tensor(hist_items, dtype=torch.long),
                'target_item': torch.tensor(target_item, dtype=torch.long),
                'label': torch.tensor(label, dtype=torch.float)
            }
        else:
            # 测试集格式: (user_id, [历史物品列表], (正样本, 负样本))
            user_id, hist_items, (pos_item, neg_item) = sample
            return {
                'user_id': torch.tensor(user_id, dtype=torch.long),
                'hist_items': torch.tensor(hist_items, dtype=torch.long),
                'pos_item': torch.tensor(pos_item, dtype=torch.long),
                'neg_item': torch.tensor(neg_item, dtype=torch.long)
            }

# 用于处理变长历史序列的collate函数
def collate_fn(batch):
    # 找到最大的历史序列长度
    max_hist_len = max(len(item['hist_items']) for item in batch)

    # 填充历史序列到相同长度
    for item in batch:
        hist_len = len(item['hist_items'])
        if hist_len < max_hist_len:
            # 填充
            item['hist_items'] = torch.cat([
                item['hist_items'],
                torch.full((max_hist_len - hist_len,), 0, dtype=torch.long)
            ])
        # 记录原始长度
        item['hist_len'] = torch.tensor(hist_len, dtype=torch.long)

    # 将batch中的各个字段组合起来（兼容训练和测试）
    return {
        'user_id': torch.stack([item['user_id'] for item in batch]),
        'hist_items': torch.stack([item['hist_items'] for item in batch]),
        'hist_len': torch.stack([item['hist_len'] for item in batch]),
        'target_item': torch.stack([item['target_item'] for item in batch]) if 'target_item' in batch[0] else None, # 仅Train使用
        'label': torch.stack([item['label'] for item in batch]) if 'label' in batch[0] else None,                   # 仅Train使用
        'pos_item': torch.stack([item['pos_item'] for item in batch]) if 'pos_item' in batch[0] else None,          # 仅测试集使用
        'neg_item': torch.stack([item['neg_item'] for item in batch]) if 'neg_item' in batch[0] else None           # 仅测试集使用
    }




def train_din(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        
        user_ids = batch['user_id'].to(device)
        history_items = batch['hist_items'].to(device)
        seq_len = batch['hist_len'].to(device)
        item_ids = batch['target_item'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(user_ids, item_ids, history_items, seq_len)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


if __name__ == "__main__":

    # 加载数据集
    dataset_path = 'DIN.2017/dataset.pkl'
    with open(dataset_path, 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    """
        train_set:
        [(0, [13179], 17993, 1), (0, [13179], 28883, 0), ...]
        [(30, [13179, 17993, 28326], 29247, 1), (30, [13179, 17993, 28326], 490, 0)]

        test_set:
        [(0, [13179, 17993, 28326, 29247], (62275, 5940)), ...]
        [(30, [13179, 17993, 28326, 29247], (490, 7657)), ...]

        cate_list:
        [738, 157, 571, 707, 799, ...] #
    """

    # 加载测试集
    test_dataset = DINDataset(test_set, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    # 加载模型权重
    embedding_dim = 64
    hidden_units = 64
    model = DIN(user_count, item_count, embedding_dim, hidden_units).to(device)
    
    # 检查模型文件是否存在
    model_path = 'din_model.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    except FileNotFoundError:
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        print("请先运行 train.py 训练模型，或确保模型文件在正确的路径下。")
        exit(1)

    # ==================================== 评估模型 ====================================
    model.eval()

    test_loss = 0.0
    all_preds = []  # 存储所有预测分数
    all_labels = []  # 存储所有标签（1表示正样本，0表示负样本）
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Evaluating"):
            user_ids = batch['user_id'].to(device)
            history_items = batch['hist_items'].to(device)
            seq_len = batch['hist_len'].to(device)
            pos_items = batch['pos_item'].to(device)
            neg_items = batch['neg_item'].to(device)

            # 正样本预测
            outputs_pos = model(user_ids, pos_items, history_items, seq_len)
            loss_pos = criterion(outputs_pos.squeeze(), torch.ones_like(outputs_pos.squeeze()))

            # 负样本预测
            outputs_neg = model(user_ids, neg_items, history_items, seq_len)
            loss_neg = criterion(outputs_neg.squeeze(), torch.zeros_like(outputs_neg.squeeze()))

            test_loss += (loss_pos + loss_neg).item() * batch['user_id'].size(0)

            # AUC
            all_preds.extend(outputs_pos.squeeze().cpu().numpy().tolist())
            all_labels.extend([1] * outputs_pos.size(0))
            all_preds.extend(outputs_neg.squeeze().cpu().numpy().tolist())
            all_labels.extend([0] * outputs_neg.size(0))

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # 计算AUC
    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"AUC Score: {auc_score:.4f}")
    
    # ======================== 添加精确率和召回率计算 ========================
    
    # 将预测分数转换为概率（通过sigmoid）
    all_probs = [1 / (1 + np.exp(-pred)) for pred in all_preds]  # sigmoid转换
    
    # 使用不同阈值计算精确率、召回率和F1分数
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n" + "="*50)
    print("精确率、召回率和F1分数 (不同阈值)")
    print("="*50)
    print(f"{'阈值':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10}")
    print("-" * 50)
    
    for threshold in thresholds:
        # 将概率转换为二元预测
        binary_preds = [1 if prob >= threshold else 0 for prob in all_probs]
        
        # 计算精确率、召回率和F1分数
        precision = precision_score(all_labels, binary_preds, zero_division=0)
        recall = recall_score(all_labels, binary_preds, zero_division=0)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        
        print(f"{threshold:<8.1f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # 计算最优阈值（基于F1分数）
    precisions, recalls, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # 避免除零
    
    # 找到最大F1分数对应的阈值
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
    best_f1 = f1_scores[best_f1_idx]
    best_precision = precisions[best_f1_idx]
    best_recall = recalls[best_f1_idx]
    
    print("\n" + "="*50)
    print("最优性能 (基于F1分数)")
    print("="*50)
    print(f"最优阈值: {best_threshold:.4f}")
    print(f"最大F1分数: {best_f1:.4f}")
    print(f"对应精确率: {best_precision:.4f}")
    print(f"对应召回率: {best_recall:.4f}")
    
    # 额外统计信息
    print("\n" + "="*50)
    print("数据集统计")
    print("="*50)
    print(f"总样本数: {len(all_labels)}")
    print(f"正样本数: {sum(all_labels)}")
    print(f"负样本数: {len(all_labels) - sum(all_labels)}")
    print(f"正样本比例: {sum(all_labels) / len(all_labels):.4f}")
    
    print("\n评估完成!")
