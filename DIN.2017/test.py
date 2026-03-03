# -*- coding: utf-8 -*-
"""
    DIN (Deep Interest Network) 模型评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tqdm
import time
import pickle
import argparse
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
from pathlib import Path

from model import DIN
from dataset import DINDataset

ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / 'dataset.pkl'
CKPT_DIR = ROOT_DIR / 'output' / 'checkpoint'


def resolve_model_path(model_path_arg: str | None) -> Path:
    if model_path_arg:
        return Path(model_path_arg).expanduser().resolve()

    best_ckpts = sorted(
        CKPT_DIR.glob('run_*/din_model_best.pth'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not best_ckpts:
        raise FileNotFoundError(f"未找到 best checkpoint: {CKPT_DIR}/run_*/din_model_best.pth")
    return best_ckpts[0]

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

# 模型参数
embedding_dim = 64
hidden_units = 64

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIN inference/evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="模型权重路径；不传则自动选择 output/checkpoint 下最新的 din_model_best.pth",
    )
    args = parser.parse_args()

    # 加载数据集
    with open(DATASET_PATH, 'rb') as f:
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
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # 加载模型权重
    model = DIN(
        num_user=user_count,
        num_item=item_count,
        num_cate=cate_count,
        cate_list=cate_list,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
    ).to(device)
    
    # 检查模型文件是否存在
    model_path = resolve_model_path(args.model_path)
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
