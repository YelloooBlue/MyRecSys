"""
    DIN (Deep Interest Network) 模型训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import time
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from model import DIN
from dataset import DINDataset

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / 'output'
TB_LOG_DIR = OUTPUT_DIR / 'tb_logs'
CKPT_DIR = OUTPUT_DIR / 'checkpoint'
DATASET_PATH = ROOT_DIR / 'dataset.pkl'

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

# TensorBoard / Checkpoint 目录（不存在则自动创建）
TB_LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# 随机数种子，方便复现代码
torch.manual_seed(1234)

# 超参数设置
train_batch_size = 256
test_batch_size = 512
num_epochs = 4
num_workers = 4         # Dataloader的工作线程数，视系统性能调整
grad_log_interval = 50  # 每多少个step记录一次梯度范数

# 模型参数
embedding_dim = 64
hidden_units = 64

if __name__ == "__main__":
    run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')
    run_log_dir = TB_LOG_DIR / run_name
    run_ckpt_dir = CKPT_DIR / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_log_dir), flush_secs=10)
    print(f"# TensorBoard run dir: {run_log_dir}")
    print(f"# Checkpoint run dir: {run_ckpt_dir}")

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

    # 创建训练集和测试集
    train_dataset = DINDataset(train_set, is_train=True)
    test_dataset = DINDataset(test_set, is_train=False)

    # =================================== 测试dataloader =================================

    # temp_train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # temp_test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # # 测试数据加载器
    # for batch in temp_train_loader:
    #     print("Train Batch:")
    #     print(f"User IDs: {batch['user_id']}")
    #     print(f"History Items: {batch['hist_items']}")
    #     print(f"Target Item: {batch['target_item']}")
    #     print(f"Labels: {batch['label']}")
    #     break

    # exit(0)  # 测试通过后退出

    # =================================== 真实数据集和数据加载器 =================================

    # 创建数据加载器
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,                # 将数据加载到CUDA时使用页锁定内存，提升性能
        persistent_workers=num_workers > 0, # worker常驻内存，减少每个epoch的加载时间
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    # 初始化模型
    model = DIN(num_user=user_count, num_item=item_count, num_cate=cate_count, cate_list=cate_list, embedding_dim=embedding_dim, hidden_units=hidden_units).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 相比BCE Loss，集成了Sigmoid函数，更适合二分类任务，并且数值稳定性更好
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda) # 使用自动混合精度训练，提升性能并节省显存

    # ================================== 训练模型 =================================

    # 记录每个epoch的损失
    train_losses = []
    test_losses = []
    best_auc = 0.0
    best_epoch = 0

    global_step = 0
    print("Start training")
    for epoch in range(num_epochs):
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")

        for batch in progress_bar:
            user_ids = batch['user_id'].to(device, non_blocking=use_cuda)       # non_blocking=True允许异步数据传输，提升性能
            history_items = batch['hist_items'].to(device, non_blocking=use_cuda)
            seq_len = batch['hist_len'].to(device, non_blocking=use_cuda)
            item_ids = batch['target_item'].to(device, non_blocking=use_cuda)
            labels = batch['label'].to(device, non_blocking=use_cuda)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_cuda):
                outputs = model(user_ids, item_ids, history_items, seq_len)
                loss = criterion(outputs.squeeze(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 记录训练损失
            writer.add_scalar('train/loss_step', loss.item(), global_step)
            writer.add_scalar('train/lr_step', optimizer.param_groups[0]['lr'], global_step)

            train_loss += loss.item() * batch['user_id'].size(0)  # 累加损失

            progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}",
                                        'epoch_loss': f"{train_loss / len(train_loader.dataset):.4f}"})
            if (global_step + 1) % grad_log_interval == 0:
                grad_sq_sum = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq_sum += p.grad.detach().float().pow(2).sum().item()
                writer.add_scalar('train/grad_global_norm', grad_sq_sum ** 0.5, global_step)
            
            global_step += 1
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        writer.add_scalar('train/loss_epoch', train_loss, epoch + 1)

        # 测试阶段
        model.eval()
        test_loss = 0.0

        all_preds = []  # 存储所有预测分数
        all_labels = []  # 存储所有标签（1表示正样本，0表示负样本）

        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(device, non_blocking=use_cuda)
                history_items = batch['hist_items'].to(device, non_blocking=use_cuda)
                seq_len = batch['hist_len'].to(device, non_blocking=use_cuda)
                pos_items = batch['pos_item'].to(device, non_blocking=use_cuda)
                neg_items = batch['neg_item'].to(device, non_blocking=use_cuda)

                # 对正样本进行预测
                outputs = model(user_ids, pos_items, history_items, seq_len)
                loss_pos = criterion(outputs.squeeze(), torch.ones_like(outputs.squeeze()))
                
                # 对负样本进行预测
                outputs_neg = model(user_ids, neg_items, history_items, seq_len)
                loss_neg = criterion(outputs_neg.squeeze(), torch.zeros_like(outputs_neg.squeeze()))

                test_loss += (loss_pos + loss_neg).item() * batch['user_id'].size(0)

                # AUC
                all_preds.extend(torch.sigmoid(outputs.squeeze()).cpu().numpy().tolist())
                all_labels.extend([1] * outputs.size(0))  # 正样本标签
                all_preds.extend(torch.sigmoid(outputs_neg.squeeze()).cpu().numpy().tolist())
                all_labels.extend([0] * outputs_neg.size(0))  # 负样本标签
                
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        # 计算AUC
        auc_score = roc_auc_score(all_labels, all_preds)
        writer.add_scalar('eval/loss_epoch', test_loss, epoch + 1)
        writer.add_scalar('eval/auc_epoch', auc_score, epoch + 1)
        writer.add_scalar('train/lr_epoch', optimizer.param_groups[0]['lr'], epoch + 1)

        if auc_score > best_auc:
            best_auc = auc_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), run_ckpt_dir / 'din_model_best.pth')

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, AUC: {auc_score:.4f}, Time: {epoch_time:.2f}s")

        # 保存模型
        torch.save(model.state_dict(), run_ckpt_dir / f'din_model_epoch{epoch+1}.pth')

    writer.add_scalar('eval/auc_best', best_auc, best_epoch if best_epoch > 0 else 0)
    writer.add_hparams(
        {
            'train_batch_size': train_batch_size,
            'test_batch_size': test_batch_size,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'embedding_dim': embedding_dim,
            'hidden_units': hidden_units,
            'lr': optimizer.param_groups[0]['lr'],
            'use_cuda': int(use_cuda),
            'amp': int(use_cuda),
        },
        {
            'hparam/best_auc': float(best_auc),
            'hparam/best_epoch': float(best_epoch),
            'hparam/final_train_loss': float(train_losses[-1]) if train_losses else 0.0,
            'hparam/final_test_loss': float(test_losses[-1]) if test_losses else 0.0,
        },
    )
    print(f"Best AUC: {best_auc:.4f} (epoch={best_epoch})")

    # 关闭TensorBoard
    writer.close()
