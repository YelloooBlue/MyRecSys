"""DIEN training script."""

from datetime import datetime
from pathlib import Path
import pickle
import time

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

from dataset import DIENDataset
from model import DIEN

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
TB_LOG_DIR = OUTPUT_DIR / "tb_logs"
CKPT_DIR = OUTPUT_DIR / "checkpoint"
DATASET_PATH = ROOT_DIR / "dataset.pkl"

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
embedding_dim = 18
gru_type = "AUGRU"
neg_mode = "first"


if __name__ == "__main__":
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_log_dir = TB_LOG_DIR / run_name
    run_ckpt_dir = CKPT_DIR / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_log_dir), flush_secs=10)
    print(f"# TensorBoard run dir: {run_log_dir}")
    print(f"# Checkpoint run dir: {run_ckpt_dir}")

    # 加载数据集
    with open(DATASET_PATH, "rb") as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    # 创建训练集和测试集
    train_dataset = DIENDataset(train_set, is_train=True)
    test_dataset = DIENDataset(test_set, is_train=False)

    # 创建数据加载器
    use_cuda = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
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
    model = DIEN(
        num_user=user_count,
        num_item=item_count,
        num_cate=cate_count,
        cate_list=cate_list.tolist() if hasattr(cate_list, "tolist") else cate_list,
        embedding_dim=embedding_dim,
        gru_type=gru_type,
        neg_mode=neg_mode,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 定义优化器和混合精度训练工具
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_auc = 0.0
    best_epoch = 0
    global_step = 0

    print("Start training")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_ctr_loss = 0.0
        train_aux_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        for batch in progress_bar:
            user_ids = batch["user_id"].to(device, non_blocking=use_cuda)
            history_items = batch["hist_items"].to(device, non_blocking=use_cuda)
            neg_history_items = batch["neg_hist_items"].to(device, non_blocking=use_cuda)
            seq_len = batch["hist_len"].to(device, non_blocking=use_cuda)
            item_ids = batch["target_item"].to(device, non_blocking=use_cuda)
            labels = batch["label"].to(device, non_blocking=use_cuda)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                _, loss, ctr_loss, aux_loss = model(
                    user_ids=user_ids,
                    item_ids=item_ids,
                    history_items=history_items,
                    seq_len=seq_len,
                    neg_history_items=neg_history_items,
                    labels=labels,
                )

            if not torch.isfinite(loss):
                print("Skip non-finite loss batch.")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/loss_step", loss.item(), global_step)
            writer.add_scalar("train/ctr_loss_step", ctr_loss.item(), global_step)
            writer.add_scalar("train/aux_loss_step", aux_loss.item(), global_step)
            writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)

            bs = batch["user_id"].size(0)
            train_loss += loss.item() * bs
            train_ctr_loss += ctr_loss.item() * bs
            train_aux_loss += aux_loss.item() * bs

            progress_bar.set_postfix(
                {
                    "batch_loss": f"{loss.item():.4f}",
                    "epoch_loss": f"{train_loss / len(train_loader.dataset):.4f}",
                }
            )

            if (global_step + 1) % grad_log_interval == 0:
                grad_sq_sum = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_sq_sum += p.grad.detach().float().pow(2).sum().item()
                writer.add_scalar("train/grad_global_norm", grad_sq_sum ** 0.5, global_step)

            global_step += 1

        train_loss /= len(train_loader.dataset)
        train_ctr_loss /= len(train_loader.dataset)
        train_aux_loss /= len(train_loader.dataset)

        writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)
        writer.add_scalar("train/ctr_loss_epoch", train_ctr_loss, epoch + 1)
        writer.add_scalar("train/aux_loss_epoch", train_aux_loss, epoch + 1)

        model.eval()
        eval_total_loss = 0.0
        eval_ctr_loss = 0.0
        eval_aux_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch["user_id"].to(device, non_blocking=use_cuda)
                history_items = batch["hist_items"].to(device, non_blocking=use_cuda)
                neg_history_items = batch["neg_hist_items"].to(device, non_blocking=use_cuda)
                seq_len = batch["hist_len"].to(device, non_blocking=use_cuda)
                pos_items = batch["pos_item"].to(device, non_blocking=use_cuda)
                neg_items = batch["neg_item"].to(device, non_blocking=use_cuda)

                pos_pred, pos_aux = model(
                    user_ids=user_ids,
                    item_ids=pos_items,
                    history_items=history_items,
                    seq_len=seq_len,
                    neg_history_items=neg_history_items,
                    labels=None,
                )
                neg_pred, neg_aux = model(
                    user_ids=user_ids,
                    item_ids=neg_items,
                    history_items=history_items,
                    seq_len=seq_len,
                    neg_history_items=neg_history_items,
                    labels=None,
                )

                # Eval loss proxy: BCE on prob + auxiliary term.
                pos_pred = torch.nan_to_num(pos_pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
                neg_pred = torch.nan_to_num(neg_pred, nan=0.5, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
                pos_bce = torch.nn.functional.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
                neg_bce = torch.nn.functional.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
                ctr_loss = pos_bce + neg_bce
                aux_loss = 0.5 * (pos_aux + neg_aux)
                total_loss = ctr_loss + aux_loss

                bs = batch["user_id"].size(0)
                eval_total_loss += total_loss.item() * bs
                eval_ctr_loss += ctr_loss.item() * bs
                eval_aux_loss += aux_loss.item() * bs

                all_preds.extend(pos_pred.squeeze(-1).cpu().numpy().tolist())
                all_labels.extend([1] * pos_pred.size(0))
                all_preds.extend(neg_pred.squeeze(-1).cpu().numpy().tolist())
                all_labels.extend([0] * neg_pred.size(0))

        eval_total_loss /= len(test_loader.dataset)
        eval_ctr_loss /= len(test_loader.dataset)
        eval_aux_loss /= len(test_loader.dataset)
        auc_score = roc_auc_score(all_labels, all_preds)

        writer.add_scalar("eval/loss_epoch", eval_total_loss, epoch + 1)
        writer.add_scalar("eval/ctr_loss_epoch", eval_ctr_loss, epoch + 1)
        writer.add_scalar("eval/aux_loss_epoch", eval_aux_loss, epoch + 1)
        writer.add_scalar("eval/auc_epoch", auc_score, epoch + 1)

        if auc_score > best_auc:
            best_auc = auc_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), run_ckpt_dir / "dien_model_best.pth")

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Eval Total: {eval_total_loss:.4f}, "
            f"Eval CTR: {eval_ctr_loss:.4f}, Eval AUX: {eval_aux_loss:.4f}, "
            f"AUC: {auc_score:.4f}, Time: {epoch_time:.2f}s"
        )

        torch.save(model.state_dict(), run_ckpt_dir / f"dien_model_epoch{epoch + 1}.pth")

    print(f"Best AUC: {best_auc:.4f} (epoch={best_epoch})")
    writer.close()
