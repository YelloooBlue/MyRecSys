"""
    DIN (Deep Interest Network) 模型训练
"""

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import DIN  # 假设 DIN 模型定义在 model.py 中
import tqdm
import time

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
                torch.full((max_hist_len - hist_len,), -1, dtype=torch.long)
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
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型
    embedding_dim = 64
    hidden_units = 64
    model = DIN(num_user=user_count, num_item=item_count, embedding_dim=embedding_dim, hidden_units=hidden_units).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ================================== 训练模型 =================================

    # 记录每个epoch的损失
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")

        for batch in progress_bar:
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

            train_loss += loss.item() * batch['user_id'].size(0)  # 累加损失

            progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}",
                                        'epoch_loss': f"{train_loss / len(train_loader.dataset):.4f}"})
            
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 测试阶段
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(device)
                history_items = batch['hist_items'].to(device)
                seq_len = batch['hist_len'].to(device)
                pos_items = batch['pos_item'].to(device)
                neg_items = batch['neg_item'].to(device)

                outputs = model(user_ids, pos_items, history_items, seq_len)
                loss_pos = criterion(outputs.squeeze(), torch.ones_like(outputs.squeeze()))
                
                outputs_neg = model(user_ids, neg_items, history_items, seq_len)
                loss_neg = criterion(outputs_neg.squeeze(), torch.zeros_like(outputs_neg.squeeze()))

                test_loss += (loss_pos + loss_neg).item() * batch['user_id'].size(0)

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Time: {epoch_time:.2f}s")



        



    # 保存模型
    torch.save(model.state_dict(), 'din_model.pth')