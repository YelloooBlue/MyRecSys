import torch
from torch.utils.data import Dataset

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
    @staticmethod
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