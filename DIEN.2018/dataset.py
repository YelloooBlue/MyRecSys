import torch
from torch.utils.data import Dataset


class DIENDataset(Dataset):
    """Dataset for DIEN.

    Train sample:
        (user_id, hist_items, neg_hist_items, target_item, label)
    Test sample:
        (user_id, hist_items, neg_hist_items, (pos_item, neg_item))
    """

    def __init__(self, data, is_train: bool = True, neg_num: int = 1):
        self.data = data
        self.is_train = is_train
        self.neg_num = neg_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.is_train:
            user_id, hist_items, neg_hist_items, target_item, label = sample
            return {
                "user_id": torch.tensor(user_id, dtype=torch.long),
                "hist_items": torch.tensor(hist_items, dtype=torch.long),
                "neg_hist_items": torch.tensor(neg_hist_items, dtype=torch.long),
                "target_item": torch.tensor(target_item, dtype=torch.long),
                "label": torch.tensor(label, dtype=torch.float),
            }

        user_id, hist_items, neg_hist_items, (pos_item, neg_item) = sample
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "hist_items": torch.tensor(hist_items, dtype=torch.long),
            "neg_hist_items": torch.tensor(neg_hist_items, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_item": torch.tensor(neg_item, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        max_hist_len = max(len(item["hist_items"]) for item in batch)

        for item in batch:
            hist_len = len(item["hist_items"])
            pad_len = max_hist_len - hist_len

            if pad_len > 0:
                item["hist_items"] = torch.cat(
                    [item["hist_items"], torch.zeros(pad_len, dtype=torch.long)]
                )
                item["neg_hist_items"] = torch.cat(
                    [item["neg_hist_items"], torch.zeros(pad_len, dtype=torch.long)]
                )

            item["hist_len"] = torch.tensor(hist_len, dtype=torch.long)

        # Model expects [B, T, NEG]. Here NEG=1.
        neg_hist_items = torch.stack([item["neg_hist_items"] for item in batch]).unsqueeze(-1)

        return {
            "user_id": torch.stack([item["user_id"] for item in batch]),
            "hist_items": torch.stack([item["hist_items"] for item in batch]),
            "neg_hist_items": neg_hist_items,
            "hist_len": torch.stack([item["hist_len"] for item in batch]),
            "target_item": torch.stack([item["target_item"] for item in batch]) if "target_item" in batch[0] else None,
            "label": torch.stack([item["label"] for item in batch]) if "label" in batch[0] else None,
            "pos_item": torch.stack([item["pos_item"] for item in batch]) if "pos_item" in batch[0] else None,
            "neg_item": torch.stack([item["neg_item"] for item in batch]) if "neg_item" in batch[0] else None,
        }
