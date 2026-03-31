import torch
import torch.nn as nn
import torch.nn.functional as F


class SENETLayer(nn.Module):
    """
    Field-wise SENet for recommendation systems.

    Input:
        x: Tensor of shape [B, F, E]
           B = batch_size
           F = num_fields
           E = embedding_dim

    Output:
        out: Tensor of shape [B, F, E]
    """
    def __init__(self, num_fields: int, reduction_ratio: int = 3):
        super().__init__()
        reduced_size = max(1, num_fields // reduction_ratio)

        self.excitation = nn.Sequential(
            nn.Linear(num_fields, reduced_size, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_size, num_fields, bias=False),
            nn.ReLU()   # 有些实现用 ReLU，也有些用 Sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, E]

        # 1) Squeeze: 对每个 field 的 embedding 做压缩
        #    从 [B, F, E] -> [B, F]
        z = torch.mean(x, dim=-1)

        # 2) Excitation: 学习每个 field 的重要性权重
        #    [B, F] -> [B, F]
        a = self.excitation(z)

        # 3) Re-weight: 回乘到原始 field embedding
        #    [B, F] -> [B, F, 1]
        a = a.unsqueeze(-1)

        out = x * a
        return out
    

class SENETLayerSigmoid(nn.Module):
    def __init__(self, num_fields: int, reduction_ratio: int = 3):
        super().__init__()
        reduced_size = max(1, num_fields // reduction_ratio)

        self.fc1 = nn.Linear(num_fields, reduced_size, bias=False)
        self.fc2 = nn.Linear(reduced_size, num_fields, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.mean(x, dim=-1)                 # [B, F]
        a = F.relu(self.fc1(z))                   # [B, F//r]
        a = torch.sigmoid(self.fc2(a))            # [B, F]
        out = x * a.unsqueeze(-1)                 # [B, F, E]
        return out
    
if __name__ == "__main__":
    batch_size = 4
    num_fields = 10
    embedding_dim = 8

    x = torch.randn(batch_size, num_fields, embedding_dim)

    senet = SENETLayer(num_fields=num_fields, reduction_ratio=3)
    y = senet(x)

    print("input shape :", x.shape)   # [4, 10, 8]
    print("output shape:", y.shape)   # [4, 10, 8]