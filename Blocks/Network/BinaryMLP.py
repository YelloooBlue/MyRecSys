import torch
import torch.nn as nn

class BinaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=[128, 64], dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

