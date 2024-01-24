import torch
import torch.nn as nn
import torch.nn.functional as F

class ResDenseLayer(nn.Module):
    def __init__(
            self,
            width,
            dropout=0.0,
    ):
        super().__init__()
        self.dense = nn.Linear(width, width)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.alpha*self.dense(x)
        y = x + self.dropout(y)
        y = F.relu(y)
        return y

class ResDenseBlock(nn.Module):
    def __init__(
            self,
            width,
            depth,
            dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([ResDenseLayer(width, dropout) for _ in range(depth)])

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
    
class ResDenseNet(nn.Module):
    def __init__(
        self,
        input_size,
        width,
        depth,
        dropout=0.0  
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.densefirst = nn.Linear(input_size, width)
        self.rdb1 = ResDenseBlock(width=width, depth=depth, dropout=dropout)
        self.denselast = nn.Linear(width, 1)

    def forward(self, x):
        y = self.densefirst(x)
        y = F.relu(y)
        y = self.rdb1(y)
        y = self.denselast(y)
        return y