# To Do
# add Lora and Dora
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class LoRALinear(nn.Module):
    """Lora Adapter for parameter efficient training
    decomposes nn.linear into 2 low rank weightes"""

    def __init__(
        self, linear_layer, rank, alpha, lora_dropout: Optional[float] = 0.0
    ) -> None:
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha

        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())

        self.lora_a = nn.Parameter(
            torch.randn(
                self.rank,
                self.in_features,
            )
            * std_dev
        )
        self.lora_b = nn.Parameter(
            torch.zeros(self.out_features, self.rank)
        )  # reverse  order due to F.linear as it process xA^T
        self.dropout = nn.Dropout(lora_dropout)

    def forward(self, x):
        lora_output = self.alpha * F.linear(F.linear(x, self.lora_a), self.lora_b)
        lora_output = self.dropout(lora_output)
        return lora_output


class MergeLoRALinear(nn.Module):
    """Lora Adapter for parameter efficient training
    decomposes nn.linear into 2 low rank weightes and combines the output of both nn.Linear and tow low rank layer output
    """

    def __init__(self, linear_layer, rank, alpha, lora_dropout=0.0):
        super().__init__()

        self.linear = linear_layer

        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.rank = rank
        self.alpha = alpha

        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())

        self.lora_a = nn.Parameter(
            torch.randn(
                self.rank,
                self.in_features,
            )
            * std_dev
        )
        self.lora_b = nn.Parameter(
            torch.zeros(self.out_features, self.rank)
        )  # reverse  order due to F.linear as it process xA^T
        self.dropout = nn.Dropout(lora_dropout)

    def forward(self, x):

        linear_output = self.linear(x)
        lora_output = self.alpha * F.linear(F.linear(x, self.lora_a), self.lora_b)
        lora_output = self.dropout(lora_output)
        return linear_output + lora_output
