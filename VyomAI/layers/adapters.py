import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class LoraLinear(nn.Module):
    """Lora Adapter for parameter efficient training
    decomposes nn.linear into 2 low rank weightes and combines the output of both nn.Linear and two low rank layer output
    """

    def __init__(
        self,
        linear_layer,
        rank: Optional[int] = 32,
        alpha: Optional[int] = 1,
        lora_dropout: Optional[float] = 0.0,
    ) -> None:
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

    def forward(self, x) -> torch.Tensor:

        linear_output = self.linear(x)
        lora_output = self.alpha * F.linear(F.linear(x, self.lora_a), self.lora_b)
        lora_output = self.dropout(lora_output)
        return linear_output + lora_output


class DoraLinear(nn.Module):
    """Dora Adapter for parameter efficient training
    decomposes nn.linear into 2 low rank weightes with direction vector multiplication
    """

    def __init__(self, linear_layer, rank: Optional[int] = 32):
        super().__init__()

        self.linear = linear_layer

        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features
        self.dora_m = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True))

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.dora_a = nn.Parameter(torch.randn(self.out_features, rank) * std_dev)
        self.dora_b = nn.Parameter(torch.zeros(rank, self.in_features))

    def forward(self, x) -> torch.Tensor:
        lora = torch.matmul(self.dora_a, self.dora_b)
        adapted = self.linear.weight + lora
        column_norm = adapted.norm(p=2, dim=0, keepdim=True)
        norm_adapted = adapted / column_norm
        calc_weights = self.dora_m * norm_adapted
        bias = getattr(self.linear, "bias", None)
        return F.linear(x, calc_weights, bias)
