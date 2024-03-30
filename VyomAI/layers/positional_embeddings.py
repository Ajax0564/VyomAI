import torch
import torch.nn as nn
from einops import rearrange, reduce
from typing import Optional,Tuple

class AbsoluteEncoding(nn.Module):
    def __init__(self,config) -> None:
        super.__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.max_size = config.max_position_embeddings

    def forward(self,hidden_state: torch.Tensor) -> torch.Tensor:
        size = hidden_state.size()[1]
        if self.max_size < size:
            raise ValueError(
                f"The hidden size ({size }) is more than the config max_position_embeddings {self.max_size}")
        return self.position_embeddings(self.position_ids[:,:size])
