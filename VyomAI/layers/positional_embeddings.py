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


class SinusoidalEncoding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size%2!=0:
             raise ValueError(f"Cannot use SinusoidalEncoding with "
                         "odd hidden dim got dim {config.hidden_size}")
        positional_encoding = torch.zeros(config.max_position_embeddings,config.hidden_size)
        position = torch.arange(0,config.max_position_embeddings ).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, config.hidden_size, 2, dtype=torch.float) *
                         -(torch.log(10000.0) / config.hidden_size)))
        
        positional_encoding[:, 0::2] = torch.sin(position.float() * div_term)
        positional_encoding[:, 1::2] = torch.cos(position.float() * div_term)

    def forward(self,hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = hidden_state.size() #b*l

        return self.positional_encoding[:batch_size, :seq_len, :]
    
class RelativePositionalEncoding(nn.Module):
    ''' each element will contain information for all forward and backward element
        it can be used inside  every attention layer or at only once before attention like  AbsoluteEncoding 

    '''
    def __init__(self, config):
        super().__init__()
        self.max_position = config.max_position_embeddings
        self.embeddings_table = nn.Parameter(torch.Tensor(self.max_position * 2 + 1,config.hidden_size))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, q: torch.Tensor,k: torch.Tensor) -> torch.Tensor:
        len_q = q.size()[1]
        len_k = k.size()[1]
        range_vec_q = torch.arange(len_q)
        range_vec_k = torch.arange(len_k)
        relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
        clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
        relative_position_matrix = clipped_relative_matrix + self.max_position
        embeddings = self.embeddings_table[relative_position_matrix]

        return embeddings

#To do : add Rope and Alibi