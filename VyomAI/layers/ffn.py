import torch
import torch.nn as nn
from einops import rearrange, reduce
from typing import Optional,Tuple,Union

class SelfOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.layerNorm(hidden_state + input_tensor)
        return hidden_state
    
_ACT_ = {
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
    "relu6": nn.ReLU6(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh()}

class FeedForward(nn.Module):
    def __init__(self, config,multiplier: Union[int,float] = 4 ) -> None:
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size,int(multiplier)*config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if _ACT_.get(config.hidden_act,None):
             self.act_fn = _ACT_[config.hidden_act]
        else:
             self.act_fn = nn.GELU()
        self.out = nn.Linear(int(multiplier)*config.hidden_size,config.hidden_size)
    def forward(self, hidden_state: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
         output = self.intermediate(hidden_state)
         output = self.act_fn(output)
         output = self.out(output)
         output = self.dropout(output)
         output = self.layerNorm(output+input_tensor)
         return output

          