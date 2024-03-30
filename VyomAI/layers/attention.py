import torch
import torch.nn as nn
from einops import rearrange, reduce
from typing import Optional,Tuple
from ffn import SelfOutput


    
class EncoderAttention(nn.Module):
    def __init__(self,config) -> None:
        super.__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.head_size = int(config.hidden_size/config.num_attention_heads)
        self.qkv = nn.Linear(self.hidden_size,3*self.hidden_size)
        self.output = SelfOutput(config=config)
        self.num_attention_heads = config.num_attention_heads
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention requires PyTorch >= 2.0")

    def forward(self,hidden_state: torch.Tensor,mask: torch.Tensor) -> torch.Tensor:
        q,k,v = self.qkv(hidden_state).chunk(3, dim = -1) #b X l X d dim =-1 or 2
        #place holder for RoPe operation
        q = rearrange(q,"b l (h d) -> b h l d",h=self.num_attention_heads)
        k = rearrange(k,"b l (h d) -> b h l d",h=self.num_attention_heads)
        v = rearrange(v,"b l (h d) -> b h l d",h=self.num_attention_heads)
        out = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask, is_causal=False)
        out = rearrange(out,"b h l d -> b l (h d)")
        out = self.output(out,hidden_state)
        return out



