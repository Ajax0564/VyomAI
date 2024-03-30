import torch
import torch.nn as nn
from einops import rearrange, reduce
from typing import Optional,Tuple
from .ffn import SelfOutput

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    
class EncoderAttention(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.head_size = int(config.hidden_size//config.num_attention_heads)
        self.qkv = nn.Linear(config.hidden_size,3*config.hidden_size)
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
    
class EncoderAttentionGqa(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention requires PyTorch >= 2.0")
        self.head_dim = int(config.hidden_size//config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config,'num_key_value_heads',4)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        if (
            self.num_attention_heads % self.num_key_value_heads != 0
            or self.num_attention_heads < self.num_key_value_heads
        ):
            raise ValueError(
                f"num_key_value_heads {self.num_key_value_heads }  should be less than equal num_attention_heads {config.num_attention_heads} and  multiple of num_attention_heads {config.num_attention_heads} "
            )
        self.output = SelfOutput(config=config)
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim)

    def forward(self,hidden_state: torch.Tensor,mask: torch.tensor) -> torch.Tensor:
        q = self.q(hidden_state)
        k = self.k(hidden_state)
        v = self.v(hidden_state)
        q = rearrange(q,"b l (h d) -> b h l d",d=self.head_dim)
        k = rearrange(k,"b l (h d) -> b h l d",d=self.head_dim)
        v = rearrange(v,"b l (h d) -> b h l d",d=self.head_dim)
        k = repeat_kv(k,n_rep=self.num_key_value_groups)
        v = repeat_kv(v,n_rep=self.num_key_value_groups)
        out = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask, is_causal=False)
        out = rearrange(out,"b h l d -> b l (h d)")
        out = self.output(out,hidden_state)
        return out






