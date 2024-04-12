import torch
import torch.nn as nn
from einops import rearrange, reduce,repeat
from typing import Optional,Tuple
from .ffn import SelfOutput
from .positional_embeddings import rotate_half,apply_rotary_pos_emb,RotaryEmbedding

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

def repeat_kv_einops(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = repeat(hidden_states,"batch num_key_value_heads slen head_dim -> batch num_key_value_heads n_rep slen head_dim",n_rep = n_rep)  #hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    return rearrange(hidden_states,"batch num_key_value_heads n_rep slen head_dim -> batch (num_key_value_heads n_rep) slen head_dim")


    
class EncoderAttention(nn.Module):
    def __init__(self,config,layer_idx: int) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.head_size = int(config.hidden_size//config.num_attention_heads)
        self.attention_bias = getattr(config,'attention_bias',False)
        self.layer_idx = layer_idx
        # self.qkv = nn.Linear(config.hidden_size,3*config.hidden_size)
        self.q = nn.Linear(config.hidden_size, config.hidden_size,bias=self.attention_bias)
        self.k = nn.Linear(config.hidden_size, config.hidden_size,bias=self.attention_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size,bias=self.attention_bias)
        self.output = SelfOutput(config=config)
        self.num_attention_heads = config.num_attention_heads
        self.rotary_emb = RotaryEmbedding(config=config) if getattr(config,'is_rope',None) else None
        if self.rotary_emb!=None and self.layer_idx==0: #avoid to print m times:
            print("Using Rotatry Embedding")
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and self.layer_idx==0: #avoid to print m times:
            print("WARNING: Flash Attention requires PyTorch >= 2.0")

    def forward(self,hidden_state: torch.Tensor,mask: torch.Tensor) -> torch.Tensor:
        q = self.q(hidden_state)
        k = self.k(hidden_state)
        v = self.v(hidden_state)
        # q,k,v = self.qkv(hidden_state).chunk(3, dim = -1) #b X l X d dim =-1 or 2
        #place holder for RoPe operation
        q = rearrange(q,"b l (h d) -> b h l d",h=self.num_attention_heads)
        k = rearrange(k,"b l (h d) -> b h l d",h=self.num_attention_heads)
        v = rearrange(v,"b l (h d) -> b h l d",h=self.num_attention_heads)
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(v)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, None)
        out = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask, is_causal=False)
        out = rearrange(out,"b h l d -> b l (h d)")
        out = self.output(out,hidden_state)
        return out
    
class EncoderAttentionGqa(nn.Module):
    def __init__(self, config,layer_idx: int) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and self.layer_idx==0: #avoid to print m times
            print("WARNING: Flash Attention requires PyTorch >= 2.0")
        self.layer_idx  = layer_idx 
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
        self.attention_bias = getattr(config,'attention_bias',False)
        self.output = SelfOutput(config=config)
        self.q = nn.Linear(config.hidden_size, config.hidden_size,bias=self.attention_bias)
        self.k = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,bias=self.attention_bias)
        self.v = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim,bias=self.attention_bias)
        self.rotary_emb = RotaryEmbedding(config=config) if getattr(config,'is_rope',None) else None
        if self.rotary_emb!=None and self.layer_idx==0 : #avoid to print m times
            print("Using Rotatry Embedding")

    def forward(self,hidden_state: torch.Tensor,mask: torch.tensor) -> torch.Tensor:
        q = self.q(hidden_state)
        k = self.k(hidden_state)
        v = self.v(hidden_state)
        q = rearrange(q,"b l (h d) -> b h l d",d=self.head_dim)
        k = rearrange(k,"b l (h d) -> b h l d",d=self.head_dim)
        v = rearrange(v,"b l (h d) -> b h l d",d=self.head_dim)
        k = repeat_kv(k,n_rep=self.num_key_value_groups)
        v = repeat_kv(v,n_rep=self.num_key_value_groups)
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(v)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, None)

        out = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask, is_causal=False)
        out = rearrange(out,"b h l d -> b l (h d)")
        out = self.output(out,hidden_state)
        return out





