import torch
import torch.nn as nn
from typing import Optional,Tuple
from ..layers.attention import EncoderAttention
from ..layers.positional_embeddings import AbsoluteEncoding,SinusoidalEncoding,RelativePositionalEncoding
from ..layers.ffn import FeedForward
from dataclasses import dataclass

_position_embeddings = {'absolute':AbsoluteEncoding,
                        'sinusoidal':SinusoidalEncoding,
                        } #'relative':RelativePositionalEncoding

@dataclass
class EncoderOutput(object):
    logits: torch.Tensor

@dataclass
class MLMOutput(object):
    hidden_state: torch.Tensor
    logits: torch.Tensor

class EncoderLayer(nn.Module):
    def __init__(self,config ) -> None:
        super().__init__()
        self.attention =  EncoderAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self,hidden_state: torch.Tensor,mask: torch.Tensor) -> torch.Tensor:
        out = self.attention(hidden_state=hidden_state,mask=mask)
        out = self.feed_forward(out,hidden_state)
        return out
    
class Embeddings(nn.Module):
    def __init__(self, config,pos_embedding: Optional[str] = 'absolute') -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        if _position_embeddings.get(pos_embedding,None):
            self.position_embeddings = _position_embeddings.get(pos_embedding)(config)
        else:
            self.position_embeddings = AbsoluteEncoding(config)
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self,input_ids: torch.Tensor) -> torch.Tensor:
        out = self.word_embeddings(input_ids)+self.position_embeddings(input_ids)
        out  = self.layerNorm(out)
        out  = self.dropout(out)
        return out

class LMHead(nn.Module):
    '''Head for masked language modelling'''
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self,hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_state)
        x = nn.GELU()(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x



class EncoderModel(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.embeddings = Embeddings(config,pos_embedding=config.position_embedding_type)
        self.all_layer = nn.ModuleList([EncoderLayer(config) 
                                     for _ in range(config.num_hidden_layers)])
        
    def forward(self,input_ids: torch.Tensor,attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_state = self.embeddings(input_ids=input_ids)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #SDPA requires attn_mask dtype to be bool or to match query dtype
        attention_mask = attention_mask.bool()

        for layer in self.all_layer:
            hidden_state = layer(hidden_state,attention_mask)
        return EncoderOutput(hidden_state)
    
    @classmethod
    def from_config(cls,config) -> nn.Module:
        return cls(config)
    

   
    
class EncoderForMaskedLM(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.encoder = EncoderModel(config)
        self.lm_head = LMHead(config=config)
    
    def forward(self,input_ids: torch.Tensor,attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        logits = self.lm_head(out.logits)
        return MLMOutput(hidden_state=out.logits,logits=logits)
    
    @classmethod
    def from_config(cls,config) -> nn.Module:
        return cls(config)
    
