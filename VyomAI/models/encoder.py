import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..layers.attention import EncoderAttention, EncoderAttentionGqa
from ..layers.positional_embeddings import (
    AbsoluteEncoding,
    SinusoidalEncoding,
    RotaryEmbedding,
)
from ..layers.ffn import FeedForward
from dataclasses import dataclass

_position_embeddings = {
    "absolute": AbsoluteEncoding,
    "sinusoidal": SinusoidalEncoding,
}  #'relative':RelativePositionalEncoding


@dataclass
class EncoderOutput(object):
    logits: torch.Tensor


@dataclass
class MLMOutput(object):
    hidden_state: torch.Tensor
    logits: torch.Tensor


class EncoderLayer(nn.Module):
    "encoder layer for encoder model"

    def __init__(self, config, layer_idx: int, attention_type: str = None) -> None:
        super().__init__()
        self.attention = (
            EncoderAttentionGqa(config, layer_idx=layer_idx)
            if attention_type == "gqa"
            else EncoderAttention(config, layer_idx=layer_idx)
        )
        if attention_type == "gqa" and layer_idx == 0:  # avoid to print m times
            print("Encoder Using GQA Attention")
        self.feed_forward = FeedForward(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: torch.Tensor of shape (batch, seq_len,embd_dim)
            attention_mask: torch.Tensor of shape (batch,1,seqlen,seqlen)
            freqs: positionl information to use in RoPE
        return:
               hidden_state: torch.Tensor of shape (batch, seq_len, embed_dim) of last layer

        """
        out = self.attention(
            hidden_state=hidden_state, attention_mask=attention_mask, freqs=freqs
        )
        out = self.feed_forward(out, hidden_state)
        return out


class LMHead(nn.Module):
    """Head for masked language modelling"""

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-6)
        )

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_state)
        x = nn.GELU()(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class EncoderModel(nn.Module):

    def __init__(
        self,
        config,
        pos_embedding_type: Optional[str] = "absolute",
        attention_type: str = None,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        if _position_embeddings.get(pos_embedding_type, None) is not None:
            self.position_embeddings = _position_embeddings.get(pos_embedding_type)(
                config
            )
        else:
            self.position_embeddings = None
        if pos_embedding_type == "rope":
            self.emb_freq = RotaryEmbedding(config)(config.max_position_embeddings)
            print(
                "Encoder Ignoring sinusoidal or absolute position embeddings because rope,is enable"
            )
        self.all_layer = nn.ModuleList(
            [
                EncoderLayer(config, layer_idx, attention_type)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / torch.sqrt(2 * len(self.all_layer))
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / torch.sqrt(2 * len(self.all_layer))
            )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: torch.LongTensor of shape (batch, seq_len) for encoder`
            attention_mask: torch.Tensor of shape (batch,seqlen) for encoder
        return:
               logits: torch.Tensor of shape (batch, seq_len, embed_dim) of last layer

        """
        bsz, seqlen = input_ids.shape
        hidden_state = self.word_embeddings(input_ids)
        freqs = None
        if self.position_embeddings is not None:
            pos_info = self.position_embeddings(seqlen)[:, :seqlen, :].to(
                input_ids.device
            )
            hidden_state = hidden_state + pos_info
        else:
            freqs = self.emb_freq[:, :seqlen].to(input_ids.device)

        if attention_mask is None:
            encoder_batch_size, encoder_sequence_length = input_ids.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            attention_mask = torch.ones(encoder_hidden_shape, device=input_ids.device)

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).type_as(hidden_state)
        attention_mask = (1.0 - attention_mask) * torch.finfo(
            hidden_state.dtype
        ).min  # invert it to to add directly to attention score

        for layer in self.all_layer:
            hidden_state = layer(hidden_state, attention_mask, freqs)
        return EncoderOutput(hidden_state)

    @classmethod
    def from_config(
        cls,
        config,
        pos_embedding_type: Optional[str] = "absolute",
        attention_type: str = None,
    ) -> nn.Module:
        return cls(config, pos_embedding_type, attention_type)


class EncoderForMaskedLM(nn.Module):

    def __init__(
        self,
        config,
        pos_embedding_type: Optional[str] = "absolute",
        attention_type: str = None,
    ) -> None:
        super().__init__()
        self.encoder = EncoderModel(
            config, pos_embedding_type=pos_embedding_type, attention_type=attention_type
        )
        self.lm_head = LMHead(config=config)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_ids: torch.LongTensor of shape (batch, seq_len) for encoder`
            attention_mask: torch.Tensor of shape (batch,seqlen) for encoder
        return:
               hidden_state: torch.Tensor of shape (batch, seq_len, embed_dim) of last layer
               logits: torch.Tensor of shape (batch,seqlen, vocab_size)

        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(out.logits)
        return MLMOutput(hidden_state=out.logits, logits=logits)

    @classmethod
    def from_config(
        cls,
        config,
        pos_embedding_type: Optional[str] = "absolute",
        attention_type: str = None,
    ) -> nn.Module:
        return cls(config, pos_embedding_type, attention_type)
