import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..layers.attention import (
    EncoderDecoderAttention,
    EncoderDecoderAttentionGqa,
    DecoderAttention,
    DecoderAttentionGqa,
)
from ..layers.positional_embeddings import (
    AbsoluteEncoding,
    SinusoidalEncoding,
    RelativePositionalEncoding,
)
from ..layers.ffn import FeedForward
from ..layers.kv_cache import DynamicCache
from .encoder import EncoderModel

from dataclasses import dataclass

_position_embeddings = {
    "absolute": AbsoluteEncoding,
    "sinusoidal": SinusoidalEncoding,
}  #'relative':RelativePositionalEncoding


@dataclass
class Seq2SeqOutput(object):
    key_value_states: torch.Tensor
    logits: torch.Tensor
    past_key_value: Optional[object]


class Seq2SeqDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.attention = (
            DecoderAttentionGqa(config, layer_idx=layer_idx)
            if getattr(config, "attention", None) == "gqa"
            else DecoderAttention(config, layer_idx=layer_idx)
        )
        if (
            getattr(config, "attention", None) == "gqa" and layer_idx == 0
        ):  # avoid to print m times
            print("Using GQA Attention")
        self.cross_attention = (
            EncoderDecoderAttentionGqa(config, layer_idx=layer_idx)
            if getattr(config, "cross_attention", None) == "gqa"
            else EncoderDecoderAttention(config, layer_idx=layer_idx)
        )
        if (
            getattr(config, "corss_attention", None) == "gqa" and layer_idx == 0
        ):  # avoid to print m times
            print("Using GQA in Cross Attention")
        self.feed_forward = FeedForward(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[object] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        out, past_key_value = self.attention(
            hidden_state=hidden_state, attention_mask=attention_mask
        )
        out, past_key_value = self.cross_attention(
            hidden_state=out,
            key_value_states=encoder_hidden_state,
            attention_mask=encoder_attention_mask,
        )

        out = self.feed_forward(out, out)
        return out, past_key_value


class Embeddings(nn.Module):
    def __init__(self, config, pos_embedding: Optional[str] = "absolute") -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        if not getattr(config, "is_rope", None) and _position_embeddings.get(
            pos_embedding, None
        ):
            self.position_embeddings = _position_embeddings.get(pos_embedding)(config)
        elif not getattr(config, "is_rope", None):
            self.position_embeddings = AbsoluteEncoding(config)
        else:
            self.position_embeddings = None
        if self.position_embeddings is None:
            print(
                "Decoder Ignoring Sinusoidal or Absolute position embeddings because RoPE is enable"
            )
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            out = out + self.position_embeddings(input_ids)
        out = self.layerNorm(out)
        out = self.dropout(out)
        return out


class LMHead(nn.Module):
    """Head for masked language modelling"""

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.vocab = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.vocab.bias = self.bias

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_state)
        x = nn.GELU()(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.vocab(x)

        return x


class Seq2SeqDecoderModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embeddings = Embeddings(
            config, pos_embedding=config.position_embedding_type
        )
        self.all_layer = nn.ModuleList(
            [
                Seq2SeqDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[object] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        hidden_state = self.embeddings(input_ids=input_ids)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # SDPA requires attn_mask dtype to be bool or to match query dtype
        attention_mask = attention_mask.bool()

        for layer in self.all_layer:
            hidden_state, past_key_value = layer(
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                encoder_hidden_state=encoder_hidden_state,
                encoder_attention_mask=encoder_attention_mask.bool(),
            )
        return hidden_state, past_key_value

    @classmethod
    def from_config(cls, config) -> nn.Module:
        return cls(config)


class EncoderDecoderModel(nn.Module):

    def __init__(self, config, encoder: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else EncoderModel(config=config)
        self.decoder = Seq2SeqDecoderModel(config=config)
        self.lm_head = LMHead(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_output: Optional[torch.FloatTensor] = None,
    ):
        if encoder_output is None:
            encoder_output = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.bool()
        decoder_output, past_key_value = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_state=encoder_output,
            encoder_attention_mask=attention_mask,
        )
        decoder_output = self.lm_head(decoder_output)
        return Seq2SeqOutput(
            key_value_states=encoder_output,
            logits=decoder_output,
            past_key_value=past_key_value,
        )

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def get_decoder(self) -> Seq2SeqDecoderModel:
        return self.decoder

    def _setup_cache(self):
        for layer in self.decoder.all_layer:
            layer.attention.past_key_value = DynamicCache()
            layer.cross_attention.past_key_value = DynamicCache()

    def _reset_cache(self):
        for layer in self.decoder.all_layer:
            layer.attention.past_key_value = None
            layer.cross_attention.past_key_value = None

    @classmethod
    def from_config(cls, config) -> nn.Module:
        return cls(config)
