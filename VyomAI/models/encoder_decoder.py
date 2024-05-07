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
    RotaryEmbedding,
)
from ..layers.ffn import FeedForward
from ..layers.kv_cache import DynamicCache, StaticCache
from .encoder import EncoderModel

from dataclasses import dataclass

_position_embeddings = {
    "absolute": AbsoluteEncoding,
    "sinusoidal": SinusoidalEncoding,
}


@dataclass
class Seq2SeqOutput(object):
    logits: torch.Tensor
    key_value_states: torch.Tensor


class Seq2SeqDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int = 0, attention_type: str = None) -> None:
        super().__init__()
        self.attention = (
            DecoderAttentionGqa(config, layer_idx=layer_idx)
            if attention_type == "gqa"
            else DecoderAttention(config, layer_idx=layer_idx)
        )
        if attention_type == "gqa" and layer_idx == 0:  # avoid to print m times
            print("Decoder Using GQA Attention")
        self.cross_attention = (
            EncoderDecoderAttentionGqa(config, layer_idx=layer_idx)
            if attention_type == "gqa" == "gqa"
            else EncoderDecoderAttention(config, layer_idx=layer_idx)
        )
        if attention_type == "gqa" and layer_idx == 0:  # avoid to print m times
            print("Using GQA in Cross Attention")
        self.feed_forward = FeedForward(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        out = self.attention(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            use_cache=use_cache,
            start_pos=start_pos,
        )
        out = self.cross_attention(
            hidden_state=out,
            key_value_states=encoder_hidden_state,
            attention_mask=encoder_attention_mask,
        )

        out = self.feed_forward(out, out)
        return out


class LMHead(nn.Module):
    """Head for masked language modelling"""

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.vocab = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.vocab.bias = self.bias

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_state)
        x = nn.GELU()(x)
        x = self.layerNorm(x)

        # project back to size of vocabulary with bias
        x = self.vocab(x)

        return x


class Seq2SeqDecoderModel(nn.Module):
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
                Seq2SeqDecoderLayer(config, layer_idx)
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
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_state: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        _bsz, seqlen = input_ids.shape
        hidden_state = self.word_embeddings(input_ids)
        freqs = None
        if self.position_embeddings is not None:
            pos_info = pos_info = self.position_embeddings(start_pos + seqlen)[
                :, start_pos : start_pos + seqlen, :
            ].to(input_ids.device)
            hidden_state = hidden_state + pos_info
        else:
            freqs = self.emb_freq[:, start_pos : start_pos + seqlen].to(
                input_ids.device
            )
        mask = None
        if seqlen > 1:
            mask = self.create_mask_for_decoder(
                input_ids=input_ids, attention_mask=attention_mask, start_pos=start_pos
            )
            mask = (1.0 - mask) * torch.finfo(
                hidden_state.dtype
            ).min  # invert it to to add directly to attention score

        for layer in self.all_layer:
            hidden_state = layer(
                hidden_state=hidden_state,
                attention_mask=mask,
                encoder_hidden_state=encoder_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                start_pos=start_pos,
            )
        return hidden_state

    def create_mask_for_decoder(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        device = input_ids.device
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = (
                torch.ones(seq_length + start_pos).repeat(batch_size, 1).to(device)
            )
        seq_ids = torch.arange(seq_length).to(device)
        causal_mask = (
            seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
            <= seq_ids[None, :, None]
        )  # 1x1xl repeat bxlxl compare to 1xlx1

        causal_mask = causal_mask.to(attention_mask.dtype)

        if start_pos > 0:  # correct the attention mask  for kv-cache operation
            causal_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, seq_length, start_pos),
                        device=device,
                        dtype=causal_mask.dtype,
                    ),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = (
            causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        )  # this is mainly if batch contains <PAD> tokens. stop casual procees before <PAD>
        return extended_attention_mask

    @classmethod
    def from_config(cls, config) -> nn.Module:
        return cls(config)


class EncoderDecoderModel(nn.Module):

    def __init__(
        self,
        encoder_config,
        decoder_config,
        encoder: Optional[nn.Module] = None,
        encoder_pos_embedding_type: Optional[str] = "absolute",
        encoder_attention_type: str = None,
        decoder_pos_embedding_type: Optional[str] = "absolute",
        decoder_attention_type: str = None,
    ) -> None:
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else EncoderModel(
                config=encoder_config,
                pos_embedding_type=encoder_pos_embedding_type,
                attention_type=encoder_attention_type,
            )
        )
        self.decoder = Seq2SeqDecoderModel(
            config=decoder_config,
            pos_embedding_type=decoder_pos_embedding_type,
            attention_type=decoder_attention_type,
        )
        self.lm_head = LMHead(config=decoder_config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_output: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        start_pos: Optional[int] = 0,
    ):

        if encoder_output is None:
            encoder_output = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        if attention_mask is None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_output.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            attention_mask = torch.ones(
                encoder_hidden_shape, device=encoder_output.device
            ).to(encoder_output.device)

        encoder_attention_mask = (
            attention_mask.unsqueeze(1).unsqueeze(2).type_as(encoder_output)
        )
        encoder_attention_mask = (1.0 - encoder_attention_mask) * torch.finfo(
            encoder_output.dtype
        ).min

        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_state=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            start_pos=start_pos,
        )
        decoder_output = self.lm_head(decoder_output)
        return Seq2SeqOutput(key_value_states=encoder_output, logits=decoder_output)

    def get_encoder(self) -> nn.Module:
        return self.encoder

    def get_encoder_output(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> object:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)

    def get_decoder(self) -> Seq2SeqDecoderModel:
        return self.decoder

    def _setup_cache(self, config, cls: Optional[object] = StaticCache) -> None:
        for layer in self.decoder.all_layer:
            layer.attention.cache = cls(config)
            layer.cross_attention.cache = cls(config)

    def _clean_cache(self) -> None:
        for layer in self.decoder.all_layer:
            layer.attention.cache = None
            layer.cross_attention.cache = None

    @classmethod
    def from_config(
        cls,
        encoder_config,
        decoder_config,
        encoder: Optional[nn.Module] = None,
        encoder_pos_embedding_type: Optional[str] = "absolute",
        encoder_attention_type: str = None,
        decoder_pos_embedding_type: Optional[str] = "absolute",
        decoder_attention_type: str = None,
    ) -> nn.Module:
        return cls(
            encoder_config,
            decoder_config,
            encoder,
            encoder_pos_embedding_type,
            encoder_attention_type,
            decoder_pos_embedding_type,
            decoder_attention_type,
        )
