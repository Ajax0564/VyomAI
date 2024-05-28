# GPT style model for  Casual language modeling
import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..layers.attention import DecoderAttention, DecoderAttentionGqa
from ..layers.positional_embeddings import (
    AbsoluteEncoding,
    SinusoidalEncoding,
    RotaryEmbedding,
)
from ..layers.ffn import FeedForward
from ..layers.kv_cache import DynamicCache, StaticCache
from dataclasses import dataclass

_position_embeddings = {
    "absolute": AbsoluteEncoding,
    "sinusoidal": SinusoidalEncoding,
}


@dataclass
class DecoderOutput(object):
    logits: torch.Tensor


class DecoderLayer(nn.Module):
    "decoder layer for decoder model"

    def __init__(
        self, config, layer_idx: int, attention_type: Optional[str] = None
    ) -> None:
        super().__init__()
        self.attention = (
            DecoderAttentionGqa(config, layer_idx=layer_idx)
            if attention_type == "gqa"
            else DecoderAttention(config, layer_idx=layer_idx)
        )
        if attention_type == "gqa" and layer_idx == 0:  # avoid to print m times
            print("Decoder Using GQA Attention")
        self.feed_forward = FeedForward(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: torch.Tensor of shape (batch, seq_len,embd_dim)
            attention_mask: torch.Tensor of shape (batch,1,seqlen,seqlen)
            freqs:Optional positionl information to use in RoPE
            start_pos: optional False  seq_no for kv-cache
        return:
               hidden_state: torch.Tensor of shape (batch, seq_len, embed_dim) of last layer

        """
        out = self.attention(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            freqs=freqs,
            use_cache=use_cache,
            start_pos=start_pos,
        )
        out = self.feed_forward(out, hidden_state)
        return out


class LMHead(nn.Module):
    """Head for language modelling"""

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


class VisionLanguageDecoderModel(nn.Module):
    "VisionLanguageModel for language modeling"

    def __init__(
        self,
        config,
        pos_embedding_type: Optional[str] = "absolute",
        attention_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.is_gqa = True if attention_type == "gqa" else False
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
                "Decoder Ignoring sinusoidal or absolute position embeddings because rope,is enable"
            )
        self.all_layer = nn.ModuleList(
            [
                DecoderLayer(config, layer_idx, attention_type)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.lm_head = LMHead(config=config)

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
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_state: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: torch.LongTensor of shape (batch, seq_len) for encoder`
            attention_mask: torch.Tensor of shape (batch,seqlen) for encoder
            use_cache: Optional False to enable kv-cache for generation
            start:pos: Optional 0, use it in case of kv-cache
        return:
               hidden_state: torch.Tensor of shape (batch, seq_len, embed_dim) of last layer
               logits: torch.Tensor of shape (batch,seqlen, vocab_size)

        """
        _bsz, seqlen = input_ids.shape
        hidden_state = self.word_embeddings(input_ids)
        if start_pos == 0:
            hidden_state = torch.cat(
                [encoder_hidden_state.unsqueeze(1), hidden_state], dim=1
            )
            if attention_mask is not None:
                extra_token = torch.full((_bsz, 1), 1).to(input_ids.device)
                attention_mask = torch.cat([extra_token, attention_mask], dim=1)
        freqs = None
        _bsz, seqlen, _ = hidden_state.shape
        if self.position_embeddings is not None:
            pos_info = self.position_embeddings(start_pos + seqlen)[
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
                hidden_state=hidden_state,
                attention_mask=attention_mask,
                start_pos=start_pos,
            )
            mask = (1.0 - mask) * torch.finfo(
                hidden_state.dtype
            ).min  # invert it to to add directly to attention score

        for layer in self.all_layer:
            hidden_state = layer(
                hidden_state,
                mask,
                freqs=freqs,
                use_cache=use_cache,
                start_pos=start_pos,
            )
        logits = self.lm_head(hidden_state)
        return DecoderOutput(logits=logits)

    def create_mask_for_decoder(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: torch.LongTensor of shape (batch, seq_len) for encoder`
            attention_mask: torch.Tensor of shape (batch,seqlen) for encoder
        return:
               attention_mask: torch.Tensor of shape (batch,1,seqlen,seqlen) for encoder
        """
        device = hidden_state.device
        batch_size, seq_length, _ = hidden_state.shape
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
        )  # # this is mainly if batch contains <PAD> tokens. stop casual procees before <PAD>
        return extended_attention_mask

    @classmethod
    def from_config(
        cls,
        config,
        pos_embedding_type: Optional[str] = "absolute",
        attention_type: Optional[str] = None,
    ) -> nn.Module:
        return cls(config, pos_embedding_type, attention_type)


class VisionLanguageModel(nn.Module):

    def __init__(
        self,
        config,
        encoder,
        pos_embedding_type: Optional[str] = "absolute",
        attention_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = VisionLanguageDecoderModel(
            config=config,
            pos_embedding_type=pos_embedding_type,
            attention_type=attention_type,
        )

    def forward(
        self,
        pixel_values: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_output: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        start_pos: Optional[int] = 0,
    ) -> DecoderOutput:
        if encoder_output is None:
            encoder_output = self.encoder(pixel_values=pixel_values).logits[
                :, 0, :
            ]  # get cls token information

        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_state=encoder_output,
            use_cache=use_cache,
            start_pos=start_pos,
        )

        return decoder_output

    def _setup_cache(self, config, cls: Optional[object] = StaticCache) -> None:
        """setup kv-cache hooks for every self-attention layer"""
        for layer in self.decoder.all_layer:
            layer.attention.cache = cls(config, is_gqa=self.is_gqa)

    def _clean_cache(self) -> None:
        """destroy kv-cache hooks for every self-attention layer"""
        for layer in self.decoder.all_layer:
            layer.attention.cache = None
