import torch
import torch.nn as nn
from typing import Optional, Tuple
from ..layers.attention import VisionAttention
from ..layers.positional_embeddings import VitAbsoluteEncoding
from ..layers.ffn import FeedForward
from dataclasses import dataclass
from einops import rearrange, repeat


_position_embeddings = {
    "absolute": VitAbsoluteEncoding
    # "sinusoidal": SinusoidalEncoding,
}


@dataclass
class EncoderOutput(object):
    logits: torch.Tensor


class EncoderLayer(nn.Module):
    "encoder layer for encoder model"

    def __init__(
        self, config, layer_idx: int, attention_type: Optional[str] = None
    ) -> None:
        super().__init__()
        self.attention = VisionAttention(config, layer_idx=layer_idx)

        self.feed_forward = FeedForward(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: torch.Tensor of shape (batch, seq_len,embd_dim)
            attention_mask: torch.Tensor of shape (batch,1,1,seqlen)
            freqs: positionl information to use in RoPE
        return:
               hidden_state: torch.Tensor of shape (batch, seq_len, embed_dim) of last layer

        """
        out = self.attention(
            hidden_state=hidden_state, attention_mask=attention_mask, freqs=freqs
        )
        out = self.feed_forward(out, hidden_state)
        return out


class Vit(nn.Module):

    def __init__(
        self,
        config,
        pos_embedding_type: Optional[str] = "absolute",
    ) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (
            self.image_size[1] // self.patch_size[1]
        )
        if _position_embeddings.get(pos_embedding_type, None) is not None:
            self.position_embeddings = _position_embeddings.get(pos_embedding_type)(
                config
            )
        else:
            self.position_embeddings = None

        self.all_layer = nn.ModuleList(
            [
                EncoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.pixel_seq = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patch_dim = config.num_channels * self.patch_size[0] * self.patch_size[1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_dim))

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
        self, pixel_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: torch.LongTensor of shape (batch, seq_len) for encoder`
            attention_mask: torch.Tensor of shape (batch,seqlen) for encoder
        return:
               logits: torch.Tensor of shape (batch, seq_len, embed_dim) of last layer

        """

        img_seq = self.pixel_seq(pixel_values)
        hidden_state = rearrange(img_seq, "b d c1 c2 -> b (c1 c2) d")
        bsz, seqlen, _ = hidden_state.shape

        # rearrange(image, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p1, p2=p2)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=bsz).to(
            hidden_state.device
        )
        hidden_state = torch.cat((cls_tokens, hidden_state), dim=1)

        freqs = None
        if self.position_embeddings is not None:
            pos_info = self.position_embeddings(hidden_state)
            hidden_state = hidden_state + pos_info
        else:
            freqs = self.emb_freq[:, :seqlen].to(pixel_values.device)

        if attention_mask is None:
            encoder_batch_size, encoder_sequence_length, _ = hidden_state.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            attention_mask = torch.ones(
                encoder_hidden_shape, device=pixel_values.device
            )

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
    ) -> nn.Module:
        return cls(config, pos_embedding_type)


if "__name__" == "__main__":
    model = Vit()
    image = torch.rand((4, 3, 224, 224))
    out = model(image)
    print(out.shape)
