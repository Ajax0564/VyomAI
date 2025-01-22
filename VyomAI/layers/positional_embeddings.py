import torch
import torch.nn as nn
from einops import rearrange, reduce
from typing import Optional, Tuple


class AbsoluteEncoding(nn.Module):
    """
    Construct the Absolute embeddings from position.
    Args:
        config (object): Configuration object containing the following attributes:
            - max_position_embeddings (int): Maximum number of position embeddings.
            - hidden_size (int): Size of the hidden layer.
            - pad_token_id (int, optional): Padding token ID.
    Attributes:
        pos_embeddings (nn.Embedding): Embedding layer for position embeddings.
        position_ids (torch.Tensor): Tensor containing position IDs.
        max_size (int): Maximum number of position embeddings.
    Methods:
        forward(size: int) -> torch.Tensor:
            Generates position embeddings for the given size.
            Args:
                size (int): The size for which to generate position embeddings.
            Returns:
                torch.Tensor: Position embeddings of the specified size.
            Raises:
                ValueError: If the specified size exceeds the maximum number of position embeddings.
    """

    """Construct the Absolute embeddings from position"""

    def __init__(self, config) -> None:
        super().__init__()
        self.pos_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.max_size = config.max_position_embeddings

    def forward(self, size: int) -> torch.Tensor:
        if self.max_size < size:
            raise ValueError(
                f"The hidden size ({size }) is more than the config max_position_embeddings {self.max_size}"
            )
        return self.pos_embeddings(self.position_ids[:, :size])


class SinusoidalEncoding(nn.Module):
    """
    Construct the Sinusoidal embeddings from word, position, and token_type embeddings.
    This class generates sinusoidal positional encodings as described in the paper
    "Attention is All You Need" by Vaswani et al. The positional encodings have the
    same dimension as the embeddings so that the two can be summed.
    Attributes:
        positional_encoding (torch.Tensor): A tensor containing the precomputed
            sinusoidal positional encodings.
        position (torch.Tensor): A tensor containing the position indices.
        div_term (torch.Tensor): A tensor containing the scaling factors for the
            sinusoidal functions.
    Args:
        config: A configuration object that contains the following attributes:
            hidden_size (int): The size of the hidden layer. Must be an even number.
            max_position_embeddings (int): The maximum number of position embeddings.
    Raises:
        ValueError: If the hidden_size is not an even number.
    Methods:
        forward(seq_len: int) -> torch.Tensor:
            Returns the positional encodings for the given sequence length.
    """

    """Construct the Sinusoidal embeddings from word, position and token_type embeddings."""

    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % 2 != 0:
            raise ValueError(
                f"Cannot use SinusoidalEncoding with "
                "odd hidden dim got dim {config.hidden_size}"
            )
        self.positional_encoding = torch.zeros(
            1, config.max_position_embeddings, config.hidden_size
        )
        self.position = torch.arange(0, config.max_position_embeddings).unsqueeze(1)
        self.div_term = torch.exp(
            (
                torch.arange(0, config.hidden_size, 2, dtype=torch.float)
                * -(torch.log(torch.tensor(10000.0)) / config.hidden_size)
            )
        )

        self.positional_encoding[:, :, 0::2] = torch.sin(
            self.position.float() * self.div_term
        )
        self.positional_encoding[:, :, 1::2] = torch.cos(
            self.position.float() * self.div_term
        )

    def forward(self, seq_len: int) -> torch.Tensor:

        return self.positional_encoding[:, :seq_len]


class RotaryEmbedding(nn.Module):
    """
    RotaryEmbedding is a PyTorch module that implements rotary positional embeddings for attention mechanisms.
    Args:
        config (object): Configuration object containing the following attributes:
            hidden_size (int): The hidden size of the model.
            num_attention_heads (int): The number of attention heads.
    Attributes:
        inv_freq (torch.Tensor): A tensor containing the inverse frequencies for the rotary embeddings.
    Methods:
        forward(seq_len):
            Computes the rotary positional embeddings for a given sequence length.
            Args:
                seq_len (int): The length of the input sequence.
            Returns:
                torch.Tensor: A tensor containing the rotary positional embeddings with shape (1, seq_len, dim).
    """

    def __init__(self, config):
        super().__init__()
        dim = int(config.hidden_size // config.num_attention_heads)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i, j -> i j", t, self.inv_freq)

        return freqs[None, :, :]


def rotate_half(x):
    """
    Rotates half the hidden dimensions of the input tensor.

    Args:
        x (torch.Tensor): The input tensor to be rotated.

    Returns:
        torch.Tensor: The tensor with half of its hidden dimensions rotated.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers
def apply_rotary_pos_emb(q, k, freqs, unsqueeze_dim=1) -> Tuple[torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        freqs: precalculated frqs for sin cos
        only_q: bool = False for encoder decoder
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype)
    sin = emb.sin().to(dtype=q.dtype)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)

    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# To do :  Alibi


class VitAbsoluteEncoding(nn.Module):
    """
    Construct the Absolute embeddings for vision model.
    This class implements the absolute positional encoding for a Vision Transformer (ViT) model.
    It generates positional embeddings for image patches and adds them to the input image sequence.
    Attributes:
        pos_embeddings (torch.nn.Parameter): Positional embeddings for the image patches.
        num_patches (torch.Tensor): Number of patches in the image.
    Args:
        config (object): Configuration object containing image size, patch size, and number of channels.
    Methods:
        forward(img_seq: torch.Tensor) -> torch.Tensor:
            Adds positional embeddings to the input image sequence.
    """

    """Construct the Absolute embeddings for vision model"""

    def __init__(self, config) -> None:
        super().__init__()
        image_height, image_width = config.image_size
        patch_height, patch_width = config.patch_size
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = config.num_channels * patch_height * patch_width
        self.pos_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, patch_dim))
        self.register_buffer(
            "num_patches",
            torch.arange(num_patches + 1).expand((1, -1)),
            persistent=False,
        )

    def forward(self, img_seq: torch.Tensor) -> torch.Tensor:
        b, n, _ = img_seq.shape

        img_seq += self.pos_embeddings[:, : (n + 1)]
        return img_seq
