from VyomAI import Vit
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass
class Config:
    hidden_size: int = 768
    num_attention_heads: int = 12
    image_size: Tuple[int] = (224, 224)
    patch_size: Tuple[int] = (16, 16)
    num_channels: int = 3
    num_hidden_layers: int = 4
    hidden_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    hidden_act: str = "gelu"


def test_vit():
    img = torch.rand((4, 3, 224, 224))
    config = Config()
    model = Vit(config)
    out = model(img)
    assert list(out.logits.shape) == [4, 197, 768]
