from itertools import product
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from VyomAI import EncoderModel, EncoderForMaskedLM
import pytest
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    max_position_embeddings: int = 514
    num_hidden_layers: int = 4
    vocab_size: int = 50265
    hidden_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-05
    hidden_act: str = "gelu"


@pytest.fixture(scope="function")  # run only once
def get_input() -> List[torch.Tensor]:
    input_ids = torch.tensor(
        [
            [0, 2387, 766, 16, 181, 967, 46035, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 12196, 16, 110, 766, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 37111, 1137, 162, 110, 766, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    ).to(device)

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    ).to(device)
    return input_ids, attention_mask
