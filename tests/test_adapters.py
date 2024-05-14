import pytest
import torch
import torch.nn as nn
from VyomAI import LoraLinear, DoraLinear
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

apply_lora = partial(LoraLinear)
apply_dora = partial(DoraLinear)


def make_input(shape):
    return torch.rand(size=shape).to(device=device)


@pytest.mark.parametrize(
    "input_shape,output_shape", [(768, 768), (768, 3072), (3072, 768)]
)
def test_lora_linear(input_shape, output_shape):
    layer = nn.Linear(in_features=input_shape, out_features=output_shape)
    lora_layer = apply_lora(layer)
    lora_layer.to(device=device)
    x = make_input((4, 32, input_shape))
    assert list(lora_layer(x).shape) == [4, 32, output_shape]


@pytest.mark.parametrize(
    "input_shape,output_shape", [(768, 768), (768, 3072), (3072, 768)]
)
def test_dora_linear(input_shape, output_shape):
    layer = nn.Linear(in_features=input_shape, out_features=output_shape)
    lora_layer = apply_dora(layer)
    lora_layer.to(device=device)
    x = make_input((4, 32, input_shape))
    assert list(lora_layer(x).shape) == [4, 32, output_shape]
