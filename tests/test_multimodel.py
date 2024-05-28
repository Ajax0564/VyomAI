from VyomAI import Vit, VisionLanguageModel
from dataclasses import dataclass
from itertools import product
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from VyomAI import DecoderModel, StaticCache, DynamicCache
from VyomAI import generate
import pytest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    hidden_size: int = 768
    num_attention_heads: int = 12
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


def get_vision_language_model(
    encoder,
    config,
    pos_embedding_type: Optional[str] = "absolute",
    attention_type: str = None,
) -> nn.Module:
    model = VisionLanguageModel(
        config=config,
        encoder=encoder,
        pos_embedding_type=pos_embedding_type,
        attention_type=attention_type,
    )
    _ = model.to(device)
    _ = model.eval()
    return model


@pytest.fixture(scope="function")
def config() -> object:
    return Config()


@pytest.fixture(scope="function")  # run only once
def all_types() -> List[Tuple]:
    """('absolute', None),
    ('absolute', 'gqa'),
    ('sinusoidal', None),
    ('sinusoidal', 'gqa'),
    ('rope', None),
    ('rope', 'gqa')"""
    pos_embeddins = ["absolute", "sinusoidal", "rope"]
    attention_type = [None, "gqa"]
    out = product(pos_embeddins, attention_type)

    return [x for x in out]


@dataclass
class VitConfig:
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


@pytest.fixture(scope="function")  # run only once
def get_vit():
    config = VitConfig()
    model = Vit(config)
    _ = model.to(device)
    _ = model.eval()
    return model


@pytest.fixture(scope="function")  # run only once
def all_types() -> List[Tuple]:
    """('absolute', None),
    ('absolute', 'gqa'),
    ('sinusoidal', None),
    ('sinusoidal', 'gqa'),
    ('rope', None),
    ('rope', 'gqa')"""
    pos_embeddins = ["absolute", "sinusoidal", "rope"]
    attention_type = [None, "gqa"]
    out = product(pos_embeddins, attention_type)

    return [x for x in out]


class TestVisionLanguageDecoderCLM(object):

    def test_absolute(self, get_vit, get_input, config, all_types):
        img = torch.rand((3, 3, 224, 224)).to(device)
        encoder = get_vit
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[0]
        model = get_vision_language_model(
            encoder=encoder,
            config=config,
            pos_embedding_type=pos_embedding_type,
            attention_type=attention_type,
        )
        with torch.no_grad():
            assert list(
                model(
                    pixel_values=img,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                ).logits.shape
            ) == [3, 18, 50265]

    def test_absolute_gqa(self, get_vit, get_input, config, all_types):
        img = torch.rand((3, 3, 224, 224)).to(device)
        encoder = get_vit
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[1]
        config.num_key_value_heads = 4
        model = get_vision_language_model(
            encoder=encoder,
            config=config,
            pos_embedding_type=pos_embedding_type,
            attention_type=attention_type,
        )
        with torch.no_grad():
            assert list(
                model(
                    pixel_values=img,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                ).logits.shape
            ) == [3, 18, 50265]

    def test_sinusoidal_none(self, get_vit, get_input, config, all_types):
        img = torch.rand((3, 3, 224, 224)).to(device)
        encoder = get_vit
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[2]
        model = get_vision_language_model(
            encoder=encoder,
            config=config,
            pos_embedding_type=pos_embedding_type,
            attention_type=attention_type,
        )
        with torch.no_grad():
            assert list(
                model(
                    pixel_values=img,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                ).logits.shape
            ) == [3, 18, 50265]

    def test_sinusoidal_gqa(self, get_vit, get_input, config, all_types):
        img = torch.rand((3, 3, 224, 224)).to(device)
        encoder = get_vit
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[3]
        config.num_key_value_heads = 4
        model = get_vision_language_model(
            encoder=encoder,
            config=config,
            pos_embedding_type=pos_embedding_type,
            attention_type=attention_type,
        )
        with torch.no_grad():
            assert list(
                model(
                    pixel_values=img,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                ).logits.shape
            ) == [3, 18, 50265]

    def test_rope_none(self, get_vit, get_input, config, all_types):
        img = torch.rand((3, 3, 224, 224)).to(device)
        encoder = get_vit
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[4]
        model = get_vision_language_model(
            encoder=encoder,
            config=config,
            pos_embedding_type=pos_embedding_type,
            attention_type=attention_type,
        )
        with torch.no_grad():
            assert list(
                model(
                    pixel_values=img,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                ).logits.shape
            ) == [3, 18, 50265]

    def test_rope_gqa(self, get_vit, get_input, config, all_types):
        img = torch.rand((3, 3, 224, 224)).to(device)
        encoder = get_vit
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[5]
        config.num_key_value_heads = 4
        model = get_vision_language_model(
            encoder=encoder,
            config=config,
            pos_embedding_type=pos_embedding_type,
            attention_type=attention_type,
        )
        with torch.no_grad():
            assert list(
                model(
                    pixel_values=img,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=attention_mask,
                ).logits.shape
            ) == [3, 18, 50265]
