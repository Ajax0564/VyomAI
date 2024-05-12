from itertools import product
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from VyomAI import EncoderDecoderModel, generate_seq2seq, StaticCache, DynamicCache
import pytest
from dataclasses import dataclass

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


def get_encoder_decoder(
    encoder_config,
    decoder_config,
    encoder: Optional[nn.Module] = None,
    encoder_pos_embedding_type: Optional[str] = "absolute",
    encoder_attention_type: str = None,
    decoder_pos_embedding_type: Optional[str] = "absolute",
    decoder_attention_type: str = None,
) -> nn.Module:
    model = EncoderDecoderModel.from_config(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        encoder=None,
        encoder_pos_embedding_type=encoder_pos_embedding_type,
        encoder_attention_type=encoder_attention_type,
        decoder_pos_embedding_type=decoder_pos_embedding_type,
        decoder_attention_type=decoder_attention_type,
    )
    _ = model.to(device)
    _ = model.eval()
    return model


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


class TestEncoderDecoderModel:
    def test_abs_none_abs_none(self, config, get_input, all_types) -> None:
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[0]
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        assert list(out.logits.shape) == [3, 17, 50265] and list(
            out.key_value_states.shape
        ) == [3, 17, 768]

    def test_abs_gqa_abs_gqa(self, config, get_input, all_types) -> None:
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[1]
        config.num_key_value_heads = 4
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        assert list(out.logits.shape) == [3, 17, 50265] and list(
            out.key_value_states.shape
        ) == [3, 17, 768]

    def test_sinusoidal_none_sinusoidal_none(
        self, config, get_input, all_types
    ) -> None:
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[2]
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        assert list(out.logits.shape) == [3, 17, 50265] and list(
            out.key_value_states.shape
        ) == [3, 17, 768]

    def test_sinusoidal_gqa_sinusoidal_gqa(self, config, get_input, all_types) -> None:
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[3]
        config.num_key_value_heads = 4
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        assert list(out.logits.shape) == [3, 17, 50265] and list(
            out.key_value_states.shape
        ) == [3, 17, 768]

    def test_rope_none_rope_none(self, config, get_input, all_types) -> None:
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[4]
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        assert list(out.logits.shape) == [3, 17, 50265] and list(
            out.key_value_states.shape
        ) == [3, 17, 768]

    def test_rope_gqa_rope_gqa(self, config, get_input, all_types) -> None:
        input_ids, attention_mask = get_input
        pos_embedding_type, attention_type = all_types[5]
        config.num_key_value_heads = 4
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
        )
        assert list(out.logits.shape) == [3, 17, 50265] and list(
            out.key_value_states.shape
        ) == [3, 17, 768]


class TestEncoderDecoderKVCache(object):
    # tokenizer.bos_token_id = 0 for roberta tokenizer

    def test_abs_none_abs_none(self, config, all_types) -> None:
        input_ids = torch.tensor([[9226, 16, 5, 1296]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        pos_embedding_type, attention_type = all_types[0]
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        encoder_output = model.get_encoder_output(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        idx = torch.tensor([0]).unsqueeze(0)
        o1 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
        )
        model._clean_cache()
        model._setup_cache(config)
        o2 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        model._clean_cache()
        model._setup_cache(config, cls=DynamicCache)
        o3 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        assert (
            torch.allclose(o1, o2) == torch.allclose(o1, o3) == torch.allclose(o2, o3)
        )

    def test_abs_gqa_abs_gqa(self, config, all_types) -> None:
        input_ids = torch.tensor([[9226, 16, 5, 1296]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        pos_embedding_type, attention_type = all_types[1]
        config.num_key_value_heads = 4  # satisfy kv cache
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        encoder_output = model.get_encoder_output(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        idx = torch.tensor([0]).unsqueeze(0)
        o1 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
        )
        model._clean_cache()
        model._setup_cache(config)
        o2 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )  # need to debug this is failing
        model._clean_cache()
        model._setup_cache(config, cls=DynamicCache)
        o3 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        assert (
            torch.allclose(o1, o3) == torch.allclose(o1, o2) == torch.allclose(o2, o3)
        )

    def test_rope_gqa_rope_gqa(self, config, all_types) -> None:
        input_ids = torch.tensor([[9226, 16, 5, 1296]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        pos_embedding_type, attention_type = all_types[5]
        config.num_key_value_heads = 4  # satisfy kv cache
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        encoder_output = model.get_encoder_output(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        idx = torch.tensor([0]).unsqueeze(0)
        o1 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
        )
        model._clean_cache()
        model._setup_cache(config)
        o2 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )  # need to debug this is failing
        model._clean_cache()
        model._setup_cache(config, cls=DynamicCache)
        o3 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        assert (
            torch.allclose(o1, o3) == torch.allclose(o1, o2) == torch.allclose(o2, o3)
        )

    def test_rope_none_rope_none(self, config, all_types) -> None:
        input_ids = torch.tensor([[9226, 16, 5, 1296]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        pos_embedding_type, attention_type = all_types[4]
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        encoder_output = model.get_encoder_output(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        idx = torch.tensor([0]).unsqueeze(0)
        o1 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
        )
        model._clean_cache()
        model._setup_cache(config)
        o2 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        model._clean_cache()
        model._setup_cache(config, cls=DynamicCache)
        o3 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        assert (
            torch.allclose(o1, o3) == torch.allclose(o1, o2) == torch.allclose(o2, o3)
        )

    def test_sinusoidal_none_sinusoidal_none(self, config, all_types) -> None:
        input_ids = torch.tensor([[9226, 16, 5, 1296]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        pos_embedding_type, attention_type = all_types[2]
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        encoder_output = model.get_encoder_output(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        idx = torch.tensor([0]).unsqueeze(0)
        o1 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
        )
        model._clean_cache()
        model._setup_cache(config)
        o2 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        model._clean_cache()
        model._setup_cache(config, cls=DynamicCache)
        o3 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        assert (
            torch.allclose(o1, o2) == torch.allclose(o1, o3) == torch.allclose(o2, o3)
        )

    def test_sinusoidal_gqa_sinusoidal_gqa(self, config, all_types) -> None:
        input_ids = torch.tensor([[9226, 16, 5, 1296]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        pos_embedding_type, attention_type = all_types[3]
        config.num_key_value_heads = 4
        model = get_encoder_decoder(
            encoder_config=config,
            decoder_config=config,
            encoder_attention_type=attention_type,
            decoder_attention_type=attention_type,
            encoder_pos_embedding_type=pos_embedding_type,
            decoder_pos_embedding_type=pos_embedding_type,
        )
        encoder_output = model.get_encoder_output(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        idx = torch.tensor([0]).unsqueeze(0)
        o1 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
        )
        model._clean_cache()
        model._setup_cache(config)
        o2 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        model._clean_cache()
        model._setup_cache(config, cls=DynamicCache)
        o3 = generate_seq2seq(
            model=model,
            encoder_attention_mask=attention_mask,
            encoder_output=encoder_output,
            decoder_start=idx,
            use_cache=True,
        )
        assert (
            torch.allclose(o1, o3) == torch.allclose(o1, o2) == torch.allclose(o2, o3)
        )
