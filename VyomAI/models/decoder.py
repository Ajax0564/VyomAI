# GPT style model for  Casual language modeling
import torch
import torch.nn as nn
from typing import Optional, List
from ..layers.positional_embeddings import (
    AbsoluteEncoding,
    SinusoidalEncoding,
    RotaryEmbedding,
)
from ..layers.ffn import FeedForward
from ..layers.kv_cache import DynamicCacheOne, StaticCacheOne
from dataclasses import dataclass
from typing import Optional, List
from ..layers.attention import repeat_kv, AttentionSelfOutput
from einops import rearrange
from ..layers.positional_embeddings import apply_rotary_pos_emb

_position_embeddings = {
    "absolute": AbsoluteEncoding,
    "sinusoidal": SinusoidalEncoding,
}


@dataclass
class DecoderOutput(object):
    logits: torch.Tensor


@dataclass
class CLMOutput(object):
    """
    CLMOutput is a data class that encapsulates the output of a causal language model (CLM).

    Attributes:
        hidden_state (torch.Tensor): The hidden states of the model.
        logits (torch.Tensor): The logits produced by the model.
    """

    hidden_state: torch.Tensor
    logits: torch.Tensor
    kv_cache: List[torch.FloatTensor] = None


class DecoderAttention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.head_size = int(config.hidden_size // config.num_attention_heads)
        self.attention_bias = getattr(config, "attention_bias", True)
        self.layer_idx = layer_idx
        # self.qkv = nn.Linear(config.hidden_size,3*config.hidden_size)
        self.query = nn.Linear(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.key = nn.Linear(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.value = nn.Linear(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.out = AttentionSelfOutput(config=config, bias=self.attention_bias)
        self.num_attention_heads = config.num_attention_heads
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash and self.layer_idx == 0:  # avoid to print m times:
            print("WARNING: Flash Attention requires PyTorch >= 2.0")

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        kv_cache: List[torch.FloatTensor] = None,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)`
            Attention_mask: torch.Tensor of shape (batch,1, seq_len, seqlen)`
            freqs: Positional freqs in case of RoPE embedding
            use_cace: Optional to use kvCache
            start_pos: in case of kvCache to get store kv-cache at start_pos
        return:
               hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)

        """
        q = self.query(hidden_state)
        k = self.key(hidden_state)
        v = self.value(hidden_state)
        # transform it into batch_size x no_of_heads x seqlen x head_dim for Multihead Attention
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_attention_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_attention_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_attention_heads)

        if freqs is not None:
            q, k = apply_rotary_pos_emb(q, k, freqs)  # apply RoPE if freqs is available

        if use_cache:
            if kv_cache is None:
                raise ValueError("you need to pass kv_cache")
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attention_mask
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.out(out, hidden_state), kv_cache


class DecoderAttentionGqa(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash and self.layer_idx == 0:  # avoid to print m times
            print("WARNING: Flash Attention requires PyTorch >= 2.0")
        self.layer_idx = layer_idx
        self.head_dim = int(config.hidden_size // config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", 4)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        if (
            self.num_attention_heads % self.num_key_value_heads != 0
            or self.num_attention_heads < self.num_key_value_heads
        ):
            raise ValueError(
                f"num_key_value_heads {self.num_key_value_heads }  should be less than equal num_attention_heads {config.num_attention_heads} and  multiple of num_attention_heads {config.num_attention_heads} "
            )
        self.attention_bias = getattr(config, "attention_bias", True)
        self.out = AttentionSelfOutput(config=config, bias=self.attention_bias)
        self.query = nn.Linear(
            config.hidden_size, config.hidden_size, bias=self.attention_bias
        )
        self.key = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.value = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        kv_cache: List[torch.FloatTensor] = None,
        start_pos: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)`
            Attention_mask: torch.Tensor of shape (batch,1, seq_len, seqlen)`
            freqs: Positional freqs in case of RoPE embedding
            use_cace: Optional to use kvCache
            start_pos: in case of kvCache to get store kv-cache at start_pos
        return:
               hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)

        """
        q = self.query(hidden_state)
        k = self.key(hidden_state)
        v = self.value(hidden_state)
        # transform it into batch_size x no_of_heads x seqlen x head_dim for Multihead Attention
        q = rearrange(q, "b l (h d) -> b h l d", d=self.head_dim)
        k = rearrange(k, "b l (h d) -> b h l d", d=self.head_dim)
        v = rearrange(v, "b l (h d) -> b h l d", d=self.head_dim)
        if freqs is not None:
            q, k = apply_rotary_pos_emb(q, k, freqs)  # apply RoPE if freqs is available

        if use_cache:
            if kv_cache is None:
                raise ValueError("you need to pass kv_cache")
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        k = repeat_kv(
            k, n_rep=self.num_key_value_groups
        )  # in case of GQA repeat k,v to make it same as q
        v = repeat_kv(v, n_rep=self.num_key_value_groups)

        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attention_mask
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.out(out, hidden_state), kv_cache


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
        kv_cache: List[torch.FloatTensor] = None,
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
        out, kv_cache = self.attention(
            hidden_state=hidden_state,
            attention_mask=attention_mask,
            freqs=freqs,
            use_cache=use_cache,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        out = self.feed_forward(out, hidden_state)
        return out, kv_cache


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


class DecoderModel(nn.Module):
    "Decoder model for language modeling"

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
        self.config = config

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
        use_cache: Optional[bool] = False,
        kv_cache: List[torch.FloatTensor] = None,
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
        freqs = None
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
                input_ids=input_ids, attention_mask=attention_mask, start_pos=start_pos
            )
            mask = (1.0 - mask) * torch.finfo(
                hidden_state.dtype
            ).min  # invert it to to add directly to attention score

        for layer in self.all_layer:
            hidden_state, kv_cache = layer(
                hidden_state,
                mask,
                freqs=freqs,
                use_cache=use_cache,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )
        logits = self.lm_head(hidden_state)
        return CLMOutput(hidden_state=hidden_state, logits=logits, kv_cache=kv_cache)

    def create_mask_for_decoder(
        self,
        input_ids: torch.Tensor,
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

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_len: int = 5,
        temperature: float = 1.0,
        use_cache: bool = True,
        do_sample: bool = False,
        use_static_cache: bool = False,
    ) -> torch.Tensor:

        device = input_ids.device

        all_prompt_size = [t.size()[0] for t in input_ids]

        min_prompt_len = min(all_prompt_size)
        max_prompt_len = max(all_prompt_size)

        max_len = (
            max_len + max_prompt_len
        )  # get  max len (prompt + to be generated token combined)

        pad_id = getattr(self.config, "pad_token_id", 1)
        bsz, _ = input_ids.size()
        tokens = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)

        kv_cache = None
        if use_cache:
            if use_static_cache:
                kv_cache = StaticCacheOne(
                    self.config, max_cache_len=max_len, batch_size=bsz
                )
            else:
                kv_cache = DynamicCacheOne(self.config)

        for k, t in enumerate(input_ids):
            tokens[k, : t.size()[0]] = t

        prev_pos = 0
        eos_reached = torch.tensor(
            [False] * bsz
        )  # to break generation if eos reached for all  prompt

        input_text_mask = tokens != pad_id  # mask to fill generated values into batch

        stop_tokens = torch.tensor(getattr(self.config, "eos_token_id", 2))
        for cur_pos in range(min_prompt_len, max_len):

            # Get the model output
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=tokens[:, prev_pos:cur_pos],
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    kv_cache=kv_cache,
                    start_pos=prev_pos,
                )
            kv_cache = outputs.kv_cache
            next_token_logits = outputs.logits[:, -1] / temperature

            if do_sample:
                next_token = torch.multinomial(next_token_logits, num_samples=1)
            else:
                _, next_token = torch.topk(next_token_logits, k=1, dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )

            if use_cache:
                prev_pos = cur_pos

            attention_mask = torch.cat(
                [attention_mask, torch.ones((bsz, 1), device=device)], dim=-1
            )
            if all(eos_reached):
                break
        return tokens
