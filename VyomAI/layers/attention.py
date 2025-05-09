import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from typing import Optional, Tuple, List
from .positional_embeddings import apply_rotary_pos_emb, RotaryEmbedding


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv_einops(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = repeat(
        hidden_states,
        "batch num_key_value_heads slen head_dim -> batch num_key_value_heads n_rep slen head_dim",
        n_rep=n_rep,
    )  # hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    return rearrange(
        hidden_states,
        "batch num_key_value_heads n_rep slen head_dim -> batch (num_key_value_heads n_rep) slen head_dim",
    )


class AttentionSelfOutput(nn.Module):
    def __init__(
        self, config, bias: Optional[bool] = True, out_features: Optional[int] = None
    ):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size if out_features is None else out_features,
            bias=bias,
        )
        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=getattr(config, "layer_norm_eps", 1e-6)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.FloatTensor of shape (batch, seq_len, embed_dim)`
            input_tensor: torch.FloatTensor of shape (batch, seq_len, embed_dim)`

        return:
               hidden_states: torch.FloatTensor of shape (batch, seq_len, embed_dim)

        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class EncoderAttention(nn.Module):
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

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)`
            attention_mask: torch.Tensor of shape (batch,1, seq_len, seqlen)`
            freqs: Positional freqs in case of RoPE embedding
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
            q, k = apply_rotary_pos_emb(
                q,
                k,
                freqs,
            )  # apply RoPE if freqs is available

        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attention_mask, is_causal=False
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out(out, hidden_state)


class EncoderAttentionGqa(nn.Module):
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
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)`
            attention_mask: torch.Tensor of shape (batch,1, seq_len, seqlen)`
            freqs: Positional freqs in case of RoPE embedding
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
            q, k = apply_rotary_pos_emb(
                q,
                k,
                freqs,
            )  # apply RoPE if freqs is available

        k = repeat_kv(
            k, n_rep=self.num_key_value_groups
        )  # in case of GQA repeat k,v to make it same as q
        v = repeat_kv(v, n_rep=self.num_key_value_groups)
        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attention_mask, is_causal=False
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.out(out, hidden_state)


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
            cache = getattr(self, "cache", None)
            if cache is None:
                raise ValueError(
                    "you need to setup cache for every attention layer with model.setup_cache()"
                )
            k, v = cache.update(k, v, start_pos)

        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attention_mask
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.out(out, hidden_state)


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
            cache = getattr(self, "cache", None)
            if cache is None:
                raise ValueError(
                    "you need to setup cache for every attention layer with model._setup_cache()"
                )
            k, v = cache.update(k, v, start_pos)

        k = repeat_kv(
            k, n_rep=self.num_key_value_groups
        )  # in case of GQA repeat k,v to make it same as q
        v = repeat_kv(v, n_rep=self.num_key_value_groups)

        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attention_mask
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.out(out, hidden_state)


class EncoderDecoderAttention(nn.Module):
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
        encoder_hidden_state: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)`
            encoder_hidden_state: torch.Tensor of shape (batch, seq_len, embed_dim)` form encoder in case of seq2seq
            encoder_attention_mask: torch.Tensor of shape (batch,1, seq_len, seqlen)`
            freqs: Positional freqs in case of RoPE embedding
            use_cace: Optional to use kvCache
            start_pos: in case of kvCache to get store kv-cache at start_pos
        return:
               hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)

        """
        q = self.query(hidden_state)
        # transform it into batch_size x no_of_heads x seqlen x head_dim for Multihead Attention
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_attention_heads)

        if use_cache == False:  # train
            k = self.key(encoder_hidden_state)
            v = self.value(encoder_hidden_state)
            k = rearrange(k, "b l (h d) -> b h l d", h=self.num_attention_heads)
            v = rearrange(v, "b l (h d) -> b h l d", h=self.num_attention_heads)

        if use_cache == True:  # kv-cache is enable
            cache = getattr(self, "cache", None)
            if cache is None:
                raise ValueError(
                    "use_cache is True please enable model._setup_cache() to use kv-cache"
                )
            if (
                cache is not None and len(cache) == 0
            ):  # first iteration witk kv-cache so store it encoder_hidden_state will be same for rest of the iteration
                k = self.key(encoder_hidden_state)
                v = self.value(encoder_hidden_state)
                k = rearrange(k, "b l (h d) -> b h l d", h=self.num_attention_heads)
                v = rearrange(v, "b l (h d) -> b h l d", h=self.num_attention_heads)
                # if freqs is not None:
                #     q, k = apply_rotary_pos_emb(
                #         q, k, freqs=freqs, k_freqs=k_freqs
                #     )  # apply RoPE if freqs is available
                # k will be fixed while q will vary in generation when using rope apply fixed positional freq on key in cross_attention
                k, v = self.cache.update(
                    k, v
                )  # store it will be same for all iteration
            elif cache is not None and len(cache) != 0:

                k, v = self.cache.get()

        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=encoder_attention_mask
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.out(out, hidden_state)


class EncoderDecoderAttentionGqa(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.is_casual = True
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
        encoder_hidden_state: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)`
            encoder_hidden_state: torch.Tensor of shape (batch, seq_len, embed_dim)` form encoder in case of seq2seq
            encoder_attention_mask: torch.Tensor of shape (batch,1, seq_len, seqlen)`
            freqs: Positional freqs in case of RoPE embedding
            use_cace: Optional to use kvCache
            start_pos: in case of kvCache to get store kv-cache at start_pos
        return:
               hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)

        """
        q = self.query(hidden_state)
        # transform it into batch_size x no_of_heads x seqlen x head_dim for Multihead Attention
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_attention_heads)
        if use_cache == False:  # train
            k = self.key(encoder_hidden_state)
            v = self.value(encoder_hidden_state)
            k = rearrange(k, "b l (h d) -> b h l d", d=self.head_dim)
            v = rearrange(v, "b l (h d) -> b h l d", d=self.head_dim)

        if use_cache == True:  # kv-cache is enable
            cache = getattr(self, "cache", None)
            if cache is None:
                raise ValueError(
                    "use_cache is True please enable model._setup_cache() to use kv-cache"
                )
            if (
                cache is not None and len(cache) == 0
            ):  # first iteration witk kv-cache so store it encoder_hidden_state will be same for rest of the iteration
                k = self.key(encoder_hidden_state)
                v = self.value(encoder_hidden_state)
                k = rearrange(k, "b l (h d) -> b h l d", d=self.head_dim)
                v = rearrange(v, "b l (h d) -> b h l d", d=self.head_dim)

                k, v = self.cache.update(
                    k, v
                )  # store it will be same for all iteration
            elif cache is not None and len(cache) != 0:

                k, v = self.cache.get()

        k = repeat_kv(
            k, n_rep=self.num_key_value_groups
        )  # in case of GQA repeat k,v to make it same as q
        v = repeat_kv(v, n_rep=self.num_key_value_groups)
        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=encoder_attention_mask
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")

        return self.out(out, hidden_state)


class VisionAttention(nn.Module):
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
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = AttentionSelfOutput(config=config, bias=self.attention_bias)
        self.num_attention_heads = config.num_attention_heads

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)`
            attention_mask: torch.Tensor of shape (batch,1, seq_len, seqlen)`
            freqs: Positional freqs in case of RoPE embedding
        return:
               hidden_states: torch.Tensor of shape (batch, seq_len, embed_dim)

        """

        q, k, v = self.qkv(hidden_state).chunk(3, dim=-1)
        # transform it into batch_size x no_of_heads x seqlen x head_dim for Multihead Attention
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_attention_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_attention_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_attention_heads)
        if freqs is not None:
            q, k = apply_rotary_pos_emb(
                q,
                k,
                freqs,
            )  # apply RoPE if freqs is available

        out = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attention_mask, is_causal=False
        )
        # transform it back into batch_size x seqlen x hidden_dim
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out(out, hidden_state)
