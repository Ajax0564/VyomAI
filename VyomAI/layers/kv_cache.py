# mainly 2way to do one keep it into the model init like llama https://github.com/meta-llama/llama/blob/main/llama/model.py
# every attention layer have its own kv-cache storage
# or keep all attention layer kv-cache into single storage like Huggingface Transformer


from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple
import torch


class DynamicCache:
    """
    A cache that grows dynamically as more tokens are generated.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, config, is_gqa: Optional[bool] = False) -> None:
        self.key_cache: torch.Tensor = None
        self.value_cache: torch.Tensor = None
        self._seen_tokens = False

    def __len__(self) -> int:
        """This value corresponds
        to the number generated tokens so for

        """
        if self.key_cache is None:
            return 0

        return self.key_cache.shape[2]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
         Updates the cache with the new `key_states` and `value_states` for the layer.

        Parameters:
            key_states: torch.Tensor
                The new key states to cache.
            value_states: torch.Tensor
                The new value states to cache.
            start_pos: int to return k,v value at specific position when kv-cache enable

        Return:
        A tuple containing the updated key and value states."""

        # Update the cache first iteration
        if self.key_cache is None:
            self._seen_tokens = True
            self.key_cache = key_states.clone()
            self.value_cache = value_states.clone()
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=-2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=-2)

        return self.key_cache, self.value_cache

    def get(self) -> Tuple[torch.Tensor]:
        """Return:
        A tuple containing the updated key and value states."""

        if self._seen_tokens:
            return self.key_cache, self.value_cache
        else:
            raise ValueError("there is no token available in kv-cache")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if self.key_cache is None:
            return 0
        return self.key_cache.shape[2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None


class StaticCache:
    """
    A cache that is size fixed suitable for torch.compile

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self, config, is_gqa: Optional[bool] = False) -> None:
        self.head_size = int(config.hidden_size // config.num_attention_heads)
        self.heads = None
        if is_gqa:
            self.heads = getattr(config, "num_key_value_heads", None)
            if self.heads is None:
                raise ValueError(
                    "you are using is_gqa=True and config.num_key_value_heads is not available"
                )
        if self.heads is None:

            self.heads = config.num_attention_heads
        self.key_cache: torch.Tensor = torch.zeros(
            1,
            self.heads,
            config.max_position_embeddings,
            self.head_size,
        )
        self.value_cache: torch.Tensor = torch.zeros(
            1,
            self.heads,
            config.max_position_embeddings,
            self.head_size,
        )
        self._seen_tokens = False

    def update(
        self, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0
    ) -> Tuple[torch.Tensor]:
        """
         Updates the cache with the new `key_states` and `value_states` for the layer.

        Parameters:
            key_states: torch.Tensor
                The new key states to cache.
            value_states: torch.Tensor
                The new value states to cache.
            start_pos: int to return k,v value at specific position when kv-cache enable

        Return:
        A tuple containing the updated key and value states."""
        self._seen_tokens = True
        bsz, head, seqlen, _ = k.shape
        self.first_update_len = seqlen
        if seqlen > self.key_cache.size()[2]:
            raise ValueError(
                f"{k.shape} is more than init k_cache size {self.key_cache}"
            )
        assert bsz == 1, "Only support batch size 1"

        self.key_cache = self.key_cache.to(k)
        self.value_cache = self.value_cache.to(v)

        self.key_cache[:bsz, :, start_pos : start_pos + seqlen] = k
        self.value_cache[:bsz, :, start_pos : start_pos + seqlen] = v

        k = self.key_cache[:bsz, :, : start_pos + seqlen]
        v = self.value_cache[:bsz, :, : start_pos + seqlen]

        return k, v

    def get(self) -> Tuple[torch.Tensor]:
        """
        Return:
        A tuple containing the updated key and value states."""
        if self._seen_tokens:
            k = self.key_cache[:, :, : self.first_update_len]
            v = self.value_cache[:, :, : self.first_update_len]

            return k, v
        else:
            raise ValueError("there is no token available in kv-cache")

    def __len__(self) -> int:
        if self._seen_tokens == False:
            return 0
        """ This value corresponds
        to the number generated tokens so for
        """
        return self.key_cache.shape[2]
