# mainly 2way to do one keep it into the model init like llama https://github.com/meta-llama/llama/blob/main/llama/model.py
# or keep it  (model-forward) enable it on inference
# inspired from Transformer cache_utils

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple
import torch


class DynamicCache:
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = (
            0  # Used in `generate` to keep tally of how many tokens the cache has seen
        )

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self) -> Generator[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self) -> int:
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache first iteration
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    # def reorder_cache(self, beam_idx: torch.LongTensor):
    #     """Reorders the cache for beam search, given the selected beam indices."""
    #     for layer_idx in range(len(self.key_cache)):
    #         device = self.key_cache[layer_idx].device
    #         self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
    #         device = self.value_cache[layer_idx].device
    #         self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    # def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
    #     """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
    #     legacy_cache = ()
    #     for layer_idx in range(len(self)):
    #         legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
    #     return legacy_cache

    # @classmethod
    # def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
    #     """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
    #     cache = cls()
    #     if past_key_values is not None:
    #         for layer_idx in range(len(past_key_values)):
    #             key_states, value_states = past_key_values[layer_idx]
    #             cache.update(key_states, value_states, layer_idx)
    #     return cache
