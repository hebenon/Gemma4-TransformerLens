"""Key-Value cache entry for TransformerLens.

This module defines the TransformerLensKeyValueCacheEntry class which stores
past keys and values for a single transformer layer.
"""

from dataclasses import dataclass
from typing import Union

import torch
from jaxtyping import Float

from transformer_lens.config.TransformerLensConfig import TransformerLensConfig


@dataclass
class TransformerLensKeyValueCacheEntry:
    past_keys: Float[torch.Tensor, "batch pos_so_far n_heads d_head"]
    past_values: Float[torch.Tensor, "batch pos_so_far n_heads d_head"]
    frozen: bool = False

    @classmethod
    def init_cache_entry(
        cls,
        cfg: TransformerLensConfig,
        device: Union[torch.device, str, None],
        batch_size: int = 1,
        block_index: int = 0,
    ):
        n_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else cfg.n_heads
        attn_types = getattr(cfg, "attn_types", None)
        d_head_global = getattr(cfg, "d_head_global", None)
        is_global = (
            d_head_global is not None
            and attn_types is not None
            and block_index < len(attn_types)
            and attn_types[block_index] == "global"
        )
        d_head = d_head_global if is_global else cfg.d_head
        return cls(
            past_keys=torch.empty(
                (batch_size, 0, n_heads, d_head), device=device, dtype=cfg.dtype
            ),
            past_values=torch.empty(
                (batch_size, 0, n_heads, d_head), device=device, dtype=cfg.dtype
            ),
        )

    def append(
        self,
        new_keys: Float[torch.Tensor, "batch new_tokens n_heads d_head"],
        new_values: Float[torch.Tensor, "batch new_tokens n_heads d_head"],
    ):
        updated_keys: Float[
            torch.Tensor, "batch pos_so_far_plus_new_tokens n_heads d_head"
        ] = torch.cat([self.past_keys, new_keys], dim=1)
        updated_values: Float[
            torch.Tensor, "batch pos_so_far_plus_new_tokens n_heads d_head"
        ] = torch.cat([self.past_values, new_values], dim=1)
        if not self.frozen:
            self.past_keys = updated_keys
            self.past_values = updated_values
        return updated_keys, updated_values
