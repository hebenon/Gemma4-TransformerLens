"""PLE (Per-Layer Embedding) Precomputer for Gemma 4.

Computes per-layer conditioning vectors once before the transformer block loop.
Each decoder block then uses its slice to gate a bottleneck projection onto the
residual stream (see TransformerBlock._apply_ple).
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components.rms_norm import RMSNorm
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class PLEPrecomputer(nn.Module):
    """Compute PLE vectors for all layers in one pass before the block loop.

    Weights mirror the HF Gemma 4 model-level PLE components:
      embed_tokens_per_layer   → W_embed  [ple_vocab_size, n_layers * d_ple]
      per_layer_model_projection → W_proj [d_model, n_layers * d_ple]
      per_layer_projection_norm  → ln      RMSNorm(d_ple)

    Forward output: [batch, pos, n_layers, d_ple] — slice [:, :, i, :] per block.

    Scale factors baked in as constants (confirmed from Kaggle enumeration 2026-04-22):
      proj_scale  = 1 / sqrt(d_model)   applied to context projection
      input_scale = 1 / sqrt(2)         applied to final sum
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        assert self.cfg.d_ple is not None, "d_ple must be set when use_ple=True"
        assert self.cfg.ple_vocab_size is not None, "ple_vocab_size must be set when use_ple=True"

        n_flat = self.cfg.n_layers * self.cfg.d_ple

        # Token identity component: separate embedding table, vocab × (n_layers * d_ple)
        self.W_embed = nn.Parameter(
            torch.empty(self.cfg.ple_vocab_size, n_flat, dtype=self.cfg.dtype)
        )
        # Context projection: d_model → (n_layers * d_ple), no bias
        self.W_proj = nn.Parameter(
            torch.empty(self.cfg.d_model, n_flat, dtype=self.cfg.dtype)
        )
        # RMSNorm applied to the d_ple dimension of the context projection
        self.ln = RMSNorm(self.cfg, length=self.cfg.d_ple)

        # Hooks for component decomposition / ablation
        self.hook_token_embeds = HookPoint()  # [batch, pos, n_layers, d_ple]
        self.hook_context_proj = HookPoint()  # [batch, pos, n_layers, d_ple]

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch pos"],
        inputs_embeds: Float[torch.Tensor, "batch pos d_model"],
    ) -> Float[torch.Tensor, "batch pos n_layers d_ple"]:
        B, L = input_ids.shape
        n_layers = self.cfg.n_layers
        d_ple = self.cfg.d_ple

        # Token identity component: embedding lookup then reshape
        token_flat = self.W_embed[input_ids]                   # [B, L, n_layers * d_ple]
        token_embeds = token_flat.reshape(B, L, n_layers, d_ple)
        token_embeds = self.hook_token_embeds(token_embeds)

        # Context component: project hidden states, scale, norm
        # proj_scale is applied before RMSNorm, which normalises to unit RMS regardless,
        # so the scale is absorbed. We keep it to mirror the HF ordering exactly — W_proj
        # weights were trained with this in the computation graph.
        proj_scale = self.cfg.d_model ** -0.5
        context_flat = (inputs_embeds @ self.W_proj) * proj_scale  # [B, L, n_layers * d_ple]
        context = context_flat.reshape(B * L * n_layers, d_ple)
        context = self.ln(context).reshape(B, L, n_layers, d_ple)
        context = self.hook_context_proj(context)

        # Combine with 1/sqrt(2) scaling (input_scale)
        ple_vecs = (token_embeds + context) * (2 ** -0.5)     # [B, L, n_layers, d_ple]
        return ple_vecs
