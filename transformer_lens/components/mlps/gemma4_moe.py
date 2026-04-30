"""Gemma 4 26B_A4B dual-branch MoE FFN component.

Each layer has TWO parallel branches (both active for every token):
  - Dense shared branch: RMSNorm → GatedMLP (hidden=moe_dense_hidden_dim) → RMSNorm
  - Sparse MoE branch:   RMSNorm → MoE (128 experts, dim=moe_expert_dim, top-k) → RMSNorm
Both outputs are summed, then passed through a shared post-norm.

The component takes the RAW residual stream (no outer ln2 pre-applied). It sets
dual_branch_moe=True so TransformerBlock bypasses its outer ln2 and apply_mlp call.

Router design (from gemma-deepmind _moe.py:MoERagged.__call__):
  x → RMSNorm(no_scale) → * rsqrt(d_model) → * router_scale[D] → @ router_W[D,E] → softmax → top-k
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class _RMSNorm(nn.Module):
    """Simple RMSNorm. with_scale=False matches the MoE router_norm (no learned scale)."""

    def __init__(self, d_model: int, with_scale: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model)) if with_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        if self.scale is not None:
            x = x * self.scale
        return x


class _GatedMLP(nn.Module):
    """Gated MLP (GELU-gated, SwiGLU style). Matches JAX FeedForward.

    Uses nn.Parameter (not nn.Linear) so state dict keys are 'W_gate', 'W_in', 'W_out'
    without a '.weight' suffix — consistent with TL convention.
    Weights are stored [d_model, hidden_dim] (TL: [in, out]).
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.W_gate = nn.Parameter(torch.empty(d_model, hidden_dim))
        self.W_in = nn.Parameter(torch.empty(d_model, hidden_dim))
        self.W_out = nn.Parameter(torch.empty(hidden_dim, d_model))
        nn.init.normal_(self.W_gate, std=0.02)
        nn.init.normal_(self.W_in, std=0.02)
        nn.init.normal_(self.W_out, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(x @ self.W_gate, approximate="tanh")   # [B, S, H]
        return (gate * (x @ self.W_in)) @ self.W_out          # [B, S, D]


class Gemma4DualBranchFFN(nn.Module):
    """Dual-branch FFN for Gemma 4 26B_A4B MoE layers.

    Receives the raw post-attention residual stream. Runs a dense shared branch and a
    sparse MoE branch in parallel, each with independent pre-norm and post-norm.
    Sums outputs and applies a shared combined post-norm.

    Sets dual_branch_moe=True so TransformerBlock.forward() bypasses its outer ln2.
    """

    dual_branch_moe: bool = True

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        assert cfg.num_experts is not None, "num_experts required"
        assert cfg.experts_per_token is not None, "experts_per_token required"
        assert cfg.moe_expert_dim is not None, "moe_expert_dim required"
        assert cfg.moe_dense_hidden_dim is not None, "moe_dense_hidden_dim required"

        D = cfg.d_model
        H_dense = cfg.moe_dense_hidden_dim
        E = cfg.num_experts
        H_expert = cfg.moe_expert_dim
        K = cfg.experts_per_token
        eps = cfg.eps

        self.D = D
        self.E = E
        self.H_expert = H_expert
        self.K = K

        # Dense shared branch (mlp2 in JAX checkpoint)
        self.ln_dense_pre = _RMSNorm(D, with_scale=True, eps=eps)
        self.dense_mlp = _GatedMLP(D, H_dense)
        self.ln_dense_post = _RMSNorm(D, with_scale=True, eps=eps)

        # MoE branch pre/post norms
        self.ln_moe_pre = _RMSNorm(D, with_scale=True, eps=eps)
        self.ln_moe_post = _RMSNorm(D, with_scale=True, eps=eps)

        # Combined post-norm (applied after dense_out + moe_out sum)
        self.ln_combined = _RMSNorm(D, with_scale=True, eps=eps)

        # MoE router: no-scale RMSNorm + learned per-feature scale + linear projection
        self.router_norm = _RMSNorm(D, with_scale=False, eps=eps)
        self.router_scale = nn.Parameter(torch.ones(D))    # [D]
        self.router_W = nn.Parameter(torch.empty(D, E))    # [D, E] TL convention [in, out]
        nn.init.normal_(self.router_W, std=0.02)

        # Expert weights — batched over all experts (not a ModuleList), matching JAX layout.
        # expert_W_gateup: fused gate+up, [E, D, 2*H_expert]
        # expert_W_down:   down projection, [E, H_expert, D]
        self.expert_W_gateup = nn.Parameter(torch.empty(E, D, 2 * H_expert))
        self.expert_W_down = nn.Parameter(torch.empty(E, H_expert, D))
        nn.init.normal_(self.expert_W_gateup, std=0.02)
        nn.init.normal_(self.expert_W_down, std=0.02)

        # Per-expert output scaling (learned scalar per expert)
        self.per_expert_scale = nn.Parameter(torch.ones(E))

        # Hook points
        self.hook_router_logits = HookPoint()   # [B, S, E]  — raw router scores
        self.hook_expert_weights = HookPoint()  # [B, S, K]  — renorm'd routing weights
        self.hook_expert_indices = HookPoint()  # [B, S, K]  — selected expert indices
        self.hook_dense_out = HookPoint()       # [B, S, D]  — post dense post-norm
        self.hook_moe_out = HookPoint()         # [B, S, D]  — post moe post-norm
        self.hook_combined_out = HookPoint()    # [B, S, D]  — final output

    def _route(self, x: torch.Tensor):
        """Compute top-K routing weights and expert indices.

        Matches JAX MoERagged.__call__: router_norm → rsqrt(D) → router_scale → router_W.
        Routing is done on the raw x, not the pre-normed moe_in.

        Returns:
            weights: [B, S, K] renormalized routing weights (sum to 1 per token)
            indices: [B, S, K] selected expert indices
        """
        h = self.router_norm(x)                                        # [B, S, D]
        h = h * (self.D ** -0.5) * self.router_scale                  # [B, S, D]
        logits = h @ self.router_W                                     # [B, S, E]
        logits = self.hook_router_logits(logits)

        probs = torch.softmax(logits.float(), dim=-1)                  # [B, S, E]
        weights, indices = torch.topk(probs, self.K, dim=-1)           # [B, S, K]

        # Renormalize: weights / sum(selected weights) so they sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights = weights.to(x.dtype)

        weights = self.hook_expert_weights(weights)
        indices = self.hook_expert_indices(indices)
        return weights, indices

    def _run_experts(
        self,
        x: torch.Tensor,        # [B, S, D] — pre-normed moe input
        weights: torch.Tensor,  # [B, S, K]
        indices: torch.Tensor,  # [B, S, K]
    ) -> torch.Tensor:          # [B, S, D]
        """Run top-K experts and combine weighted outputs.

        Uses a loop over K choices (K=8 for 26B). Per-token expert lookup with
        batched matmuls. Correct for mechinterp; see plan appendix for a vectorized
        alternative if throughput matters.
        """
        B, S, D = x.shape
        out = torch.zeros(B, S, D, dtype=x.dtype, device=x.device)

        for k in range(self.K):
            expert_idx = indices[..., k]                               # [B, S]
            w = weights[..., k:k+1]                                    # [B, S, 1]

            # Gather weights for selected expert — [B, S, D, 2H] and [B, S, H, D]
            W_gateup = self.expert_W_gateup[expert_idx]               # [B, S, D, 2H]
            W_down = self.expert_W_down[expert_idx]                   # [B, S, H, D]

            # Expert forward: gated MLP with GELU
            h = torch.einsum("bsd,bsdh->bsh", x, W_gateup)           # [B, S, 2H]
            gate, up = h.chunk(2, dim=-1)                             # [B, S, H] each
            h = F.gelu(gate, approximate="tanh") * up                 # [B, S, H]

            # Per-expert output scale (learned scalar)
            scale = self.per_expert_scale[expert_idx]                 # [B, S]
            h = h * scale.unsqueeze(-1)                               # [B, S, H]

            # Down projection and weighted accumulation
            expert_out = torch.einsum("bsh,bshd->bsd", h, W_down)    # [B, S, D]
            out = out + w * expert_out

        return out

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Dense shared branch (mlp2)
        dense = self.ln_dense_pre(x)
        dense = self.dense_mlp(dense)
        dense = self.ln_dense_post(dense)
        dense = self.hook_dense_out(dense)

        # Sparse MoE branch — router uses raw x, experts use pre-normed moe_in
        moe_in = self.ln_moe_pre(x)
        weights, indices = self._route(x)
        moe = self._run_experts(moe_in, weights, indices)
        moe = self.ln_moe_post(moe)
        moe = self.hook_moe_out(moe)

        # Combine and apply shared post-norm
        combined = self.ln_combined(dense + moe)
        return self.hook_combined_out(combined)
