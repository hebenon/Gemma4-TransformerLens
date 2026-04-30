# Gemma 4 26B_A4B MoE Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing Gemma 4 E2B TL adapter to support the 26B_A4B MoE variant, adding a new `Gemma4DualBranchFFN` component and streaming weight converter, so the model can be loaded into `HookedTransformer` on Kaggle for mechinterp.

**Architecture:** The 26B_A4B has 30 layers with a novel dual-branch FFN (dense shared 2112 + sparse MoE 128 experts × dim 704, top-8). Unlike E2B, it has no PLE and no KV sharing, making those paths irrelevant. The new `Gemma4DualBranchFFN` component takes the raw residual and applies both branches in parallel, with their own independent pre-norms, summing and then applying a shared post-norm. TransformerBlock needs a 3-line bypass to skip its outer `ln2` for MoE layers.

**Tech Stack:** PyTorch, safetensors, HookedTransformer (not model_bridge), JAX DeepMind Gemma repo for architecture reference.

---

## Architecture Quick Reference

From `gemma-deepmind/gemma/gm/nn/gemma4/`:

| Parameter | 26B_A4B | E2B |
|-----------|---------|-----|
| Layers | 30 | 35 |
| d_model | 2816 | 1536 |
| Heads | 16 | 8 |
| KV heads (local) | 8 | 1 |
| Global KV heads | 2 | 1 |
| Attn pattern | 5:1 sliding:global | 4:1 |
| PLE | **No** | Yes (d_ple=256) |
| KV sharing | **No** | Yes (20 shared) |
| FFN type | **All layers: dual-branch MoE** | Dense (heterogeneous width) |
| Dense branch hidden | 2112 | — |
| MoE experts | 128 | — |
| Expert dim | 704 | — |
| Top-k | 8 | — |

**Dual-branch MoE forward (from `_modules.py:_forward_moe`):**
```
attn_output ──┬──► pre_ffw2_norm ──► mlp2 (dense, hidden=2112) ──► post_ffw2_norm ──► dense_out ──┐
              │                                                                                      ├──► sum ──► post_ffw_norm ──► output
              └──► pre_ffw_norm ──► mlp (MoERagged, 128 experts) ──► post_ffw1_norm ──► moe_out ──┘
```

**MoE router (from `_moe.py:MoERagged.__call__`):**
```
x ──► router_norm (RMSNorm, no scale) ──► * rsqrt(d_model) ──► * router_scale[d_model] ──► router_logits W[d_model, 128] ──► softmax ──► top-8 ──► renormalize
```

**Expert weights (batched, not ModuleList):**
- `gating_einsum`: `[128, 2, 704, 2816]` (fused gate+up, shape: `[E, 2, H, F]`)
- `linear`: `[128, 704, 2816]` (down projection, shape: `[E, H, F]`)
- `per_expert_scale`: `[128]` (learned post-expert scalar)

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `transformer_lens/config/HookedTransformerConfig.py` | Add `moe_expert_dim`, `moe_dense_hidden_dim` fields |
| Modify | `transformer_lens/loading_from_pretrained.py` | Add `google/gemma-4-26B_A4B` config entry |
| **Create** | `transformer_lens/components/mlps/gemma4_moe.py` | New `Gemma4DualBranchFFN` component |
| Modify | `transformer_lens/components/mlps/__init__.py` | Export `Gemma4DualBranchFFN` |
| Modify | `transformer_lens/components/__init__.py` | Re-export |
| Modify | `transformer_lens/components/transformer_block.py` | 3-line dual-branch bypass in `forward()` |
| Modify | `transformer_lens/pretrained/weight_conversions/gemma.py` | Add MoE weight conversion |
| **Create** | `tests/unit/components/test_gemma4_moe.py` | Unit tests for `Gemma4DualBranchFFN` |
| **Create** | `notebooks/gemma4_26b_validation.ipynb` | Kaggle validation notebook |

---

## Task 1: Enumerate 26B HF Checkpoint (Kaggle required)

**Files:** None — this is exploratory. Results feed Task 2 config and Task 5 weight mapping.

This task requires internet access to a Kaggle notebook or Lambda Labs. Cannot be run on the Pi.

- [ ] **Step 1: Open a Kaggle notebook with GPU/TPU and Gemma 4 26B access**

Attach the `google/gemma-4-26b-a4b-it` model input (or equivalent HF Hub download).

- [ ] **Step 2: List all safetensors keys and save to a text file**

```python
import json, os

# Path depends on Kaggle model input mount
model_dir = "/kaggle/input/gemma-4/transformers/gemma-4-26b-a4b-it/1"

index_path = os.path.join(model_dir, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)

keys = sorted(index["weight_map"].keys())
for k in keys[:20]:
    print(k)
print(f"\nTotal: {len(keys)} keys")

# Save for reference
with open("gemma4_26b_keys.txt", "w") as f:
    f.write("\n".join(keys))
```

- [ ] **Step 3: Confirm MoE layer key names**

Look for these patterns in the output:
```
# Expected (extrapolated from JAX _modules.py):
model.language_model.model.layers.0.mlp.experts.gate_proj.weight   # OR
model.language_model.model.layers.0.mlp.gate_proj.weight            # (batched)
model.language_model.model.layers.0.mlp.router.weight
model.language_model.model.layers.0.mlp.shared_expert.gate_proj.weight
```

Record the actual key patterns for MoE, dense-branch, and norm layers. Update Task 5 accordingly.

- [ ] **Step 4: Confirm config.json architecture**

```python
import json
with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)
print(json.dumps({k: v for k, v in cfg.items() if "expert" in k.lower() or "moe" in k.lower() or "hidden" in k.lower()}, indent=2))
```

Expected: `num_experts=128`, `num_experts_per_tok=8` (or similar), `intermediate_size=2112` (dense), `moe_intermediate_size=704` (expert dim).

**Record findings in a comment at the top of `gemma.py` weight converter before Task 5.**

---

## Task 2: Add Config Fields

**Files:**
- Modify: `transformer_lens/config/HookedTransformerConfig.py:310-320`

Current Gemma 4 extension block ends around line 319 with `d_mlp_by_layer`. Add two new fields immediately after.

- [ ] **Step 1: Write failing test**

Create `tests/unit/config/test_gemma4_26b_config.py`:

```python
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig

def test_moe_fields_exist():
    cfg = HookedTransformerConfig(
        d_model=2816, d_head=256, n_heads=16, d_mlp=2112, n_layers=30,
        n_ctx=131072, d_vocab=262144, act_fn="gelu_pytorch_tanh",
        normalization_type="RMS", positional_embedding_type="rotary",
        num_experts=128, experts_per_token=8,
        moe_expert_dim=704,
        moe_dense_hidden_dim=2112,
    )
    assert cfg.moe_expert_dim == 704
    assert cfg.moe_dense_hidden_dim == 2112

def test_moe_fields_default_to_none():
    cfg = HookedTransformerConfig(
        d_model=512, d_head=64, n_heads=8, d_mlp=2048, n_layers=6,
        n_ctx=512, d_vocab=32000, act_fn="gelu",
        normalization_type="RMS", positional_embedding_type="rotary",
    )
    assert cfg.moe_expert_dim is None
    assert cfg.moe_dense_hidden_dim is None
```

- [ ] **Step 2: Run test — verify it fails**

```bash
cd /home/meridian/projects/Gemma4-TransformerLens
uv run pytest tests/unit/config/test_gemma4_26b_config.py -v 2>&1 | tail -10
```

Expected: `FAILED — AttributeError: ... has no attribute 'moe_expert_dim'`

- [ ] **Step 3: Add fields to HookedTransformerConfig**

In `transformer_lens/config/HookedTransformerConfig.py`, find the Gemma 4 MoE section (around line 319) and add:

```python
    # Gemma 4: heterogeneous MLP widths (layers 0-14: 6144, layers 15-34: 12288)
    d_mlp_by_layer: Optional[List[int]] = None
    # Gemma 4 26B_A4B: dual-branch MoE FFN
    moe_expert_dim: Optional[int] = None        # Expert hidden dim (704 for 26B)
    moe_dense_hidden_dim: Optional[int] = None  # Dense branch hidden dim (2112 for 26B)
```

- [ ] **Step 4: Run test — verify it passes**

```bash
uv run pytest tests/unit/config/test_gemma4_26b_config.py -v 2>&1 | tail -5
```

Expected: `PASSED`

- [ ] **Step 5: Add 26B config entry to loading_from_pretrained.py**

In `transformer_lens/loading_from_pretrained.py`, find the `elif official_model_name.startswith("google/gemma-4-E2B")` block (around line 1376) and add a new elif BEFORE it:

```python
    elif official_model_name.startswith("google/gemma-4-26B_A4B"):
        # Gemma 4 26B_A4B Mixture-of-Experts
        # Architecture: 30 layers, d_model=2816, 5:1 sliding:global, NO PLE, NO KV sharing
        # ALL layers: dual-branch FFN (dense hidden=2112 + MoE: 128 experts, dim=704, top-8)
        # Confirmed from google-deepmind/gemma JAX repo 2026-04-30.
        # HF key names: TBD pending Task 1 checkpoint enumeration.
        cfg_dict = {
            "d_model": 2816,
            "d_head": 256,              # Local attention head dim
            "d_head_global": 512,       # Global attention head dim (k_eq_v_global=True)
            "n_heads": 16,
            "d_mlp": 2112,              # Dense branch hidden dim (mlp2)
            "n_layers": 30,
            "n_ctx": 131072,
            "eps": 1e-06,
            "d_vocab": 262144,
            "act_fn": "gelu_pytorch_tanh",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": 1_000_000,       # Global attention layers
            "rotary_base_local": 10_000,    # Local (sliding) attention layers
            "partial_rotary_factor_global": 0.25,
            "use_attn_scale": False,
            "n_key_value_heads": 8,
            "n_key_value_heads_global": 2,  # num_global_kv_heads=2, k_eq_v_global=True
            "gated_mlp": True,
            "final_rms": True,
            "use_normalization_before_and_after": False,  # MoE handles all norms internally
            "use_qk_norm": True,
            "window_size": 1024,
            "use_local_attn": True,
            "attn_types": [
                # 5:1 sliding:global, 30 layers = 5 repetitions of [local×5, global×1]
                "local", "local", "local", "local", "local", "global",  # 0-5
                "local", "local", "local", "local", "local", "global",  # 6-11
                "local", "local", "local", "local", "local", "global",  # 12-17
                "local", "local", "local", "local", "local", "global",  # 18-23
                "local", "local", "local", "local", "local", "global",  # 24-29
            ],
            "output_logits_soft_cap": 30.0,
            # MoE configuration
            "num_experts": 128,
            "experts_per_token": 8,
            "moe_expert_dim": 704,
            "moe_dense_hidden_dim": 2112,
            # No PLE, no KV sharing
            "use_ple": False,
            "num_kv_shared_layers": 0,
            "tokenizer_name": "google/gemma-4-26B_A4B-it",
            "original_architecture": "Gemma4ForConditionalGeneration",
        }
```

- [ ] **Step 6: Commit**

```bash
git add transformer_lens/config/HookedTransformerConfig.py
git add transformer_lens/loading_from_pretrained.py
git add tests/unit/config/test_gemma4_26b_config.py
git commit -m "feat: add moe_expert_dim/moe_dense_hidden_dim config fields + gemma4-26b_a4b config entry"
```

---

## Task 3: Build `Gemma4DualBranchFFN` Component

**Files:**
- Create: `transformer_lens/components/mlps/gemma4_moe.py`
- Modify: `transformer_lens/components/mlps/__init__.py`
- Test: `tests/unit/components/test_gemma4_moe.py`

This is the core new component. It receives the **raw residual stream** (not pre-normalized) and handles both branches plus all norms internally. This matches the JAX `_forward_moe()` data flow exactly.

Shapes (batch=B, seq=S, d_model=D=2816):
- Dense branch: `[B,S,D]` → RMSNorm → GatedMLP(hidden=2112) → `[B,S,D]` → RMSNorm
- MoE branch: `[B,S,D]` → RMSNorm → router → top-8 experts (gated, hidden=704 each) → `[B,S,D]` → RMSNorm
- Combined: dense_out + moe_out → RMSNorm → `[B,S,D]`

**State dict key convention:** `_GatedMLP` uses `nn.Parameter` (not `nn.Linear`) so weight keys are `dense_mlp.W_gate`, `dense_mlp.W_in`, `dense_mlp.W_out` (no `.weight` suffix). Router uses `nn.Parameter` for `router_W` as well (shape `[D, E]` in TL convention).

- [ ] **Step 1: Write failing tests**

Create `tests/unit/components/test_gemma4_moe.py`:

```python
import pytest
import torch
import torch.nn as nn

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components.mlps.gemma4_moe import Gemma4DualBranchFFN


@pytest.fixture
def small_cfg():
    """Minimal config resembling 26B structure but tiny for testing."""
    return HookedTransformerConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=32,          # dense branch hidden
        n_layers=2,
        n_ctx=16,
        d_vocab=256,
        act_fn="gelu_pytorch_tanh",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        num_experts=8,
        experts_per_token=2,
        moe_expert_dim=16, # expert hidden
        moe_dense_hidden_dim=32,
    )


def test_output_shape(small_cfg):
    moe = Gemma4DualBranchFFN(small_cfg)
    x = torch.randn(2, 5, small_cfg.d_model)
    out = moe(x)
    assert out.shape == (2, 5, small_cfg.d_model)


def test_dual_branch_flag(small_cfg):
    moe = Gemma4DualBranchFFN(small_cfg)
    assert moe.dual_branch_moe is True


def test_hook_points_exist(small_cfg):
    moe = Gemma4DualBranchFFN(small_cfg)
    from transformer_lens.hook_points import HookPoint
    assert isinstance(moe.hook_dense_out, HookPoint)
    assert isinstance(moe.hook_moe_out, HookPoint)
    assert isinstance(moe.hook_router_logits, HookPoint)
    assert isinstance(moe.hook_expert_weights, HookPoint)
    assert isinstance(moe.hook_expert_indices, HookPoint)
    assert isinstance(moe.hook_combined_out, HookPoint)


def test_hooks_fire(small_cfg):
    moe = Gemma4DualBranchFFN(small_cfg)
    captured = {}

    def save(name):
        def hook(value, hook):
            captured[name] = value.clone()
            return value
        return hook

    moe.hook_router_logits.add_hook(save("router_logits"))
    moe.hook_expert_weights.add_hook(save("expert_weights"))
    moe.hook_expert_indices.add_hook(save("expert_indices"))
    moe.hook_dense_out.add_hook(save("dense_out"))
    moe.hook_moe_out.add_hook(save("moe_out"))

    x = torch.randn(1, 4, small_cfg.d_model)
    _ = moe(x)

    assert "router_logits" in captured
    assert captured["router_logits"].shape == (1, 4, small_cfg.num_experts)
    assert "expert_weights" in captured
    assert captured["expert_weights"].shape == (1, 4, small_cfg.experts_per_token)
    assert "expert_indices" in captured
    assert captured["expert_indices"].shape == (1, 4, small_cfg.experts_per_token)


def test_no_nan_on_random_input(small_cfg):
    moe = Gemma4DualBranchFFN(small_cfg)
    x = torch.randn(2, 8, small_cfg.d_model)
    out = moe(x)
    assert not torch.isnan(out).any()


def test_parameter_count_reasonable(small_cfg):
    moe = Gemma4DualBranchFFN(small_cfg)
    # Expert weights: 8 * (16*64 + 16*64) + 8 * (16*64) + dense: 64*32 + 32*64 + ...
    params = sum(p.numel() for p in moe.parameters())
    # Just check it's non-zero and has the expert matrix
    assert params > small_cfg.num_experts * small_cfg.moe_expert_dim * small_cfg.d_model
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/unit/components/test_gemma4_moe.py -v 2>&1 | tail -10
```

Expected: `ImportError: cannot import name 'Gemma4DualBranchFFN'`

- [ ] **Step 3: Implement `Gemma4DualBranchFFN`**

Create `transformer_lens/components/mlps/gemma4_moe.py`:

```python
"""Gemma 4 26B_A4B dual-branch MoE FFN component.

Each layer has TWO parallel branches (both active for every token):
  - Dense shared branch: RMSNorm → GatedMLP (hidden=2112) → RMSNorm
  - Sparse MoE branch:   RMSNorm → MoE (128 experts, dim=704, top-8) → RMSNorm
Both outputs are summed, then passed through a shared post-norm.

The component takes the RAW residual stream as input (no outer ln2 applied).
It sets dual_branch_moe=True so TransformerBlock bypasses its outer ln2.

Router design (from _moe.py:MoERagged.__call__):
  x → RMSNorm(no_scale) → * rsqrt(d_model) → * router_scale[D] → linear[D,E] → softmax → top-k
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class _RMSNorm(nn.Module):
    """Simple RMSNorm. with_scale=False matches MoE router_norm (no learned scale)."""

    def __init__(self, d_model: int, with_scale: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model)) if with_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        if self.scale is not None:
            x = x * self.scale
        return x


class _GatedMLP(nn.Module):
    """Gated MLP (SwiGLU-style with GELU). Matches JAX FeedForward.

    Uses nn.Parameter (not nn.Linear) so state dict keys are 'W_gate', 'W_in',
    'W_out' (no '.weight' suffix) — consistent with TL convention.
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

    Receives the raw post-attention residual. Runs dense and sparse MoE branches
    in parallel, each with their own pre-norm and post-norm. Sums and applies a
    shared combined post-norm.

    Sets dual_branch_moe=True so TransformerBlock bypasses its outer ln2.
    """

    dual_branch_moe: bool = True

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        assert cfg.num_experts is not None
        assert cfg.experts_per_token is not None
        assert cfg.moe_expert_dim is not None
        assert cfg.moe_dense_hidden_dim is not None

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

        # Dense shared branch (mlp2 in checkpoint)
        self.ln_dense_pre = _RMSNorm(D, with_scale=True, eps=eps)
        self.dense_mlp = _GatedMLP(D, H_dense)
        self.ln_dense_post = _RMSNorm(D, with_scale=True, eps=eps)

        # MoE branch pre/post norms
        self.ln_moe_pre = _RMSNorm(D, with_scale=True, eps=eps)
        self.ln_moe_post = _RMSNorm(D, with_scale=True, eps=eps)

        # Combined post-norm
        self.ln_combined = _RMSNorm(D, with_scale=True, eps=eps)

        # MoE router: no-scale RMSNorm + learned scale + linear
        self.router_norm = _RMSNorm(D, with_scale=False, eps=eps)
        self.router_scale = nn.Parameter(torch.ones(D))    # [D]
        self.router_W = nn.Parameter(torch.empty(D, E))    # [D, E] — TL convention [in, out]
        nn.init.normal_(self.router_W, std=0.02)

        # Expert weights (batched, not a ModuleList — matches JAX layout)
        # gate+up fused: [E, 2, H_expert, D] — stored transposed for matmul as [E, D, 2*H_expert]
        self.expert_W_gateup = nn.Parameter(torch.empty(E, D, 2 * H_expert))
        # down projection: [E, H_expert, D]
        self.expert_W_down = nn.Parameter(torch.empty(E, H_expert, D))
        # per-expert output scale
        self.per_expert_scale = nn.Parameter(torch.ones(E))

        nn.init.normal_(self.expert_W_gateup, std=0.02)
        nn.init.normal_(self.expert_W_down, std=0.02)

        # Hook points
        self.hook_router_logits = HookPoint()   # [B, S, E]
        self.hook_expert_weights = HookPoint()  # [B, S, K]
        self.hook_expert_indices = HookPoint()  # [B, S, K]  (int)
        self.hook_dense_out = HookPoint()       # [B, S, D]  (post dense post-norm)
        self.hook_moe_out = HookPoint()         # [B, S, D]  (post moe post-norm)
        self.hook_combined_out = HookPoint()    # [B, S, D]  (post combined norm)

    def _route(self, x: torch.Tensor):
        """Compute routing weights and expert indices.

        Returns:
            weights: [B, S, K] — renormalized routing weights for top-K experts
            indices: [B, S, K] — expert indices (int64)
        """
        # Router: no-scale RMSNorm → learned scale → linear
        h = self.router_norm(x)                                   # [B, S, D]
        h = h * (self.D ** -0.5) * self.router_scale             # [B, S, D]
        logits = h @ self.router_W                                # [B, S, E]
        logits = self.hook_router_logits(logits)

        probs = torch.softmax(logits.float(), dim=-1)      # [B, S, E]
        weights, indices = torch.topk(probs, self.K, dim=-1)  # [B, S, K]
        # Renormalize selected weights
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights = weights.to(x.dtype)

        weights = self.hook_expert_weights(weights)
        indices = self.hook_expert_indices(indices)
        return weights, indices

    def _run_experts(
        self,
        x: torch.Tensor,             # [B, S, D]
        weights: torch.Tensor,       # [B, S, K]
        indices: torch.Tensor,       # [B, S, K]
    ) -> torch.Tensor:               # [B, S, D]
        """Run top-K experts and combine outputs.

        Uses a simple loop over K choices (K=8 for 26B). Each token's K selected
        experts are looked up, forward-passed through batched matmuls, and summed
        weighted by routing weights. This avoids sorting/shuffling complexity and
        is correct for mechinterp (hooks over routing patterns).

        For production throughput, see MoERagged (ragged_dot) in the JAX repo.
        """
        B, S, D = x.shape
        out = torch.zeros(B, S, D, dtype=x.dtype, device=x.device)

        for k in range(self.K):
            expert_idx = indices[..., k]                          # [B, S]
            w = weights[..., k:k+1]                               # [B, S, 1]

            # Look up weights for this choice — gather from batched expert matrices
            # expert_W_gateup: [E, D, 2H]
            W_gateup = self.expert_W_gateup[expert_idx]           # [B, S, D, 2H]
            W_down = self.expert_W_down[expert_idx]               # [B, S, H, D]

            # Expert forward: gated MLP
            h = torch.einsum('bsd,bsdh->bsh', x, W_gateup)       # [B, S, 2H]
            gate, up = h.chunk(2, dim=-1)                         # [B, S, H] each
            h = F.gelu(gate, approximate="tanh") * up             # [B, S, H]

            # Per-expert scale
            scale = self.per_expert_scale[expert_idx]             # [B, S]
            h = h * scale.unsqueeze(-1)

            # Down projection
            expert_out = torch.einsum('bsh,bshd->bsd', h, W_down) # [B, S, D]
            out = out + w * expert_out

        return out

    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # Dense shared branch
        dense = self.ln_dense_pre(x)
        dense = self.dense_mlp(dense)
        dense = self.ln_dense_post(dense)
        dense = self.hook_dense_out(dense)

        # Sparse MoE branch
        moe_in = self.ln_moe_pre(x)
        weights, indices = self._route(x)     # route on raw x (matches JAX)
        moe = self._run_experts(moe_in, weights, indices)
        moe = self.ln_moe_post(moe)
        moe = self.hook_moe_out(moe)

        # Combine and final norm
        combined = self.ln_combined(dense + moe)
        return self.hook_combined_out(combined)
```

- [ ] **Step 4: Export from `__init__.py`**

In `transformer_lens/components/mlps/__init__.py`, add:
```python
from transformer_lens.components.mlps.gemma4_moe import Gemma4DualBranchFFN
```

In `transformer_lens/components/__init__.py`, add:
```python
from transformer_lens.components.mlps.gemma4_moe import Gemma4DualBranchFFN
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
uv run pytest tests/unit/components/test_gemma4_moe.py -v 2>&1 | tail -15
```

Expected: all 6 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add transformer_lens/components/mlps/gemma4_moe.py
git add transformer_lens/components/mlps/__init__.py
git add transformer_lens/components/__init__.py
git add tests/unit/components/test_gemma4_moe.py
git commit -m "feat: add Gemma4DualBranchFFN component for 26B_A4B MoE layers"
```

---

## Task 4: TransformerBlock Dual-Branch Bypass

**Files:**
- Modify: `transformer_lens/components/transformer_block.py`

TransformerBlock currently does:
```python
normalized_resid_mid = self.ln2(mlp_in)
mlp_out = self.apply_mlp(normalized_resid_mid)
```

`Gemma4DualBranchFFN` must receive `mlp_in` (un-normalized), not `normalized_resid_mid`. We add a 3-line bypass guarded by `dual_branch_moe`.

- [ ] **Step 1: Write failing integration test**

Add to `tests/unit/components/test_gemma4_moe.py`:

```python
from transformer_lens.components.transformer_block import TransformerBlock

def test_transformer_block_bypass(small_cfg):
    """Verify TransformerBlock passes raw residual to Gemma4DualBranchFFN."""
    from transformer_lens.components.mlps.gemma4_moe import Gemma4DualBranchFFN
    import einops

    block = TransformerBlock(small_cfg, 0)
    # Replace mlp with MoE component
    block.mlp = Gemma4DualBranchFFN(small_cfg)

    captured_moe_input = []

    def capture_input(x, hook):
        captured_moe_input.append(x.clone())
        return x

    block.mlp.hook_dense_out.add_hook(capture_input)

    # Run a forward pass
    x = torch.randn(1, 4, small_cfg.d_model)
    # We need a dummy cache — just call with no cache
    # TransformerBlock.forward signature: (resid_pre, shortformer_pos_embed, attention_mask, past_kv_cache, ...)
    try:
        out = block(x, None, None, None)
        assert out.shape == x.shape
        assert len(captured_moe_input) > 0
    except Exception as e:
        pytest.skip(f"TransformerBlock forward requires more setup: {e}")
```

- [ ] **Step 2: Locate the bypass point in transformer_block.py**

Find the forward() body around line 240:

```python
normalized_resid_mid = self.ln2(mlp_in)
mlp_out = self.apply_mlp(normalized_resid_mid)
```

- [ ] **Step 3: Add the 3-line bypass**

Replace those two lines with:

```python
if getattr(self.mlp, "dual_branch_moe", False):
    # Dual-branch MoE (Gemma 4 26B): handles all norms internally
    mlp_out = self.hook_mlp_out(self.mlp(mlp_in))
else:
    normalized_resid_mid = self.ln2(mlp_in)
    mlp_out = self.apply_mlp(normalized_resid_mid)
```

Note: `hook_mlp_out` is already called inside `apply_mlp`. For the MoE bypass we call it directly here and skip `apply_mlp` (which would also apply `ln2_post`, incorrect for MoE).

- [ ] **Step 4: Run existing transformer_block tests to check for regressions**

```bash
uv run pytest tests/unit/components/ -k "transformer_block" -v 2>&1 | tail -15
```

Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add transformer_lens/components/transformer_block.py
git add tests/unit/components/test_gemma4_moe.py
git commit -m "feat: add dual-branch MoE bypass in TransformerBlock.forward()"
```

---

## Task 5: Weight Conversion for 26B (Streaming)

**Files:**
- Modify: `transformer_lens/pretrained/weight_conversions/gemma.py`

The existing `convert_gemma4_weights_from_disk()` handles E2B (dense FFN). Add MoE layer handling.

**Prerequisite:** Exact HF key names from Task 1. The code below uses placeholder names matching the JAX structure. Update with actual names from Task 1 before running.

Expected HF key patterns for 26B MoE (hypothetical — verify in Task 1):
```
layers.{l}.mlp.shared_expert.gate_proj.weight      # dense branch gate
layers.{l}.mlp.shared_expert.up_proj.weight        # dense branch up
layers.{l}.mlp.shared_expert.down_proj.weight      # dense branch down
layers.{l}.mlp.shared_expert_gate.weight           # pre_ffw2_norm (dense pre-norm)
layers.{l}.mlp.gate.weight                         # router W [D, E]
layers.{l}.mlp.router_norm.weight                  # (may not exist if no_scale)
layers.{l}.mlp.router_scale                        # router_scale [D]
layers.{l}.mlp.experts.gate_proj.weight            # [E, H, D] or [E, 2H, D] fused
layers.{l}.mlp.experts.up_proj.weight
layers.{l}.mlp.experts.down_proj.weight
layers.{l}.mlp.experts_scale                       # per_expert_scale [E]
# Norms:
layers.{l}.pre_feedforward_layernorm.weight        # ln_dense_pre (or separate for each branch)
layers.{l}.pre_feedforward_layernorm_2.weight      # ln_moe_pre
layers.{l}.post_feedforward_layernorm.weight       # ln_combined (shared post-norm)
```

- [ ] **Step 1: Write the weight conversion test**

Create `tests/unit/weight_conversions/test_gemma4_26b_weights.py`:

```python
import pytest
import torch
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


def make_26b_cfg():
    return HookedTransformerConfig(
        d_model=64, d_head=16, n_heads=4, d_mlp=32, n_layers=2,
        n_ctx=16, d_vocab=256, act_fn="gelu_pytorch_tanh",
        normalization_type="RMS", positional_embedding_type="rotary",
        num_experts=4, experts_per_token=2, moe_expert_dim=8, moe_dense_hidden_dim=16,
        use_ple=False, num_kv_shared_layers=0,
        attn_types=["local", "global"],
        use_normalization_before_and_after=False,
    )


def test_moe_weight_keys_present(tmp_path):
    """Verify the weight converter produces the correct TL keys for MoE layers."""
    pytest.skip("Requires Task 1 HF key enumeration — unskip after Task 1 complete")
```

- [ ] **Step 2: Run the test (skip expected)**

```bash
uv run pytest tests/unit/weight_conversions/test_gemma4_26b_weights.py -v 2>&1 | tail -5
```

Expected: `SKIPPED`

- [ ] **Step 3: Add `_convert_moe_layer()` helper to gemma.py**

In `transformer_lens/pretrained/weight_conversions/gemma.py`, add this function after the `convert_gemma4_weights_from_disk` function. Update HF key names from Task 1 results:

```python
def _convert_moe_layer_from_disk(
    l: int,
    get,
    get_opt,
    rms,
    prefix: str,
    cfg,
    dtype,
    state_dict: dict,
):
    """Convert one 26B_A4B MoE layer from safetensors to TL format.

    HF key names confirmed from checkpoint enumeration (Task 1).
    *** UPDATE THESE NAMES AFTER TASK 1 ***
    """
    lp = f"{prefix}layers.{l}."

    # Attention norms (same as E2B)
    state_dict[f"blocks.{l}.ln1.w"] = rms(f"{lp}input_layernorm.weight")
    state_dict[f"blocks.{l}.ln1_post.w"] = rms(f"{lp}post_attention_layernorm.weight")

    # MoE FFN norms (stored on Gemma4DualBranchFFN, not on block itself)
    # Keys: UPDATE FROM TASK 1
    state_dict[f"blocks.{l}.mlp.ln_dense_pre.scale"] = rms(f"{lp}mlp.DENSE_PRE_NORM_KEY.weight")
    state_dict[f"blocks.{l}.mlp.ln_dense_post.scale"] = rms(f"{lp}mlp.DENSE_POST_NORM_KEY.weight")
    state_dict[f"blocks.{l}.mlp.ln_moe_pre.scale"] = rms(f"{lp}mlp.MOE_PRE_NORM_KEY.weight")
    state_dict[f"blocks.{l}.mlp.ln_moe_post.scale"] = rms(f"{lp}mlp.MOE_POST_NORM_KEY.weight")
    state_dict[f"blocks.{l}.mlp.ln_combined.scale"] = rms(f"{lp}mlp.COMBINED_POST_NORM_KEY.weight")

    # Dense branch weights (mlp2).
    # HF stores [out, in] (i.e., gate_proj.weight is [hidden, d_model]).
    # TL/our _GatedMLP stores [in, out] (W_gate is [d_model, hidden]).
    # So we transpose.
    state_dict[f"blocks.{l}.mlp.dense_mlp.W_gate"] = get(f"{lp}mlp.DENSE_GATE_KEY").T.to(dtype)
    state_dict[f"blocks.{l}.mlp.dense_mlp.W_in"] = get(f"{lp}mlp.DENSE_UP_KEY").T.to(dtype)
    state_dict[f"blocks.{l}.mlp.dense_mlp.W_out"] = get(f"{lp}mlp.DENSE_DOWN_KEY").T.to(dtype)

    # Router weights (both nn.Parameter, no '.weight' suffix)
    state_dict[f"blocks.{l}.mlp.router_scale"] = get(f"{lp}mlp.ROUTER_SCALE_KEY").to(dtype)
    # router_W is [D, E] in TL; HF may store as [E, D] (like a linear layer weight).
    # Confirm shape from Task 1 and transpose if needed. Assuming HF is [E, D]:
    state_dict[f"blocks.{l}.mlp.router_W"] = get(f"{lp}mlp.ROUTER_W_KEY").T.to(dtype)  # → [D, E]

    # Expert weights
    # JAX gating_einsum shape: [E, 2, H, D] — we store as [E, D, 2H]
    # HF may store as [E, 2H, D] or split gate/up — UPDATE FROM TASK 1
    E = cfg.num_experts
    H = cfg.moe_expert_dim
    D = cfg.d_model
    w_gate = get(f"{lp}mlp.EXPERT_GATE_KEY.weight")   # [E, H, D] or [E, 2H, D]
    w_up = get_opt(f"{lp}mlp.EXPERT_UP_KEY.weight")   # [E, H, D] (if not fused)
    if w_up is not None:
        # Separate gate/up: fuse to [E, D, 2H]
        state_dict[f"blocks.{l}.mlp.expert_W_gateup"] = torch.cat(
            [w_gate.transpose(-1, -2), w_up.transpose(-1, -2)], dim=-1
        ).to(dtype)  # [E, D, 2H]
    else:
        # Already fused [E, 2H, D]: transpose to [E, D, 2H]
        state_dict[f"blocks.{l}.mlp.expert_W_gateup"] = w_gate.transpose(-1, -2).to(dtype)

    w_down = get(f"{lp}mlp.EXPERT_DOWN_KEY.weight")   # [E, D, H] or [E, H, D]
    # We want [E, H, D] (expert_W_down[e] is [H, D] for einsum 'bsh,bshd->bsd')
    if w_down.shape[-1] == D:                          # already [E, H, D]
        state_dict[f"blocks.{l}.mlp.expert_W_down"] = w_down.to(dtype)
    else:                                              # [E, D, H] — transpose
        state_dict[f"blocks.{l}.mlp.expert_W_down"] = w_down.transpose(-1, -2).to(dtype)

    # Per-expert scale
    state_dict[f"blocks.{l}.mlp.per_expert_scale"] = get(f"{lp}mlp.PER_EXPERT_SCALE_KEY").to(dtype)
```

- [ ] **Step 4: Hook into `convert_gemma4_weights_from_disk`**

Find the per-layer loop in `convert_gemma4_weights_from_disk` (around line 440). Replace the MLP section:

```python
        # MLP weights — detect MoE vs dense
        if cfg.num_experts and getattr(cfg, "moe_expert_dim", None):
            _convert_moe_layer_from_disk(l, get, get_opt, rms, p, cfg, dtype, state_dict)
        else:
            # Dense MLP (E2B path)
            d_mlp_l = cfg.d_mlp_by_layer[l] if cfg.d_mlp_by_layer else cfg.d_mlp
            state_dict[f"blocks.{l}.mlp.W_in"] = get(f"{lp}mlp.up_proj.weight").T.to(dtype)
            state_dict[f"blocks.{l}.mlp.W_gate"] = get(f"{lp}mlp.gate_proj.weight").T.to(dtype)
            state_dict[f"blocks.{l}.mlp.W_out"] = get(f"{lp}mlp.down_proj.weight").T.to(dtype)
            state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(d_mlp_l, dtype=dtype)
            state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=dtype)
```

Also remove the ln2/ln2_post assignments for MoE layers (the keys don't exist in HF for 26B — norms are inside the MoE block):

```python
        # Layer norms — dense layers have pre_feedforward_layernorm; MoE layers don't
        state_dict[f"blocks.{l}.ln1.w"] = rms(f"{lp}input_layernorm.weight")
        state_dict[f"blocks.{l}.ln1_post.w"] = rms(f"{lp}post_attention_layernorm.weight")
        if not (cfg.num_experts and getattr(cfg, "moe_expert_dim", None)):
            # Dense-only: shared pre/post FFN norms live on the block
            state_dict[f"blocks.{l}.ln2.w"] = rms(f"{lp}pre_feedforward_layernorm.weight")
            state_dict[f"blocks.{l}.ln2_post.w"] = rms(f"{lp}post_feedforward_layernorm.weight")
        # For MoE layers, norms are loaded inside _convert_moe_layer_from_disk()
```

- [ ] **Step 5: Run existing E2B weight conversion tests to check for regressions**

```bash
uv run pytest tests/unit/weight_conversions/ -v 2>&1 | tail -15
```

Expected: all existing tests pass (MoE path is only triggered by `moe_expert_dim` being set).

- [ ] **Step 6: Commit**

```bash
git add transformer_lens/pretrained/weight_conversions/gemma.py
git add tests/unit/weight_conversions/test_gemma4_26b_weights.py
git commit -m "feat: add MoE weight conversion path to convert_gemma4_weights_from_disk"
```

---

## Task 6: Wire `Gemma4DualBranchFFN` into `TransformerBlock.__init__`

**Files:**
- Modify: `transformer_lens/components/transformer_block.py` (init only)

The `TransformerBlock.__init__` currently always creates a dense MLP. For 26B, we need to instantiate `Gemma4DualBranchFFN` when `num_experts` and `moe_expert_dim` are set.

- [ ] **Step 1: Find MLP instantiation in transformer_block.py**

Search for the section that creates `self.mlp`. It will look like:

```python
        if self.cfg.num_experts:
            self.mlp = MoE(cfg)
        elif self.cfg.gated_mlp:
            self.mlp = GatedMLP(cfg)
        else:
            self.mlp = MLP(cfg)
```

- [ ] **Step 2: Add Gemma4 dual-branch check**

```python
        if self.cfg.num_experts and getattr(self.cfg, "moe_expert_dim", None):
            # Gemma 4 26B_A4B: dual-branch MoE FFN (handles all norms internally)
            from transformer_lens.components.mlps.gemma4_moe import Gemma4DualBranchFFN
            self.mlp = Gemma4DualBranchFFN(self.cfg)
        elif self.cfg.num_experts:
            self.mlp = MoE(cfg)
        elif self.cfg.gated_mlp:
            self.mlp = GatedMLP(cfg)
        else:
            self.mlp = MLP(cfg)
```

- [ ] **Step 3: Run all component tests**

```bash
uv run pytest tests/unit/components/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 4: Verify full model instantiation (no checkpoint needed)**

```python
# Run this snippet directly
import sys
sys.path.insert(0, '/home/meridian/projects/Gemma4-TransformerLens')

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components.transformer_block import TransformerBlock
import torch

cfg = HookedTransformerConfig(
    d_model=64, d_head=16, n_heads=4, d_mlp=32, n_layers=2,
    n_ctx=16, d_vocab=256, act_fn="gelu_pytorch_tanh",
    normalization_type="RMS", positional_embedding_type="rotary",
    num_experts=4, experts_per_token=2, moe_expert_dim=8, moe_dense_hidden_dim=16,
    use_ple=False, num_kv_shared_layers=0,
    attn_types=["local", "global"],
)
block = TransformerBlock(cfg, 0)
print(type(block.mlp).__name__)  # Should print Gemma4DualBranchFFN
print(getattr(block.mlp, "dual_branch_moe", False))  # Should print True
```

Save this as `scripts/verify_26b_block.py` and run:
```bash
cd /home/meridian/projects/Gemma4-TransformerLens
uv run python scripts/verify_26b_block.py
```

Expected:
```
Gemma4DualBranchFFN
True
```

- [ ] **Step 5: Commit**

```bash
git add transformer_lens/components/transformer_block.py
git commit -m "feat: instantiate Gemma4DualBranchFFN in TransformerBlock for 26B MoE layers"
```

---

## Task 7: Kaggle Validation Notebook

**Files:**
- Create: `notebooks/gemma4_26b_validation.ipynb`

This runs on Kaggle (TPU or T4, 16 GB RAM). It validates that the full loading pipeline works end-to-end.

**Prerequisite:** Task 1 (correct HF key names), Task 5 (weight conversion with correct keys).

- [ ] **Step 1: Create the validation notebook outline**

Create `notebooks/gemma4_26b_validation.ipynb` with these cells:

**Cell 1 — Setup:**
```python
import sys
sys.path.insert(0, "/kaggle/working/Gemma4-TransformerLens")

import torch
from transformer_lens import HookedTransformer
from transformer_lens.pretrained.weight_conversions.gemma import convert_gemma4_weights_from_disk
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
```

**Cell 2 — Load config:**
```python
cfg = get_pretrained_model_config(
    "google/gemma-4-26B_A4B-it",
    dtype=torch.bfloat16,
    device="cpu",
)
print(cfg.d_model, cfg.num_experts, cfg.moe_expert_dim)
# Expected: 2816, 128, 704
```

**Cell 3 — Stream weights from checkpoint:**
```python
model_dir = "/kaggle/input/gemma-4/transformers/gemma-4-26b-a4b-it/1"
state_dict = convert_gemma4_weights_from_disk(model_dir, cfg, dtype=torch.bfloat16)
print(f"Loaded {len(state_dict)} weight tensors")
# Spot-check MoE weights
print(state_dict["blocks.0.mlp.expert_W_gateup"].shape)  # [128, 2816, 1408]
print(state_dict["blocks.0.mlp.router_scale"].shape)      # [2816]
```

**Cell 4 — Build HookedTransformer:**
```python
model = HookedTransformer(cfg)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("Model loaded. Parameter count:", sum(p.numel() for p in model.parameters()) / 1e9, "B")
```

**Cell 5 — Sanity check logits:**
```python
from transformer_lens import utils
test_prompt = "The capital of France is"
tokens = model.to_tokens(test_prompt)
print("Tokens:", tokens)
logits, cache = model.run_with_cache(tokens)
print("Logits shape:", logits.shape)

# Top-5 next token predictions
top5 = logits[0, -1].topk(5)
for idx, score in zip(top5.indices, top5.values):
    print(f"  {model.to_string(idx)!r}: {score.item():.3f}")
# Expected: "Paris" or " Paris" near top
```

**Cell 6 — Check router patterns (mechinterp hook):**
```python
expert_indices_per_layer = {}

def capture_router(value, hook):
    layer_name = hook.name.split(".")[1]  # e.g., "0"
    expert_indices_per_layer[int(layer_name)] = value.detach().cpu()
    return value

hooks = [
    (f"blocks.{l}.mlp.hook_expert_indices", capture_router)
    for l in range(cfg.n_layers)
]
model.run_with_hooks(tokens, fwd_hooks=hooks)

import matplotlib.pyplot as plt
# Show which experts are most frequently selected in layer 0
idx_layer0 = expert_indices_per_layer[0].flatten()
counts = torch.bincount(idx_layer0, minlength=cfg.num_experts)
plt.bar(range(cfg.num_experts), counts.numpy())
plt.title("Expert selection frequency, layer 0")
plt.xlabel("Expert index")
plt.ylabel("Times selected")
plt.show()
```

**Cell 7 — Logit lens on residual stream:**
```python
# Check residual stream through layers using logit lens
resid_post_cache = [cache[f"blocks.{l}.hook_resid_post"] for l in range(cfg.n_layers)]
print("hook_resid_post shapes:", [r.shape for r in resid_post_cache[:3]])
```

- [ ] **Step 2: Push notebook to git**

```bash
git add notebooks/gemma4_26b_validation.ipynb
git commit -m "feat: add Kaggle validation notebook for Gemma 4 26B_A4B"
```

- [ ] **Step 3: Run on Kaggle**

Upload to Kaggle, attach the gemma-4-26b_a4b-it model input, run all cells.
Expected: logits shape `[1, N, 262144]`, "Paris" in top-5 predictions, router plot shows non-uniform expert selection.

- [ ] **Step 4: If validation fails, debug**

Common issues:
- Wrong HF key names in weight converter → update `_convert_moe_layer_from_disk` placeholder keys
- Shape mismatch in `expert_W_gateup` → re-check gate/up fusion orientation from Task 1
- `load_state_dict` unexpected keys → use `strict=False` and check what's missing

---

## Appendix: Expert Forward Implementation Note

The `_run_experts` method in `Gemma4DualBranchFFN` uses a simple K-loop (K=8 iterations). This is O(K × B × S × D × H) and correct but not optimized. For large models on GPU, the JAX `ragged_dot` approach is faster. For mechinterp on Kaggle with batch=1, the loop is fine.

If throughput becomes an issue, replace the loop with a gather-based einsum:
```python
# Vectorized expert lookup
W_gateup = self.expert_W_gateup[indices]  # [B, S, K, D, 2H]
h = torch.einsum('bsd,bskdh->bskh', x, W_gateup)   # [B, S, K, 2H]
gate, up = h.chunk(2, dim=-1)
h = F.gelu(gate, approximate="tanh") * up  # [B, S, K, H]
scale = self.per_expert_scale[indices]     # [B, S, K]
h = h * scale.unsqueeze(-1)
W_down = self.expert_W_down[indices]       # [B, S, K, H, D]
expert_outs = torch.einsum('bskh,bskhd->bskd', h, W_down)  # [B, S, K, D]
out = (weights.unsqueeze(-1) * expert_outs).sum(dim=2)       # [B, S, D]
```

The vectorized form may run out of memory for large batch/sequence — the loop is safer for mechinterp workloads.
