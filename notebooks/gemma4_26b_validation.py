"""Gemma 4 26B_A4B TL Validation Script (run as Kaggle notebook).

PURPOSE: Validate the full loading pipeline end-to-end:
  1. Load the 26B_A4B config
  2. Stream weights from the Kaggle checkpoint (no full-model RAM required)
  3. Build HookedTransformer and verify forward pass
  4. Check router hook patterns (mechinterp sanity check)
  5. Logit lens across layers

BEFORE RUNNING:
  - Complete Task 1 (checkpoint enumeration) and update HF key names
    in transformer_lens/pretrained/weight_conversions/gemma.py
    (_convert_moe_layer_from_disk placeholder keys → real keys)
  - Attach the gemma-4-26b-a4b-it model to the Kaggle notebook
  - Clone Gemma4-TransformerLens into /kaggle/working/

EXPECTED OUTPUTS:
  - Config: d_model=2816, num_experts=128, moe_expert_dim=704
  - Forward pass: logits shape [1, N, 262144]
  - "Paris" or " Paris" in top-5 predictions for the test prompt
  - Router plot: non-uniform expert selection (load imbalance visible)
"""

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import sys
import os

# Add the cloned repo to the path
sys.path.insert(0, "/kaggle/working/Gemma4-TransformerLens")

import torch
from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.pretrained.weight_conversions.gemma import (
    convert_gemma4_weights_from_disk,
)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Cell 2: Load config ────────────────────────────────────────────────────────
cfg = get_pretrained_model_config(
    "google/gemma-4-26B_A4B-it",
    dtype=torch.bfloat16,
    device="cpu",
    fold_ln=False,
)

print(f"Model config loaded:")
print(f"  d_model:       {cfg.d_model}")
print(f"  n_layers:      {cfg.n_layers}")
print(f"  n_heads:       {cfg.n_heads}")
print(f"  num_experts:   {cfg.num_experts}")
print(f"  experts/token: {cfg.experts_per_token}")
print(f"  moe_expert_dim: {cfg.moe_expert_dim}")
print(f"  moe_dense_hidden_dim: {cfg.moe_dense_hidden_dim}")
print(f"  use_ple:       {cfg.use_ple}")
print(f"  num_kv_shared: {cfg.num_kv_shared_layers}")

# Expected: 2816, 30, 16, 128, 8, 704, 2112, False, 0


# ── Cell 3: Stream weights from checkpoint ─────────────────────────────────────
# Adjust model_dir to match your Kaggle input mount path.
# Common paths: /kaggle/input/gemma-4/... or /kaggle/input/gemma-4-26b-a4b-it/...
model_dir = "/kaggle/input/gemma-4/transformers/gemma-4-26b-a4b-it/1"

if not os.path.exists(model_dir):
    # Try alternative paths
    for candidate in [
        "/kaggle/input/gemma-4-26b-a4b-it",
        "/kaggle/input/gemma4-26b-a4b/transformers/gemma-4-26b-a4b-it/1",
    ]:
        if os.path.exists(candidate):
            model_dir = candidate
            break
    else:
        raise FileNotFoundError(
            f"Could not find 26B_A4B checkpoint. Searched:\n"
            f"  {model_dir}\n"
            "Attach the model input to this Kaggle notebook."
        )

print(f"Loading from: {model_dir}")
state_dict = convert_gemma4_weights_from_disk(model_dir, cfg, dtype=torch.bfloat16)
print(f"Loaded {len(state_dict)} weight tensors")

# Spot-check MoE weights for layer 0
print(f"\nLayer 0 MoE weight shapes:")
print(f"  expert_W_gateup: {state_dict['blocks.0.mlp.expert_W_gateup'].shape}")
# Expected: [128, 2816, 1408]  (1408 = 2*704)
print(f"  expert_W_down:   {state_dict['blocks.0.mlp.expert_W_down'].shape}")
# Expected: [128, 704, 2816]
print(f"  router_W:        {state_dict['blocks.0.mlp.router_W'].shape}")
# Expected: [2816, 128]
print(f"  router_scale:    {state_dict['blocks.0.mlp.router_scale'].shape}")
# Expected: [2816]
print(f"  per_expert_scale:{state_dict['blocks.0.mlp.per_expert_scale'].shape}")
# Expected: [128]


# ── Cell 4: Build HookedTransformer ───────────────────────────────────────────
model = HookedTransformer(cfg)
model.load_state_dict(state_dict, strict=False)
model.eval()

n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"Model loaded. Parameter count: {n_params:.1f}B")

# Move to device if CUDA available
if device == "cuda":
    model = model.to(device)


# ── Cell 5: Sanity check logits ────────────────────────────────────────────────
test_prompt = "The capital of France is"

# Tokenize manually using tokenizer loaded from the checkpoint dir
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokens = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
print(f"Prompt: {test_prompt!r}")
print(f"Tokens: {tokens.tolist()}")

with torch.no_grad():
    logits = model(tokens)

print(f"\nLogits shape: {logits.shape}")  # Expected: [1, N, 262144]

# Top-5 next token predictions
top5 = logits[0, -1].topk(5)
print(f"\nTop-5 next token predictions:")
for idx, score in zip(top5.indices, top5.values):
    tok = tokenizer.decode([idx.item()])
    print(f"  {tok!r}: {score.item():.3f}")
# Expected: "Paris" or " Paris" in top-3


# ── Cell 6: Check router patterns (mechinterp hook) ────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

expert_indices_per_layer = {}

def capture_router(value, hook):
    # Extract layer index from hook name: "blocks.{l}.mlp.hook_expert_indices"
    l = int(hook.name.split(".")[1])
    expert_indices_per_layer[l] = value.detach().cpu()
    return value

hooks = [
    (f"blocks.{l}.mlp.hook_expert_indices", capture_router)
    for l in range(cfg.n_layers)
]

with torch.no_grad():
    model.run_with_hooks(tokens, fwd_hooks=hooks)

# Plot expert selection frequency across layers
fig, axes = plt.subplots(5, 6, figsize=(18, 15))
axes = axes.flatten()

for l in range(cfg.n_layers):
    ax = axes[l]
    if l in expert_indices_per_layer:
        idx_flat = expert_indices_per_layer[l].flatten().numpy()
        counts = np.bincount(idx_flat, minlength=cfg.num_experts)
        ax.bar(range(cfg.num_experts), counts, width=1.0)
        ax.set_title(f"Layer {l}", fontsize=8)
        ax.set_xlabel("Expert", fontsize=6)
        ax.tick_params(labelsize=6)
    else:
        ax.set_visible(False)

plt.suptitle("Expert selection frequency per layer (Gemma 4 26B_A4B)", fontsize=12)
plt.tight_layout()
plt.savefig("expert_selection.png", dpi=100, bbox_inches="tight")
plt.show()
print("Router plot saved to expert_selection.png")


# ── Cell 7: Logit lens on residual stream ─────────────────────────────────────
with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)

print(f"\nResidual stream cache keys (first 5):")
resid_keys = [k for k in cache.keys() if "hook_resid_post" in k]
for k in resid_keys[:5]:
    print(f"  {k}: {cache[k].shape}")

# Logit lens: project each layer's residual stream through unembed
print(f"\nLogit lens top token per layer:")
for l in range(cfg.n_layers):
    key = f"blocks.{l}.hook_resid_post"
    if key in cache:
        resid = cache[key][0, -1]   # Last token position, no batch
        logits_l = model.unembed(resid.unsqueeze(0).unsqueeze(0))[0, 0]
        top_tok = tokenizer.decode([logits_l.argmax().item()])
        print(f"  Layer {l:2d}: {top_tok!r}")


# ── Cell 8: Memory cleanup ─────────────────────────────────────────────────────
import gc
del state_dict
del cache
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
print("Cleanup done.")
