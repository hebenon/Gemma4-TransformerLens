"""
Unit tests for Gemma 4 model support.

Tests cover:
1. Model registration in OFFICIAL_MODEL_NAMES
2. Core architectural parameters (E2B)
3. Hybrid local/global attention (4:1 pattern, 35 layers)
4. Dual head dimensions (256 local / 512 global)
5. Partial RoPE for global layers (factor 0.25, 1M theta)
6. QKV normalization (q_norm, k_norm, v_norm)
7. Normalization style (pre- and post-sublayer RMS)
8. Logit soft-capping (30.0)
9. Shared KV layers (last 20 borrow from layers 13/14)
10. Per-Layer Embeddings (PLE gated bottleneck)

No HF network access required — Gemma 4 config is hardcoded in loading_from_pretrained.py.
"""

import pytest

from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformer_lens.supported_models import OFFICIAL_MODEL_NAMES

# ============================================================================
# Constants
# ============================================================================

GLOBAL_LAYER_INDICES = [4, 9, 14, 19, 24, 29, 34]
LOCAL_LAYER_INDICES = [i for i in range(35) if i not in GLOBAL_LAYER_INDICES]


# ============================================================================
# Test: Model registration
# ============================================================================


class TestGemma4ModelRegistration:
    """Gemma 4 model names appear in OFFICIAL_MODEL_NAMES."""

    def test_e2b_it_registered(self):
        assert "google/gemma-4-E2B-it" in OFFICIAL_MODEL_NAMES


# ============================================================================
# Test: Core architecture
# ============================================================================


class TestGemma4ConfigGeneration:
    """Core architectural parameters match Gemma 4 E2B spec."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_d_model(self, e2b_cfg):
        assert e2b_cfg.d_model == 1536

    def test_n_heads(self, e2b_cfg):
        assert e2b_cfg.n_heads == 8

    def test_n_layers(self, e2b_cfg):
        assert e2b_cfg.n_layers == 35

    def test_d_head_local(self, e2b_cfg):
        assert e2b_cfg.d_head == 256

    def test_n_key_value_heads(self, e2b_cfg):
        assert e2b_cfg.n_key_value_heads == 1

    def test_d_mlp(self, e2b_cfg):
        assert e2b_cfg.d_mlp == 6144

    def test_d_vocab(self, e2b_cfg):
        assert e2b_cfg.d_vocab == 262144

    def test_n_ctx(self, e2b_cfg):
        assert e2b_cfg.n_ctx == 131072

    def test_act_fn(self, e2b_cfg):
        assert e2b_cfg.act_fn == "gelu_pytorch_tanh"

    def test_gated_mlp(self, e2b_cfg):
        assert e2b_cfg.gated_mlp is True

    def test_original_architecture(self, e2b_cfg):
        assert e2b_cfg.original_architecture == "Gemma4ForConditionalGeneration"


# ============================================================================
# Test: Hybrid attention
# ============================================================================


class TestGemma4HybridAttention:
    """4:1 sliding:full attention pattern — 7 global layers in 35 layers."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_use_local_attn(self, e2b_cfg):
        assert e2b_cfg.use_local_attn is True

    def test_attn_types_length(self, e2b_cfg):
        assert len(e2b_cfg.attn_types) == 35

    def test_global_layer_count(self, e2b_cfg):
        assert e2b_cfg.attn_types.count("global") == 7

    def test_local_layer_count(self, e2b_cfg):
        assert e2b_cfg.attn_types.count("local") == 28

    def test_global_layer_positions(self, e2b_cfg):
        global_indices = [i for i, t in enumerate(e2b_cfg.attn_types) if t == "global"]
        assert global_indices == GLOBAL_LAYER_INDICES

    def test_pattern_is_4_1_repeating(self, e2b_cfg):
        # Each block of 5 is [local, local, local, local, global]
        for rep in range(7):
            base = rep * 5
            assert e2b_cfg.attn_types[base:base + 4] == ["local", "local", "local", "local"]
            assert e2b_cfg.attn_types[base + 4] == "global"

    def test_window_size(self, e2b_cfg):
        assert e2b_cfg.window_size == 512


# ============================================================================
# Test: Dual head dimensions
# ============================================================================


class TestGemma4DualHeadDim:
    """Global attention layers use d_head=512; local layers use d_head=256."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_local_d_head(self, e2b_cfg):
        assert e2b_cfg.d_head == 256

    def test_global_d_head_field_exists(self, e2b_cfg):
        assert hasattr(e2b_cfg, "d_head_global")

    def test_global_d_head_value(self, e2b_cfg):
        assert e2b_cfg.d_head_global == 512

    def test_rotary_dim_global_computed(self, e2b_cfg):
        # rotary_dim_global = d_head_global * partial_rotary_factor_global = 512 * 0.25 = 128
        assert hasattr(e2b_cfg, "rotary_dim_global")
        assert e2b_cfg.rotary_dim_global == 128

    def test_rotary_dim_local(self, e2b_cfg):
        # Local: full rotary → rotary_dim = d_head = 256
        assert e2b_cfg.rotary_dim == 256


# ============================================================================
# Test: Partial RoPE
# ============================================================================


class TestGemma4PartialRoPE:
    """Global attention uses partial rotary encoding (factor 0.25, 1M theta);
    local attention uses full rotary (10K theta)."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_rotary_base_global(self, e2b_cfg):
        assert e2b_cfg.rotary_base == 1_000_000

    def test_rotary_base_local(self, e2b_cfg):
        assert e2b_cfg.rotary_base_local == 10_000

    def test_partial_rotary_factor_global(self, e2b_cfg):
        assert hasattr(e2b_cfg, "partial_rotary_factor_global")
        assert e2b_cfg.partial_rotary_factor_global == 0.25


# ============================================================================
# Test: QKV normalization
# ============================================================================


class TestGemma4QKVNorm:
    """All three projections normalized: q_norm, k_norm, v_norm."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_use_qk_norm(self, e2b_cfg):
        assert e2b_cfg.use_qk_norm is True


# ============================================================================
# Test: Normalization style
# ============================================================================


class TestGemma4Normalization:
    """Pre- and post-sublayer RMS norms (Gemma 3 pattern)."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_normalization_before_and_after(self, e2b_cfg):
        assert e2b_cfg.use_normalization_before_and_after is True

    def test_normalization_type(self, e2b_cfg):
        assert e2b_cfg.normalization_type == "RMS"

    def test_final_rms(self, e2b_cfg):
        assert e2b_cfg.final_rms is True


# ============================================================================
# Test: Logit soft-capping
# ============================================================================


class TestGemma4LogitSoftcap:
    """final_logit_softcapping=30.0 → output_logits_soft_cap=30.0."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_output_logits_soft_cap(self, e2b_cfg):
        assert e2b_cfg.output_logits_soft_cap == 30.0


# ============================================================================
# Test: Shared KV layers
# ============================================================================


class TestGemma4SharedKV:
    """Last 20 layers borrow K/V from source layers (13 for local, 14 for global)."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_num_kv_shared_layers(self, e2b_cfg):
        assert e2b_cfg.num_kv_shared_layers == 20

    def test_kv_shared_layer_sources_exists(self, e2b_cfg):
        assert hasattr(e2b_cfg, "kv_shared_layer_sources")
        assert e2b_cfg.kv_shared_layer_sources is not None

    def test_shared_layer_count(self, e2b_cfg):
        assert len(e2b_cfg.kv_shared_layer_sources) == 20

    def test_local_borrowers_point_to_layer_13(self, e2b_cfg):
        sources = e2b_cfg.kv_shared_layer_sources
        local_borrowers = [l for l in range(15, 35) if e2b_cfg.attn_types[l] == "local"]
        assert all(sources[l] == 13 for l in local_borrowers)

    def test_global_borrowers_point_to_layer_14(self, e2b_cfg):
        sources = e2b_cfg.kv_shared_layer_sources
        global_borrowers = [l for l in range(15, 35) if e2b_cfg.attn_types[l] == "global"]
        assert all(sources[l] == 14 for l in global_borrowers)

    def test_source_layers_not_in_shared_map(self, e2b_cfg):
        sources = e2b_cfg.kv_shared_layer_sources
        assert 13 not in sources
        assert 14 not in sources

    def test_layers_0_to_14_not_in_shared_map(self, e2b_cfg):
        sources = e2b_cfg.kv_shared_layer_sources
        for l in range(15):
            assert l not in sources


# ============================================================================
# Test: Per-Layer Embeddings
# ============================================================================


class TestGemma4PLE:
    """Per-Layer Embeddings: gated bottleneck at every decoder layer."""

    @pytest.fixture(scope="class")
    def e2b_cfg(self):
        return get_pretrained_model_config("google/gemma-4-E2B-it", fold_ln=False)

    def test_use_ple(self, e2b_cfg):
        assert e2b_cfg.use_ple is True

    def test_d_ple(self, e2b_cfg):
        assert e2b_cfg.d_ple == 256

    def test_ple_vocab_size(self, e2b_cfg):
        assert e2b_cfg.ple_vocab_size == 262144
