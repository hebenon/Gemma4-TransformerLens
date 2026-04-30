"""Tests for Gemma4DualBranchFFN — the dual-branch MoE FFN for Gemma 4 26B_A4B."""

import pytest
import torch

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.components.mlps.gemma4_moe import Gemma4DualBranchFFN
from transformer_lens.hook_points import HookPoint


@pytest.fixture
def cfg():
    """Minimal config matching 26B structure but tiny for fast testing."""
    return HookedTransformerConfig(
        d_model=64,
        d_head=16,
        n_heads=4,
        d_mlp=32,             # dense branch hidden (moe_dense_hidden_dim)
        n_layers=2,
        n_ctx=16,
        d_vocab=256,
        act_fn="gelu_pytorch_tanh",
        normalization_type="RMS",
        positional_embedding_type="rotary",
        num_experts=8,
        experts_per_token=2,
        moe_expert_dim=16,
        moe_dense_hidden_dim=32,
    )


class TestGemma4DualBranchFFN:
    def test_output_shape(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        x = torch.randn(2, 5, cfg.d_model)
        out = moe(x)
        assert out.shape == (2, 5, cfg.d_model)

    def test_dual_branch_flag(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        assert moe.dual_branch_moe is True

    def test_hook_points_exist(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        assert isinstance(moe.hook_dense_out, HookPoint)
        assert isinstance(moe.hook_moe_out, HookPoint)
        assert isinstance(moe.hook_router_logits, HookPoint)
        assert isinstance(moe.hook_expert_weights, HookPoint)
        assert isinstance(moe.hook_expert_indices, HookPoint)
        assert isinstance(moe.hook_combined_out, HookPoint)

    def test_router_logits_hook_shape(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        captured = {}

        def save(value, hook):
            captured["router_logits"] = value.clone()
            return value

        moe.hook_router_logits.add_hook(save)
        moe(torch.randn(1, 4, cfg.d_model))
        assert captured["router_logits"].shape == (1, 4, cfg.num_experts)

    def test_expert_weights_hook_shape(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        captured = {}

        def save(value, hook):
            captured["w"] = value.clone()
            return value

        moe.hook_expert_weights.add_hook(save)
        moe(torch.randn(1, 4, cfg.d_model))
        assert captured["w"].shape == (1, 4, cfg.experts_per_token)

    def test_expert_indices_hook_shape(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        captured = {}

        def save(value, hook):
            captured["idx"] = value.clone()
            return value

        moe.hook_expert_indices.add_hook(save)
        moe(torch.randn(1, 4, cfg.d_model))
        assert captured["idx"].shape == (1, 4, cfg.experts_per_token)

    def test_expert_weights_sum_to_one(self, cfg):
        """Top-k weights are renormalized to sum to 1 per token."""
        moe = Gemma4DualBranchFFN(cfg)
        captured = {}

        def save(value, hook):
            captured["w"] = value.clone()
            return value

        moe.hook_expert_weights.add_hook(save)
        moe(torch.randn(2, 6, cfg.d_model))
        sums = captured["w"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_no_nan_on_random_input(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        x = torch.randn(2, 8, cfg.d_model)
        out = moe(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_dense_branch_fires(self, cfg):
        """hook_dense_out fires and has the correct shape."""
        moe = Gemma4DualBranchFFN(cfg)
        captured = {}

        def save(value, hook):
            captured["dense"] = value.clone()
            return value

        moe.hook_dense_out.add_hook(save)
        moe(torch.randn(1, 3, cfg.d_model))
        assert "dense" in captured
        assert captured["dense"].shape == (1, 3, cfg.d_model)

    def test_moe_branch_fires(self, cfg):
        """hook_moe_out fires and has the correct shape."""
        moe = Gemma4DualBranchFFN(cfg)
        captured = {}

        def save(value, hook):
            captured["moe"] = value.clone()
            return value

        moe.hook_moe_out.add_hook(save)
        moe(torch.randn(1, 3, cfg.d_model))
        assert "moe" in captured
        assert captured["moe"].shape == (1, 3, cfg.d_model)

    def test_combined_hook_fires(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        captured = {}

        def save(value, hook):
            captured["combined"] = value.clone()
            return value

        moe.hook_combined_out.add_hook(save)
        out = moe(torch.randn(1, 3, cfg.d_model))
        assert "combined" in captured
        assert torch.allclose(out, captured["combined"])

    def test_parameter_count_has_expert_weights(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        params = {n: p for n, p in moe.named_parameters()}
        assert "expert_W_gateup" in params
        assert "expert_W_down" in params
        assert "per_expert_scale" in params
        assert "router_scale" in params
        assert "router_W" in params

    def test_expert_W_gateup_shape(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        params = dict(moe.named_parameters())
        # [E, D, 2*H_expert]
        assert params["expert_W_gateup"].shape == (
            cfg.num_experts, cfg.d_model, 2 * cfg.moe_expert_dim
        )

    def test_expert_W_down_shape(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        params = dict(moe.named_parameters())
        # [E, H_expert, D]
        assert params["expert_W_down"].shape == (
            cfg.num_experts, cfg.moe_expert_dim, cfg.d_model
        )

    def test_dense_mlp_uses_moe_dense_hidden_dim(self, cfg):
        moe = Gemma4DualBranchFFN(cfg)
        params = dict(moe.named_parameters())
        assert params["dense_mlp.W_gate"].shape == (cfg.d_model, cfg.moe_dense_hidden_dim)
        assert params["dense_mlp.W_in"].shape == (cfg.d_model, cfg.moe_dense_hidden_dim)
        assert params["dense_mlp.W_out"].shape == (cfg.moe_dense_hidden_dim, cfg.d_model)
