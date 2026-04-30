"""MLP Factory

Centralized location for creating any MLP needed within TransformerLens
"""

from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.components.mlps.gated_mlp import GatedMLP
from transformer_lens.components.mlps.gated_mlp_4bit import GatedMLP4Bit
from transformer_lens.components.mlps.gemma4_moe import Gemma4DualBranchFFN
from transformer_lens.components.mlps.gpt_oss_moe import GptOssMoE
from transformer_lens.components.mlps.mlp import MLP
from transformer_lens.components.mlps.moe import MoE
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


class MLPFactory:
    @staticmethod
    def create_mlp(cfg: HookedTransformerConfig) -> CanBeUsedAsMLP:
        if cfg.num_experts and getattr(cfg, "moe_expert_dim", None):
            # Gemma 4 26B_A4B: dual-branch MoE (dense shared + sparse MoE), all norms internal
            return Gemma4DualBranchFFN(cfg)
        elif cfg.num_experts:
            if cfg.original_architecture == "GptOssForCausalLM":
                return GptOssMoE(cfg)
            return MoE(cfg)
        elif cfg.gated_mlp:
            return GatedMLP(cfg) if not cfg.load_in_4bit else GatedMLP4Bit(cfg)
        else:
            return MLP(cfg)
