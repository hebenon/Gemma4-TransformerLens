import einops
import torch

from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig


def convert_gemma_weights(gemma, cfg: HookedTransformerConfig):
    state_dict = {}

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Check if this is a multimodal model (Gemma3ForConditionalGeneration)
    # Multimodal models have language_model attribute, text-only models don't
    is_multimodal = hasattr(gemma, "language_model")

    # Get the actual model
    # For multimodal: gemma.language_model.model is Gemma3TextModel which has layers/embed_tokens
    # For text-only: gemma has .model which contains layers/embed_tokens
    if is_multimodal:
        # Multimodal structure: gemma.language_model.model contains the text transformer
        # We skip gemma.vision_tower entirely to save memory
        if hasattr(gemma.language_model, "model"):
            base_model = gemma.language_model.model
        else:
            # Fallback if structure is different
            base_model = gemma.language_model
    else:
        # Text-only Gemma3ForCausalLM has .model wrapper
        base_model = gemma.model

    # Gemma Models scale embeddings by multiplying by sqrt(d_model), use hidden state type to match
    # HF implementation
    state_dict["embed.W_E"] = base_model.embed_tokens.weight * torch.tensor(
        cfg.d_model**0.5, dtype=cfg.dtype
    )

    # Gemma has no biases anywhere
    for l in range(cfg.n_layers):
        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        state_dict[f"blocks.{l}.ln1.w"] = base_model.layers[
            l
        ].input_layernorm.weight.float() + torch.ones_like(
            base_model.layers[l].input_layernorm.weight, dtype=torch.float32
        )
        if cfg.use_normalization_before_and_after:
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln1_post.w"] = base_model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].input_layernorm.weight, dtype=torch.float32
            )

        W_Q = base_model.layers[l].self_attn.q_proj.weight
        W_K = base_model.layers[l].self_attn.k_proj.weight
        W_V = base_model.layers[l].self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        # Load q_norm and k_norm if they exist (Gemma 3)
        # Gemma3RMSNorm adds 1 to weights in forward(), so we pre-add it here
        if cfg.use_qk_norm:
            state_dict[f"blocks.{l}.attn.q_norm.w"] = base_model.layers[
                l
            ].self_attn.q_norm.weight.float() + torch.ones_like(
                base_model.layers[l].self_attn.q_norm.weight, dtype=torch.float32
            )
            state_dict[f"blocks.{l}.attn.k_norm.w"] = base_model.layers[
                l
            ].self_attn.k_norm.weight.float() + torch.ones_like(
                base_model.layers[l].self_attn.k_norm.weight, dtype=torch.float32
            )

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=W_Q.device
        )
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=W_K.device
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=W_V.device
        )

        W_O = base_model.layers[l].self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=W_O.device
        )

        # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
        if not cfg.use_normalization_before_and_after:
            # Only applies for Gemma 1. Confusingly post_attention_layernorm is applied to mlp_input in Gemma 1 and attn_out in Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = base_model.layers[
                l
            ].post_attention_layernorm.weight.float() + torch.ones_like(
                base_model.norm.weight, dtype=torch.float32
            )
        else:
            # Only applies for Gemma 2
            state_dict[f"blocks.{l}.ln2.w"] = base_model.layers[
                l
            ].pre_feedforward_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].pre_feedforward_layernorm.weight, dtype=torch.float32
            )
            state_dict[f"blocks.{l}.ln2_post.w"] = base_model.layers[
                l
            ].post_feedforward_layernorm.weight.float() + torch.ones_like(
                base_model.layers[l].post_feedforward_layernorm.weight, dtype=torch.float32
            )

        state_dict[f"blocks.{l}.mlp.W_in"] = base_model.layers[l].mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = base_model.layers[l].mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=base_model.layers[l].mlp.up_proj.weight.device
        )

        state_dict[f"blocks.{l}.mlp.W_out"] = base_model.layers[l].mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=base_model.layers[l].mlp.down_proj.weight.device
        )

    # GemmaRMSNorm adds 1 to weights before multiplying by input, keep RMS calcs in float32
    state_dict["ln_final.w"] = base_model.norm.weight.float() + torch.ones_like(
        base_model.norm.weight, dtype=torch.float32
    )

    # For multimodal models, lm_head might not exist or be tied to embeddings
    if hasattr(gemma, "lm_head"):
        state_dict["unembed.W_U"] = gemma.lm_head.weight.T
        unembed_device = gemma.lm_head.weight.device
    else:
        # Multimodal models might use tied embeddings
        state_dict["unembed.W_U"] = base_model.embed_tokens.weight.T
        unembed_device = base_model.embed_tokens.weight.device
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=unembed_device)

    return state_dict


def _rms_weight(norm_or_tensor) -> torch.Tensor:
    """Extract weight from a Gemma4RMSNorm (or tensor) and pre-add 1.

    Gemma4RMSNorm adds 1 in forward() — bake that into the stored weight so TL
    can use a plain multiply. Accepts either a raw tensor or a norm module,
    auto-discovering the weight parameter name (varies across transformers versions).
    """
    if isinstance(norm_or_tensor, torch.Tensor):
        w = norm_or_tensor
    else:
        for attr in ("weight", "w", "scale"):
            candidate = getattr(norm_or_tensor, attr, None)
            if isinstance(candidate, torch.Tensor):
                w = candidate
                break
        else:
            params = list(norm_or_tensor._parameters.keys())
            raise AttributeError(
                f"No weight param found on {type(norm_or_tensor).__name__} "
                f"(tried weight/w/scale; actual params: {params})"
            )
    return w.float() + torch.ones_like(w, dtype=torch.float32)


def convert_gemma4_weights(gemma, cfg: HookedTransformerConfig):
    """Convert Gemma 4 weights to TransformerLens format.

    Gemma 4 extends Gemma 3 with:
    - Dual head dimensions (local d_head=256, global d_head=512)
    - v_norm alongside q_norm/k_norm
    - Per-Layer Embeddings (PLE): gated bottleneck at each decoder layer
    - layer_scalar: learned per-layer output scale (plain tensor, not nn.Parameter)
    - Shared KV cache: handled at forward-pass level, no weight changes needed

    Architecture confirmed from Kaggle enumeration 2026-04-22.
    Module path: gemma.model.language_model = Gemma4TextModel (base_model)
    """
    assert cfg.n_key_value_heads is not None
    assert cfg.d_mlp is not None

    # Handle both Gemma4ForConditionalGeneration (.model.language_model)
    # and bare Gemma4Model (.language_model) returned by AutoModel.from_pretrained.
    if hasattr(gemma, "model") and hasattr(gemma.model, "language_model"):
        base_model = gemma.model.language_model
    elif hasattr(gemma, "language_model"):
        base_model = gemma.language_model
    else:
        raise ValueError(f"Cannot find language_model in {type(gemma).__name__}")

    state_dict = {}

    # Embeddings: scaled by sqrt(d_model), same as Gemma 3
    state_dict["embed.W_E"] = base_model.embed_tokens.weight * torch.tensor(
        cfg.d_model**0.5, dtype=cfg.dtype
    )

    # PLE model-level weights
    if cfg.use_ple:
        state_dict["ple.W_embed"] = base_model.embed_tokens_per_layer.weight
        # Linear weight is [out=n_layers*d_ple, in=d_model]; transpose for TL [d_model, n_layers*d_ple]
        state_dict["ple.W_proj"] = base_model.per_layer_model_projection.weight.T
        state_dict["ple.ln.w"] = _rms_weight(base_model.per_layer_projection_norm)
        # Scale factors are fixed constants (confirmed 2026-04-22): proj_scale=1/sqrt(1536),
        # input_scale=1/sqrt(2). Baked into PLEPrecomputer.forward(), not stored here.

    for l in range(cfg.n_layers):
        layer = base_model.layers[l]
        is_global = cfg.attn_types is not None and cfg.attn_types[l] == "global"
        d_head_l = cfg.d_head_global if (is_global and cfg.d_head_global) else cfg.d_head

        # Layer norms (pre- and post-sublayer, same as Gemma 3 use_normalization_before_and_after)
        state_dict[f"blocks.{l}.ln1.w"] = _rms_weight(layer.input_layernorm)
        state_dict[f"blocks.{l}.ln1_post.w"] = _rms_weight(layer.post_attention_layernorm)
        state_dict[f"blocks.{l}.ln2.w"] = _rms_weight(layer.pre_feedforward_layernorm)
        state_dict[f"blocks.{l}.ln2_post.w"] = _rms_weight(layer.post_feedforward_layernorm)

        # Attention weights
        # einops infers h = total_dim / n_heads, so global layers (W_Q=[4096,1536]) correctly
        # produce [8, 1536, 512] without needing to specify h explicitly.
        W_Q = einops.rearrange(layer.self_attn.q_proj.weight, "(n h) m -> n m h", n=cfg.n_heads)
        W_O = einops.rearrange(layer.self_attn.o_proj.weight, "m (n h) -> n h m", n=cfg.n_heads)

        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        # Bias shapes must use per-layer d_head (global layers: 512, local: 256)
        dev = W_Q.device
        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(cfg.n_heads, d_head_l, dtype=cfg.dtype, device=dev)
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(cfg.d_model, dtype=cfg.dtype, device=dev)

        # Shared KV layers (15–34) may not have k_proj/v_proj in the HF model
        is_shared_kv = cfg.kv_shared_layer_sources is not None and l in cfg.kv_shared_layer_sources
        if not is_shared_kv:
            W_K = einops.rearrange(layer.self_attn.k_proj.weight, "(n h) m -> n m h", n=cfg.n_key_value_heads)
            W_V = einops.rearrange(layer.self_attn.v_proj.weight, "(n h) m -> n m h", n=cfg.n_key_value_heads)
            state_dict[f"blocks.{l}.attn._W_K"] = W_K
            state_dict[f"blocks.{l}.attn._W_V"] = W_V
            state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(cfg.n_key_value_heads, d_head_l, dtype=cfg.dtype, device=dev)
            state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(cfg.n_key_value_heads, d_head_l, dtype=cfg.dtype, device=dev)

        # Q/K/V norms (Gemma 4 adds v_norm alongside q_norm and k_norm)
        if cfg.use_qk_norm:
            state_dict[f"blocks.{l}.attn.q_norm.w"] = _rms_weight(layer.self_attn.q_norm)
            state_dict[f"blocks.{l}.attn.k_norm.w"] = _rms_weight(layer.self_attn.k_norm)
            if hasattr(layer.self_attn, "v_norm") and len(list(layer.self_attn.v_norm.parameters())) > 0:
                state_dict[f"blocks.{l}.attn.v_norm.w"] = _rms_weight(layer.self_attn.v_norm)

        # MLP weights
        state_dict[f"blocks.{l}.mlp.W_in"] = layer.mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_gate"] = layer.mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_out"] = layer.mlp.down_proj.weight.T
        mlp_dev = layer.mlp.up_proj.weight.device
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype, device=mlp_dev)
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(cfg.d_model, dtype=cfg.dtype, device=mlp_dev)

        # PLE per-block weights
        if cfg.use_ple:
            state_dict[f"blocks.{l}.ple_gate.W"] = layer.per_layer_input_gate.weight.T
            state_dict[f"blocks.{l}.ple_up.W"] = layer.per_layer_projection.weight.T
            state_dict[f"blocks.{l}.ple_ln.w"] = _rms_weight(layer.post_per_layer_input_norm)
            # layer_scalar: plain torch.Tensor (not nn.Parameter) — copy as-is
            state_dict[f"blocks.{l}.layer_scale"] = layer.layer_scalar.clone()

    state_dict["ln_final.w"] = _rms_weight(base_model.norm)

    # tie_word_embeddings=True for Gemma 4; lm_head exists on the outer wrapper
    if hasattr(gemma, "lm_head"):
        state_dict["unembed.W_U"] = gemma.lm_head.weight.T
        unembed_dev = gemma.lm_head.weight.device
    else:
        state_dict["unembed.W_U"] = base_model.embed_tokens.weight.T
        unembed_dev = base_model.embed_tokens.weight.device
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=unembed_dev)

    return state_dict
