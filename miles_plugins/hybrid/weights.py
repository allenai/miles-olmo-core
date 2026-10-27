"""Exact release HF/native tensor-name interchange; tensor layouts are unchanged."""

_HYBRID_SMALL_TOP_LEVEL_TO_HF: dict[str, str] = {
    "embeddings.weight": "model.embed_tokens.weight",
    "embedding_norm.weight": "model.embed_norm.weight",
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight": "lm_head.weight",
}

_HYBRID_SMALL_COMMON_LAYER_TO_HF: dict[str, str] = {
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "feed_forward_norm.weight": "ffn_layernorm.weight",
    "post_feed_forward_norm.weight": "post_feedforward_layernorm.weight",
}

_HYBRID_SMALL_GDN_LAYER_TO_HF: dict[str, str] = {
    "attention.w_q.weight": "linear_attn.q_proj.weight",
    "attention.w_k.weight": "linear_attn.k_proj.weight",
    "attention.w_v.weight": "linear_attn.v_proj.weight",
    "attention.w_out.weight": "linear_attn.o_proj.weight",
    "attention.w_g.weight": "linear_attn.g_proj.weight",
    "attention.w_a.weight": "linear_attn.a_proj.weight",
    "attention.w_b.weight": "linear_attn.b_proj.weight",
    "attention.o_norm.weight": "linear_attn.o_norm.weight",
    "attention.q_conv1d.weight": "linear_attn.q_conv1d.weight",
    "attention.k_conv1d.weight": "linear_attn.k_conv1d.weight",
    "attention.v_conv1d.weight": "linear_attn.v_conv1d.weight",
    "attention.A_log": "linear_attn.A_log",
    "attention.dt_bias": "linear_attn.dt_bias",
}

_HYBRID_SMALL_FULL_ATTN_LAYER_TO_HF: dict[str, str] = {
    "attention.w_q.weight": "self_attn.q_proj.weight",
    "attention.w_k.weight": "self_attn.k_proj.weight",
    "attention.w_v.weight": "self_attn.v_proj.weight",
    "attention.w_out.weight": "self_attn.o_proj.weight",
    "attention.w_g.weight": "self_attn.attn_gate.weight",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
}


def native_to_hf_names(layer_types):
    mapping = dict(_HYBRID_SMALL_TOP_LEVEL_TO_HF)
    for index, kind in enumerate(layer_types):
        if kind not in ("linear_attention", "full_attention"):
            raise ValueError(f"Unsupported hybrid layer type: {kind}")
        layer = dict(_HYBRID_SMALL_COMMON_LAYER_TO_HF)
        layer.update(
            _HYBRID_SMALL_GDN_LAYER_TO_HF if kind == "linear_attention" else _HYBRID_SMALL_FULL_ATTN_LAYER_TO_HF
        )
        mapping.update({f"blocks.{index}.{native}": f"model.layers.{index}.{hf}" for native, hf in layer.items()})
    return mapping


def strip_checkpoint_wrapper(name):
    return ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")


def from_hf(config, state):
    mapping = native_to_hf_names(config.layer_types)
    missing = set(mapping.values()) - state.keys()
    unexpected = state.keys() - set(mapping.values())
    if missing or unexpected:
        raise ValueError(
            f"Hybrid checkpoint tensor mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return {native: state[hf] for native, hf in mapping.items()}
