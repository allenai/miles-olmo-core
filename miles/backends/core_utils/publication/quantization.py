"""Resolve publication precision from the actual rollout checkpoints."""

from copy import deepcopy

from transformers import PretrainedConfig

from miles.backends.core_utils.quantized_export import export_format


def resolve(args, *, deployment=None):
    if not getattr(args.olmo_core, "quantized_export", False):
        return None
    if not args.olmo_core.stream_moe_export or args.olmo_core.expert_publication != "per_expert":
        raise ValueError("core.quantized_export requires streaming per_expert publication")
    if getattr(args.olmo_core, "weight_sync_mode", "flattened") != "flattened":
        raise ValueError("core.quantized_export requires flattened byte transport")
    if deployment is None:
        # SGLang's config resolver imports its CUDA runtime. Keep pure exporter
        # and disabled-mode use available without that optional dependency.
        from miles.backends.sglang_utils.sglang_config import resolve_sglang_config

        deployment = resolve_sglang_config(args)
    configs = []
    for model in deployment.models:
        if not model.update_weights:
            continue
        for group in model.server_groups:
            if group.worker_type == "placeholder":
                continue
            path = group.overrides.get("model_path", model.model_path or args.hf_checkpoint)
            descriptor, _ = PretrainedConfig.get_config_dict(path)
            config = descriptor.get("quantization_config")
            if descriptor.get("model_type") != "olmo3moe" or not isinstance(config, dict):
                raise ValueError("Core quantized export requires an Olmo MoE serialized quantized rollout checkpoint")
            fmt = export_format(config)
            selection = group.overrides.get("quantization", getattr(args, "sglang_quantization", None))
            expected = "fp8" if fmt == "fp8" else "modelopt_fp4"
            if selection not in (None, expected):
                raise ValueError(f"Rollout quantization override {selection!r} conflicts with {expected}")
            configs.append(config)
    if not configs or any(config != configs[0] for config in configs[1:]):
        raise ValueError("All updated rollout engines must use the same serialized quantization config")
    return deepcopy(configs[0])
