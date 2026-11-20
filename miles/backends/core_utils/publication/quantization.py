"""Resolve publication precision from the actual rollout checkpoints."""

from copy import deepcopy

from transformers import PretrainedConfig

from miles.backends.core_utils.quantized_export import export_format


def resolve(args, *, deployment=None):
    if deployment is None:
        # Defer the optional SGLang runtime dependency for pure exporter tests.
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
            selection = group.overrides.get("quantization", getattr(args, "sglang_quantization", None))
            if config is None:
                if selection is not None:
                    raise ValueError(
                        "Weight quantization requires a matching serialized rollout checkpoint; convert it first with tools/convert_core_hf_quantized.py. KV-cache-only quantization should use kv_cache_dtype instead."
                    )
                configs.append(None)
                continue
            if descriptor.get("model_type") != "olmo3moe" or not isinstance(config, dict):
                raise ValueError("Core quantized export requires an Olmo MoE serialized quantized rollout checkpoint")
            fmt = export_format(config)
            if (
                fmt == "fp8"
                and group.overrides.get("moe_runner_backend", getattr(args, "sglang_moe_runner_backend", None))
                != "triton"
            ):
                raise ValueError("Core block FP8 refresh requires explicit moe_runner_backend=triton")
            expected = "fp8" if fmt == "fp8" else "modelopt_fp4"
            if selection not in (None, expected):
                raise ValueError(f"Rollout quantization override {selection!r} conflicts with {expected}")
            configs.append(config)
    if not configs:
        return None
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("All updated rollout engines must use the same serialized quantization config")
    if configs[0] is not None:
        if not args.olmo_core.stream_moe_export or args.olmo_core.expert_publication != "per_expert":
            raise ValueError("Quantized export requires streaming per_expert publication")
        if getattr(args.olmo_core, "weight_sync_mode", "flattened") != "flattened":
            raise ValueError("Quantized export requires flattened byte transport")
    return deepcopy(configs[0])
