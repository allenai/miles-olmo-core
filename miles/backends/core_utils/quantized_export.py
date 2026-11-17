"""Sender-side quantization of Core's per-expert HF export.

The trainer retains its ordinary weights. Only publication tensors are quantized;
the receiver must have loaded a serialized checkpoint with the same recipe.
"""

import re

import torch

from miles.utils import nvfp4
from miles.utils.fp8_export import quantize_per_tensor

_EXPERT = re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")


def export_format(config):
    if config.get("quant_method") == "fp8":
        if config.get("activation_scheme") != "dynamic" or config.get("weight_block_size") is not None:
            raise ValueError("Core FP8 export requires dynamic activations and per-tensor weight scaling")
        if config.get("fmt", "e4m3") != "e4m3":
            raise ValueError("Core FP8 export requires e4m3")
        return "fp8"
    if config.get("quant_algo") == "NVFP4" and config.get("quant_method") == "modelopt":
        if config.get("group_size", 16) != 16 or config.get("config_groups"):
            raise ValueError("Core NVFP4 export requires the flat group_size=16 recipe")
        return "nvfp4"
    raise ValueError("Core export supports serialized fp8 or modelopt NVFP4 checkpoints")


def _exclusion_rules(config, fmt):
    # Reject richer policies rather than silently differing from the receiver.
    rules = (
        config.get("ignored_layers", config.get("modules_to_not_convert", []))
        if fmt == "fp8"
        else config.get("ignore", config.get("modules_to_not_convert", config.get("exclude_modules", [])))
    )
    if not isinstance(rules, list) or any(
        not isinstance(rule, str) or any(c in rule for c in "*?[]:") for rule in rules
    ):
        raise ValueError("Core quantized export requires literal module exclusion paths")
    return frozenset(rule.strip(".") for rule in rules)


def _excluded(name, rules, fmt):
    module = name.removesuffix(".weight")
    parts = module.split(".")
    if fmt == "fp8":
        return any(
            ".".join(parts[start:end]) in rules
            for start in range(len(parts))
            for end in range(start + 1, len(parts) + 1)
        )
    return module in rules or any(part in rules for part in parts)


def _outputs(name, result):
    yield name, result[0]
    yield name.removesuffix(".weight") + ".weight_scale", result[1]
    if len(result) == 3:
        yield name.removesuffix(".weight") + ".weight_scale_2", result[2].reshape(1)


def iter_quantized_state(state, config):
    """Stream routed experts, retaining at most one gate/up pair.

    Core emits up, gate, down for each expert. Both gate-first and up-first pairs
    are accepted, but incomplete/interleaved pairs fail instead of accumulating
    an unbounded model-sized staging dictionary. Quantize before bucketing.
    """
    fmt = export_format(config)
    rules = _exclusion_rules(config, fmt)
    pending = None
    seen = set()
    for name, weight in state:
        if name in seen:
            raise ValueError(f"Duplicate export tensor: {name}")
        seen.add(name)
        match = _EXPERT.fullmatch(name)
        excluded = _excluded(name, rules, fmt)
        if match and excluded:
            raise ValueError("Core quantized export currently requires all routed experts quantized")
        if pending is not None:
            other_name = pending[0]
            expected = (
                other_name.replace(".up_proj.", ".gate_proj.")
                if ".up_proj." in other_name
                else other_name.replace(".gate_proj.", ".up_proj.")
            )
            if name != expected or excluded:
                raise ValueError(f"Expected complete equally quantized gate/up pair: {other_name}, {name}")
        if not match or excluded:
            if not excluded and name.endswith(".weight") and weight.ndim == 2 and "embed_tokens" not in name:
                raise ValueError(f"Non-expert matrix must be excluded in the rollout quantization config: {name}")
            yield name, weight
            continue
        if weight.ndim != 2 or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"Expected an unquantized 2D expert weight: {name}")
        if fmt == "nvfp4" and weight.shape[1] % nvfp4.NVFP4_GROUP_SIZE:
            raise ValueError(f"NVFP4 requires K divisible by 16: {name}")
        if fmt == "fp8":
            yield from _outputs(name, quantize_per_tensor(weight))
        elif match[1] == "down_proj":
            yield from _outputs(name, nvfp4.nvfp4_quantize_1d(weight))
        elif pending is None:
            # Own storage: Core is allowed to reuse its export scratch buffer.
            pending = name, weight.detach().clone()
        else:
            first, second = nvfp4.nvfp4_quantize_1d_pair(pending[1], weight)
            yield from _outputs(pending[0], first)
            yield from _outputs(name, second)
            pending = None
    if pending is not None:
        raise ValueError(f"Incomplete NVFP4 gate/up pair: {pending[0]}")
