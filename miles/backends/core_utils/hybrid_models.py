"""Temporary MainlineLadder native learner, publication, and checkpoint adapter."""

import json
from pathlib import Path

import torch
from olmo_core.nn import lm_head, transformer
from torch.distributed.tensor import DTensor

from miles.backends.core_utils import standard_models
from miles_plugins.hybrid import weights

validate_training_options = standard_models.validate_training_options
save_native = standard_models.save_native
load_native = standard_models.load_native
replay_context = standard_models.replay_context


def _override_attention_backend(value, backend):
    if isinstance(value, dict):
        if value.get("_CLASS_", "").endswith("AttentionConfig"):
            value["backend"] = str(backend)
        for child in value.values():
            _override_attention_backend(child, backend)
    elif isinstance(value, list):
        for child in value:
            _override_attention_backend(child, backend)


def model_config_from_hf(hf, options):
    if options.model_config:
        path = Path(options.model_config)
    else:
        source = Path(hf._name_or_path)
        path = None
        for parent in source.parents:
            if parent.name == "hf" and source.relative_to(parent).parts[0] == "v1":
                path = parent.with_name("olmo-core") / source.relative_to(parent) / "config.json"
                break
        if path is None:
            raise ValueError("MainlineLadder requires core.model_config or the paired hf/v1 and olmo-core/v1 trees")
    document = json.loads(path.read_text())
    model = document.get("model", document)
    _override_attention_backend(model, options.attention_backend)
    config = transformer.TransformerConfig.from_dict(model)
    config.lm_head.loss_implementation = lm_head.LMLossImplementation.default
    if (
        config.n_layers != len(hf.layer_types)
        or config.d_model != hf.hidden_size
        or config.vocab_size != hf.vocab_size
    ):
        raise ValueError("Native hybrid config does not match HF checkpoint dimensions")
    return config


def build_train_module(args, *, common, optim, hf_config, hf_state):
    return standard_models.build_train_module(
        args,
        common=common,
        optim=optim,
        hf_config=hf_config,
        hf_state=hf_state,
        state_converter=weights.from_hf,
    )


def iter_export_state(module, hf, *, stream_moe=True, fused_experts=False):
    if fused_experts:
        raise ValueError("Hybrid checkpoints have no routed experts")
    names = weights.native_to_hf_names(hf.layer_types)
    seen = set()
    for name, value in module.model.state_dict().items():
        canonical = weights.strip_checkpoint_wrapper(name)
        if canonical not in names:
            raise ValueError(f"Unexpected native hybrid tensor: {name}")
        tensor = value.full_tensor() if isinstance(value, DTensor) else value
        seen.add(canonical)
        yield names[canonical], tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
    if seen != names.keys():
        raise ValueError(f"Missing native hybrid tensors: {sorted(names.keys() - seen)}")
