"""Validate every released tensor name and wrapper normalization without weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles_plugins.hybrid import weights

ROOT = Path(__file__).with_name("hybrid_release")


def test_release_interchange_covers_actual_checkpoint_header():
    config = SimpleNamespace(**json.loads((ROOT / "hf_config.json").read_text()))
    shapes = json.loads((ROOT / "tensor_shapes.json").read_text())
    mapping = weights.native_to_hf_names(config.layer_types)
    assert set(mapping.values()) == shapes.keys()
    assert len(mapping) == 192
    state = {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
    native = weights.from_hf(config, state)
    for name, tensor in native.items():
        assert tensor is state[mapping[name]]
    # Name mapping must work on a streamed individual tensor and AC wrappers.
    wrapped = "blocks.0._checkpoint_wrapped_module.attention.w_q.weight"
    assert mapping[weights.strip_checkpoint_wrapper(wrapped)] == "model.layers.0.linear_attn.q_proj.weight"
    full_layer = config.layer_types.index("full_attention")
    assert mapping[f"blocks.{full_layer}.attention.w_q.weight"] == f"model.layers.{full_layer}.self_attn.q_proj.weight"


def test_release_interchange_rejects_incomplete_state():
    config = SimpleNamespace(layer_types=["linear_attention"])
    with pytest.raises(ValueError, match="tensor mismatch"):
        weights.from_hf(config, {})


def test_release_interchange_rejects_unknown_layer():
    with pytest.raises(ValueError, match="layer type"):
        weights.native_to_hf_names(["mamba"])
