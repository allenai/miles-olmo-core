"""MILES opts into Core scoring parity and accepts equivalent legacy manifests."""

from types import SimpleNamespace

import pytest

pytest.importorskip("olmo_core")

from miles.backends.core_utils import checkpoint, moe_models


def test_scoring_rounding_enabled_for_named_blocks_and_overrides():
    blocks = [SimpleNamespace(routed_experts=SimpleNamespace(match_eager_rounding=False)) for _ in range(3)]
    config = SimpleNamespace(block={"kda": blocks[0], "attention": blocks[1]}, block_overrides={7: blocks[2]})
    options = SimpleNamespace(activation_checkpointing=False, row_specialization="dynamic")
    moe_models.prepare_model_config(config, SimpleNamespace(layer_types=[]), options)
    assert all(block.routed_experts.match_eager_rounding for block in blocks)


def test_legacy_manifest_rounding_matches_explicit_opt_in():
    legacy = {"block": {"routed_experts": {"hidden_size": 128}}}
    enabled = {"block": {"routed_experts": {"hidden_size": 128, "match_eager_rounding": True}}}
    disabled = {"block": {"routed_experts": {"hidden_size": 128, "match_eager_rounding": False}}}
    assert checkpoint.comparable_model_config(legacy) == checkpoint.comparable_model_config(enabled)
    assert checkpoint.comparable_model_config(legacy) != checkpoint.comparable_model_config(disabled)
    disabled["block"]["routed_experts"]["match_eager_rounding"] = "false"
    with pytest.raises(ValueError, match="match_eager_rounding"):
        checkpoint.comparable_model_config(disabled)
