"""Retained NumPy routes and refreshed-policy spans survive artifact serialization."""

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from miles.ray.rollout import debug_data
from miles.utils import types


def test_replay_arrays_and_refreshed_policy_spans_round_trip(tmp_path: Path):
    template = str(tmp_path / "rollout_{rollout_id}.pt")
    original = types.Sample(
        tokens=[3, 10, 11],
        response_length=2,
        rollout_log_probs=[-0.125, -0.25],
        rollout_routed_experts=np.arange(8, dtype=np.int32).reshape(2, 2, 2),
        weight_versions=[
            types.WeightVersionsPerCall(spans=[types.WeightVersionSpan("3", 1, 2), types.WeightVersionSpan("4", 2, 3)])
        ],
        metadata={"tensor": torch.tensor([[1, 2]], dtype=torch.int64)},
    )
    debug_data.save_debug_rollout_data(
        Namespace(save_debug_rollout_data=template, save_debug_trajectory_data=None, rollout_sample_rate=1.0),
        [original],
        rollout_id=0,
        evaluation=False,
    )
    [loaded], _ = debug_data.load_debug_rollout_data(
        Namespace(load_debug_rollout_data=template, load_debug_rollout_data_subsample=None), rollout_id=0
    )
    assert loaded.tokens == original.tokens
    assert loaded.response_length == original.response_length
    assert loaded.rollout_log_probs == original.rollout_log_probs
    assert loaded.weight_versions == original.weight_versions
    np.testing.assert_array_equal(loaded.rollout_routed_experts, original.rollout_routed_experts, strict=True)
    assert torch.equal(loaded.metadata["tensor"], original.metadata["tensor"])
