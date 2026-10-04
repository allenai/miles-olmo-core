import argparse
import random
from collections import Counter

import pytest
import torch

from miles.ray.rollout import debug_data
from miles.utils.types import Sample


def samples():
    return [
        Sample(group_index=g, index=g * 4 + j, prompt=f"prompt {g}", tokens=[1, 2], response_length=1)
        for g in range(40)
        for j in range(4)
    ]


def args(tmp_path, **extra):
    return argparse.Namespace(
        save_debug_rollout_data=str(tmp_path / "rollouts" / "{rollout_id}.pt"),
        save_debug_trajectory_data=None,
        rollout_seed=42,
        **extra,
    )


@pytest.mark.parametrize("rate", [None, 0])
def test_disabled_capture_performs_no_serialization(tmp_path, monkeypatch, rate):
    settings = args(tmp_path, **({} if rate is None else {"rollout_sample_rate": rate}))
    monkeypatch.setattr(torch, "save", lambda *a, **kw: pytest.fail("capture is disabled"))
    debug_data.save_debug_rollout_data(settings, None, 0, evaluation=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rate", [1, 2.5])
def test_full_capture_clamps_above_one(tmp_path, rate):
    batch = samples()
    debug_data.save_debug_rollout_data(args(tmp_path, rollout_sample_rate=rate), batch, 0, evaluation=False)
    saved = torch.load(tmp_path / "rollouts/0.pt", weights_only=False)
    assert [s["index"] for s in saved["samples"]] == [s.index for s in batch]


def test_partial_capture_preserves_siblings_and_full_training_batch(tmp_path, monkeypatch):
    batch = samples()
    before = [s.to_dict() for s in batch]
    rng = random.getstate()
    dashboard = []
    monkeypatch.setattr(debug_data, "save_dashboard_columns", lambda data, path: dashboard.extend(data))
    debug_data.save_debug_rollout_data(args(tmp_path, rollout_sample_rate=0.3), batch, 7, evaluation=False)
    saved = torch.load(tmp_path / "rollouts/7.pt", weights_only=False)["samples"]
    counts = Counter(s["group_index"] for s in saved)
    assert 0 < len(counts) < 40
    assert set(counts.values()) == {4}
    assert [s.index for s in dashboard] == [s["index"] for s in saved]
    assert [s.to_dict() for s in batch] == before
    assert random.getstate() == rng
    selected = debug_data.sample_rollout_groups(batch[::-1], 0.3, seed=42, rollout_id=(False, 7), dataset="train")
    assert {s.index for s in selected} == {s["index"] for s in saved}


def test_eval_groups_without_indices_are_sampled_by_prompt(tmp_path):
    batch = samples()
    for sample in batch:
        sample.group_index = None
    debug_data.save_debug_rollout_data(
        args(tmp_path, rollout_sample_rate=0.3), {"math": {"samples": batch}}, 3, evaluation=True
    )
    saved = torch.load(tmp_path / "rollouts/eval_3.pt", weights_only=False)["samples"]
    counts = Counter(s["prompt"] for s in saved)
    assert 0 < len(counts) < 40
    assert set(counts.values()) == {4}


@pytest.mark.parametrize("rate", [-0.1, float("nan"), float("inf"), True])
def test_invalid_rate(tmp_path, rate):
    with pytest.raises(ValueError, match="rollout_sample_rate"):
        debug_data.save_debug_rollout_data(args(tmp_path, rollout_sample_rate=rate), samples(), 0, evaluation=False)


def test_empty_selection_creates_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_data, "sample_rollout_groups", lambda *a, **kw: [])
    debug_data.save_debug_rollout_data(args(tmp_path, rollout_sample_rate=0.1), samples(), 0, evaluation=False)
    assert list(tmp_path.iterdir()) == []
