"""Expert scheduling preserves samples, update membership and pack constraints."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from tests.fast.backends.core_utils.helpers import histograms, hook_args, sample_groups

from miles.backends.core_utils import expert_schedule as schedule


def test_disabled_hook_leaves_samples_untouched_without_model_inputs():
    args = SimpleNamespace(olmo_core=SimpleNamespace(expert_balanced_packing=False))
    groups = sample_groups()
    original = [list(group) for group in groups]
    schedule.reorder_samples(args, groups)
    assert all(
        actual is expected
        for group, old_group in zip(groups, original, strict=True)
        for actual, expected in zip(group, old_group, strict=True)
    )


def test_plan_improves_split_routing_and_preserves_every_sample():
    samples = sum(sample_groups(), [])
    lengths = [len(s.tokens) for s in samples]
    params = dict(world=4, ep_degree=2, max_tokens=12)
    order, before, after = schedule.plan_order(lengths, histograms(samples), **params)
    assert sorted(order) == list(range(16))
    assert order == schedule.plan_order(lengths, histograms(samples), **params)[0]
    assert after["skew_mean"] < before["skew_mean"]
    assert after["critical_work_proxy"] < before["critical_work_proxy"]
    membership = schedule.schedule(order, lengths, world=4, max_tokens=12)
    assert all(sum(lengths[i] for i in pack) <= 12 for rank in membership for pack in rank)
    assert all(sum(map(len, rank)) == 4 for rank in membership)
    json.dumps(after, allow_nan=False)


def test_actual_boundaries_and_world_wide_equalization():
    # Near-equal row lengths, equal pack counts, different pack membership.
    lengths = [56, 56, 45, 44, 44, 44]
    ranks = schedule.schedule(list(range(6)), lengths, world=2, max_tokens=100)
    assert ranks == [[[0], [2, 4]], [[1, 3], [5]]]
    # Equalization must include other EP groups, not just each group's ranks.
    lengths = [2, 2, 9, 9] * 4
    ranks = schedule.schedule(list(range(16)), lengths, world=4, max_tokens=12)
    assert [len(rank) for rank in ranks] == [4, 4, 4, 4]


@pytest.mark.parametrize("malformation", ["missing", "length", "dtype", "negative", "range"])
def test_invalid_replay_fails(malformation):
    sample = sample_groups()[0][0]
    routes = sample.rollout_routed_experts
    if malformation == "missing":
        sample.rollout_routed_experts = None
    elif malformation == "length":
        sample.rollout_routed_experts = routes[:-1]
    elif malformation == "dtype":
        sample.rollout_routed_experts = routes.astype(float)
    else:
        routes[0, 1, 0] = -1 if malformation == "negative" else 4
    with pytest.raises(ValueError):
        histograms([sample])


def test_hook_preserves_blocks_metadata_and_trimmed_tail(tmp_path):
    args = hook_args(tmp_path)
    groups = sample_groups(36)
    original = sum(groups, [])
    metadata = {
        s.index: (s.group_index, s.reward, list(s.weight_versions), s.rollout_routed_experts.copy()) for s in original
    }
    schedule.reorder_samples(args, groups)
    reordered = sum(groups, [])
    assert [s.index for s in reordered] != [s.index for s in original]
    for start in [0, 16]:
        assert {s.index for s in reordered[start : start + 16]} == set(range(start, start + 16))
    assert all(a is b for a, b in zip(reordered[32:], original[32:], strict=True))
    for s in reordered:
        group, reward, versions, routes = metadata[s.index]
        assert (s.group_index, s.reward, s.weight_versions) == (group, reward, versions)
        np.testing.assert_array_equal(s.rollout_routed_experts, routes)


@pytest.mark.parametrize(
    "field,value", [("group_index", None), ("index", None), ("rollout_id", 3), ("rollout_routed_experts", None)]
)
def test_hook_rejects_unsafe_inputs_before_mutating(tmp_path, field, value):
    args = hook_args(tmp_path)
    groups = sample_groups()
    original = [s.index for s in sum(groups, [])]
    setattr(groups[-1][-1], field, value)
    with pytest.raises(ValueError):
        schedule.reorder_samples(args, groups)
    if field != "index":
        assert [s.index for s in sum(groups, [])] == original


def test_identity_fallback_never_regresses_selected_load_metrics():
    rng = np.random.default_rng(71)
    for _ in range(40):
        lengths = rng.integers(2, 13, size=16).tolist()
        counts = np.stack([rng.multinomial(n * 2, [0.2, 0.8], size=3) for n in lengths])
        order, before, after = schedule.plan_order(lengths, counts, world=4, ep_degree=2, max_tokens=12)
        for key in before:
            assert after[key] <= before[key]
        assert sorted(order) == list(range(16))
    identical = np.ones((16, 2, 2), dtype=np.int64)
    assert schedule.plan_order([6] * 16, identical, world=4, ep_degree=2, max_tokens=12)[0] == list(range(16))
