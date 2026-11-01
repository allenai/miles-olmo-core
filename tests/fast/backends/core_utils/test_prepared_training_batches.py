"""Prepared packs retain storage while receiving newly computed training fields."""

import pytest
import torch

from miles.backends.core_utils import data, packing


@pytest.mark.parametrize("packed", [False, True])
def test_bind_matches_fresh_preparation_after_scoring_and_advantages(packed):
    rollout = {
        "tokens": [torch.arange(4), torch.arange(5), torch.arange(3)],
        "total_lengths": [4, 5, 3],
        "response_lengths": [2, 3, 1],
        "loss_masks": [torch.ones(2), torch.ones(3), torch.ones(1)],
        "weight_versions": [[1], [1], [1]],
    }
    samples = data.sample_batches(rollout, 12)
    plan = [[0, 1], [2]] if packed else [[0], [1], [2]]
    prepared = [packing.combine(samples, indices) for indices in plan] if packed else samples
    pointers = [b["tokens"].data_ptr() for b in prepared]
    for key in ("advantages", "returns", "log_probs", "ref_log_probs"):
        rollout[key] = [torch.randn(n) for n in rollout["response_lengths"]]
    packing.bind_training_fields([prepared], rollout)
    fresh = data.sample_batches(rollout, 12)
    fresh = [packing.combine(fresh, indices) for indices in plan] if packed else fresh
    assert [b["tokens"].data_ptr() for b in prepared] == pointers
    for got, expected in zip(prepared, fresh, strict=True):
        assert got.keys() == expected.keys()
        for key in got:
            if isinstance(got[key], torch.Tensor):
                torch.testing.assert_close(got[key], expected[key])
            elif isinstance(got[key], list) and got[key] and isinstance(got[key][0], torch.Tensor):
                for a, b in zip(got[key], expected[key], strict=True):
                    torch.testing.assert_close(a, b)
            else:
                assert got[key] == expected[key]
