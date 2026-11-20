"""Tail statistics of the trainer-vs-rollout log-probability gap.

``train_rollout_logprob_abs_diff`` is a masked mean; these cover the masked
maximum and the fractions of masked tokens above 0.5 and 1.0, and the MAX
reduction of the maximum in ``aggregate_train_losses``.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_parallel_state
from tests.fast.backends.training_utils.test_true_on_policy_loss_metrics import (
    _make_args,
    _patch_single_rank_loss_helpers,
)
from torch import distributed as dist

from miles.backends.training_utils.loss.hub import losses as loss_utils
from miles.backends.training_utils.metrics import log_utils

MAX_KEY = "train_rollout_logprob_abs_diff_max"


def test_tail_metrics_cover_only_masked_response_tokens(monkeypatch):
    args = _make_args(use_rollout_logprobs=False)
    train_log_probs = torch.tensor([-0.10, -0.20, -3.00, -0.40, -9.00])
    rollout_log_probs = torch.tensor([-0.12, -0.90, -1.50, -0.40, -0.10])
    # |diff| = 0.02, 0.70, 1.50, 0.00 and a masked 8.90 that must not count.
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    batch = {
        "advantages": [torch.ones(5)],
        "log_probs": [train_log_probs.clone()],
        "rollout_log_probs": [rollout_log_probs],
        "unconcat_tokens": [torch.arange(6)],
        "response_lengths": [5],
        "total_lengths": [6],
        "loss_masks": [mask],
    }
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(tp=SimpleNamespace(group=None)))
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {"log_probs": [train_log_probs.clone()], "entropy": [torch.zeros(5)]},
    )

    _, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 6, 8)),
        sum_of_sample_mean=lambda tensor: (tensor.float() * mask).sum() / mask.sum(),
    )

    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff"], torch.tensor(2.22 / 4))
    torch.testing.assert_close(metrics[MAX_KEY], torch.tensor(1.5))
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff_frac_gt_0p5"], torch.tensor(0.5))
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff_frac_gt_1p0"], torch.tensor(0.25))


def test_tis_rejected_token_sets_neither_maximum_nor_fractions(monkeypatch):
    """The maximum and the fractions describe the same tokens as the mean."""
    args = _make_args(use_rollout_logprobs=False)
    args.get_mismatch_metrics = True
    args.custom_tis_function_path = "rejects.largest.gap"
    train_log_probs = torch.tensor([-0.10, -0.20, -3.00, -0.40, -9.00])
    rollout_log_probs = torch.tensor([-0.12, -0.90, -1.50, -0.40, -0.10])
    mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    # The hook rejects the 1.50 gap; the remaining active gaps are 0.02, 0.70 and 0.00.
    kept = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    batch = {
        "advantages": [torch.ones(5)],
        "log_probs": [train_log_probs.clone()],
        "rollout_log_probs": [rollout_log_probs],
        "unconcat_tokens": [torch.arange(6)],
        "response_lengths": [5],
        "total_lengths": [6],
        "loss_masks": [mask],
    }
    monkeypatch.setattr(loss_utils, "get_parallel_state", lambda: SimpleNamespace(tp=SimpleNamespace(group=None)))
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {"log_probs": [train_log_probs.clone()], "entropy": [torch.zeros(5)]},
    )
    monkeypatch.setattr(
        loss_utils, "load_function", lambda path: lambda pg_loss, **kwargs: (pg_loss, [kept.clone()], {})
    )
    # Decoupled rejection: the numerator uses the kept tokens, the denominator the original mask.
    monkeypatch.setattr(
        loss_utils,
        "get_sum_of_sample_mean",
        lambda *a, **kw: lambda tensor: (tensor.float() * a[2][0]).sum() / mask.sum(),
    )

    _, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 6, 8)),
        sum_of_sample_mean=lambda tensor: (tensor.float() * mask).sum() / mask.sum(),
    )

    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff"], torch.tensor(0.72 / 4))
    torch.testing.assert_close(metrics[MAX_KEY], torch.tensor(0.7))
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff_frac_gt_0p5"], torch.tensor(0.25))
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff_frac_gt_1p0"], torch.tensor(0.0))


def _microbatch(count, values):
    return {"keys": list(values), "values": torch.tensor([count, *values.values()])}


@pytest.mark.parametrize("num_rollouts", [None, 4])
def test_aggregate_takes_maximum_without_divisor(monkeypatch, num_rollouts):
    make_parallel_state()
    reductions = []
    monkeypatch.setattr(
        log_utils.MultiPGUtil,
        "all_reduce",
        staticmethod(lambda tensor, groups, op: reductions.append((op, tensor.numel()))),
    )
    assert MAX_KEY in log_utils.MAX_REDUCED_LOSS_KEYS
    losses = [
        _microbatch(2, {"loss": 2.0, MAX_KEY: 0.75, "after": 1.0}),
        _microbatch(2, {"loss": 6.0, MAX_KEY: 1.5, "after": 3.0}),
        _microbatch(2, {"loss": 0.0, MAX_KEY: 0.25, "after": 0.0}),
    ]

    out = log_utils.aggregate_train_losses(losses, num_rollouts=num_rollouts)

    divisor = 6 if num_rollouts is None else num_rollouts
    assert math.isclose(out[MAX_KEY], 1.5)
    assert math.isclose(out["loss"], 8.0 / divisor)
    assert math.isclose(out["after"], 4.0 / divisor)
    # One SUM over [count, sums...] and one MAX over the extrema, per group.
    assert reductions == [(dist.ReduceOp.SUM, 4), (dist.ReduceOp.MAX, 1)]
    # The caller's per-microbatch payloads are not modified.
    assert losses[0]["values"].tolist() == [2.0, 2.0, 0.75, 1.0]
