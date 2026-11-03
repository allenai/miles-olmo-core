"""Behavior drift uses real token weights, masks and cross-rank reductions."""

import math
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from miles.backends.core_utils import metrics, observability


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        length = 4 if rank == 0 else 4097
        scores = torch.zeros(length, dtype=torch.float64)
        scores[:3] = (
            torch.tensor([math.log(4), math.log(0.25), 0]) if rank == 0 else torch.tensor([math.log(2), 0, 100])
        )
        mask = torch.zeros(length)
        mask[: 3 if rank == 0 else 2] = 1
        spans = (
            [dict(version=0, start=0, end=1), dict(version=10, start=1, end=2), dict(version=12, start=2, end=length)]
            if rank == 0
            else [dict(version=0, start=0, end=1), dict(version=12, start=1, end=length)]
        )
        rollout = dict(
            log_probs=[scores],
            rollout_log_probs=[torch.zeros_like(scores)],
            loss_masks=[mask],
            metadata=[dict(policy_refresh=dict(spans=spans, replay_version=12))],
        )
        for empty_rank in (False, True):
            if empty_rank and rank == 1:
                mask.zero_()
            result = observability.behavior_metrics(rollout, current_version=12, clip_low=0.5, clip_high=2.01)
            weights = [4, 0.25, 1] if empty_rank else [4, 0.25, 1, 2, 1]
            assert result["behavior/all/tokens"] == len(weights)
            assert result["behavior/all/ess_ratio"] == pytest.approx(
                sum(weights) ** 2 / (len(weights) * sum(w * w for w in weights))
            )
            assert result["behavior/all/tis_clip_low_fraction"] == pytest.approx(1 / len(weights))
            assert result["behavior/all/tis_clip_high_fraction"] == pytest.approx(1 / len(weights))
            assert result["behavior/age/0/tokens"] == (1 if empty_rank else 2)
            assert result["behavior/age/9_12/tokens"] == (1 if empty_rank else 2)
            assert result["behavior/age/13_plus/tokens"] == 0
            assert result["behavior/age/13_plus/ess_ratio"] == 0
            assert result["behavior/length/le_4096/tokens"] == 3
            assert result["behavior/length/4097_8192/tokens"] == (0 if empty_rank else 2)
            assert result["behavior/historical_prefix/tokens"] == (2 if empty_rank else 3)
            assert result["behavior/length/4097_8192/historical_prefix/tokens"] == (0 if empty_rank else 1)
            assert result["behavior/all/max_abs_logratio"] == pytest.approx(math.log(4))
            assert result["behavior/all/p95_abs_logratio_overflow"] == 1
            assert all(math.isfinite(v) for v in result.values())
        # Equal, enormous ratios still have ESS one; exp(delta) would overflow.
        scores.fill_(1000)
        result = observability.behavior_metrics(rollout, current_version=12, clip_low=0, clip_high=2)
        assert result["behavior/all/ess_ratio"] == pytest.approx(1)
        assert result["behavior/all/tis_clip_high_fraction"] == 1
        with mock.patch.object(
            torch.cuda,
            "memory_stats",
            return_value={
                "allocated_bytes.all.current": 100 + rank,
                "reserved_bytes.all.current": 1000 + rank,
                "reserved_bytes.all.peak": 2000 + rank,
                "inactive_split_bytes.all.current": 300 + rank,
                "num_alloc_retries": 11 + rank,
                "num_ooms": 2,
            },
        ):
            result = observability.allocator_metrics({"num_alloc_retries": 10, "num_ooms": 2}, device="cpu")
        assert result["allocator/rank_0/alloc_retries"] == 1
        assert result["allocator/rank_1/alloc_retries"] == 2
        assert result["allocator/alloc_retries_sum"] == 3
        assert result["allocator/ooms_sum"] == 0
        assert result["allocator/reserved_bytes_max"] == 1001
    finally:
        dist.destroy_process_group()


def test_global_behavior_buckets_and_allocator_deltas(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_length_and_behavior_diagnostics_reach_tracking():
    profile = dict(
        max_abs=2,
        p50_upper=0.01,
        p95_upper=None,
        p99_upper=None,
        groups={"length_gt_4096": {"tokens": 20, "mean_abs": 0.02}},
        behavior_metrics={"behavior/age/9_12/ess_ratio": 0.7},
    )
    scores = metrics.score_metrics(0.01, 22, profile)
    args = SimpleNamespace(entropy_coef=0, observe_training_entropy=False)
    with mock.patch.object(metrics.dist, "get_rank", return_value=0), mock.patch.object(
        metrics.tracking, "log"
    ) as log:
        result = metrics.log_step(
            args,
            losses={},
            summary={},
            scores=scores,
            clock=SimpleNamespace(completed_steps=2, published_step=1),
            rollout_id=1,
            lr_used=[],
            lr_next=[],
            optimizer_metrics={},
        )
    assert result["train/collection_score/length_gt_4096/mean_abs"] == 0.02
    assert result["train/behavior/age/9_12/ess_ratio"] == 0.7
    log.assert_called_once_with(args, result, step_key="train/step")
