from types import SimpleNamespace

import pytest

from miles.ray import train_actor
from miles.ray.train_actor import TrainRayActor
from miles.utils.tracking_utils import tracking, wandb_utils
from miles.utils.tracking_utils.base import TrackingManager, WandbBackend

_TRAIN_PARALLEL_CONFIG = {"dp_size": 2, "cp_size": 1}


def _make_actor(*, rank: int) -> TrainRayActor:
    actor = object.__new__(TrainRayActor)
    actor.args = SimpleNamespace(rank=rank)
    actor.train_parallel_config = _TRAIN_PARALLEL_CONFIG
    return actor


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("finish_fails", [False, True])
def test_trainer_teardown_closes_distributed_and_tracking_resources(monkeypatch, initialized, finish_fails):
    """Core inherits this teardown; W&B's service must close even when finishing a run fails."""
    calls = []

    def finish():
        calls.append("finish")
        if finish_fails:
            raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(
        train_actor,
        "dist",
        SimpleNamespace(
            is_initialized=lambda: initialized,
            get_rank=lambda: 0,
            get_world_size=lambda: 1,
            destroy_process_group=lambda: calls.append("destroy"),
        ),
    )
    manager = TrackingManager({})
    manager._backends = [WandbBackend()]
    monkeypatch.setattr(tracking, "_manager", manager)
    monkeypatch.setattr(wandb_utils.wandb, "finish", finish)
    monkeypatch.setattr(wandb_utils.wandb, "teardown", lambda: calls.append("teardown"))

    _make_actor(rank=0).dispose()

    assert calls == (["destroy"] if initialized else []) + ["finish", "teardown"]
    assert manager._backends == []
