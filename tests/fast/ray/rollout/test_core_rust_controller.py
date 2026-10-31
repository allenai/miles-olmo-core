"""Core reconciles Rust membership only for the instances it published to."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.fast.ray.rollout.test_inference_controller import _make_controller, _RecordingServer


def cell(instance, pending=False):
    return SimpleNamespace(
        meta=SimpleNamespace(workers_hash=instance),
        is_pending_weights=pending,
        is_pending_weights_or_serving=True,
        is_serving=not pending,
        mark_weights_ready=AsyncMock(),
    )


@pytest.mark.parametrize("abort", [False, True])
async def test_publication_checks_instance_and_reconciles_serving_workers(abort):
    cells = {
        "serving": cell("one"),
        "pending": cell("two", True),
        "replaced": cell("new"),
        "unpublished": cell("four"),
    }
    controller = _make_controller({"default": _RecordingServer(cells)})
    controller.args.train_backend = "olmo_core"
    await controller.context_lock.acquire()
    controller.context_lock.detach()
    if abort:
        await controller.abort_update_weights()
    else:
        await controller.end_update_weights({"serving": "one", "pending": "two", "replaced": "old"})
    for name, worker in cells.items():
        assert worker.mark_weights_ready.await_count == int(not abort and name in ("serving", "pending"))


async def test_membership_failure_waits_for_other_registrations_before_unlocking():
    cells = {"failed": cell("one"), "slow": cell("two")}
    controller = _make_controller({"default": _RecordingServer(cells)})
    controller.args.train_backend = "olmo_core"
    cells["failed"].mark_weights_ready.side_effect = TimeoutError("router")
    settled = []

    async def finish_registration():
        await asyncio.sleep(0.01)
        assert controller.context_lock.locked
        settled.append(True)

    cells["slow"].mark_weights_ready.side_effect = finish_registration
    await controller.context_lock.acquire()
    controller.context_lock.detach()
    with pytest.raises(TimeoutError, match="router"):
        await controller.end_update_weights({"failed": "one", "slow": "two"})
    assert settled == [True]
    assert not controller.context_lock.locked
