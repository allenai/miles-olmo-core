"""Prepared batch task ownership independent of Open Instruct and GPUs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.backends.core_utils.rollout import prepared_batch


def executor():
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(publication_mode="refresh", prepared_batch_overlap=True),
        fully_async=True,
        offload_rollout=False,
        rollout_batch_size=128,
        n_samples_per_prompt=8,
        global_batch_size=1024,
        update_weights_interval=1,
        async_data_buffer_capacity_factor=4,
    )
    return SimpleNamespace(
        args=args,
        _weight_versions_of_model_id={None: 11},
        generate_rollout=SimpleNamespace(release_reservation=AsyncMock()),
    )


async def test_one_version_ahead_and_wait_overlaps_work(monkeypatch):
    e = executor()
    ready = asyncio.Event()
    batch = object()

    async def prepare(rollout_id, **kwargs):
        assert rollout_id == 7
        assert kwargs == {"weight_version": 12, "reserve": True}
        ready.set()
        return batch

    e._prepare_rollout = prepare
    monkeypatch.setattr(prepared_batch.tracking, "log", Mock())
    monkeypatch.setattr(prepared_batch, "compute_rollout_step", lambda *_: 7)
    p = prepared_batch.PreparedBatch(e, 7)
    await ready.wait()  # a trainer could be running here
    e._weight_versions_of_model_id[None] = 12
    assert await p.consume(7) is batch
    e.generate_rollout.release_reservation.assert_awaited_once_with(consume=True)


@pytest.mark.parametrize("ready", [False, True])
async def test_close_joins_and_releases_only_completed_storage(monkeypatch, ready):
    e = executor()
    started = asyncio.Event()
    joined = asyncio.Event()
    batch = object()

    async def prepare(*_args, **_kwargs):
        started.set()
        try:
            if not ready:
                await asyncio.Event().wait()
            return batch
        finally:
            joined.set()

    e._prepare_rollout = prepare
    remove = Mock()
    monkeypatch.setattr(prepared_batch, "remove_rollout_data_refs", remove)
    p = prepared_batch.PreparedBatch(e, 1)
    await started.wait()
    await p.close()
    assert joined.is_set()
    e.generate_rollout.release_reservation.assert_awaited_once_with(consume=False)
    assert remove.call_count == int(ready)


async def test_cancel_consumer_does_not_orphan_preparation(monkeypatch):
    e = executor()
    started = asyncio.Event()

    async def prepare(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    e._prepare_rollout = prepare
    p = prepared_batch.PreparedBatch(e, 1)
    waiter = asyncio.create_task(p.consume(1))
    await started.wait()
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    assert not p.task.done()
    await p.close()
    assert p.task.cancelled()
