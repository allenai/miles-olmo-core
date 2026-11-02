"""Prepared batch task ownership independent of Open Instruct and GPUs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.backends.core_utils.rollout import prepared_batch
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.audit_utils.event_logger import logger as event_logger
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity


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
        generate_rollout=SimpleNamespace(release_reservation=AsyncMock(return_value=[])),
        data_source=SimpleNamespace(acknowledge_groups=Mock(), requeue_pending_groups=Mock()),
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
    e.data_source.acknowledge_groups.assert_called_once_with([])


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


@pytest.mark.parametrize("failure", ["tracking", "event_logger", "logging"])
async def test_telemetry_failure_does_not_discard_ready_batch(monkeypatch, failure):
    e = executor()
    batch = object()
    e._prepare_rollout = AsyncMock(return_value=batch)
    monkeypatch.setattr(prepared_batch, "compute_rollout_step", lambda *_: 7)
    monkeypatch.setattr(prepared_batch.tracking._manager, "log", Mock())
    monkeypatch.setattr(prepared_batch.tracking, "is_event_logger_initialized", lambda: failure == "event_logger")
    error = Mock(side_effect=OSError("telemetry unavailable"))
    if failure == "tracking":
        monkeypatch.setattr(prepared_batch.tracking._manager, "log", error)
    elif failure == "event_logger":
        monkeypatch.setattr(prepared_batch.tracking, "get_event_logger", lambda: SimpleNamespace(log=error))
    else:
        monkeypatch.setattr(prepared_batch.logger, "info", error)
        monkeypatch.setattr(prepared_batch.logger, "warning", error)
    p = prepared_batch.PreparedBatch(e, 7)
    e._weight_versions_of_model_id[None] = 12
    assert await p.consume(7) is batch
    assert error.called
    e.data_source.acknowledge_groups.assert_called_once_with([])
    e.data_source.requeue_pending_groups.assert_not_called()


async def test_prepared_handoff_carries_rollout_audit_context(tmp_path, monkeypatch):
    logger = event_logger.EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="rollout_executor"))
    monkeypatch.setattr(event_logger, "_event_logger", logger)
    batch = object()

    async def consume(rollout_id):
        assert logger._context_var.get() == {"rollout_id": rollout_id}
        return batch

    e = SimpleNamespace(
        _core_prepared_batch=SimpleNamespace(consume=consume),
        _rollouts_since_publish_of_model_id={None: 0},
    )
    assert await RolloutExecutor.core_get_prepared(e, 7) is batch
    assert logger._context_var.get({}) == {}
    assert e._core_prepared_batch is None and e.rollout_id == 7
