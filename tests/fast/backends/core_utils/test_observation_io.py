"""Slow observation storage must not stall delivery or lose cancelled attempts."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from miles.backends.core_utils.rollout import async_rollout, observation_io, refreshing_rollout, sibling_timing
from miles.utils.types import Sample


def test_slow_writes_are_serial_and_cancellation_drains_waiting_records(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    native = observation_io._append
    loop_thread = threading.get_ident()

    def blocked(rows):
        assert threading.get_ident() != loop_thread
        calls.append(rows)
        entered.set()
        assert release.wait(5)
        native(rows)

    monkeypatch.setattr(observation_io, "_append", blocked)

    async def scenario():
        writer = observation_io.ObservationWriter(max_pending=1)
        path = tmp_path / "rows.jsonl"
        first = asyncio.create_task(writer.append([(path, {"n": 1})]))
        second = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            second = asyncio.create_task(writer.append([(path, {"n": 2})]))
            await asyncio.sleep(0)
            second.cancel()
            await asyncio.sleep(0)
            second.cancel()
            # Ready delivery runs while one write executes and another waits.
            assert await asyncio.create_task(asyncio.sleep(0, result="ready")) == "ready"
            assert len(calls) == 1
            assert first.done() and not second.done()
            assert len(writer._pending) == 1
            release.set()
            await first
            with pytest.raises(asyncio.CancelledError):
                await second
            await writer.flush()
            assert [json.loads(line) for line in path.read_text().splitlines()] == [{"n": 1}, {"n": 2}]
        finally:
            release.set()
            await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)
            await writer.close()
        with pytest.raises(RuntimeError, match="closed"):
            await writer.append([(path, {})])

    asyncio.run(scenario())


def test_sibling_snapshot_is_detached_and_timestamped_before_io(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    native = observation_io._append
    captured = []

    def blocked(rows):
        captured.extend(rows)
        entered.set()
        assert release.wait(5)
        native(rows)

    monkeypatch.setattr(observation_io, "_append", blocked)

    async def scenario():
        producer = SimpleNamespace(args=SimpleNamespace(save=str(tmp_path)))
        sample = SimpleNamespace(group_index=1, index=0, metadata={})
        records = sibling_timing.start_group([sample])
        records[0]["engine_weight_versions"] = [{"version": 1}]
        task = asyncio.create_task(sibling_timing.write_group(producer, records, "cancelled"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            stamp = captured[0][1]["group_finished_monotonic"]
            records[0]["engine_weight_versions"][0]["version"] = 9
            records[0]["outcome"] = "mutated"
            release.set()
            await task
            await producer._observation_writer.flush()
            row = json.loads(next(tmp_path.glob("sibling_timing_*.jsonl")).read_text())
            assert row["group_finished_monotonic"] == stamp
            assert row["samples"][0]["engine_weight_versions"] == [{"version": 1}]
            assert "outcome" not in row["samples"][0]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await observation_io.close(producer)

    asyncio.run(scenario())


def test_io_failure_is_reported_at_flush_and_later_rows_survive(monkeypatch, tmp_path, caplog):
    native = observation_io._append

    def broken(rows):
        if rows[0][1]["n"] == 1:
            raise OSError("storage unavailable")
        native(rows)

    monkeypatch.setattr(observation_io, "_append", broken)

    async def scenario():
        writer = observation_io.ObservationWriter()
        path = tmp_path / "rows.jsonl"
        try:
            await writer.append([(path, {"n": 1})])
            await writer.append([(path, {"n": 2})])
            await writer.flush()
            assert "storage unavailable" in caplog.text
            assert json.loads(path.read_text()) == {"n": 2}
        finally:
            await writer.close()

    asyncio.run(scenario())


def producer(tmp_path, monkeypatch):
    def parent_init(fn, input):
        fn.args = SimpleNamespace(
            save=str(tmp_path),
            olmo_core=SimpleNamespace(pipeline_observation_interval=1),
            sglang_server_concurrency=8,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
        )
        fn.state = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1]))
        fn._scheduler = SimpleNamespace(samples_in_flight=0)
        fn._active_tasks, fn._producing_groups = set(), {}
        fn._worker = fn._output = None
        fn._stopping = fn._publication_paused = False
        fn._producer_resumed = asyncio.Event()
        fn._producer_resumed.set()

    # Use the real refresh constructor while avoiding unrelated data/engine setup.
    monkeypatch.setattr(async_rollout.ManagedFullyAsyncRolloutFn, "__init__", parent_init)
    return refreshing_rollout.RefreshingRolloutFn(None)


def test_64_finished_groups_and_publication_do_not_wait_for_storage(monkeypatch, tmp_path):
    entered, release = threading.Event(), threading.Event()
    native = observation_io._append

    def blocked(rows):
        entered.set()
        assert release.wait(5)
        native(rows)

    async def complete(self, samples):
        return samples

    monkeypatch.setattr(observation_io, "_append", blocked)
    monkeypatch.setattr(async_rollout.ManagedFullyAsyncRolloutFn, "_generate_group", complete)

    async def scenario():
        fn = producer(tmp_path, monkeypatch)
        try:
            groups = [[Sample(prompt="q", group_index=i, index=i)] for i in range(64)]
            result = await asyncio.wait_for(asyncio.gather(*(fn._generate_group(g) for g in groups)), 2)
            assert len(result) == 64
            assert await asyncio.to_thread(entered.wait, 2)
            # The filesystem has completed zero writes, but both publication
            # boundaries must return while the 64 timing rows remain queued.
            await asyncio.wait_for(fn.prepare_publication(), 1)
            await asyncio.wait_for(fn.finish_publication(), 1)
            assert fn._producer_resumed.is_set()
            assert not release.is_set()
            assert len(fn._observation_writer._pending) == 66
            closing = asyncio.create_task(fn.dispose())
            await asyncio.sleep(0)
            assert not closing.done()
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await closing
            rows = [
                json.loads(line)
                for path in tmp_path.glob("sibling_timing_*.jsonl")
                for line in path.read_text().splitlines()
            ]
            assert len(rows) == 64
            lifecycle = [json.loads(line) for line in (tmp_path / "pipeline_lifecycle.jsonl").read_text().splitlines()]
            assert [r["event"] for r in lifecycle] == ["generation_paused", "generation_resumed"]
            assert [r["producer_admission_open"] for r in lifecycle] == [False, True]
            assert all(r["schema_version"] == 2 for r in lifecycle)
        finally:
            release.set()
            await fn.dispose()

    asyncio.run(scenario())


@pytest.mark.parametrize("serving_failure", [True, False])
def test_cancellation_during_full_queue_preserves_serving_error(monkeypatch, tmp_path, serving_failure):
    release = threading.Event()
    native = observation_io._append

    def blocked(rows):
        assert release.wait(5)
        native(rows)

    async def finish(self, samples):
        if serving_failure:
            raise ValueError("original serving failure")
        return samples

    monkeypatch.setattr(observation_io, "_append", blocked)
    monkeypatch.setattr(async_rollout.ManagedFullyAsyncRolloutFn, "_generate_group", finish)

    async def scenario():
        fn = producer(tmp_path, monkeypatch)
        fn._observation_writer = observation_io.ObservationWriter(max_pending=1)
        await fn._observation_writer.append([(tmp_path / "dummy", {})])
        task = asyncio.create_task(fn._generate_group([Sample(prompt="q", group_index=0, index=0)]))
        try:
            async with asyncio.timeout(2):
                while not fn._observation_writer._lock.locked():
                    await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with pytest.raises(ValueError if serving_failure else asyncio.CancelledError):
                await task
            await fn._observation_writer.flush()
            row = json.loads(next(tmp_path.glob("sibling_timing_*.jsonl")).read_text())
            assert row["outcome"] == ("ValueError" if serving_failure else "completed")
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await fn.dispose()

    asyncio.run(scenario())
