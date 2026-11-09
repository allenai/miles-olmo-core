"""Slow observation storage must not stall delivery or lose cancelled attempts."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from miles.backends.core_utils.rollout import observation_io, sibling_timing


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
        writer = observation_io.ObservationWriter()
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
            assert not first.done() and not second.done()
            release.set()
            await first
            with pytest.raises(asyncio.CancelledError):
                await second
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
            row = json.loads(next(tmp_path.glob("sibling_timing_*.jsonl")).read_text())
            assert row["group_finished_monotonic"] == stamp
            assert row["samples"][0]["engine_weight_versions"] == [{"version": 1}]
            assert "outcome" not in row["samples"][0]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await observation_io.close(producer)

    asyncio.run(scenario())


def test_cancelled_io_failure_preserves_cancellation_and_reports_missing_record(monkeypatch, tmp_path, caplog):
    entered, release = threading.Event(), threading.Event()

    def broken(rows):
        entered.set()
        assert release.wait(5)
        raise OSError("storage unavailable")

    monkeypatch.setattr(observation_io, "_append", broken)

    async def scenario():
        writer = observation_io.ObservationWriter()
        task = asyncio.create_task(writer.append([(tmp_path / "rows", {})]))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert "storage unavailable" in caplog.text
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await writer.close()

    asyncio.run(scenario())
