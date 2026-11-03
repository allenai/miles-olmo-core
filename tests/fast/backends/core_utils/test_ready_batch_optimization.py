"""Fast draining still owns failure, staleness and cancellation decisions."""

import asyncio
from types import SimpleNamespace

import pytest

from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer
from miles.backends.core_utils.rollout.async_rollout import ManagedFullyAsyncRolloutFn
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample, WeightVersionsPerCall


def make_buffer(size, dropped):
    args = SimpleNamespace(
        rollout_batch_size=size,
        n_samples_per_prompt=1,
        global_batch_size=size,
        max_weight_staleness=2,
        async_data_buffer_capacity_factor=4,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )
    return HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, dropped.extend))


def entry(index, version=0):
    sample = Sample(
        index=index,
        group_index=index,
        tokens=[1, 2],
        response_length=1,
        reward=1.0,
        weight_versions=[WeightVersionsPerCall.from_dicts([dict(version=str(version), abs_start=1, abs_end=2)])],
        status=Sample.Status.COMPLETED,
    )
    return DataBufferInput(prompt_group=[sample], group=[sample])


def producer(buffer):
    result = ManagedFullyAsyncRolloutFn.__new__(ManagedFullyAsyncRolloutFn)
    result._output = buffer
    result._worker = asyncio.create_task(asyncio.Event().wait())
    result._reserved_groups = None
    result._draining_groups = []
    result._interrupted_groups = []
    result._drain_yield_at = float("inf")
    return result


def test_ready_drain_filters_stale_and_does_not_schedule_per_entry():
    async def scenario():
        dropped = []
        buffer = make_buffer(128, dropped)
        for i in range(512):
            await buffer.put(entry(i, 0 if i < 384 else 3))
        worker = producer(buffer)
        callbacks = []
        asyncio.get_running_loop().call_soon(callbacks.append, "yielded")
        try:
            found = [await worker._next_group(3) for _ in range(128)]
            assert [x.group[0].index for x in found] == list(range(384, 512))
            assert len(dropped) == 384
            assert len(worker._draining_groups) == 128
            assert callbacks == []
            assert await buffer.get_ready(current_version=3) is None
            metrics = buffer.get_metrics()
            assert metrics["rollout/fully_async/stale_groups_filtered"] == 384
        finally:
            worker._worker.cancel()
            await asyncio.gather(worker._worker, return_exceptions=True)

    asyncio.run(scenario())


def test_ready_drain_checks_worker_failure_before_consumption():
    async def scenario():
        buffer = make_buffer(1, [])
        await buffer.put(entry(0))
        worker = producer(buffer)
        worker._worker.cancel()
        await asyncio.gather(worker._worker, return_exceptions=True)
        with pytest.raises(asyncio.CancelledError):
            await worker._next_group(0)
        assert len(buffer._delegate._buffer) == 1

    asyncio.run(scenario())


def test_empty_drain_cancel_retains_unconsumed_work():
    async def scenario():
        buffer = make_buffer(1, [])
        worker = producer(buffer)
        task = asyncio.create_task(worker._next_group(0))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await buffer.put(entry(1))
        assert (await worker._next_group(0)).group[0].index == 1
        worker._worker.cancel()
        await asyncio.gather(worker._worker, return_exceptions=True)

    asyncio.run(scenario())
