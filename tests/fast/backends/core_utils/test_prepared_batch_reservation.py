"""Real producer/buffer ownership, including the production refresh polling wrapper."""

import asyncio
from types import SimpleNamespace

import pytest

from miles.backends.core_utils.publication import policy_refresh
from miles.backends.core_utils.rollout import rollout_errors
from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer, RefreshPolicyDataBuffer
from miles.backends.core_utils.rollout.async_rollout import ManagedFullyAsyncRolloutFn
from miles.backends.core_utils.rollout.refreshing_rollout import RefreshingRolloutFn
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample


class Ledger:
    def __init__(self):
        self.pending = set()
        self.requeued = []

    def acknowledge_groups(self, groups):
        for group in groups:
            self.pending.remove(group[0].group_index)

    def requeue_pending_groups(self, identities):
        assert set(identities) <= self.pending
        self.requeued.extend(identities)


@pytest.fixture(params=[ManagedFullyAsyncRolloutFn, RefreshingRolloutFn])
async def producer(request):
    cls = request.param
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=1,
        global_batch_size=2,
        max_weight_staleness=2,
        async_data_buffer_capacity_factor=1,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )
    p = cls.__new__(cls)
    p.args, p.data_source = args, Ledger()
    buffer = RefreshPolicyDataBuffer if cls is RefreshingRolloutFn else HomogeneousPolicyDataBuffer
    p._output = buffer(DataBufferConstructorInput(args, lambda groups: p.data_source.acknowledge_groups([groups])))
    p._publication_paused = p._refreshing = p._stopping = False
    p._draining_groups = p._reserved_groups = p._reservation_task = None
    p._interrupted_groups = []
    p._errors = rollout_errors.RolloutErrors()
    p._sample_filter = None
    p._drain_yield_at = float("inf")
    p._producer_resumed = asyncio.Event()
    p._worker = asyncio.create_task(asyncio.Event().wait())
    yield p
    await p.release_reservation(consume=False)
    p._worker.cancel()
    await asyncio.gather(p._worker, return_exceptions=True)


async def seed(p, index):
    sample = Sample(
        index=index,
        group_index=index,
        prompt="question",
        response="answer",
        tokens=[9, 10],
        response_length=1,
        rollout_log_probs=[-0.5],
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )
    policy_refresh.record_response(
        sample,
        dict(
            weight_version="0", weight_versions=[dict(version="0", start=0, end=1)], output_token_logprobs=[[-0.5, 10]]
        ),
    )
    p.data_source.pending.add(index)
    await p._output.put(DataBufferInput(prompt_group=[sample], group=[sample]))


async def wait_reserved(p, count):
    async def wait():
        while p._reserved_groups is None or len(p._reserved_groups) != count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 2)


@pytest.mark.parametrize("count", [0, 1, 2])
async def test_cancel_reservation_recovers_partial_and_full_batches(producer, count):
    p = producer
    for index in range(count):
        await seed(p, index)
    wrapper = asyncio.create_task(p.reserve(RolloutFnTrainInput(rollout_id=0, weight_version=0)))
    await wait_reserved(p, count)
    if count == 2:
        await wrapper
    else:
        owned = p._reservation_task
        wrapper.cancel()
        await asyncio.gather(wrapper, return_exceptions=True)
        # Cancellation belongs to release_reservation, not the cross-thread wrapper.
        assert not owned.done()
    assert p._output._delegate._capacity == 2 - count
    await p.release_reservation(consume=False)
    assert p._output._delegate._capacity == 2
    assert p._reserved_groups is p._draining_groups is p._reservation_task is None
    assert p.data_source.pending == set(range(count))
    assert p.data_source.requeued == list(range(count))
    await p.release_reservation(consume=False)
    assert p._output._delegate._capacity == 2


async def test_transfer_keeps_ledger_pending_until_consumer_commits(producer):
    p = producer
    for index in range(2):
        await seed(p, index)
    result = await p.reserve(RolloutFnTrainInput(rollout_id=0, weight_version=0))
    assert len(result.samples) == 2 and p._output._delegate._capacity == 0
    groups = await p.release_reservation(consume=True)
    assert p.data_source.pending == {0, 1}
    assert p._output._delegate._capacity == 2
    p.data_source.acknowledge_groups(groups)
    assert not p.data_source.pending


async def test_cancel_after_dequeue_recovers_group_once(producer, monkeypatch):
    p = producer
    original = p._output.get
    entered = asyncio.Event()
    parent = None

    async def get_then_cancel(**kwargs):
        entered.set()
        result = await original(**kwargs)
        # The queue get completes, but cancellation wins before _next_group
        # resumes. Its finally block must retain the successful dequeue.
        parent.cancel()
        return result

    monkeypatch.setattr(p._output, "get", get_then_cancel)
    wrapper = asyncio.create_task(p.reserve(RolloutFnTrainInput(rollout_id=0, weight_version=0)))
    await asyncio.wait_for(entered.wait(), 2)
    parent = p._reservation_task
    await seed(p, 0)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(wrapper, 2)
    assert [g[0].group_index for g in p._reserved_groups] == [0]
    await p.release_reservation(consume=False)
    assert p.data_source.requeued == [0] and p.data_source.pending == {0}
    assert p._output._delegate._capacity == 2


@pytest.mark.parametrize("ready", [False, True])
async def test_capacity_failure_preserves_prompt_without_extra_credit(producer, monkeypatch, ready):
    p = producer
    if ready:
        await seed(p, 0)
    entered = asyncio.Event()
    original = p._output.get

    async def observed_get(**kwargs):
        entered.set()
        return await original(**kwargs)

    monkeypatch.setattr(p._output, "get", observed_get)
    wrapper = asyncio.create_task(p.reserve(RolloutFnTrainInput(rollout_id=0, weight_version=0)))
    if not ready:
        await asyncio.wait_for(entered.wait(), 2)
        await seed(p, 0)
    # Inject the otherwise-unreachable zero-capacity invariant violation.
    p._output._delegate._capacity = 0
    with pytest.raises(RuntimeError, match="exceeds completed-buffer capacity"):
        await asyncio.wait_for(wrapper, 2)
    assert p._reserved_groups == []
    await p.release_reservation(consume=False)
    assert p._output._delegate._capacity == 0
    assert p.data_source.requeued == [0] and p.data_source.pending == {0}
