"""CPU checks that the Core async adapter speaks the unused-samples handler contract."""

import asyncio
from argparse import Namespace

from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer, MeasuredDataBuffer
from miles.backends.core_utils.rollout.async_rollout import ManagedFullyAsyncRolloutFn
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.filter_hub.base_types import FilterOutput, MetricGatherer
from miles.rollout.filter_hub.common_filters import FilterReason
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.rollout.never_give_up import NeverGiveUp
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class _Ledger:
    def __init__(self):
        self.acknowledged = []

    def acknowledge_groups(self, groups):
        self.acknowledged.extend(groups)


def _group(size: int) -> list:
    return [Sample(group_index=7, index=i) for i in range(size)]


def test_drop_retires_every_unused_group_but_not_a_kept_one():
    """A kept group retires when its batch drains; retiring it here too would fail that drain."""
    rollout = object.__new__(ManagedFullyAsyncRolloutFn)
    rollout.data_source = _Ledger()
    group = _group(2)

    rollout._acknowledge_unused(group, group=group, reason=FilterReason.kept)
    assert rollout.data_source.acknowledged == []

    for reason in (FilterReason.aborted, FilterReason.stale):
        rollout._acknowledge_unused(group, group=group, reason=reason)
    assert rollout.data_source.acknowledged == [group, group]


def test_a_rejected_policy_group_reaches_the_handler_as_aborted():
    """The retry handler regenerates only aborted and stale groups, so this rejection must be one of them."""
    calls = []
    buffer = object.__new__(HomogeneousPolicyDataBuffer)
    buffer._unused = lambda prompt_group, *, group, reason: calls.append((prompt_group, group, reason))
    buffer._samples = 2
    buffer._rejected = buffer._incomplete = buffer._mixed = 0
    group = _group(1)

    asyncio.run(buffer.put(DataBufferInput(prompt_group=group, group=group)))

    assert calls == [(group, group, FilterReason.aborted)]
    assert buffer._incomplete == 1


def test_drop_leaves_a_filter_rejection_to_the_producer():
    """The producer retires a group whose put returns False; retiring it here too would fail that call."""
    rollout = object.__new__(ManagedFullyAsyncRolloutFn)
    rollout.data_source = _Ledger()
    group = _group(2)

    rollout._acknowledge_unused(group, group=group, reason="zero_std")

    assert rollout.data_source.acknowledged == []


def test_a_filter_rejection_reaches_the_handler_and_the_producer_retires_it():
    """never_give_up only retries what the handler hears about, and Core runs the filter itself."""
    calls = []
    buffer = object.__new__(MeasuredDataBuffer)
    buffer._args = Namespace(reward_key=None)
    buffer._records = None
    buffer._group_filter = lambda args, group, **_: FilterOutput(keep=False, reason="zero_std")
    buffer._metric_gatherer = MetricGatherer()
    buffer._unused_handler_fn = lambda prompt_group, *, group, reason: calls.append(reason)
    group = _group(2)
    for sample in group:
        sample.reward = 0.0

    assert asyncio.run(buffer.put(DataBufferInput(prompt_group=group, group=group))) is False
    assert calls == ["zero_std"]


class _NeverGiveUp:
    def __init__(self, pending: bool):
        self.pending, self.calls = pending, []

    def has_pending_chain(self, chain_id):
        return self.pending

    def __call__(self, prompt_group, *, group, reason):
        self.calls.append(reason)


def _never_give_up_rollout(pending: bool) -> ManagedFullyAsyncRolloutFn:
    rollout = object.__new__(ManagedFullyAsyncRolloutFn)
    rollout.data_source = _Ledger()
    rollout._never_give_up = _NeverGiveUp(pending)
    return rollout


def test_never_give_up_retires_an_aborted_attempt_it_drops():
    """A dropped prompt that stays outstanding would be regenerated after every resume."""
    rollout, group = _never_give_up_rollout(pending=False), _group(2)

    rollout._never_give_up_unused(group, group=group, reason=FilterReason.aborted)

    assert rollout._never_give_up.calls == [FilterReason.aborted]
    assert rollout.data_source.acknowledged == [group]


def test_never_give_up_keeps_an_aborted_attempt_it_retries_outstanding():
    """The retry reuses the chain's group_index, which must still be outstanding when it trains."""
    for reason in (FilterReason.aborted, FilterReason.stale, FilterReason.kept, "zero_std"):
        rollout, group = _never_give_up_rollout(pending=reason == FilterReason.aborted), _group(2)

        rollout._never_give_up_unused(group, group=group, reason=reason)

        assert rollout.data_source.acknowledged == []


class _LedgerDataSource(RolloutDataSourceWithBuffer):
    """The restart ledger of open-instruct's adapter: drawn prompts stay outstanding until retired."""

    def __init__(self, args):
        super().__init__(args)
        self.pending = {}

    def get_samples(self, num_samples):
        groups = super().get_samples(num_samples)
        for group in groups:
            self.pending.setdefault(group[0].group_index, group)
        return groups

    def acknowledge_groups(self, groups):
        for group in groups:
            del self.pending[group[0].group_index]  # KeyError: the group was not outstanding


def _versioned_group(rewards, *, first_index):
    versions = [WeightVersionsPerCall(spans=[WeightVersionSpan("1", 0, 1)])]
    return [
        Sample(
            group_index=0,
            index=first_index + i,
            prompt="prompt",
            response="response",
            response_length=1,
            reward=reward,
            status=Sample.Status.COMPLETED,
            weight_versions=list(versions),
        )
        for i, reward in enumerate(rewards)
    ]


def test_a_never_give_up_chain_stays_in_the_ledger_until_its_merged_group_drains():
    """The retry reuses the chain's group_index after the producer retired the rejected attempt."""
    args = Namespace(
        rollout_global_dataset=False,
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        global_batch_size=2,
        buffer_filter_path=None,
        rollout_seed=0,
        reward_key=None,
        dynamic_sampling_filter_path=f"{__name__}.nonzero_std",
        ngu_requeue_probability=1.0,
        ngu_solved_reward=1.0,
        async_data_buffer_capacity_factor=1000.0,
        max_weight_staleness=4,
    )
    source = _LedgerDataSource(args)
    rollout = object.__new__(ManagedFullyAsyncRolloutFn)
    rollout.data_source = source
    rollout._never_give_up = NeverGiveUp(args, data_source=source)
    buffer = HomogeneousPolicyDataBuffer(
        DataBufferConstructorInput(args=args, unused_handler_fn=rollout._never_give_up_unused)
    )

    async def run():
        first = _versioned_group([0.0, 0.0], first_index=0)
        source.pending[0] = first  # drawn by the producer
        assert await buffer.put(DataBufferInput(prompt_group=first, group=first)) is False
        source.acknowledge_groups([first])  # what the producer does on False

        [retry] = source.get_samples(1)
        for sample, reward in zip(retry, [1.0, 0.0], strict=True):
            sample.reward, sample.status = reward, Sample.Status.COMPLETED
            sample.weight_versions = first[0].weight_versions
        assert await buffer.put(DataBufferInput(prompt_group=retry, group=retry)) is True
        return await buffer.get(current_version=1)

    entry = asyncio.run(run())
    assert len(entry.group) == 4  # both attempts train together
    source.acknowledge_groups([entry.prompt_group])  # what the drain does
    assert source.pending == {}


def nonzero_std(_args, group, **_kwargs):
    keep = len({sample.reward for sample in group}) > 1
    return FilterOutput(keep=keep, reason=None if keep else "zero_std")
