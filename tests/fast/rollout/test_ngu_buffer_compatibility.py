"""Attempt pruning must not change whole-group staleness outside NGU."""

from argparse import Namespace

import pytest

from miles.rollout.filter_hub.common_filters import FilterReason
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.rollout.group_metrics import group_size_metrics
from miles.rollout.unused_samples import make_unused_samples_handler
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


class _RecordingHandler:
    """Exercise a real policy's dequeue hook independently of its enqueue behavior."""

    def __init__(self, pre_staleness_filter, calls):
        self.pre_staleness_filter = pre_staleness_filter
        self.calls = calls

    def __call__(self, prompt_group, **context):
        self.calls.append(context)


@pytest.mark.parametrize("handler", ["drop", "retry", "never_give_up", "custom", "legacy"])
@pytest.mark.parametrize("current_version", [3, 5])
async def test_pre_staleness_hook_preserves_final_staleness_check(handler, current_version):
    args = Namespace(
        async_unused_samples_handler=handler,
        async_data_buffer_capacity_factor=1,
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        dynamic_sampling_filter_path=None,
        reward_key=None,
        max_weight_staleness=1,
        rollout_seed=0,
    )
    calls = []

    def custom_hook(group, *, current_version):
        assert current_version in (3, 5)
        return group[2:]

    if handler == "legacy":

        def unused(prompt_group, **context):
            calls.append(context)

    else:
        hook = (
            custom_hook if handler == "custom" else make_unused_samples_handler(args, Namespace()).pre_staleness_filter
        )
        unused = _RecordingHandler(hook, calls)
    buffer = DefaultDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=unused))
    group = [
        Sample(
            index=i,
            group_index=0,
            reward=float(i % 2),
            status=Sample.Status.COMPLETED,
            weight_versions=[WeightVersionsPerCall(spans=[WeightVersionSpan(str(version), 0, 1)])],
        )
        for i, version in enumerate([1, 1, 3, 3])
    ]
    await buffer.put(DataBufferInput(prompt_group=group, group=group))
    entry = await buffer.get(current_version=current_version, wait=False)

    prunes = handler in ("never_give_up", "custom")
    if prunes and current_version == 3:
        assert [sample.index for sample in entry.group] == [2, 3]
        assert len(calls) == 1
    else:
        assert entry is None
        assert calls[-1] == {"group": group[2:] if prunes else group, "reason": FilterReason.stale}
        assert len(group) == 4


@pytest.mark.parametrize("handler", ["drop", "retry"])
def test_default_hook_preserves_nested_group_identity(handler):
    policy = make_unused_samples_handler(Namespace(async_unused_samples_handler=handler), Namespace())
    group = [[Sample(), Sample()], [Sample()]]
    assert policy.pre_staleness_filter(group, current_version=3) is group


def test_group_metrics_accept_nested_non_ngu_trajectories():
    group = [[Sample(), Sample()], [Sample()]]
    metrics = group_size_metrics([group], attempt_size=2, stage="training")
    assert metrics["rollout/groups/training/groups"] == 1
    assert metrics["rollout/groups/training/chain_attempts_mean"] == 1
