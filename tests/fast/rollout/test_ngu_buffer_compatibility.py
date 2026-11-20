"""Attempt pruning must not change whole-group staleness outside NGU."""

from argparse import Namespace

import pytest

from miles.rollout.filter_hub.common_filters import FilterReason
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.rollout.group_metrics import group_size_metrics
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


@pytest.mark.parametrize("handler", ["drop", "retry", "never_give_up"])
async def test_only_ngu_prunes_old_attempts(handler):
    args = Namespace(
        async_unused_samples_handler=handler,
        async_data_buffer_capacity_factor=1,
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        dynamic_sampling_filter_path=None,
        reward_key=None,
        max_weight_staleness=1,
    )
    calls = []
    buffer = DefaultDataBuffer(
        DataBufferConstructorInput(args=args, unused_handler_fn=lambda prompt_group, **context: calls.append(context))
    )
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
    entry = await buffer.get(current_version=3, wait=False)

    if handler == "never_give_up":
        assert [sample.index for sample in entry.group] == [2, 3]
        assert len(calls) == 1
    else:
        assert entry is None
        assert calls[-1] == {"group": group, "reason": FilterReason.stale}
        assert len(group) == 4


def test_group_metrics_accept_nested_non_ngu_trajectories():
    group = [[Sample(), Sample()], [Sample()]]
    metrics = group_size_metrics([group], attempt_size=2, stage="training")
    assert metrics["rollout/groups/training/groups"] == 1
    assert metrics["rollout/groups/training/chain_attempts_mean"] == 1
