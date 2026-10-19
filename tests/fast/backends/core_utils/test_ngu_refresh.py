"""NGU retains each attempt's original behavior provenance under policy refresh."""

import asyncio
from argparse import Namespace

import pytest
import torch

from miles.backends.core_utils.publication import policy_refresh
from miles.backends.core_utils.rollout.async_buffer import RefreshPolicyDataBuffer
from miles.backends.core_utils.rollout.refreshing_rollout import RefreshingRolloutFn
from miles.rollout import fully_async_rollout
from miles.rollout.base_types import RolloutFnConstructorInput
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.rollout.filter_hub.base_types import FilterOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.rollout.never_give_up import (
    NGU_BASELINE_SAMPLE_COUNT_KEY,
    NeverGiveUp,
    chain_rewards_and_std,
    fit_to_sample_budget,
)
from miles.utils.types import Sample


def nonzero_std(_args, group, **_kwargs):
    keep = len({sample.reward for sample in group}) > 1
    return FilterOutput(keep=keep, reason=None if keep else "zero_std")


def setup():
    args = Namespace(
        rollout_global_dataset=False,
        rollout_batch_size=4,
        n_samples_per_prompt=8,
        global_batch_size=32,
        buffer_filter_path=None,
        rollout_seed=0,
        reward_key=None,
        ngu_requeue_probability=None,
        ngu_max_budget=4,
        ngu_solved_reward=1.0,
        dynamic_sampling_filter_path=f"{__name__}.nonzero_std",
        async_data_buffer_capacity_factor=2,
        max_weight_staleness=4,
    )
    source = RolloutDataSourceWithBuffer(args)
    ngu = NeverGiveUp(args, data_source=source)
    buffer = RefreshPolicyDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=ngu))
    return source, ngu, buffer


def complete(group, *, versions, success=False):
    for index, sample in enumerate(group):
        sample.tokens, sample.response_length = [9, 10, 11, 12], 3
        sample.rollout_log_probs = [-0.8, -1.2, -0.6]
        sample.reward, sample.status = float(success and index == 0), Sample.Status.COMPLETED
        policy_refresh.record_response(
            sample,
            dict(
                weight_version=str(versions[-1]),
                weight_versions=[
                    dict(version=versions[0], start=0, end=2),
                    dict(version=versions[-1], start=2, end=3),
                ],
                output_token_logprobs=[[-0.8, 10], [-1.2, 11], [-0.6, 12]],
            ),
        )
    return group


def provenance(group):
    return policy_refresh.validate_batch(
        dict(
            metadata=[s.train_metadata for s in group],
            response_lengths=[s.response_length for s in group],
            weight_versions=[s.weight_versions for s in group],
        )
    )


def test_refresh_producer_keeps_the_ngu_handler_through_publication(monkeypatch):
    async def run():
        source, _, buffer = setup()
        args = buffer._delegate._args
        args.olmo_core = Namespace(publication_mode="refresh")
        args.async_unused_samples_handler = "never_give_up"
        args.rollout_submission_granularity = "sample"
        args.rollout_sample_filter_path = None
        args.sglang_server_concurrency = 2
        args.rollout_num_gpus = 1
        args.rollout_num_gpus_per_engine = 1
        monkeypatch.setattr(fully_async_rollout, "GenerateState", lambda args: Namespace())
        fn = RefreshingRolloutFn(RolloutFnConstructorInput(args=args, data_source=source))
        first = complete([Sample(group_index=7, index=i, prompt="prompt") for i in range(8)], versions=(1, 2))
        fn._handle_unused(first, group=first, reason="zero_std")
        await fn.begin_refresh()
        assert fn._never_give_up.has_pending_chain(7)
        assert not fn._producer_resumed.is_set()
        await fn.end_refresh()
        assert fn._producer_resumed.is_set()
        retry = complete(source.get_samples(1)[0], versions=(2, 3), success=True)
        fn._handle_unused(retry, group=retry, reason="kept")
        assert len(retry) == 16
        assert len(provenance(retry)) == 16
        assert fn._ngu_handler.collect_metrics()["rollout/ngu/accepted_retried_chains"] == 1

    asyncio.run(run())


@pytest.mark.parametrize("current_version,expected_size", [(4, 16), (6, 8)])
def test_refresh_retries_merge_then_prune_by_the_oldest_behavior_token(current_version, expected_size):
    async def run():
        source, ngu, buffer = setup()
        first = complete([Sample(group_index=7, index=i, prompt="prompt") for i in range(8)], versions=(1, 2))
        assert await buffer.put(DataBufferInput(prompt_group=first, group=first)) is False
        retry = source.get_samples(1)[0]
        assert all(s.train_metadata is None and s.rollout_log_probs is None for s in retry)
        complete(retry, versions=(3, 4), success=True)
        assert await buffer.put(DataBufferInput(prompt_group=retry, group=retry)) is True
        entry = await buffer.get(current_version=current_version)
        return entry.group, ngu.collect_metrics()

    group, metrics = asyncio.run(run())
    assert len(group) == expected_size
    assert group[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 16
    assert all(s.rollout_log_probs == [-0.8, -1.2, -0.6] for s in group)
    spans = provenance(group)
    assert spans[-1][0]["version"] == 3
    assert spans[0][0]["version"] == (1 if expected_size == 16 else 3)
    fitted = fit_to_sample_budget([group], budget=8, attempt_size=8)[0]
    assert len(provenance(fitted)) == 8
    assert fitted[0].metadata[NGU_BASELINE_SAMPLE_COUNT_KEY] == 16
    assert metrics["rollout/ngu/accepted_retried_chains"] == 1
    # Mixed-policy rewards remain compatible with the chain-wide advantage baseline.
    from_rewards = torch.tensor([s.reward for s in fitted])
    shifted, _ = chain_rewards_and_std(fitted, from_rewards, from_rewards.std())
    assert shifted.mean().item() == pytest.approx(1 / 16)


def test_refresh_failure_budget_exhausts_after_32_samples():
    async def run():
        source, ngu, buffer = setup()
        group = [Sample(group_index=7, index=i, prompt="prompt") for i in range(8)]
        for attempt in range(4):
            complete(group, versions=(attempt, attempt + 1))
            assert await buffer.put(DataBufferInput(prompt_group=group, group=group)) is False
            if attempt < 3:
                group = source.get_samples(1)[0]
        assert not source.buffer
        assert not ngu.has_pending_chain(7)
        return ngu.collect_metrics()

    metrics = asyncio.run(run())
    assert metrics["rollout/ngu/filter_retries"] == 3
    assert metrics["rollout/ngu/budget_exhausted_chains"] == 1
    assert metrics["rollout/ngu/accepted_chains"] == 0
