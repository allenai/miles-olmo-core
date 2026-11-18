"""Linear TITO episodes retain native provenance across Core publications."""

import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from miles.backends.core_utils.publication import policy_refresh
from miles.backends.core_utils.rollout.async_buffer import RefreshPolicyDataBuffer
from miles.backends.core_utils.rollout.async_rollout import ManagedFullyAsyncRolloutFn
from miles.backends.core_utils.rollout.generation_admission import GenerationAdmission
from miles.backends.core_utils.rollout.refreshing_rollout import RefreshingRolloutFn
from miles.rollout.base_types import GenerateFnOutput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.rollout.session.samples import merge
from miles.rollout.session.types import SessionRecord
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def trajectory():
    return Sample(
        group_index=0,
        index=0,
        tokens=[90, 91, 10, 11, 20, 21, 12, 13],
        response_length=6,
        rollout_log_probs=[-0.1, -0.2, 0.0, 0.0, -0.3, -0.4],
        loss_mask=[1, 1, 0, 0, 1, 1],
        reward=1.0,
        status=Sample.Status.COMPLETED,
        weight_versions=[
            WeightVersionsPerCall([WeightVersionSpan("0", 2, 4)]),
            WeightVersionsPerCall([WeightVersionSpan("1", 6, 8)]),
        ],
    )


def batch(sample):
    return dict(
        metadata=[sample.train_metadata],
        response_lengths=[sample.response_length],
        total_lengths=[len(sample.tokens)],
        weight_versions=[sample.weight_versions],
        loss_masks=[torch.tensor(sample.loss_mask)],
    )


def test_native_tool_gaps_preserve_tokens_scores_masks_and_versions():
    sample = trajectory()
    before = sample.to_dict()
    record = policy_refresh.record_trajectory(sample)
    assert record["spans"] == [dict(version=0, start=0, end=2), dict(version=1, start=4, end=6)]
    assert sample.to_dict() == before | {"train_metadata": sample.train_metadata}
    restored = Sample.from_dict(sample.to_dict())
    assert policy_refresh.validate_batch(batch(restored)) == [record["spans"]]
    assert restored.oldest_weight_version == 0


@pytest.mark.parametrize("change", ["uncovered", "overlap", "out_of_bounds", "no_scores", "no_mask"])
def test_agent_trajectory_requires_provenance_for_every_loss_token(change):
    sample = trajectory()
    if change == "uncovered":
        sample.loss_mask[2] = 1
    elif change == "overlap":
        sample.weight_versions[1] = WeightVersionsPerCall([WeightVersionSpan("1", 3, 8)])
    elif change == "out_of_bounds":
        sample.weight_versions[0] = WeightVersionsPerCall([WeightVersionSpan("0", 1, 4)])
    elif change == "no_scores":
        sample.rollout_log_probs = None
    else:
        sample.loss_mask = None
    with pytest.raises(ValueError):
        policy_refresh.record_trajectory(sample)


def test_missing_batch_mask_cannot_turn_tool_gaps_into_trainable_tokens():
    sample = trajectory()
    policy_refresh.record_trajectory(sample)
    data = batch(sample)
    data.pop("loss_masks")
    with pytest.raises(ValueError, match="lost loss masks"):
        policy_refresh.validate_batch(data)
    data["loss_masks"] = [torch.ones(6)]
    with pytest.raises(ValueError, match="every trainable"):
        policy_refresh.validate_batch(data)


def test_same_version_across_multiple_calls_is_normalized_without_losing_gaps():
    sample = trajectory()
    sample.weight_versions[1] = WeightVersionsPerCall([WeightVersionSpan("0", 6, 7), WeightVersionSpan("0", 7, 8)])
    record = policy_refresh.record_trajectory(sample)
    assert policy_refresh.validate_batch(batch(sample)) == [record["spans"]]
    assert len(record["spans"]) == 2


def test_agent_buffer_accepts_mixed_versions_but_retries_an_old_episode():
    async def scenario():
        args = SimpleNamespace(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            global_batch_size=1,
            max_weight_staleness=1,
            async_data_buffer_capacity_factor=1,
            dynamic_sampling_filter_path=None,
            reward_key=None,
        )
        unused = []
        buffer = RefreshPolicyDataBuffer(DataBufferConstructorInput(args, unused.append))
        sample = trajectory()
        policy_refresh.record_trajectory(sample)
        entry = DataBufferInput(prompt_group=[Sample(group_index=0)], group=[sample])
        await buffer.put(entry)
        assert await buffer.get(current_version=1) is entry
        await buffer.put(entry)
        assert await buffer.get_ready(current_version=2) is None
        assert unused == [entry.prompt_group]
        aborted = DataBufferInput(
            prompt_group=[Sample(group_index=1)], group=[Sample(group_index=1, status=Sample.Status.ABORTED)]
        )
        await buffer.put(aborted)
        assert unused[-1] is aborted.prompt_group

    asyncio.run(scenario())


def test_constructor_retains_native_agent_and_unwraps_one_tito_leaf(monkeypatch):
    async def native(input):
        return GenerateFnOutput(samples=[trajectory()])

    def initialize(self, input):
        self.args = SimpleNamespace(
            custom_generate_function_path="miles.rollout.generate_hub.agentic_tool_call.generate",
            sglang_server_concurrency=1,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
        )
        self.state = SimpleNamespace(generate_function=native, tokenizer=object())

    monkeypatch.setattr(ManagedFullyAsyncRolloutFn, "__init__", initialize)
    producer = RefreshingRolloutFn(None)
    result = asyncio.run(producer.state.generate_function(None))
    assert isinstance(result.samples, Sample)
    assert result.samples.train_metadata["policy_refresh"]["kind"] == "agent"
    assert producer._agent_generate is native


def test_agent_episode_survives_publication_while_waiting_for_a_tool():
    async def scenario():
        started, tool_finished = asyncio.Event(), asyncio.Event()
        sandbox = object()
        observed = []
        p = RefreshingRolloutFn.__new__(RefreshingRolloutFn)
        p._refreshing = p._publication_paused = p._stopping = False
        p._producer_resumed = asyncio.Event()
        p._producer_resumed.set()
        p._producing_groups = {0: [Sample(group_index=0)]}
        p._generation_admission = GenerationAdmission(1)
        p._output = None
        p.args = SimpleNamespace()

        async def native(input):
            observed.append(sandbox)
            started.set()
            await tool_finished.wait()
            observed.append(sandbox)
            return GenerateFnOutput(samples=[trajectory()])

        async def episode():
            async with p._generation_admission:
                return await p._generate_agent(None)

        p._agent_generate = native
        task = asyncio.create_task(episode())
        await started.wait()
        # This legacy boundary is already stricter than the current driver's
        # publication path, which does not call the producer at all in refresh.
        await p.begin_refresh()
        assert not task.done() and p._generation_admission.active == 1
        await p.end_refresh()
        tool_finished.set()
        result = await asyncio.wait_for(task, 1)
        assert observed == [sandbox, sandbox]
        assert result.samples.oldest_weight_version == 0
        assert result.samples.train_metadata["policy_refresh"]["replay_version"] == 1

    asyncio.run(scenario())


def test_agent_branching_is_rejected_before_training():
    async def native(input):
        return GenerateFnOutput(samples=[trajectory(), trajectory()])

    p = RefreshingRolloutFn.__new__(RefreshingRolloutFn)
    p._agent_generate = native
    with pytest.raises(ValueError, match="branching"):
        asyncio.run(p._generate_agent(None))


def test_shared_engine_boundary_waits_for_whole_episode_including_tools():
    async def scenario():
        tool_started, tool_finished = asyncio.Event(), asyncio.Event()
        p = RefreshingRolloutFn.__new__(RefreshingRolloutFn)
        p._refreshing = p._publication_paused = p._stopping = False
        p._boundary_capacity = None
        p._output = None
        p._producer_resumed = asyncio.Event()
        p._producer_resumed.set()
        p._generation_admission = GenerationAdmission(1)
        p._producing_groups = {0: []}
        p.args = SimpleNamespace(olmo_core=SimpleNamespace(engine_drain_timeout=1))
        admitted = []

        async def episode(index):
            async with p._generation_admission:
                admitted.append(index)
                if index == 0:
                    tool_started.set()
                    await tool_finished.wait()

        active = asyncio.create_task(episode(0))
        await tool_started.wait()
        p._worker = asyncio.create_task(asyncio.Event().wait())
        boundary = asyncio.create_task(p.prepare_publication())
        await asyncio.sleep(0)
        queued = asyncio.create_task(episode(1))
        await asyncio.sleep(0)
        assert not boundary.done() and not queued.done()
        tool_finished.set()
        await asyncio.wait_for(boundary, 1)
        await active
        assert admitted == [0] and not queued.done()
        await p.finish_publication()
        await asyncio.wait_for(queued, 1)
        assert admitted == [0, 1]
        p._worker.cancel()
        await asyncio.gather(p._worker, return_exceptions=True)

    asyncio.run(scenario())


def test_native_session_assembly_preserves_moe_routes_and_tool_masks():
    routes = np.arange(1, 29, dtype=np.int32).reshape(7, 2, 2)

    def record(prompt, output, version, start, end):
        return SessionRecord(
            timestamp=0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={
                "messages": [{"role": "user", "content": "task"}],
                "input_ids": prompt,
                "routed_experts_start_len": start,
            },
            response={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                        "meta_info": {
                            "weight_version": str(version),
                            "output_token_logprobs": [[-0.5, t] for t in output],
                            "routed_experts": base64.b64encode(routes[start:end].tobytes()).decode(),
                        },
                    }
                ]
            },
        )

    records = [record([90, 91], [10, 11], 0, 0, 3), record([90, 91, 10, 11, 20, 21], [12, 13], 1, 3, 7)]
    args = SimpleNamespace(save_debug_trajectory_data=None, sglang_speculative_algorithm=None, num_layers=2)
    tokenizer = SimpleNamespace(decode=lambda ids: str(ids))
    samples = merge.compute_samples_from_openai_records(args, records, tokenizer, use_addition_r3=True)
    merged = merge.merge_samples_with_addition_r3(args, samples, records, tokenizer)
    policy_refresh.record_trajectory(merged)
    assert merged.loss_mask == [1, 1, 0, 0, 1, 1]
    assert merged.rollout_log_probs == [-0.5, -0.5, 0, 0, -0.5, -0.5]
    np.testing.assert_array_equal(merged.rollout_routed_experts, routes)
    assert policy_refresh.validate_batch(batch(merged))[0] == [
        dict(version=0, start=0, end=2),
        dict(version=1, start=4, end=6),
    ]
