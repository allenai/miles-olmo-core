"""Exercise Core publication modes through the native async consumer interface."""

import asyncio
from types import SimpleNamespace

import pytest

from miles.backends.core_utils.publication import policy_refresh
from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer, RefreshPolicyDataBuffer
from miles.backends.core_utils.rollout.draining_rollout import DrainingRolloutFn
from miles.backends.core_utils.rollout.refreshing_rollout import RefreshingRolloutFn
from miles.rollout.base_types import RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample


@pytest.mark.parametrize("trainer_model_id", [None, "unsupported-policy"])
@pytest.mark.parametrize(
    ("producer_type", "buffer_type"),
    [(RefreshingRolloutFn, RefreshPolicyDataBuffer), (DrainingRolloutFn, HomogeneousPolicyDataBuffer)],
)
def test_native_drain_preserves_core_single_policy_contract(trainer_model_id, producer_type, buffer_type):
    async def scenario():
        args = SimpleNamespace(
            rollout_global_dataset=True,
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            global_batch_size=1,
            max_weight_staleness=1,
            async_data_buffer_capacity_factor=1,
            dynamic_sampling_filter_path=None,
            reward_key=None,
        )
        sample = Sample(
            index=0,
            group_index=0,
            prompt="question",
            response="answer",
            tokens=[9, 10],
            response_length=1,
            rollout_log_probs=[-0.8],
            reward=1.0,
            status=Sample.Status.COMPLETED,
        )
        policy_refresh.record_response(
            sample,
            dict(
                weight_version="0",
                weight_versions=[dict(version="0", start=0, end=1)],
                output_token_logprobs=[[-0.8, 10]],
            ),
        )
        producer = producer_type.__new__(producer_type)
        producer.args = args
        producer._output = buffer_type(DataBufferConstructorInput(args, lambda samples: None))
        producer.controller = SimpleNamespace(check=lambda: None)
        producer._sample_filter = None
        producer._publication_paused = producer._refreshing = producer._stopping = False
        producer._producer_resumed = asyncio.Event()
        producer._reserved_groups = None
        producer._draining_groups = []
        producer._interrupted_groups = []
        producer._worker = asyncio.create_task(asyncio.Event().wait())
        batch = [sample]
        await producer._output.put(DataBufferInput(prompt_group=batch, group=batch))
        request = RolloutFnTrainInput(rollout_id=0, weight_version=0, trainer_model_id=trainer_model_id)
        try:
            if trainer_model_id is not None:
                with pytest.raises(ValueError, match="Core rollout does not support multi-policy training"):
                    await asyncio.wait_for(producer._drain(request), timeout=1)
                assert producer._draining_groups == []
                assert len(producer._output._delegate._buffer) == 1
            else:
                result = await asyncio.wait_for(producer._drain(request), timeout=1)
                assert result.samples == [batch]
                assert producer._draining_groups == [batch]
                assert producer._producer_resumed.is_set()
                assert "rollout/fully_async/completed_queue/consumer_wait_seconds" in result.metrics
        finally:
            producer._worker.cancel()
            await asyncio.gather(producer._worker, return_exceptions=True)

    asyncio.run(scenario())
