"""A timed-out sample can be regenerated alone while its siblings keep their responses."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from miles.backends.core_utils.rollout import prompt_preparation, refreshing_rollout
from miles.backends.core_utils.rollout.errors import GenerationRequestTimeout
from miles.backends.core_utils.rollout.generation_admission import GenerationAdmission
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils.types import Sample


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def make_args(tmp_path, **core):
    return SimpleNamespace(
        olmo_core=SimpleNamespace(pipeline_observation_interval=1, refresh_request_timeout=0.05, **core),
        save=str(tmp_path),
        sglang_router_ip="localhost",
        sglang_router_port=1234,
        sglang_router_policy="round_robin",
        rollout_max_response_len=10,
        rollout_max_context_len=100,
        rollout_top_logprobs_num=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        sglang_speculative_algorithm=None,
        partial_rollout=False,
        group_rm=False,
        sglang_enable_deterministic_inference=False,
        rollout_seed=0,
    )


@pytest.fixture
def serving(monkeypatch):
    """Sample 0 hangs for its first ``hangs`` attempts; every other request finishes."""
    calls = []
    state = SimpleNamespace(hangs=0)

    async def post(url, payload, **kwargs):
        sample_index = payload["input_ids"][-1]
        calls.append((sample_index, payload["rid"]))
        if sample_index == 0 and sum(index == 0 for index, _ in calls) <= state.hangs:
            if getattr(state, "drain_while_hung", False):
                # A lifecycle drain closes admission while this attempt is in flight.
                state.fn._publication_paused = True
            await asyncio.Event().wait()
        return {"meta_info": {"finish_reason": {"type": "stop"}}}

    async def update(args, sample, payload, output):
        sample.tokens = list(payload["input_ids"])
        sample.response_length = 1

    async def reward(args, sample):
        return 1

    monkeypatch.setattr(refreshing_rollout, "post", post)
    monkeypatch.setattr(refreshing_rollout, "update_sample_from_response", update)
    # Make each sample's prompt end in its own index so the fake server can tell siblings apart.
    monkeypatch.setattr(
        refreshing_rollout, "compute_prompt_ids_from_sample", lambda state, sample: [*b"q", sample.index]
    )
    monkeypatch.setattr(
        refreshing_rollout.policy_refresh, "record_response", lambda *a: {"spans": [], "replay_version": 0}
    )
    monkeypatch.setattr(inference_rollout_common, "async_rm", reward)
    return SimpleNamespace(calls=calls, state=state)


def run_group(args, serving=None):
    async def scenario():
        fn = refreshing_rollout.RefreshingRolloutFn.__new__(refreshing_rollout.RefreshingRolloutFn)
        fn.args = args
        if serving is not None:
            serving.state.fn = fn
        fn._worker = None
        fn._scheduler = SimpleNamespace(sample_done_callback=lambda: None)
        fn._sample_timeout_retries = refreshing_rollout.sample_timeout_retries(args)
        fn.state = SimpleNamespace(
            args=args,
            aborted=False,
            sampling_params={},
            generate_function=fn._generate_response,
            generate_fn_semaphore=GenerationAdmission(8),
        )
        # Token-list prompts skip shared preparation and use compute_prompt_ids_from_sample.
        fn._prompt_preparer = prompt_preparation.PromptPreparer(Tokenizer)
        fn._prepared_prompt_ids = {}
        samples = [Sample(prompt=[1], index=i, group_index=5) for i in range(8)]
        try:
            return samples, await fn._generate_group(samples)
        finally:
            assert fn.state.generate_fn_semaphore.active == 0
            # Flushes sibling timing and closes the prompt preparer.
            await fn.dispose()

    return asyncio.run(scenario())


def timing_rows(tmp_path):
    return [
        json.loads(line) for path in tmp_path.glob("sibling_timing_*.jsonl") for line in path.read_text().splitlines()
    ]


def test_timed_out_sample_is_regenerated_alone_and_group_completes(tmp_path, serving):
    serving.state.hangs = 1
    samples, result = run_group(make_args(tmp_path, refresh_sample_timeout_retries=1))

    assert len(result.group) == 8 and all(s.reward == 1 for s in result.group)
    attempts = [rid for index, rid in serving.calls if index == 0]
    # Only the timed-out sample was resent, with a new request id; no sibling was regenerated.
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    assert sorted(index for index, _ in serving.calls) == [0, 0, 1, 2, 3, 4, 5, 6, 7]
    [row] = timing_rows(tmp_path)
    assert row["outcome"] == "completed"
    retried = {s["sample_index"]: s.get("timeout_retries", 0) for s in row["samples"]}
    assert retried == {0: 1, **{i: 0 for i in range(1, 8)}}
    assert next(s for s in row["samples"] if s["sample_index"] == 0)["request_id"] == attempts[1]


def test_default_keeps_whole_group_failure(tmp_path, serving):
    serving.state.hangs = 1
    with pytest.raises(GenerationRequestTimeout, match="refresh_request_timeout"):
        run_group(make_args(tmp_path))
    assert sum(index == 0 for index, _ in serving.calls) == 1


def test_exhausted_retries_fall_back_to_group_failure(tmp_path, serving):
    serving.state.hangs = 3
    with pytest.raises(GenerationRequestTimeout, match="refresh_request_timeout"):
        run_group(make_args(tmp_path, refresh_sample_timeout_retries=2))
    assert sum(index == 0 for index, _ in serving.calls) == 3


def test_timeout_after_lifecycle_drain_started_is_not_retried(tmp_path, serving):
    serving.state.hangs = 1
    serving.state.drain_while_hung = True
    with pytest.raises(GenerationRequestTimeout, match="refresh_request_timeout"):
        run_group(make_args(tmp_path, refresh_sample_timeout_retries=2), serving)
    # Only the first request id for the sample was submitted; recovery owns the group.
    assert sum(index == 0 for index, _ in serving.calls) == 1


@pytest.mark.parametrize("value", [-1, 1.0, True, "1", None])
def test_retry_count_must_be_a_non_negative_integer(value):
    args = SimpleNamespace(olmo_core=SimpleNamespace(refresh_sample_timeout_retries=value))
    with pytest.raises(ValueError, match="refresh_sample_timeout_retries"):
        refreshing_rollout.sample_timeout_retries(args)


def test_retry_count_defaults_to_zero_when_unset():
    assert refreshing_rollout.sample_timeout_retries(SimpleNamespace(olmo_core=SimpleNamespace())) == 0
    assert refreshing_rollout.sample_timeout_retries(SimpleNamespace()) == 0
