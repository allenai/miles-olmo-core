"""Prompt preparation preserves tokens and cannot monopolize the producer loop."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from miles.backends.core_utils.rollout import async_rollout, prompt_preparation, refreshing_rollout
from miles.backends.core_utils.rollout.generation_admission import GenerationAdmission
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils.types import Sample


def test_encoding_is_deduplicated_off_loop_and_scoped_to_one_call():
    calls = []
    loop_thread = threading.get_ident()

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            assert threading.get_ident() != loop_thread
            calls.append(text)
            return list(text.encode())

    def factory():
        assert threading.get_ident() == loop_thread
        return Tokenizer()

    async def scenario():
        worker = prompt_preparation.PromptPreparer(factory)
        try:
            result = await worker.prepare(["café"] * 8 + ["other"])
            assert result == {"café": tuple("café".encode()), "other": tuple(b"other")}
            assert calls == ["café", "other"]
            await worker.prepare(["café"] * 8)
            assert calls == ["café", "other", "café"]
        finally:
            await worker.close()
        with pytest.raises(RuntimeError, match="closed"):
            await worker.prepare(["new"])

    asyncio.run(scenario())


def test_cancellation_keeps_capacity_until_worker_finishes_and_loop_stays_responsive():
    entered, release = threading.Event(), threading.Event()
    calls = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            calls.append(text)
            entered.set()
            assert release.wait(5)
            return [1]

    async def scenario():
        worker = prompt_preparation.PromptPreparer(Tokenizer)
        first = asyncio.create_task(worker.prepare(["first"] * 8))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()  # Repeated lifecycle cancellation must also join.
            second = asyncio.create_task(worker.prepare(["second"]))
            await asyncio.sleep(0)
            assert calls == ["first"]
            assert not first.done() and not second.done()
            # The ready consumer runs while the tokenizer is still blocked.
            assert await asyncio.create_task(asyncio.sleep(0, result="ready")) == "ready"
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert await second == {"second": (1,)}
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)
            await worker.close()

    asyncio.run(scenario())


def test_encoding_error_propagates_and_releases_capacity():
    class Tokenizer:
        def encode(self, text, **kwargs):
            if text == "bad":
                raise ValueError("tokenizer failed")
            return [2]

    async def scenario():
        worker = prompt_preparation.PromptPreparer(Tokenizer)
        try:
            with pytest.raises(ValueError, match="tokenizer failed"):
                await worker.prepare(["bad"])
            assert await worker.prepare(["good"]) == {"good": (2,)}
        finally:
            await worker.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("paused", [False, True])
def test_refresh_requests_share_encoding_but_not_mutable_tokens(monkeypatch, paused, tmp_path):
    calls, payloads = [], []

    class Tokenizer:
        def encode(self, text, **kwargs):
            calls.append(text)
            return list(text.encode())

    args = SimpleNamespace(
        olmo_core=SimpleNamespace(pipeline_observation_interval=1, refresh_request_timeout=10),
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
        sglang_enable_deterministic_inference=True,
        rollout_seed=17,
    )

    async def post(url, payload, **kwargs):
        payloads.append(payload)
        await asyncio.sleep(0)
        return {"meta_info": {"finish_reason": {"type": "stop"}}}

    async def update(args, sample, payload, output):
        sample.tokens = payload["input_ids"]
        sample.tokens.append(sample.index)

    async def reward(args, sample):
        assert sample.tokens == [*b"same", sample.index]
        return sample.index % 2

    monkeypatch.setattr(refreshing_rollout, "post", post)
    monkeypatch.setattr(refreshing_rollout, "update_sample_from_response", update)
    monkeypatch.setattr(
        refreshing_rollout.policy_refresh, "record_response", lambda *a: {"spans": [], "replay_version": 0}
    )
    monkeypatch.setattr(inference_rollout_common, "async_rm", reward)

    async def scenario():
        fn = refreshing_rollout.RefreshingRolloutFn.__new__(refreshing_rollout.RefreshingRolloutFn)
        fn.args = args
        fn._worker = None
        completed = []
        fn._scheduler = SimpleNamespace(sample_done_callback=lambda: completed.append(1))
        fn.state = SimpleNamespace(
            args=args,
            aborted=False,
            sampling_params={},
            generate_function=fn._generate_response,
            generate_fn_semaphore=GenerationAdmission(8),
        )
        fn._prompt_preparer = prompt_preparation.PromptPreparer(Tokenizer)
        fn._prepared_prompt_ids = {}
        samples = [Sample(prompt="same", index=i, group_index=0) for i in range(8)]
        try:
            if paused:
                fn.state.generate_fn_semaphore.pause()
            pending = asyncio.create_task(fn._generate_group(samples))
            if paused:
                async with asyncio.timeout(5):
                    while fn.state.generate_fn_semaphore.waiting < 8:
                        await asyncio.sleep(0.001)
                assert calls == ["same"] and payloads == []
                fn.state.generate_fn_semaphore.resume()
            result = await pending
            assert calls == ["same"]
            assert len({p["rid"] for p in payloads}) == 8
            assert len({id(p["input_ids"]) for p in payloads}) == 8
            assert all(s.tokens == [*b"same", s.index] for s in samples)
            assert [p["sampling_params"]["sampling_seed"] for p in payloads] == list(range(17, 25))
            assert [s.reward for s in result.group] == [i % 2 for i in range(8)]
            assert len(completed) == 8
            assert fn.state.generate_fn_semaphore.active == 0
            assert fn._prepared_prompt_ids == {}
            await fn.dispose()
            rows = [
                json.loads(line)
                for path in tmp_path.glob("sibling_timing_*.jsonl")
                for line in path.read_text().splitlines()
            ]
            assert len(rows) == 1
            assert all(r["prompt_preparation_seconds"] >= 0 for r in rows[0]["samples"])
        finally:
            await fn._prompt_preparer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [ValueError, asyncio.CancelledError])
def test_group_failure_releases_prepared_prefixes(monkeypatch, failure):
    async def fail(self, samples):
        assert len(self._prepared_prompt_ids) == 8
        raise failure()

    monkeypatch.setattr(async_rollout.ManagedFullyAsyncRolloutFn, "_generate_group", fail)

    async def scenario():
        fn = refreshing_rollout.RefreshingRolloutFn.__new__(refreshing_rollout.RefreshingRolloutFn)
        fn.args = SimpleNamespace(olmo_core=SimpleNamespace(pipeline_observation_interval=0))
        fn._prompt_preparer = prompt_preparation.PromptPreparer(lambda: SimpleNamespace(encode=lambda *a, **kw: [1]))
        fn._prepared_prompt_ids = {}
        samples = [Sample(prompt="same", index=i, group_index=0) for i in range(8)]
        try:
            with pytest.raises(failure):
                await fn._generate_group(samples)
            assert fn._prepared_prompt_ids == {}
            assert all(not s.tokens for s in samples)
        finally:
            await fn._prompt_preparer.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", ["pause", "stop", "interrupt"])
def test_refill_yields_and_rechecks_lifecycle(stop):
    async def scenario():
        fn = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
        fn._producer_resumed = asyncio.Event()
        fn._producer_resumed.set()
        fn._stopping = False
        fn._retry_after = 0
        signals = {"interrupted": False}
        fn._interrupted = lambda: signals["interrupted"]
        fn._max_in_flight_groups = lambda: 24
        fn._scheduler = SimpleNamespace(has_capacity=lambda **kw: True)
        fn._submit_managed_group = object
        active = set()

        def signal():
            if stop == "pause":
                fn._producer_resumed.clear()
            elif stop == "stop":
                fn._stopping = True
            else:
                signals["interrupted"] = True

        asyncio.get_running_loop().call_soon(signal)
        await fn._refill(active)
        assert len(active) == 4

    asyncio.run(scenario())
