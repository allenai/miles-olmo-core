"""Exercise outage tolerance and isolated retry limits through the real producer."""

import asyncio
import collections
import json
from types import SimpleNamespace

import httpx
import pytest

from miles.backends.core_utils.rollout import async_rollout, rollout_errors


def http_failure(status, code):
    request = httpx.Request("POST", "http://router/generate")
    response = httpx.Response(status, request=request, json={"error": {"code": code}})
    return httpx.HTTPStatusError("serving failure", request=request, response=response)


class Source:
    """Mirror the consumer's front-of-buffer pristine retries with stable IDs."""

    def __init__(self):
        self.buffer = collections.deque()
        self.next_index = 7
        self.requeued = []

    def get_samples(self, n):
        assert n == 1
        if self.buffer:
            index = self.buffer.popleft()
        else:
            index = self.next_index
            self.next_index += 1
        return [[SimpleNamespace(group_index=index)]]

    def requeue_pending_groups(self, identities):
        self.buffer.extendleft(reversed(identities))
        self.requeued.extend(identities)


@pytest.fixture
def producer(monkeypatch, tmp_path):
    async def observe(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(rollout_errors, "observe", observe)
    monkeypatch.setattr(rollout_errors, "RETRY_DELAY_SECONDS", 0)
    monkeypatch.setattr(rollout_errors.tracking, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rollout.pipeline_observer, "observe", observe)
    monkeypatch.setattr(async_rollout.pipeline_observer, "observe_engines", observe)
    monkeypatch.setattr(async_rollout.pipeline_observer, "completion_counts", lambda items: {})
    fn = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
    fn.args = SimpleNamespace(save=str(tmp_path))
    fn.data_source = Source()
    fn._errors = rollout_errors.RolloutErrors()
    fn._retry_after = 0
    fn._producing_groups = {}
    fn._task_groups = {}
    fn._producer_resumed = asyncio.Event()
    fn._producer_resumed.set()
    fn._producer_idle = asyncio.Event()
    fn._stopping = fn._publication_paused = False
    fn._ready_completion_counts = {}
    fn._shutdown_unqueued_counts = {}
    fn._max_in_flight_groups = lambda: 1
    fn._scheduler = SimpleNamespace(
        has_capacity=lambda pending_groups, group_budget: pending_groups < group_budget,
        on_submit=lambda groups: None,
        wait_for_progress=lambda active: asyncio.wait(active),
    )
    return fn


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        http_failure(503, "no_available_workers"),
        http_failure(429, "overloaded"),
        http_failure(500, "read_response_body_failed"),
    ],
)
def test_same_prompt_short_outage_recovers_and_clears_retry_history(producer, error):
    attempts = []
    delivered = []

    async def generate(samples):
        index = samples[0].group_index
        attempts.append(index)
        if len(attempts) <= 4:
            raise error
        return SimpleNamespace(prompt_group=samples, group=[])

    async def deliver(item):
        delivered.append(item.prompt_group[0].group_index)
        producer._stopping = True
        return True

    producer._generate_group = generate
    producer._put_or_stop = deliver
    asyncio.run(asyncio.wait_for(producer._worker_loop(), timeout=5))
    assert attempts == [7] * 5 and delivered == [7]
    assert producer.data_source.requeued == [7] * 4
    assert producer._errors.failures.total() == 4 and producer._errors.successes == 1
    assert producer._errors._group_retries == {}
    assert not producer._active_tasks and not producer._task_groups


def test_isolated_failure_stops_with_http_detail_and_final_attempt_counted(producer, tmp_path):
    error = http_failure(500, "read_response_body_failed")
    delivered = []
    producer._max_in_flight_groups = lambda: 2

    async def generate(samples):
        if samples[0].group_index == 7:
            raise error
        return SimpleNamespace(prompt_group=samples, group=[])

    async def deliver(item):
        delivered.append(item.prompt_group[0].group_index)
        return True

    producer._generate_group = generate
    producer._put_or_stop = deliver
    with pytest.raises(RuntimeError, match="after 3 isolated retries") as caught:
        asyncio.run(asyncio.wait_for(producer._worker_loop(), timeout=5))
    assert caught.value.__cause__ is error
    assert producer.data_source.requeued == [7] * 4
    assert list(producer.data_source.buffer) == [7]
    assert 7 not in producer._producing_groups and 7 not in delivered
    assert len(delivered) == 3
    assert producer._errors.successes == 4
    assert producer._errors.failures["transport"] == 4
    row = json.loads((tmp_path / "rollout_errors.jsonl").read_text().splitlines()[-1])
    assert row["errors/completed_group_attempts"] == 8 and row["errors/requeued_groups"] == 4
    assert row["errors/fatal"] == 1 and "read_response_body_failed" in row["fatal"]
    assert '"status": 500' in row["fatal"]
    assert not producer._active_tasks and not producer._task_groups


def test_same_prompt_prolonged_outage_uses_global_stop_rule(producer, monkeypatch):
    clock = SimpleNamespace(now=0)
    monkeypatch.setattr(rollout_errors, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now))

    async def generate(samples):
        clock.now += 60
        raise httpx.ConnectError("refused")

    producer._generate_group = generate
    with pytest.raises(RuntimeError, match="50% for 300 seconds"):
        asyncio.run(asyncio.wait_for(producer._worker_loop(), timeout=5))
    assert producer.data_source.requeued == [7] * 6
    assert producer._errors.failures.total() == 6 and producer._errors.successes == 0


def test_unread_overload_response_still_requeues(producer):
    request = httpx.Request("POST", "http://router/generate")
    response = httpx.Response(429, request=request, stream=httpx.ByteStream(b"unread"))
    error = httpx.HTTPStatusError("overloaded", request=request, response=response)
    producer._producing_groups[7] = object()
    assert producer._requeue_generation_failure(7, error)
    assert producer.data_source.requeued == [7]
