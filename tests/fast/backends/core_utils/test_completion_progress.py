"""Finished groups must reach the consumer even when an event loop is delayed."""

import asyncio
import time
from types import SimpleNamespace

import pytest
import uvloop

from miles.backends.core_utils.rollout import async_rollout, rollout_errors
from miles.rollout.submission_scheduler import SampleBackfillSubmission


@pytest.mark.parametrize("loop_factory", [asyncio.new_event_loop, uvloop.new_event_loop])
def test_completed_group_cannot_starve_behind_timed_progress_wait(monkeypatch, loop_factory):
    async def observe(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(rollout_errors, "observe", observe)
    monkeypatch.setattr(rollout_errors.RolloutErrors, "report", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rollout.pipeline_observer, "observe", observe)
    monkeypatch.setattr(async_rollout.pipeline_observer, "observe_engines", observe)

    async def scenario():
        producer = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
        group = [SimpleNamespace(group_index=7, response_length=1)]
        entry = SimpleNamespace(prompt_group=group, group=group)

        async def complete():
            return entry

        completed = asyncio.create_task(complete())
        await completed
        submitted, delivered = [], []

        class LoadedScheduler(SampleBackfillSubmission):
            def has_capacity(self, *, pending_groups, group_budget):
                if not submitted:
                    return True
                # Model synchronous response decode delaying uvloop past the
                # producer's existing 0.5s timeout, with a group already done.
                asyncio.get_running_loop().call_soon(time.sleep, 0.6)
                return False

            async def wait_for_progress(self, pendings):
                if len(submitted) >= 3:
                    pytest.fail("completed group repeatedly lost to timed progress wait")
                submitted.append("wait")
                return await super().wait_for_progress(pendings)

        def submit():
            submitted.append("submitted")
            producer._task_groups[completed] = 7
            return completed

        async def deliver(item):
            delivered.append(item)
            producer._stopping = True
            return True

        producer.args = SimpleNamespace()
        producer._scheduler = LoadedScheduler(1)
        producer._submit_managed_group = submit
        producer._max_in_flight_groups = lambda: 1
        producer._put_or_stop = deliver
        producer._errors = rollout_errors.RolloutErrors()
        producer._retry_after = 0
        producer._producer_resumed = asyncio.Event()
        producer._producer_resumed.set()
        producer._producer_idle = asyncio.Event()
        producer._stopping = producer._publication_paused = False
        producer._producing_groups = {7: group}
        producer._task_groups = {}
        producer._ready_completion_counts = {}
        producer._shutdown_unqueued_counts = {"groups": 0, "samples": 0, "response_tokens": 0}
        await producer._worker_loop()
        assert delivered == [entry]
        assert producer._errors.successes == 1
        assert not producer._active_tasks

    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(scenario())
