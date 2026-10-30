"""Distinguish completed but uncollected groups from staged queue writes."""

import asyncio
from types import SimpleNamespace

from miles.backends.core_utils.rollout import pipeline_observer


def test_ready_stages_exclude_already_buffered_entries():
    async def scenario():
        entries = [SimpleNamespace(group=[SimpleNamespace(response_length=n)]) for n in (2, 3, 5)]
        completed = asyncio.get_running_loop().create_future()
        completed.set_result(entries[2])
        resumed = asyncio.Event()
        resumed.set()
        producer = SimpleNamespace(
            args=SimpleNamespace(sglang_server_concurrency=512, rollout_num_gpus=19, rollout_num_gpus_per_engine=1),
            state=SimpleNamespace(generate_fn_semaphore=asyncio.Semaphore(9728)),
            _output=SimpleNamespace(_buffer=[entries[0]], _capacity=256),
            _ready_completion_counts={id(entry): pipeline_observer.completion_counts([entry]) for entry in entries[:2]},
            _active_tasks={completed},
            _producing_groups={1: [], 2: [], 3: []},
            _producer_resumed=resumed,
            _scheduler=SimpleNamespace(samples_in_flight=0),
        )
        row = pipeline_observer.snapshot(producer)
        assert row["completed_queue"]["groups"] == 1
        assert row["producer_staged_completion_groups"] == 1
        assert row["producer_unharvested_completion_groups"] == 1
        assert row["producer_ready"] == dict(groups=2, samples=2, response_tokens=8)
        assert producer._output._buffer == [entries[0]]
        assert len(producer._active_tasks) == 1

    asyncio.run(scenario())
