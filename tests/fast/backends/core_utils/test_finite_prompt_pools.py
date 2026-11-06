"""Finite application pools drain ready work and terminate without polling forever."""

import asyncio

import pytest

from miles.backends.core_utils.rollout.errors import PromptPoolExhausted
from tests.fast.backends.core_utils import test_prepared_batch_reservation

producer = test_prepared_batch_reservation.producer
seed = test_prepared_batch_reservation.seed


async def test_ready_work_is_delivered_before_exhaustion(producer):
    p = producer
    p._pool_exposure = True
    p._active_tasks, p._ready_completion_counts = set(), {}
    p.data_source.pool_exhausted = lambda: True
    await seed(p, 0)
    entry = await p._next_group(0)
    assert entry.prompt_group[0].group_index == 0
    with pytest.raises(PromptPoolExhausted):
        await asyncio.wait_for(p._next_group(0), 2)
    assert not p.data_source.pending


async def test_outstanding_generation_prevents_premature_exhaustion(producer):
    p = producer
    p._pool_exposure = True
    active = asyncio.create_task(asyncio.Event().wait())
    p._active_tasks, p._ready_completion_counts = {active}, {}
    p.data_source.pool_exhausted = lambda: True
    waiting = asyncio.create_task(p._next_group(0))
    try:
        await asyncio.sleep(0.6)
        assert not waiting.done()
        p._active_tasks.clear()
        with pytest.raises(PromptPoolExhausted):
            await asyncio.wait_for(waiting, 2)
    finally:
        active.cancel()
        waiting.cancel()
        await asyncio.gather(active, waiting, return_exceptions=True)
