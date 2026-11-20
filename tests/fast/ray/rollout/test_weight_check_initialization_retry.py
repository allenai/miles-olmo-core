"""Preserve checkpoint references when an engine mutates weights before a cancelled reply."""

import asyncio

import pytest
from tests.fast.ray.rollout import test_server_cell_state_machine as state_tests

from miles.ray.rollout.server_cell import ServerCell

cell_env = state_tests.cell_env
pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")


async def test_cancelled_reset_cannot_replace_the_original_checkpoint_reference(cell_env, monkeypatch):
    cell = state_tests._make_cell(args_overrides={"check_weight_update_equal": True})
    weights = {"live": 17, "reference": None}
    actions = []
    reset_started = asyncio.Event()

    async def check_weights(self, action, **kwargs):
        actions.append(action)
        if action == "snapshot":
            weights["reference"] = weights["live"]
        elif action == "reset_tensors":
            weights["live"] = 99
            if actions.count("reset_tensors") == 1:
                reset_started.set()
                await asyncio.Event().wait()

    monkeypatch.setattr(ServerCell, "check_weights", check_weights)
    await cell.init()
    tick = asyncio.create_task(cell.tick())
    try:
        await asyncio.wait_for(reset_started.wait(), timeout=5)
    finally:
        tick.cancel()
        await asyncio.gather(tick, return_exceptions=True)
    assert tick.cancelled()
    assert cell.is_initializing
    assert weights == {"live": 99, "reference": 17}

    await cell.tick()

    assert cell.is_pending_weights
    assert actions == ["snapshot", "reset_tensors", "reset_tensors"]
    weights["live"] = 17  # The trainer publishes the original checkpoint.
    assert weights["live"] == weights["reference"]
