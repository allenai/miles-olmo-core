"""Engine initialization cannot monopolize the rollout controller lock."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from tests.fast.ray.rollout import test_inference_controller_tick as tick_tests
from tests.fast.ray.rollout import test_server_cell_state_machine as state_tests

from miles.ray.rollout import inference_controller as controller_module
from miles.ray.rollout.rollout_server import RolloutServer

cell_env = state_tests.cell_env
pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")


async def test_abort_reaches_serving_engine_while_replacement_is_initializing(cell_env, monkeypatch):
    serving = state_tests._make_cell(args_overrides={"debug_rollout_only": True}, cell_id="serving")
    await serving.init()
    await serving.tick()
    initializing = state_tests._make_cell(args_overrides={"check_weight_update_equal": True})
    await initializing.init()
    server = RolloutServer.__new__(RolloutServer)
    server.server_cells = {"serving": serving, "replacement": initializing}
    controller = tick_tests._make_controller({"default": server})
    server.context_lock = controller.context_lock
    entered, release = asyncio.Event(), asyncio.Event()

    async def check_weights(action, **kwargs):
        if action == "snapshot":
            entered.set()
            await release.wait()

    monkeypatch.setattr(initializing, "check_weights", check_weights)
    abort = AsyncMock()
    monkeypatch.setattr(serving, "abort_all", abort)
    await asyncio.wait_for(controller._tick_cells(), 1)
    await asyncio.wait_for(entered.wait(), 1)
    task = initializing._initialization_task
    await asyncio.wait_for(controller._tick_cells(), 1)
    assert initializing._initialization_task is task
    await asyncio.wait_for(controller.abort_all(), 1)
    abort.assert_awaited_once()
    assert initializing.is_initializing
    release.set()
    await asyncio.wait_for(task, 1)
    assert initializing.is_initializing
    await controller._tick_cells()
    assert initializing.is_pending_weights


async def test_expired_background_preparation_cannot_mark_cell_ready(cell_env, monkeypatch):
    cell = state_tests._make_cell()
    await cell.init()
    controller = tick_tests._make_controller({"default": tick_tests._StubServer({"engine": cell})})
    monkeypatch.setattr(controller_module, "CELL_INITIALIZATION_TICK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(cell, "_prepare_initialization", asyncio.Event().wait)
    await controller._tick_cells()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(cell._initialization_task), 1)
    await controller._tick_cells()
    assert cell._initialization_task is None
    assert cell.is_initializing
