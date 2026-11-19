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


async def test_missing_snapshot_reply_fails_both_readiness_waits_without_another_copy(cell_env, monkeypatch):
    cell = state_tests._make_cell(args_overrides={"check_weight_update_equal": True})
    await cell.init()
    server = RolloutServer.__new__(RolloutServer)
    server.server_cells = {"replacement": cell}
    server.model_name = "default"
    controller = tick_tests._make_controller({"default": server})
    controller.args = cell.args
    server.context_lock = controller.context_lock
    started = asyncio.Event()
    actions = []

    async def never_reply(action, **kwargs):
        actions.append(action)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(cell, "check_weights", never_reply)
    monkeypatch.setattr("miles.ray.rollout.server_cell.SNAPSHOT_REPLY_TIMEOUT_SECONDS", 0.05)
    await controller._tick_cells()
    await asyncio.wait_for(started.wait(), 1)
    with pytest.raises(RuntimeError, match="no snapshot reply"):
        await asyncio.wait_for(asyncio.shield(cell._initialization_task), 1)
    for _ in range(3):
        await controller._tick_cells()
    with pytest.raises(RuntimeError, match=cell.meta.cell_id):
        await server.wait_init_expected_num_cells()
    async with controller.context_lock:
        with pytest.raises(RuntimeError, match=cell.meta.workers_hash):
            await controller._ensure_cells_ready()
    assert actions == ["snapshot"]
