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
    await reset_started.wait()
    tick.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tick
    assert cell.is_initializing
    assert weights == {"live": 99, "reference": 17}

    await cell.tick()

    assert cell.is_pending_weights
    assert actions == ["snapshot", "reset_tensors", "reset_tensors"]
    weights["live"] = 17  # The trainer publishes the original checkpoint.
    assert weights["live"] == weights["reference"]


@pytest.mark.parametrize("error", [TimeoutError("reset reply timed out"), RuntimeError("reset reply lost")])
async def test_failed_reset_retries_without_resnapshotting(cell_env, monkeypatch, error):
    cell = state_tests._make_cell(args_overrides={"check_weight_update_equal": True})
    actions = []

    async def check_weights(self, action, **kwargs):
        actions.append(action)
        if action == "reset_tensors" and actions.count(action) == 1:
            raise error

    monkeypatch.setattr(ServerCell, "check_weights", check_weights)
    await cell.init()
    with pytest.raises(type(error), match=str(error)):
        await cell.tick()

    await cell.tick()

    assert cell.is_pending_weights
    assert actions == ["snapshot", "reset_tensors", "reset_tensors"]


async def test_unacknowledged_snapshot_is_retryable_before_any_reset(cell_env, monkeypatch):
    cell = state_tests._make_cell(args_overrides={"check_weight_update_equal": True})
    actions = []

    async def check_weights(self, action, **kwargs):
        actions.append(action)
        if action == "snapshot" and actions.count(action) == 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(ServerCell, "check_weights", check_weights)
    await cell.init()
    with pytest.raises(asyncio.CancelledError):
        await cell.tick()

    await cell.tick()

    assert cell.is_pending_weights
    assert actions == ["snapshot", "snapshot", "reset_tensors"]


async def test_timed_out_snapshot_rejoins_the_same_inflight_rpc(cell_env, monkeypatch):
    cell = state_tests._make_cell(args_overrides={"check_weight_update_equal": True})
    actions = []
    snapshot_started = asyncio.Event()
    finish_snapshot = asyncio.Event()

    async def check_weights(self, action, **kwargs):
        actions.append(action)
        if action == "snapshot":
            snapshot_started.set()
            await finish_snapshot.wait()

    monkeypatch.setattr(ServerCell, "check_weights", check_weights)
    await cell.init()
    first_tick = asyncio.create_task(cell.tick())
    await snapshot_started.wait()
    first_tick.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_tick
    snapshot_task = cell._startup_snapshot_task
    assert not snapshot_task.done()

    second_tick = asyncio.create_task(cell.tick())
    await asyncio.sleep(0)
    assert cell._startup_snapshot_task is snapshot_task
    assert actions == ["snapshot"]
    finish_snapshot.set()
    await second_tick

    assert cell.is_pending_weights
    assert actions == ["snapshot", "reset_tensors"]


async def test_disposal_settles_the_inflight_snapshot(cell_env, monkeypatch):
    cell = state_tests._make_cell(args_overrides={"check_weight_update_equal": True})
    snapshot_started = asyncio.Event()

    async def check_weights(self, action, **kwargs):
        snapshot_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ServerCell, "check_weights", check_weights)
    await cell.init()
    tick = asyncio.create_task(cell.tick())
    await snapshot_started.wait()
    tick.cancel()
    with pytest.raises(asyncio.CancelledError):
        await tick

    await cell.dispose()

    assert cell._startup_snapshot_task.cancelled()


@pytest.mark.parametrize(
    "args_overrides,meta_overrides",
    [
        ({"debug_rollout_only": True}, {}),
        ({}, {"update_weights": False}),
    ],
)
async def test_serving_without_publication_skips_snapshot_and_reset(
    cell_env, monkeypatch, args_overrides, meta_overrides
):
    cell = state_tests._make_cell(
        args_overrides={"check_weight_update_equal": True, **args_overrides},
        **meta_overrides,
    )
    actions = []

    async def check_weights(self, action, **kwargs):
        actions.append(action)

    monkeypatch.setattr(ServerCell, "check_weights", check_weights)
    await cell.init()
    await cell.tick()

    assert cell.is_serving
    assert actions == []


async def test_replacement_cell_at_the_same_address_takes_a_fresh_snapshot(cell_env, monkeypatch):
    actions = []

    async def check_weights(self, action, **kwargs):
        actions.append((self.meta.workers_hash, action))

    monkeypatch.setattr(ServerCell, "check_weights", check_weights)
    original = state_tests._make_cell(args_overrides={"check_weight_update_equal": True})
    await original.init()
    await original.tick()
    original_url = original.server_url
    await original.dispose()
    replacement = state_tests._make_cell(
        args_overrides={"check_weight_update_equal": True}, workers_hash="replacement-generation"
    )
    await replacement.init()
    assert replacement.server_url == original_url
    await replacement.tick()

    assert actions == [
        ("pseudo-hash-0", "snapshot"),
        ("pseudo-hash-0", "reset_tensors"),
        ("replacement-generation", "snapshot"),
        ("replacement-generation", "reset_tensors"),
    ]
