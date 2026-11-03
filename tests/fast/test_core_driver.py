"""Fork checkpoint contracts at the real driver boundary, using CPU worker fakes."""

import asyncio

import pytest
import train as train_driver
from tests.fast.test_train import _install_driver_fakes, _make_args

from miles.utils.async_utils import with_disposer


@pytest.mark.parametrize("backend", ["megatron", "fsdp", "olmo_core"])
@pytest.mark.parametrize("fail_finalization", [False, True])
async def test_checkpoint_saves_both_states_before_awaiting_backend_finalization(
    monkeypatch, backend, fail_finalization
):
    events = []
    args = _make_args(num_rollout=1, save_interval=1, train_backend=backend)
    components = _install_driver_fakes(monkeypatch, args, events)
    entered, release = asyncio.Event(), asyncio.Event()

    async def finalize(rollout_id):
        assert events.index("actor_save:0") < events.index("executor_save:0")
        assert rollout_id == 0
        events.append("finalize")
        entered.set()
        await release.wait()
        if fail_finalization:
            raise RuntimeError("checkpoint incomplete")

    components.actor_model.finalize_checkpoint = finalize
    task = asyncio.create_task(with_disposer(train_driver.train, args))
    try:
        if backend == "olmo_core":
            await asyncio.wait_for(entered.wait(), 2)
            assert not task.done()
            assert "actor_dispose" not in events
            release.set()
        if backend == "olmo_core" and fail_finalization:
            with pytest.raises(RuntimeError, match="checkpoint incomplete"):
                await asyncio.wait_for(task, 2)
        else:
            await asyncio.wait_for(task, 2)
        assert ("finalize" in events) == (backend == "olmo_core")
        assert "actor_save:0" in events and "executor_save:0" in events
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
