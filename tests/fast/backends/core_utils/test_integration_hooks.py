"""Contracts that must survive upstream orchestration changes."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.backends.core_utils import integration
from miles.backends.hooks import get_backend_hooks
from miles.utils.async_utils import get_async_loop


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["megatron", "fsdp", None])
async def test_native_hooks_preserve_paths_and_do_not_call_core_operations(backend):
    args = SimpleNamespace(rollout_function_path="upstream.Rollout", eval_function_path="custom.Eval")
    producer = SimpleNamespace(shutdown=AsyncMock())
    actor = SimpleNamespace(finalize_checkpoint=AsyncMock())
    hooks = get_backend_hooks(backend)
    hooks.configure_rollout(args, "custom.Eval")
    hooks.shutdown_rollout(producer)
    await hooks.finalize_checkpoint(actor, 3)
    assert vars(args) == {"rollout_function_path": "upstream.Rollout", "eval_function_path": "custom.Eval"}
    producer.shutdown.assert_not_called()
    actor.finalize_checkpoint.assert_not_called()


def test_hooks_load_without_optional_backend_dependencies():
    script = """
import sys
from miles.backends.hooks import get_backend_hooks
for backend in ("megatron", "fsdp", "olmo_core"):
    get_backend_hooks(backend)
assert not any(name.split(".")[0] in ("torch", "open_instruct", "olmo_core", "sglang") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=20)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "begin,refresh,method",
    [
        (True, False, "prepare_publication"),
        (False, False, "finish_publication"),
        (True, True, "begin_refresh"),
        (False, True, "end_refresh"),
    ],
)
async def test_publication_uses_producer_loop(begin, refresh, method):
    caller_loop = asyncio.get_running_loop()

    async def boundary():
        assert asyncio.get_running_loop() is get_async_loop().loop
        assert asyncio.get_running_loop() is not caller_loop
        return "acknowledged"

    producer = SimpleNamespace(**{method: boundary})
    assert await integration.publication_boundary(producer, begin, refresh=refresh) == "acknowledged"


@pytest.mark.asyncio
async def test_missing_barrier_is_optional_but_refresh_requires_managed_producer():
    assert await integration.publication_boundary(object(), True) is None
    with pytest.raises(RuntimeError, match="managed refreshing producer"):
        await integration.publication_boundary(object(), True, refresh=True)
    with pytest.raises(RuntimeError, match="draining producer"):
        await integration.engine_drain(object(), "step", set_weight_version=lambda version: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["initialize", "step", "publish", "status"])
async def test_drain_mutates_producer_then_updates_version_on_caller_loop(operation):
    caller_loop = asyncio.get_running_loop()
    versions = []

    async def control(received_operation, **kwargs):
        assert asyncio.get_running_loop() is get_async_loop().loop
        assert received_operation == operation
        assert kwargs == {"rollout_id": 3}
        assert versions == []
        return {"optimizer_step": 7}

    def set_version(version):
        assert asyncio.get_running_loop() is caller_loop
        versions.append(version)

    result = await integration.engine_drain(
        SimpleNamespace(engine_drain_control=control), operation, set_weight_version=set_version, rollout_id=3
    )
    assert result == {"optimizer_step": 7}
    assert versions == ([7] if operation in ("initialize", "step", "publish") else [])


@pytest.mark.asyncio
async def test_failed_drain_does_not_publish_version():
    versions = []
    producer = SimpleNamespace(engine_drain_control=AsyncMock(side_effect=RuntimeError("publish failed")))
    with pytest.raises(RuntimeError, match="publish failed"):
        await integration.engine_drain(producer, "publish", set_weight_version=versions.append)
    assert versions == []


def test_shutdown_waits_for_producer_loop():
    completed = []

    async def shutdown():
        assert asyncio.get_running_loop() is get_async_loop().loop
        await asyncio.sleep(0)
        completed.append(True)

    get_backend_hooks("olmo_core").shutdown_rollout(SimpleNamespace(shutdown=shutdown))
    assert completed == [True]
    get_backend_hooks("olmo_core").shutdown_rollout(None)


@pytest.mark.asyncio
async def test_finalization_waits_for_acknowledgement_and_propagates_failure():
    entered, release = asyncio.Event(), asyncio.Event()

    async def finalize(rollout_id):
        assert rollout_id == 4
        entered.set()
        await release.wait()
        raise RuntimeError("checkpoint incomplete")

    task = asyncio.create_task(
        get_backend_hooks("olmo_core").finalize_checkpoint(SimpleNamespace(finalize_checkpoint=finalize), 4)
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not task.done()
        release.set()
        with pytest.raises(RuntimeError, match="checkpoint incomplete"):
            await asyncio.wait_for(task, 2)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
