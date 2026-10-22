"""Exercise the public Ray methods and executor shutdown ordering on CPU."""

from types import SimpleNamespace

import pytest

from miles.ray.rollout import rollout_executor


@pytest.mark.asyncio
async def test_publication_and_drain_rpc_delegates_preserve_results_and_version():
    executor = object.__new__(rollout_executor.RolloutExecutor.__ray_actor_class__)
    calls = []

    async def refresh():
        calls.append("refresh")
        return "ready"

    async def control(operation, **kwargs):
        calls.append((operation, kwargs))
        return {"optimizer_step": 8}

    executor.generate_rollout = SimpleNamespace(begin_refresh=refresh, engine_drain_control=control)
    executor.set_weight_version = lambda version: calls.append(("version", version))
    assert await executor.core_publication_boundary(True, refresh=True) == "ready"
    assert await executor.core_engine_drain("step", rollout_id=3) == {"optimizer_step": 8}
    assert calls == ["refresh", ("step", {"rollout_id": 3}), ("version", 8)]


@pytest.mark.parametrize("backend", ["megatron", "fsdp", "olmo_core"])
def test_executor_stops_core_producer_before_shared_cleanup(monkeypatch, backend):
    executor = object.__new__(rollout_executor.RolloutExecutor.__ray_actor_class__)
    calls = []

    async def shutdown():
        calls.append("producer")

    executor.args = SimpleNamespace(train_backend=backend)
    executor.generate_rollout = SimpleNamespace(shutdown=shutdown)
    executor.data_source = SimpleNamespace(close=lambda: calls.append("data"))
    executor._metric_checker = None
    executor.eval_generate_rollout = None
    monkeypatch.setattr(
        rollout_executor.event_analyzer, "run_analysis_from_args", lambda args: calls.append("analysis")
    )
    monkeypatch.setattr(rollout_executor, "finish_tracking", lambda: calls.append("tracking"))
    executor.dispose()
    assert calls == (["producer"] if backend == "olmo_core" else []) + ["data", "analysis", "tracking"]
