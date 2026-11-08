import asyncio
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.backends.core_utils.publication import cache_protocol, policy_refresh


def engine(version, **overrides):
    state = {"disable_radix_cache": False, "radix_cache_backend": cache_protocol.BACKEND, "weight_version": version}
    state.update(overrides)
    return SimpleNamespace(get_server_info=AsyncMock(return_value={"internal_states": [state]}))


def test_only_owned_backend_relaxes_existing_guard():
    args = SimpleNamespace(pause_generation_mode="in_place", colocate=False, sglang_disable_radix_cache=False)
    with pytest.raises(ValueError):
        policy_refresh.validate_pause_mode(args)
    args.sglang_radix_cache_backend = cache_protocol.BACKEND
    policy_refresh.validate_pause_mode(args)
    assert cache_protocol.enabled(args)
    args.sglang_radix_cache_backend = "unrelated"
    with pytest.raises(ValueError):
        policy_refresh.validate_pause_mode(args)


@pytest.mark.parametrize("previous", [None, "default", "0", "3"])
def test_valid_before_and_after_publication(previous):
    asyncio.run(cache_protocol.validate_engine(engine(previous), 3, committed=False))
    asyncio.run(cache_protocol.validate_engine(engine("3"), 3, committed=True))


@pytest.mark.parametrize("observed", [None, "default", "2", "4", 3])
def test_uncommitted_version_rejected(observed):
    with pytest.raises(RuntimeError):
        asyncio.run(cache_protocol.validate_engine(engine(observed), 3, committed=True))


def test_wrong_backend_and_version_rollback_rejected():
    with pytest.raises(RuntimeError):
        asyncio.run(cache_protocol.validate_engine(engine("0", radix_cache_backend="ordinary"), 3, committed=False))
    with pytest.raises(RuntimeError):
        asyncio.run(cache_protocol.validate_engine(engine("4"), 3, committed=False))


@pytest.mark.parametrize("observed", ["3", "2", "4", None])
def test_actor_resumes_only_after_verified_version(monkeypatch, observed):
    pytest.importorskip("olmo_core")
    actor = import_module("miles.backends.core_utils.actor")
    events = []

    async def readback():
        events.append("readback")
        current = "2" if events.count("readback") == 1 else observed
        return await engine(current).get_server_info()

    def operation(name):
        async def call(*args, **kwargs):
            events.append(name)
            return {"success": True}

        return AsyncMock(side_effect=call)

    client = SimpleNamespace(
        get_server_info=readback,
        **{
            name: operation(name)
            for name in (
                "pause_generation",
                "begin_weight_update",
                "end_weight_update",
                "update_weight_version",
                "continue_generation",
            )
        }
    )
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(
        pause_generation_mode="in_place",
        sglang_disable_radix_cache=False,
        sglang_radix_cache_backend=cache_protocol.BACKEND,
        colocate=False,
        update_weight_buffer_size=1024,
        save=None,
        olmo_core=SimpleNamespace(stream_moe_export=False, expert_publication="fused", weight_sync_mode="flattened"),
    )
    worker.clock = SimpleNamespace(completed_steps=3, published_step=2, published=Mock())
    worker.weight_updater = Mock()
    worker.train_module = worker.hf_config = None
    monkeypatch.setattr(actor.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(actor.dist, "barrier", lambda: None)
    monkeypatch.setattr(actor.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor.models, "iter_export_state", lambda *a, **kw: iter(()))
    info = SimpleNamespace(
        rollout_engines=[client],
        engine_gpu_counts=[1],
        engine_gpu_offsets=[0],
        snapshot_cell_id_to_hashes={"e": "hash"},
    )
    if observed == "3":
        worker.update_weights(info)
        assert events == [
            "readback",
            "pause_generation",
            "begin_weight_update",
            "end_weight_update",
            "update_weight_version",
            "readback",
            "continue_generation",
        ]
    else:
        with pytest.raises(RuntimeError):
            worker.update_weights(info)
        client.continue_generation.assert_not_awaited()
        worker.clock.published.assert_not_called()
