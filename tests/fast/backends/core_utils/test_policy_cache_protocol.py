import asyncio
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.backends.core_utils.publication import cache_protocol, policy_refresh


def test_versioned_opt_in_preserves_default_guard():
    args = SimpleNamespace(pause_generation_mode="in_place", colocate=False, sglang_disable_radix_cache=False)
    with pytest.raises(ValueError):
        policy_refresh.validate_pause_mode(args)
    args.sglang_enable_policy_versioned_cache = True
    policy_refresh.validate_pause_mode(args)


def engine(epoch, **overrides):
    state = {"disable_radix_cache": False, "policy_cache": {"protocol": 1, "epoch": epoch, "updating": False}}
    state["policy_cache"].update(overrides)
    return SimpleNamespace(get_server_info=AsyncMock(return_value={"internal_states": [state]}))


@pytest.mark.parametrize("epoch,overrides", [(2, {}), (4, {}), (3, {"updating": True}), (3, {"protocol": 0})])
def test_invalid_commit_never_accepted(epoch, overrides):
    with pytest.raises(RuntimeError):
        asyncio.run(cache_protocol.verify_commit(engine(epoch, **overrides), 2))


def test_committed_epoch_and_missing_capability():
    asyncio.run(cache_protocol.verify_commit(engine(3), 2))
    missing = SimpleNamespace(get_server_info=AsyncMock(return_value={}))
    with pytest.raises(RuntimeError):
        asyncio.run(cache_protocol.read_epoch(missing))


@pytest.mark.parametrize("observed", [3, 2, 4, None])
def test_publication_only_resumes_after_verified_cache_commit(monkeypatch, observed):
    pytest.importorskip("olmo_core")
    actor = import_module("miles.backends.core_utils.actor")
    events = []

    async def readback():
        events.append("readback")
        epoch = 2 if events.count("readback") == 1 else observed
        return {
            "internal_states": [
                {
                    "disable_radix_cache": False,
                    "policy_cache": {
                        "protocol": 1,
                        "epoch": epoch,
                        "updating": False,
                    },
                }
            ]
        }

    def operation(name):
        async def call(*args, **kwargs):
            events.append(name)
            return {"success": True}

        return AsyncMock(side_effect=call)

    engine = SimpleNamespace(
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
        sglang_enable_policy_versioned_cache=True,
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
        rollout_engines=[engine],
        engine_gpu_counts=[1],
        engine_gpu_offsets=[0],
        snapshot_cell_id_to_hashes={"engine": "hash"},
    )
    if observed == 3:
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
        engine.continue_generation.assert_not_awaited()
        worker.clock.published.assert_not_called()
