"""Core must retain live state only on the explicit, safe publication path."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.backends.core_utils import actor
from miles.backends.core_utils.publication import policy_refresh


@pytest.mark.parametrize("mode", ["retract", "in_place"])
@pytest.mark.parametrize("failure", [None, "pause_generation", "begin_weight_update", "end_weight_update"])
def test_publication_order_and_failed_update_stays_paused(monkeypatch, mode, failure):
    events = []

    def method(name):
        async def call(*args, **kwargs):
            events.append((name, kwargs))
            if name == failure:
                raise RuntimeError("injected publication failure")

        return AsyncMock(side_effect=call)

    names = (
        "pause_generation",
        "flush_cache",
        "begin_weight_update",
        "end_weight_update",
        "update_weight_version",
        "continue_generation",
    )
    engine = SimpleNamespace(**{name: method(name) for name in names})
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(
        pause_generation_mode=mode,
        sglang_disable_radix_cache=True,
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
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            worker.update_weights(info)
        engine.continue_generation.assert_not_awaited()
        worker.clock.published.assert_not_called()
    else:
        assert worker.update_weights(info) == 3
        expected = ["pause_generation"] + (["flush_cache"] if mode == "retract" else [])
        assert [name for name, _ in events] == expected + [
            "begin_weight_update",
            "end_weight_update",
            "update_weight_version",
            "continue_generation",
        ]
        engine.update_weight_version.assert_awaited_once_with("3", abort_all_requests=False)
        worker.clock.published.assert_called_once()
    engine.pause_generation.assert_awaited_once_with(mode=mode)
    if mode == "in_place":
        engine.flush_cache.assert_not_awaited()


@pytest.mark.parametrize("colocate,radix_disabled", [(True, True), (False, False)])
def test_reject_unsafe_retained_state(colocate, radix_disabled):
    with pytest.raises(ValueError):
        policy_refresh.validate_pause_mode(
            SimpleNamespace(
                pause_generation_mode="in_place", colocate=colocate, sglang_disable_radix_cache=radix_disabled
            )
        )
