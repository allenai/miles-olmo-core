"""Core must retain live state only on the explicit, safe publication path."""

import json
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.backends.core_utils.publication import policy_refresh


@pytest.fixture
def actor():
    # Lightweight CPU CI omits the optional Core trainer. Keep the pure policy
    # guards below runnable there, and exercise actor ordering in the runtime.
    pytest.importorskip("olmo_core")
    return import_module("miles.backends.core_utils.actor")


@pytest.mark.parametrize("mode", ["retract", "in_place"])
@pytest.mark.parametrize(
    "failure", [None, "pause_generation", "begin_weight_update", "end_weight_update", "negative_ack"]
)
def test_publication_order_and_failed_update_stays_paused(monkeypatch, actor, mode, failure):
    events = []

    def method(name):
        async def call(*args, **kwargs):
            events.append((name, kwargs))
            if name == failure:
                raise RuntimeError("injected publication failure")
            if name == "end_weight_update" and failure == "negative_ack":
                return {"success": False, "message": "injected negative acknowledgment"}

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
        engine.update_weight_version.assert_not_awaited()
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


def config_args(tmp_path, models, **overrides):
    path = tmp_path / "engines.json"
    path.write_text(json.dumps({"sglang": models}))
    values = dict(
        pause_generation_mode="in_place",
        sglang_disable_radix_cache=True,
        sglang_config=str(path),
        starts_inference_engines=True,
        colocate=False,
        offload_rollout=False,
        debug_rollout_only=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        num_gpus_per_node=8,
        rollout_num_gpus=sum(g["num_gpus"] for model in models for g in model["server_groups"]),
        rollout_num_gpus_per_engine=1,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        hf_checkpoint="/policy",
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("worker_type", ["regular", "prefill", "decode"])
@pytest.mark.parametrize("disabled", [False, None, "false"])
def test_reject_cache_override_in_later_policy_group(tmp_path, worker_type, disabled):
    models = [
        dict(
            name="policy",
            server_groups=[
                dict(worker_type="regular", num_gpus=1),
                dict(worker_type=worker_type, num_gpus=1, overrides={"disable_radix_cache": disabled}),
            ],
        )
    ]
    args = config_args(tmp_path, models)
    with pytest.raises(ValueError, match="model 'policy', server group 1"):
        policy_refresh.validate_pause_mode(args)


@pytest.mark.parametrize("global_disabled", [False, True])
def test_safe_policy_override_allows_frozen_cache_and_placeholders(tmp_path, global_disabled):
    models = [
        dict(
            name="policy",
            server_groups=[
                dict(worker_type="regular", num_gpus=1, overrides={"disable_radix_cache": True}),
                dict(worker_type="placeholder", num_gpus=1, overrides={"disable_radix_cache": False}),
            ],
        ),
        dict(
            name="reference",
            update_weights=False,
            server_groups=[dict(worker_type="regular", num_gpus=1, overrides={"disable_radix_cache": False})],
        ),
        dict(
            name="judge",
            model_path="/judge",
            server_groups=[dict(worker_type="regular", num_gpus=1, overrides={"disable_radix_cache": False})],
        ),
    ]
    policy_refresh.validate_pause_mode(config_args(tmp_path, models, sglang_disable_radix_cache=global_disabled))


def test_inferred_policy_in_later_model_cannot_enable_cache(tmp_path):
    models = [
        dict(name="frozen", update_weights=False, server_groups=[dict(worker_type="regular", num_gpus=1)]),
        dict(
            name="second_policy",
            server_groups=[dict(worker_type="regular", num_gpus=1, overrides={"disable_radix_cache": False})],
        ),
    ]
    with pytest.raises(ValueError, match="model 'second_policy'"):
        policy_refresh.validate_pause_mode(config_args(tmp_path, models))


@pytest.mark.parametrize("updated,group_override", [(False, False), (True, False), (True, True)])
def test_updated_eval_model_uses_resolved_eval_and_group_precedence(tmp_path, updated, group_override):
    models = [
        dict(name="policy", server_groups=[dict(worker_type="regular", num_gpus=1)]),
        dict(
            name="eval",
            update_weights=updated,
            server_groups=[
                dict(
                    worker_type="regular",
                    num_gpus=1,
                    overrides={"disable_radix_cache": True} if group_override else {},
                )
            ],
        ),
    ]
    args = config_args(tmp_path, models, rollout_num_gpus=1, eval_num_gpus=1, eval_sglang_disable_radix_cache=False)
    if updated and not group_override:
        with pytest.raises(ValueError, match="model 'eval'"):
            policy_refresh.validate_pause_mode(args)
    else:
        policy_refresh.validate_pause_mode(args)


def test_retract_allows_policy_cache_override(tmp_path):
    models = [
        dict(
            name="policy",
            server_groups=[dict(worker_type="regular", num_gpus=1, overrides={"disable_radix_cache": False})],
        )
    ]
    policy_refresh.validate_pause_mode(config_args(tmp_path, models, pause_generation_mode="retract"))
