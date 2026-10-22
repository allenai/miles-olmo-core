"""Fork checkpoint contracts at the real driver boundary, using CPU worker fakes."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import train as train_driver
from tests.fast.fixtures.driver_fakes import FakeInferenceController, FakeRolloutExecutor, FakeTrainingModel


def _make_args(**overrides: Any) -> SimpleNamespace:
    args = SimpleNamespace(
        api_server_host="127.0.0.1",
        api_server_port=None,
        check_weight_update_allow_quant_error=False,
        check_weight_update_equal=False,
        check_weight_update_selector=None,
        check_weight_update_skip_list=None,
        colocate_memory_peak_device="cpu",
        debug_exit_after_rollout=None,
        eval_interval=None,
        eval_uses_snapshots=False,
        ft_components=[],
        fully_async=False,
        hf_checkpoint="/base/checkpoint",
        num_critic_only_steps=0,
        num_rollout=0,
        offload_rollout=False,
        offload_rollout_level="",
        offload_train=False,
        save_interval=None,
        save_trigger_sentinel=None,
        skip_eval_before_train=False,
        start_rollout_id=0,
        use_critic=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _install_driver_fakes(
    monkeypatch: pytest.MonkeyPatch, args: SimpleNamespace, events: list[str]
) -> SimpleNamespace:
    components = SimpleNamespace(
        inference_controller=FakeInferenceController(events),
        rollout_executor=FakeRolloutExecutor(events),
        actor_model=FakeTrainingModel(events, "actor"),
        critic_model=FakeTrainingModel(events, "critic") if args.use_critic else None,
    )

    async def create_rollout_components(_args: SimpleNamespace) -> tuple[Any, Any, int]:
        return components.inference_controller, components.rollout_executor, 4

    async def create_training_models(_args: SimpleNamespace, _controller: Any, _executor: Any) -> tuple[Any, Any]:
        return components.actor_model, components.critic_model

    async def update_weights(_model: Any, _executor: Any, rollout_id: int | None = None) -> None:
        events.append(f"update_weights:{rollout_id}")

    monkeypatch.setattr(train_driver, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_driver, "maybe_start_periodic_pyspy_dump", lambda: None)
    monkeypatch.setattr(train_driver, "launch_worker_manager", lambda _args: None)
    monkeypatch.setattr(train_driver.object_store, "init_instance", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(train_driver, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_driver, "create_rollout_components", create_rollout_components)
    monkeypatch.setattr(train_driver, "create_training_models", create_training_models)
    monkeypatch.setattr(train_driver, "maybe_start_mini_ft_controller", lambda _args: None)
    monkeypatch.setattr(train_driver, "update_weights", update_weights)
    monkeypatch.setattr(train_driver, "remove_rollout_data_refs", lambda *_args, **_kwargs: None)
    return components


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
    task = asyncio.create_task(train_driver.train(args))
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
