"""Trainer initialization can finish before the rollout executor exists."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.ray import placement_group


@pytest.mark.parametrize("critic", [False, True])
async def test_initialize_then_connect_preserves_restore_and_parallel_metadata(monkeypatch, critic):
    actor = SimpleNamespace(get_train_parallel_config=AsyncMock(return_value={"dp_size": 2}))
    value = SimpleNamespace()
    configs = [SimpleNamespace(role="actor", trainer_id="actor")]
    if critic:
        configs.append(SimpleNamespace(role="critic", trainer_id="critic"))
    handles = {"actor": actor, "critic": value}
    args = SimpleNamespace(use_critic=critic, start_rollout_id=None)
    monkeypatch.setattr(placement_group, "compute_trainer_configs", lambda _: configs)
    monkeypatch.setattr(placement_group, "create_trainer_handles", lambda *a, **k: handles)
    monkeypatch.setattr(placement_group, "take_over_trainers", AsyncMock(return_value=False))
    monkeypatch.setattr(placement_group, "compute_trainer_args", lambda a, c: a)

    async def initialize(args, *, handle, **kwargs):
        return placement_group.TrainerInfo(handle=handle, restored_rollout_id=7, start_rollout_id=7)

    monkeypatch.setattr(placement_group, "create_training_model", initialize)
    models = await placement_group.initialize_training_models(args)
    assert models.actor.handle is actor
    assert (models.critic.handle is value) if critic else models.critic is None
    assert args.start_rollout_id is None  # No shared-state mutation while serving initializes.
    actor.get_train_parallel_config.assert_not_awaited()
    executor = SimpleNamespace(set_train_parallel_config=AsyncMock(), load=AsyncMock())
    await placement_group.connect_training_models(args, models, executor)
    assert args.start_rollout_id == 7
    executor.set_train_parallel_config.assert_awaited_once_with({"dp_size": 2})
    executor.load.assert_awaited_once_with(6)


async def test_legacy_entrypoint_still_initializes_before_connection(monkeypatch):
    actor, critic = object(), object()
    models = placement_group.InitializedTrainingModels(
        actor=placement_group.TrainerInfo(actor, 4, 4), critic=placement_group.TrainerInfo(critic, 4, 4)
    )
    events = []

    async def initialize(args):
        events.append("initialize")
        return models

    async def connect(args, actual, executor):
        assert actual is models
        events.append("connect")

    monkeypatch.setattr(placement_group, "initialize_training_models", initialize)
    monkeypatch.setattr(placement_group, "connect_training_models", connect)
    assert await placement_group.create_training_models(object(), object()) == (actor, critic)
    assert events == ["initialize", "connect"]
