"""Native save/restore preserves the token clock; legacy totals remain unknown."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytest.importorskip("olmo_core")

from miles.backends.core_utils import checkpoint
from miles.backends.core_utils.publication.state import PolicyClock


@pytest.fixture
def saved_actor(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.tensor([1], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: None)
    monkeypatch.setattr(checkpoint.models, "save_native", lambda *args: None)
    load_native = mock.Mock()
    monkeypatch.setattr(checkpoint.models, "load_native", load_native)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            save=str(tmp_path),
            load=str(tmp_path),
            rollout_global_dataset=True,
            olmo_core=SimpleNamespace(
                expert_parallel_size=1, checkpoint_profile=False, checkpoint_keep_last=None, checkpoint_keep_every=None
            ),
        ),
        _agree=lambda fn: fn(),
        train_module=object(),
        lr_scheduler=mock.Mock(state_dict=lambda: {"step": 12}),
        model_config=SimpleNamespace(as_config_dict=lambda: {"width": 32}),
        hf_config=SimpleNamespace(to_dict=lambda: {}),
        clock=PolicyClock(completed_steps=12, published_step=12, next_rollout_id=12),
        _observed_response_tokens=123456789,
    )
    cursor = tmp_path / "rollout" / "global_dataset_state_dict_11.pt"
    cursor.parent.mkdir()
    cursor.write_bytes(b"cursor")
    checkpoint.save(actor, 11)
    checkpoint.finalize(actor, 11)
    return actor, checkpoint.checkpoint_path(tmp_path, 11), load_native


def test_token_clock_restores_and_continues_across_checkpoints(saved_actor):
    actor, path, load_native = saved_actor
    assert torch.load(path / "rank_0.pt", weights_only=False)["consumed_response_tokens"] == 123456789
    actor._observed_response_tokens = 0
    actor.clock = PolicyClock()
    checkpoint.restore(actor)
    assert actor._observed_response_tokens == 123456789
    assert actor.clock.completed_steps == 12
    load_native.assert_called_once()
    actor._observed_response_tokens += 900
    checkpoint.save(actor, 12)
    state = torch.load(checkpoint.checkpoint_path(actor.args.save, 12) / "rank_0.pt", weights_only=False)
    assert state["consumed_response_tokens"] == 123457689


@pytest.mark.parametrize("value", ["absent", None, 0])
def test_legacy_unknown_and_zero_are_distinct(saved_actor, value, caplog):
    actor, path, _ = saved_actor
    state = torch.load(path / "rank_0.pt", weights_only=False)
    if value == "absent":
        del state["consumed_response_tokens"]
    else:
        state["consumed_response_tokens"] = value
    torch.save(state, path / "rank_0.pt")
    checkpoint.restore(actor)
    expected = 0 if value == 0 else None
    assert actor._observed_response_tokens == expected
    if expected is None:
        assert "pre-resume total is unknown" in caplog.text
    checkpoint.save(actor, 12)
    saved = torch.load(checkpoint.checkpoint_path(actor.args.save, 12) / "rank_0.pt", weights_only=False)
    assert saved["consumed_response_tokens"] == expected


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "123"])
def test_corrupt_counter_fails_before_native_restore(saved_actor, invalid):
    actor, path, load_native = saved_actor
    state = torch.load(path / "rank_0.pt", weights_only=False)
    state["consumed_response_tokens"] = invalid
    torch.save(state, path / "rank_0.pt")
    with pytest.raises(ValueError, match="consumed_response_tokens"):
        checkpoint.restore(actor)
    load_native.assert_not_called()
