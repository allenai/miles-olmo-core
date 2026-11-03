import pytest

from miles.utils.tracking_utils import wandb_utils
from miles.utils.tracking_utils.base import WandbBackend


def test_wandb_backend_finish_tears_down_process_service(monkeypatch):
    calls = []
    monkeypatch.setattr(wandb_utils.wandb, "finish", lambda: calls.append("finish"))
    monkeypatch.setattr(wandb_utils.wandb, "teardown", lambda: calls.append("teardown"))

    WandbBackend().finish()

    assert calls == ["finish", "teardown"]


def test_wandb_backend_finish_tears_down_after_finish_error(monkeypatch):
    calls = []

    def fail_finish():
        calls.append("finish")
        raise RuntimeError("finish failed")

    monkeypatch.setattr(wandb_utils.wandb, "finish", fail_finish)
    monkeypatch.setattr(wandb_utils.wandb, "teardown", lambda: calls.append("teardown"))

    with pytest.raises(RuntimeError, match="finish failed"):
        WandbBackend().finish()

    assert calls == ["finish", "teardown"]
