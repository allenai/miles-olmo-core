import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from miles.utils import infra_timeouts


@pytest.fixture(autouse=True)
def reset_multiplier(monkeypatch):
    monkeypatch.delenv("MILES_INFRA_TIMEOUT_MULTIPLIER", raising=False)


def test_deadlines_preserve_none_and_scale_each_http_component(monkeypatch):
    assert infra_timeouts.seconds(30) == 30
    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", "5")
    assert infra_timeouts.seconds(None) is None
    timeout = infra_timeouts.http_timeout(httpx.Timeout(None, connect=10, write=30, pool=2))
    assert timeout.as_dict() == {"connect": 50, "read": None, "write": 150, "pool": 10}


@pytest.mark.parametrize("factor", ["0", "-1", "nan", "inf", "garbage"])
def test_invalid_factor_rejected(monkeypatch, factor):
    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", factor)
    with pytest.raises(ValueError):
        infra_timeouts.seconds(1)


async def test_slow_request_warns_but_completes_inside_extended_deadline(monkeypatch, caplog):
    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", "5")
    with caplog.at_level("WARNING"):
        assert (
            await infra_timeouts.wait_for(asyncio.sleep(0.03, result="ready"), 0.01, operation="engine worker-7")
            == "ready"
        )
    assert "operation=engine worker-7" in caplog.text
    assert "deadline_s=0.05" in caplog.text
    assert "finished" in caplog.text


async def test_hard_timeout_still_cancels_underlying_operation(monkeypatch):
    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", "5")
    cancelled = asyncio.Event()

    async def stuck():
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    with pytest.raises(TimeoutError):
        await infra_timeouts.wait_for(stuck(), 0.001, operation="stuck")
    assert cancelled.is_set()


async def test_request_scales_once_and_hides_credentials(monkeypatch, caplog):
    monkeypatch.setenv("MILES_INFRA_TIMEOUT_MULTIPLIER", "5")

    async def slow(*args, **kwargs):
        await asyncio.sleep(0.02)
        return kwargs["timeout"]

    client = type("Client", (), {"post": AsyncMock(side_effect=slow)})()
    with caplog.at_level("WARNING"):
        timeout = await infra_timeouts.request(
            client, "post", "http://user:secret@engine:1234/add_worker?token=private", timeout=0.001
        )
    assert timeout == 0.005
    assert "engine:1234/add_worker" in caplog.text
    assert "secret" not in caplog.text
    assert "private" not in caplog.text


async def test_caller_cancellation_propagates_and_joins_watchdog():
    baseline = asyncio.all_tasks()
    task = asyncio.create_task(infra_timeouts.wait_for(asyncio.sleep(100), 30, operation="cancel"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert asyncio.all_tasks() == baseline
