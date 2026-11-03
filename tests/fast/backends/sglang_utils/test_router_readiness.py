from unittest import mock

import httpx
import pytest

sglang_router = pytest.importorskip("sglang_router")

from miles.backends.sglang_utils import sglang_router_api_client as module
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.utils.http_utils import GeneralHttpClientProvider

ROUTER_URL = "http://router-host:9000"
WORKER_URL = "http://fake-host:1234"


class _FakeResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200):
        self._payload = payload if payload is not None else {"ok": True}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"status {self.status_code}", request=None, response=None)

    def json(self):
        return self._payload


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []
        self.responses: list[_FakeResponse] = []

    def install(self, monkeypatch, responses: list[_FakeResponse] | None = None):
        self.responses = list(responses or [])
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: self)

    async def get(self, url, **kwargs):
        return self._record("get", url, kwargs)

    async def post(self, url, **kwargs):
        return self._record("post", url, kwargs)

    async def delete(self, url, **kwargs):
        return self._record("delete", url, kwargs)

    def _record(self, verb: str, url: str, kwargs: dict) -> _FakeResponse:
        self.calls.append((verb, url, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse()


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    rec.install(monkeypatch)
    return rec


@pytest.fixture
def client():
    return SGLangRouterApiClient(router_url=ROUTER_URL)


async def test_completed_weight_update_explicitly_confirms_native_router_readiness(client, recorder):
    await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True, weights_ready=True)
    assert recorder.calls[0][1] == f"{ROUTER_URL}/add_worker?url={WORKER_URL}&weights_ready=true"


@pytest.mark.parametrize("failure", [httpx.ReadTimeout, httpx.ReadError])
async def test_native_readiness_retries_transport_failures(client, monkeypatch, caplog, failure):
    post = mock.AsyncMock(side_effect=[failure("lost response"), _FakeResponse()])
    fake = type("Client", (), {"post": post})()
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: fake)
    sleep = mock.AsyncMock()
    monkeypatch.setattr(module, "_retry_sleep", sleep)
    await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True, weights_ready=True)
    assert post.await_count == 2
    assert post.await_args_list[0] == post.await_args_list[1]
    assert "weights_ready=true" in post.await_args.args[0]
    assert WORKER_URL in caplog.text
    assert failure.__name__ in caplog.text
    sleep.assert_awaited_once_with(1)


async def test_native_readiness_exhaustion_fails_closed(client, monkeypatch):
    post = mock.AsyncMock(side_effect=httpx.ReadTimeout("unresponsive"))
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: type("Client", (), {"post": post})())
    monkeypatch.setattr(module, "_retry_sleep", mock.AsyncMock())
    with pytest.raises(httpx.ReadTimeout):
        await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True, weights_ready=True)
    assert post.await_count == 3


async def test_native_readiness_does_not_retry_http_rejection(client, monkeypatch):
    post = mock.AsyncMock(return_value=_FakeResponse(status_code=409))
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: type("Client", (), {"post": post})())
    with pytest.raises(httpx.HTTPStatusError):
        await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True, weights_ready=True)
    assert post.await_count == 1
