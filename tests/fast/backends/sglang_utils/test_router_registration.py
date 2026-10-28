import asyncio
import json

import httpx
import pytest

from miles.backends.sglang_utils import router_registration as module

URL = "http://engine:30000"
PAYLOAD = {"url": URL, "worker_type": "regular"}
RECEIPT = {"worker_id": "worker-1", "url": URL, "location": "/workers/worker-1"}


def worker(*, status="processing", healthy=False, message=None):
    return {
        "id": "worker-1",
        "url": URL,
        "worker_type": "regular",
        "is_healthy": healthy,
        "job_status": {"status": status, "message": message},
    }


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    async def pause(_):
        await asyncio.sleep(0)

    monkeypatch.setattr(module, "_sleep", pause)


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_response", [False, True])
async def test_acceptance_is_not_admission_and_lost_response_is_reconciled(lost_response):
    entered, release = asyncio.Event(), asyncio.Event()
    posts = []

    async def handler(request):
        if request.method == "POST":
            posts.append(json.loads(request.content))
            if lost_response and len(posts) == 1:
                raise httpx.ReadTimeout("response lost after acceptance", request=request)
            return httpx.Response(202, json=RECEIPT)
        if request.url.path == "/workers/worker-1":
            entered.set()
            await release.wait()
            # A duplicate job can fail after the original succeeded.
            return httpx.Response(200, json=worker(status="failed", healthy=True, message="already exists"))
        return httpx.Response(200, json={"workers": [worker(healthy=True)]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(module.register_worker(client, "http://router", PAYLOAD))
        await entered.wait()
        assert not task.done()
        release.set()
        await task
    assert posts == [PAYLOAD] * (2 if lost_response else 1)


@pytest.mark.asyncio
async def test_failed_registration_retries_then_reports_the_underlying_error():
    posts = []

    async def handler(request):
        if request.method == "POST":
            posts.append(request)
            return httpx.Response(202, json=RECEIPT)
        return httpx.Response(200, json=worker(status="failed", message="discover_metadata timed out"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(module.RegistrationFailed, match="discover_metadata timed out"):
            await module.register_worker(client, "http://router", PAYLOAD)
    assert len(posts) == module.REGISTRATION_ATTEMPTS


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["id", "url"])
async def test_a_different_worker_cannot_satisfy_registration(mismatch):
    async def handler(request):
        if request.method == "POST":
            return httpx.Response(202, json=RECEIPT)
        return httpx.Response(200, json={**worker(healthy=True), mismatch: "replacement"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="identity changed"):
            await module.register_worker(client, "http://router", PAYLOAD)


@pytest.mark.asyncio
async def test_registration_has_a_deadline_even_if_job_never_finishes(monkeypatch):
    monkeypatch.setattr(module, "REGISTRATION_TIMEOUT_SECONDS", 0.02)

    async def handler(request):
        return httpx.Response(202, json=RECEIPT) if request.method == "POST" else httpx.Response(200, json=worker())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TimeoutError, match="admission deadline"):
            await module.register_worker(client, "http://router", PAYLOAD)


@pytest.mark.asyncio
async def test_cancellation_does_not_resubmit_registration():
    entered = asyncio.Event()
    posts = []

    async def handler(request):
        if request.method == "POST":
            posts.append(request)
            return httpx.Response(202, json=RECEIPT)
        entered.set()
        await asyncio.Future()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        task = asyncio.create_task(module.register_worker(client, "http://router", PAYLOAD))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(posts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status, attempts", [(503, 3), (429, 3), (400, 1), (401, 1)])
async def test_transient_http_errors_are_bounded_and_client_errors_fail_immediately(status, attempts):
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "injected"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await module.register_worker(client, "http://router", PAYLOAD)
    assert len(calls) == attempts
