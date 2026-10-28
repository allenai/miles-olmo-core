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
        task = asyncio.create_task(
            module.register_worker(client, "http://router", PAYLOAD, state=module.Registration())
        )
        await entered.wait()
        assert not task.done()
        release.set()
        await task
    assert posts == [PAYLOAD]


@pytest.mark.asyncio
async def test_failed_registration_retries_then_reports_the_underlying_error():
    posts = []

    async def handler(request):
        if request.method == "POST":
            posts.append(request)
            return httpx.Response(202, json=RECEIPT)
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": []})
        return httpx.Response(200, json=worker(status="failed", message="discover_metadata timed out"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(module.RegistrationFailed, match="discover_metadata timed out"):
            await module.register_worker(client, "http://router", PAYLOAD, state=module.Registration())
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
            await module.register_worker(client, "http://router", PAYLOAD, state=module.Registration())


@pytest.mark.asyncio
async def test_registration_has_a_deadline_even_if_job_never_finishes(monkeypatch):
    monkeypatch.setattr(module, "REGISTRATION_TIMEOUT_SECONDS", 0.02)

    async def handler(request):
        return httpx.Response(202, json=RECEIPT) if request.method == "POST" else httpx.Response(200, json=worker())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(TimeoutError, match="admission deadline"):
            await module.register_worker(client, "http://router", PAYLOAD, state=module.Registration())


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
        task = asyncio.create_task(
            module.register_worker(client, "http://router", PAYLOAD, state=module.Registration())
        )
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
            await module.register_worker(client, "http://router", PAYLOAD, state=module.Registration())
    assert len(calls) == attempts


@pytest.mark.asyncio
async def test_cleanup_does_not_forget_an_unresolved_post_with_no_receipt(monkeypatch):
    monkeypatch.setattr(module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    state = module.Registration(pending=True)

    async def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json={"workers": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(module.RegistrationCleanupError):
            await module.unregister_worker(client, "http://router", URL, state=state)
        assert state.pending and state.retiring
        with pytest.raises(module.RegistrationCleanupError):
            await module.register_worker(client, "http://router", PAYLOAD, state=state)


@pytest.mark.asyncio
async def test_lost_delete_response_is_reconciled_without_duplicate_removal_jobs():
    state = module.Registration(worker_id="worker-1")
    registered = True
    deletes = []

    async def handler(request):
        if request.method == "DELETE":
            deletes.append(request)
            raise httpx.ReadTimeout("lost deletion receipt", request=request)
        return httpx.Response(200, json={"workers": [worker(healthy=True)] if registered else []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(module.RegistrationCleanupError):
            await module.unregister_worker(client, "http://router", URL, state=state)
        assert state.removing and state.retiring
        registered = False
        await module.unregister_worker(client, "http://router", URL, state=state)
    assert len(deletes) == 1


@pytest.mark.asyncio
async def test_registry_read_failure_after_job_completion_does_not_resubmit():
    posts, reads = [], []

    async def handler(request):
        if request.method == "POST":
            posts.append(request)
            return httpx.Response(202, json=RECEIPT)
        if request.url.path == "/workers/worker-1":
            return httpx.Response(200, json=worker(status="completed", healthy=True))
        reads.append(request)
        if len(reads) == 1:
            raise httpx.ReadTimeout("registry read failed", request=request)
        return httpx.Response(200, json={"workers": [worker(healthy=True)]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await module.register_worker(client, "http://router", PAYLOAD, state=module.Registration())
    assert len(posts) == 1


@pytest.mark.asyncio
async def test_cleanup_can_retry_a_delete_that_never_connected():
    state = module.Registration(worker_id="worker-1")
    registered = True
    deletes = []

    async def handler(request):
        nonlocal registered
        if request.method == "DELETE":
            deletes.append(request)
            if len(deletes) == 1:
                raise httpx.ConnectError("connection refused", request=request)
            registered = False
            return httpx.Response(202)
        return httpx.Response(200, json={"workers": [worker(healthy=True)] if registered else []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(module.RegistrationCleanupError):
            await module.unregister_worker(client, "http://router", URL, state=state)
        assert not state.removing
        await module.unregister_worker(client, "http://router", URL, state=state)
    assert len(deletes) == 2
