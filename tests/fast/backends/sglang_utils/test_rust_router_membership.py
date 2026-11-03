"""Publication owns registration; lost acknowledgements never imply readiness."""

import asyncio

import httpx
import pytest

from miles.backends.sglang_utils import rust_router_membership as module
from miles.utils.http_utils import GeneralHttpClientProvider


@pytest.fixture
def registry(monkeypatch):
    workers, calls = {}, []
    state = {"post_delay": 0, "delete_delay": 0, "lose_post": False, "lose_delete": False, "healthy": True}
    tasks = []

    async def later(delay, action):
        await asyncio.sleep(delay)
        action()

    async def respond(request):
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"workers": list(workers.values())})
        if request.method == "POST":
            if state.get("reject_post"):
                return httpx.Response(state["reject_post"], json={})
            worker = dict(id="worker-1", url="http://engine", is_healthy=state["healthy"])
            tasks.append(asyncio.create_task(later(state["post_delay"], lambda: workers.update({"worker-1": worker}))))
            if state["lose_post"]:
                raise httpx.ReadError("acknowledgement lost")
            return httpx.Response(202, json={"worker_id": "worker-1"})
        if request.method == "DELETE":
            if state.get("already_removed"):
                workers.clear()
                return httpx.Response(404, json={})
            tasks.append(asyncio.create_task(later(state["delete_delay"], lambda: workers.pop("worker-1", None))))
            if state["lose_delete"]:
                raise httpx.ReadError("acknowledgement lost")
            return httpx.Response(202, json={})
        raise AssertionError(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: client)
    monkeypatch.setattr(module, "POLL_SECONDS", 0.001)
    monkeypatch.setattr(module, "MEMBERSHIP_TIMEOUT_SECONDS", 0.15)
    return module.RustRouterMembership("http://router", "http://engine"), workers, calls, state, tasks, client


@pytest.mark.parametrize("lost_ack", [False, True])
async def test_async_registration_and_removal_are_confirmed(registry, lost_ack):
    member, workers, calls, state, tasks, client = registry
    state.update(post_delay=0.02, delete_delay=0.02, lose_post=lost_ack, lose_delete=lost_ack)
    try:
        await member.ensure_registered()
        assert member.worker_id == "worker-1" and workers
        await member.ensure_registered()
        assert calls.count(("POST", "/workers")) == 1
        await member.remove()
        assert not workers and member.worker_id is None
        assert calls.count(("DELETE", "/workers/worker-1")) == 1
    finally:
        await asyncio.gather(*tasks)
        await client.aclose()


async def test_cancelled_registration_is_removed_even_if_it_appears_late(registry):
    member, workers, calls, state, tasks, client = registry
    state["post_delay"] = 0.03
    registering = asyncio.create_task(member.ensure_registered())
    while ("POST", "/workers") not in calls:
        await asyncio.sleep(0)
    registering.cancel()
    await asyncio.gather(registering, return_exceptions=True)
    assert not workers
    await member.remove(retire=True)
    assert not workers
    with pytest.raises(RuntimeError, match="Retired"):
        await member.ensure_registered()
    await asyncio.gather(*tasks)
    await client.aclose()


async def test_registration_acceptance_without_health_is_not_readiness(registry):
    member, workers, calls, state, tasks, client = registry
    state["healthy"] = False
    with pytest.raises(TimeoutError):
        await member.ensure_registered()
    assert calls.count(("POST", "/workers")) == 1
    await member.remove(retire=True)
    assert not workers
    await client.aclose()


async def test_unknown_or_replaced_worker_is_never_adopted_or_deleted(registry):
    member, workers, calls, state, tasks, client = registry
    workers["other"] = dict(id="other", url="http://engine", is_healthy=True)
    with pytest.raises(RuntimeError, match="Unowned"):
        await member.ensure_registered()
    with pytest.raises(RuntimeError, match="unowned"):
        await member.remove()
    assert all(method == "GET" for method, _ in calls)
    member.worker_id = "old-instance"
    with pytest.raises(RuntimeError, match="identity changed"):
        await member.ensure_registered()
    await client.aclose()


async def test_empty_registry_reconciles_only_on_explicit_publication(registry):
    member, workers, calls, state, tasks, client = registry
    await member.ensure_registered()
    workers.clear()  # router restarted; no health callback registers it
    await asyncio.sleep(0.005)
    assert not workers
    await member.ensure_registered()
    assert workers and calls.count(("POST", "/workers")) == 2
    await member.remove(retire=True)
    await client.aclose()


async def test_failed_removal_cannot_finish_or_allow_retired_registration(registry):
    member, workers, calls, state, tasks, client = registry
    await member.ensure_registered()
    state["delete_delay"] = 0.25
    with pytest.raises(TimeoutError):
        await member.remove(retire=True)
    with pytest.raises(RuntimeError, match="Retired"):
        await member.ensure_registered()
    await asyncio.gather(*tasks)
    await client.aclose()


async def test_concurrent_retirement_serializes_after_registration(registry):
    member, workers, calls, state, tasks, client = registry
    state["post_delay"] = 0.02
    registering = asyncio.create_task(member.ensure_registered())
    while ("POST", "/workers") not in calls:
        await asyncio.sleep(0)
    await member.remove(retire=True)
    await registering
    assert not workers
    await client.aclose()


async def test_rejected_registration_can_be_retired_without_waiting(registry):
    member, workers, calls, state, tasks, client = registry
    state["reject_post"] = 400
    with pytest.raises(httpx.HTTPStatusError):
        await member.ensure_registered()
    assert not member._registration_pending
    await member.remove(retire=True)
    assert not workers
    await client.aclose()


async def test_removal_accepts_404_only_after_confirming_absence(registry):
    member, workers, calls, state, tasks, client = registry
    await member.ensure_registered()
    state["already_removed"] = True
    await member.remove(retire=True)
    assert not workers and member.worker_id is None
    await client.aclose()
