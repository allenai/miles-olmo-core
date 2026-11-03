"""Pinned Rust API: parent reservation plus independently visible engine@rank IDs."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from miles.backends.sglang_utils import rust_router_membership as module
from miles.utils.http_utils import GeneralHttpClientProvider


@pytest.fixture
async def registry(monkeypatch):
    workers, calls, tasks = {}, [], []
    state = dict(size=2, healthy=True, lose_post=False, lose_delete=False, partial=False)

    async def register():
        for rank in range(state["size"]):
            if rank and state.get("rank_gate") is not None:
                await state["rank_gate"].wait()
            await asyncio.sleep(0.01)
            if state["partial"] and rank:
                return
            workers[f"http://engine@{rank}"] = dict(
                url=f"http://engine@{rank}", id=f"rank-{rank}", is_healthy=state["healthy"] or rank == 0
            )

    async def remove():
        for url in list(workers):
            await asyncio.sleep(0.01)
            if url.startswith("http://engine@"):
                workers.pop(url)

    async def respond(request):
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"workers": list(workers.values())})
        if request.method == "POST":
            tasks.append(asyncio.create_task(register()))
            if state["lose_post"]:
                raise httpx.ReadError("lost parent registration identity")
            return httpx.Response(202, json={"worker_id": "parent"})
        if request.method == "DELETE":
            # The pinned router applies base@* matching. A rank ID would instead
            # match engine@rank@*, remove nothing and falsely accept the request.
            assert request.url.path == "/workers/parent"
            tasks.append(asyncio.create_task(remove()))
            if state["lose_delete"]:
                raise httpx.ReadError("lost delete acknowledgement")
            return httpx.Response(202, json={})
        raise AssertionError(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: client)
        monkeypatch.setattr(module, "POLL_SECONDS", 0.001)
        monkeypatch.setattr(module, "MEMBERSHIP_TIMEOUT_SECONDS", 0.5)
        yield SimpleNamespace(workers=workers, calls=calls, tasks=tasks, state=state)
        await asyncio.gather(*tasks)


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("lost_delete", [False, True])
async def test_all_ranks_ready_and_removed_with_parent_identity(registry, size, lost_delete):
    registry.state.update(size=size, lose_delete=lost_delete)
    member = module.RustRouterMembership("http://router", "http://engine", dp_size=size)
    await member.ensure_registered()
    assert member.worker_id == "parent"
    assert member.worker_ids == {f"http://engine@{rank}": f"rank-{rank}" for rank in range(size)}
    await member.ensure_registered()
    assert registry.calls.count(("POST", "/workers")) == 1
    await member.remove(retire=True)
    assert not registry.workers and not member.worker_ids
    assert member.worker_id is None


async def test_cancelled_registration_waits_for_late_siblings_before_removal(registry):
    registry.state["rank_gate"] = asyncio.Event()
    member = module.RustRouterMembership("http://router", "http://engine", dp_size=2)
    task = asyncio.create_task(member.ensure_registered())
    while not registry.workers:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert len(registry.workers) == 1
    removing = asyncio.create_task(member.remove(retire=True))
    await asyncio.sleep(0)
    assert not removing.done()
    registry.state["rank_gate"].set()
    await removing
    assert not registry.workers


@pytest.mark.parametrize("partial", [False, True])
async def test_incomplete_or_unhealthy_ranks_never_become_ready(registry, partial):
    registry.state.update(partial=partial, healthy=partial)
    member = module.RustRouterMembership("http://router", "http://engine", dp_size=2)
    with pytest.raises(TimeoutError):
        await member.ensure_registered()
    if partial:
        with pytest.raises(TimeoutError):
            await member.remove(retire=True)
        assert not any(method == "DELETE" for method, _ in registry.calls)
    else:
        await member.remove(retire=True)
        assert not registry.workers


async def test_lost_parent_ack_fails_closed_without_reposting_or_deleting_rank_ids(registry):
    registry.state["lose_post"] = True
    member = module.RustRouterMembership("http://router", "http://engine", dp_size=2)
    with pytest.raises(RuntimeError, match="identity unknown"):
        await member.ensure_registered()
    with pytest.raises(RuntimeError, match="identity unknown"):
        await member.remove(retire=True)
    assert registry.calls.count(("POST", "/workers")) == 1
    assert not any(method == "DELETE" for method, _ in registry.calls)


async def test_unowned_and_replaced_ranks_are_never_adopted_or_removed(registry):
    member = module.RustRouterMembership("http://router", "http://engine", dp_size=2)
    registry.workers["http://engine@0"] = dict(url="http://engine@0", id="stranger", is_healthy=True)
    with pytest.raises(RuntimeError, match="Unowned"):
        await member.ensure_registered()
    with pytest.raises(RuntimeError, match="unowned"):
        await member.remove()
    registry.workers.clear()
    await member.ensure_registered()
    registry.workers["http://engine@1"]["id"] = "replacement"
    with pytest.raises(RuntimeError, match="identity changed"):
        await member.ensure_registered()
    with pytest.raises(RuntimeError, match="identity changed"):
        await member.remove()
    assert not any(method == "DELETE" for method, _ in registry.calls)


async def test_empty_registry_reconciles_at_publication_and_ignores_other_engines(registry):
    member = module.RustRouterMembership("http://router", "http://engine", dp_size=2)
    await member.ensure_registered()
    registry.workers.clear()
    registry.workers["http://engine-other@0"] = dict(url="http://engine-other@0", id="other", is_healthy=True)
    await member.ensure_registered()
    assert registry.calls.count(("POST", "/workers")) == 2
    await member.remove()
    assert list(registry.workers) == ["http://engine-other@0"]
