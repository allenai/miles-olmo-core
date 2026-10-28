"""Own asynchronous SGLang registrations until admission or confirmed removal.

A lost POST response is ambiguous: reconcile it instead of submitting a second
AddWorker job. The router deduplicates worker IDs, but not background jobs.
"""

import asyncio
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from miles.utils import infra_timeouts, logger_utils

logger = logger_utils.setup_logger(__name__)

REGISTRATION_TIMEOUT_SECONDS = 180.0
REGISTRATION_ATTEMPTS = 3
CLEANUP_TIMEOUT_SECONDS = 25.0
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_sleep = asyncio.sleep


class RegistrationFailed(RuntimeError):
    pass


class RegistrationCleanupError(RuntimeError):
    """The caller must retain ownership or terminate the router before forgetting this worker."""


@dataclass
class Registration:
    worker_id: str | None = None
    submitted: bool = False
    pending: bool = False
    retiring: bool = False
    removing: bool = False


async def _workers(client, router_url):
    response = await infra_timeouts.request(client, "get", router_url + "/workers", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["workers"]


def _identify(state, worker, worker_url):
    if worker.get("url") != worker_url or not worker.get("id"):
        raise ValueError(f"Router registration identity changed or is invalid for {worker_url}: {worker}")
    if state.worker_id is not None and worker["id"] != state.worker_id:
        raise ValueError(f"Router registration identity changed for {worker_url}: {worker}")
    state.worker_id = worker["id"]


async def _status(client, router_url, worker_url, state):
    if state.worker_id is None:
        # A lost receipt provides no ID lookup. The registered URL can recover
        # it, but an empty registry cannot prove the AddWorker job is absent.
        rows = [row for row in await _workers(client, router_url) if row.get("url") == worker_url]
        if not rows:
            return None
        _identify(state, rows[0], worker_url)
    location = f"/workers/{quote(state.worker_id, safe='')}"
    response = await infra_timeouts.request(client, "get", router_url + location, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    worker = response.json()
    _identify(state, worker, worker_url)
    status = (worker.get("job_status") or {}).get("status")
    if status in ("completed", "failed") or (
        status is None and worker.get("worker_type") in ("regular", "prefill", "decode")
    ):
        # Router 0.3.2 returns job_status only for placeholders. A materialized
        # worker resource has passed registration; our single submission cannot
        # insert another worker after removal. Newer routers may retain status.
        state.pending = False
    elif status not in (None, "queued", "pending", "processing"):
        raise ValueError(f"Unknown router registration state for {worker_url}: {worker}")
    return worker


async def _submit(client, router_url, payload, state):
    state.submitted = True
    state.pending = True  # Record intent before a request can be accepted or cancelled.
    try:
        response = await infra_timeouts.request(
            client, "post", router_url + "/workers", json=payload, timeout=REQUEST_TIMEOUT
        )
    except (httpx.ConnectError, httpx.ConnectTimeout):
        state.submitted = False
        state.pending = False  # No connection was established; resubmission is safe.
        raise
    if response.is_client_error:
        state.submitted = False
        state.pending = False  # An explicit client rejection did not enqueue a job.
    response.raise_for_status()
    if response.status_code != 202:
        raise ValueError(f"Expected asynchronous registration receipt, got {response.status_code}")
    receipt = response.json()
    _identify(state, {"id": receipt.get("worker_id"), "url": receipt.get("url")}, payload["url"])
    if receipt.get("location") != f"/workers/{quote(state.worker_id, safe='')}":
        raise ValueError(f"Invalid router registration location: {receipt}")


async def _wait_registered(client, router_url, payload, state):
    while True:
        worker = await _status(client, router_url, payload["url"], state)
        if worker is not None:
            # Wait for the only submitted job to finish, even if it has already
            # inserted a registry entry. Removal must not race its remaining work.
            if not state.pending:
                rows = (
                    await _workers(client, router_url)
                    if worker.get("is_healthy") is True and worker.get("worker_type") == payload["worker_type"]
                    else []
                )
                if any(
                    row.get("id") == state.worker_id
                    and row.get("url") == payload["url"]
                    and row.get("worker_type") == payload["worker_type"]
                    and row.get("is_healthy") is True
                    for row in rows
                ):
                    return
                job = worker.get("job_status") or {}
                if job.get("status") == "failed":
                    # A failed job may still have inserted a worker. Only retry
                    # submission if that completed attempt left no registry entry.
                    state.submitted = any(
                        row.get("url") == payload["url"] for row in await _workers(client, router_url)
                    )
                    raise RegistrationFailed(f"Router registration failed for {payload['url']}: {job.get('message')}")
        await _sleep(1)


async def register_worker(client, router_url, payload, *, state):
    """Retry known failures; reconcile ambiguous submissions without duplicate jobs."""
    if state.retiring:
        raise RegistrationCleanupError(f"Registration cleanup is unresolved for {payload['url']}")
    try:
        async with asyncio.timeout(infra_timeouts.seconds(REGISTRATION_TIMEOUT_SECONDS)):
            for attempt in range(1, REGISTRATION_ATTEMPTS + 1):
                try:
                    if not state.submitted:
                        await _submit(client, router_url, payload, state)
                    await _wait_registered(client, router_url, payload, state)
                    logger.info("Router admission confirmed: worker=%s attempt=%s", payload["url"], attempt)
                    return
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.HTTPStatusError,
                    RegistrationFailed,
                ) as error:
                    if (
                        isinstance(error, httpx.HTTPStatusError)
                        and error.response.status_code < 500
                        and error.response.status_code != 429
                    ):
                        raise
                    logger.warning(
                        "Router admission retry: worker=%s attempt=%s error=%s", payload["url"], attempt, error
                    )
                    if attempt == REGISTRATION_ATTEMPTS:
                        raise
                    await _sleep(attempt)
    except TimeoutError as error:
        raise TimeoutError(f"Router admission deadline exceeded for {payload['url']}") from error


async def unregister_worker(client, router_url, worker_url, *, state):
    """Settle the owned AddWorker job, then delete and confirm registry removal.

    Never treat an empty registry as success while an accepted/ambiguous POST is
    outstanding. On a bounded cleanup failure, keep state for a later disposal
    attempt; callers must not forget the cell or reuse its URL in the meantime.
    """
    state.retiring = True
    try:
        async with asyncio.timeout(infra_timeouts.seconds(CLEANUP_TIMEOUT_SECONDS)):
            while state.pending:
                await _status(client, router_url, worker_url, state)
                if state.pending:
                    await _sleep(1)
            if state.worker_id is None:
                return  # Submission was explicitly rejected or never connected.
            rows = [row for row in await _workers(client, router_url) if row.get("url") == worker_url]
            if not rows:
                return  # AddWorker has settled, so it cannot register later.
            _identify(state, rows[0], worker_url)
            if not state.removing:
                state.removing = True
                try:
                    response = await infra_timeouts.request(
                        client,
                        "delete",
                        f"{router_url}/workers/{quote(state.worker_id, safe='')}",
                        timeout=REQUEST_TIMEOUT,
                    )
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    state.removing = False
                    raise
                if response.is_client_error:
                    state.removing = False
                response.raise_for_status()
            while any(row.get("url") == worker_url for row in await _workers(client, router_url)):
                await _sleep(1)
    except Exception as error:
        raise RegistrationCleanupError(f"Could not confirm router registration cleanup for {worker_url}") from error
