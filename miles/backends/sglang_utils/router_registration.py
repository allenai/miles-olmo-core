"""Confirm asynchronous SGLang router admission using its worker resource.

The >=0.3.2 router deduplicates worker IDs by URL, but duplicate submissions can
leave a failed AddWorker job attached to an already registered healthy worker.
The registry is authoritative in that case; a 202 receipt alone never is.
"""

import asyncio
from urllib.parse import quote

import httpx

from miles.utils import infra_timeouts, logger_utils

logger = logger_utils.setup_logger(__name__)

REGISTRATION_TIMEOUT_SECONDS = 180.0
REGISTRATION_ATTEMPTS = 3
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_sleep = asyncio.sleep


class RegistrationFailed(RuntimeError):
    pass


async def _wait_registered(client, router_url, receipt, *, worker_url, worker_type):
    worker_id = receipt["worker_id"]
    location = f"/workers/{quote(worker_id, safe='')}"
    if receipt.get("url") != worker_url or receipt.get("location") != location:
        raise ValueError(f"Router registration receipt does not identify {worker_url}: {receipt}")
    while True:
        response = await infra_timeouts.request(client, "get", router_url + location, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        worker = response.json()
        if worker.get("id") != worker_id or worker.get("url") != worker_url:
            raise ValueError(f"Router registration identity changed for {worker_url}: {worker}")
        # GET /workers/{id} also returns placeholders for jobs not yet admitted.
        # Confirm membership in the actual registry before accepting readiness.
        if worker.get("is_healthy") is True and worker.get("worker_type") == worker_type:
            response = await infra_timeouts.request(client, "get", router_url + "/workers", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            if any(
                row.get("id") == worker_id
                and row.get("url") == worker_url
                and row.get("worker_type") == worker_type
                and row.get("is_healthy") is True
                for row in response.json()["workers"]
            ):
                return
        job = worker.get("job_status") or {}
        if job.get("status") == "failed":
            raise RegistrationFailed(f"Router registration failed for {worker_url}: {job.get('message')}")
        if job.get("status") not in (None, "queued", "pending", "processing", "completed"):
            raise ValueError(f"Unknown router registration state for {worker_url}: {job}")
        await _sleep(1)


async def register_worker(client, router_url, payload):
    """Bound the whole admission, including transport retries and reconciliation.

    Only this adapter uses the router's URL-deduplicated submission contract. A lost
    POST response may be retried safely; readiness still requires registry evidence.
    """
    worker_url = payload["url"]
    try:
        async with asyncio.timeout(infra_timeouts.seconds(REGISTRATION_TIMEOUT_SECONDS)):
            for attempt in range(1, REGISTRATION_ATTEMPTS + 1):
                try:
                    response = await infra_timeouts.request(
                        client, "post", router_url + "/workers", json=payload, timeout=REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    if response.status_code != 202:
                        raise ValueError(f"Expected asynchronous registration receipt, got {response.status_code}")
                    await _wait_registered(
                        client, router_url, response.json(), worker_url=worker_url, worker_type=payload["worker_type"]
                    )
                    logger.info("Router admission confirmed: worker=%s attempt=%s", worker_url, attempt)
                    return
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError, RegistrationFailed) as error:
                    if (
                        isinstance(error, httpx.HTTPStatusError)
                        and error.response.status_code < 500
                        and error.response.status_code != 429
                    ):
                        raise
                    logger.warning("Router admission retry: worker=%s attempt=%s error=%s", worker_url, attempt, error)
                    if attempt == REGISTRATION_ATTEMPTS:
                        raise
                    await _sleep(attempt)
    except TimeoutError as error:
        raise TimeoutError(f"Router admission deadline exceeded for {worker_url}") from error
