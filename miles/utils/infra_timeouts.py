"""Opt-in infrastructure deadline scaling with periodic slow-operation warnings."""

import asyncio
import contextlib
import math
import os
import time
from urllib.parse import urlsplit

import httpx

from miles.utils import logging_utils

logger = logging_utils.logger


def seconds(value):
    factor = float(os.environ.get("MILES_INFRA_TIMEOUT_MULTIPLIER", "1"))
    if not math.isfinite(factor) or factor < 1:
        raise ValueError("MILES_INFRA_TIMEOUT_MULTIPLIER must be finite and >= 1")
    return None if value is None else value * factor


def http_timeout(value):
    value = value if isinstance(value, httpx.Timeout) else httpx.Timeout(value)
    return httpx.Timeout(**{key: seconds(limit) for key, limit in value.as_dict().items()})


@contextlib.asynccontextmanager
async def watch(operation, *, previous_timeout=None, deadline=None):
    threshold = min(30.0, previous_timeout) if previous_timeout and previous_timeout > 0 else 30.0
    started = time.monotonic()

    async def warn():
        await asyncio.sleep(threshold)
        while True:
            logger.warning(
                "Slow infrastructure operation: operation=%s elapsed_s=%.1f warning_s=%s deadline_s=%s",
                operation,
                time.monotonic() - started,
                threshold,
                deadline,
            )
            await asyncio.sleep(30.0)

    task = asyncio.create_task(warn())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        elapsed = time.monotonic() - started
        if elapsed >= threshold:
            logger.warning(
                "Slow infrastructure operation finished: operation=%s elapsed_s=%.1f deadline_s=%s",
                operation,
                elapsed,
                deadline,
            )


async def wait_for(awaitable, timeout, *, operation):
    deadline = seconds(timeout)
    async with watch(operation, previous_timeout=timeout, deadline=deadline):
        return await asyncio.wait_for(awaitable, timeout=deadline)


async def request(client, method, url, **kwargs):
    previous = kwargs.get("timeout", getattr(client, "timeout", None))
    previous = previous if isinstance(previous, httpx.Timeout) else httpx.Timeout(previous)
    # Preserve the client's existing defaults when the multiplier is unset.
    if "timeout" in kwargs or seconds(1) != 1:
        raw = kwargs.get("timeout", previous)
        kwargs["timeout"] = http_timeout(raw) if isinstance(raw, httpx.Timeout) else seconds(raw)
    # Log the endpoint only: URLs can contain API credentials in queries/userinfo.
    parsed = urlsplit(url)
    operation = f"HTTP {method.upper()} {parsed.hostname}:{parsed.port}{parsed.path}"
    async with watch(operation, previous_timeout=previous.read, deadline=seconds(previous.read)):
        return await getattr(client, method)(url, **kwargs)
