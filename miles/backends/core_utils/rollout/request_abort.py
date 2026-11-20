"""Abort serving requests the producer has stopped waiting for."""

import asyncio
import time
from collections.abc import Awaitable, Callable

import httpx

from miles.utils import logger_utils

logger = logger_utils.setup_logger(__name__)

# Engines change only on recovery; a stale list misses at most a few aborts.
_URL_CACHE_SECONDS = 60.0
# Pool waits are bounded by the posts ahead of them, not by a deadline.
_ABORT_TIMEOUT = httpx.Timeout(10.0, pool=None)
_ABORT_LIMITS = httpx.Limits(max_connections=32, max_keepalive_connections=32)


class RequestAborter:
    """Send SGLang a per-request abort for generations the producer abandoned.

    A request deadline or a failed sibling cancels the producer's await, but the
    cancellation does not reach the engine: a distributed HTTP worker keeps the
    connection open, so SGLang decodes the response to the end for nobody. The
    router does not route aborts by request id and the producer does not know
    which engine received the request, so every engine gets the abort. SGLang
    ignores an abort for a request id it does not hold.

    ``abort`` is synchronous and never raises, so cancellation and timeout paths
    can call it before re-raising. Posts run in one background task per batch.
    """

    def __init__(self, get_urls: Callable[[], Awaitable[list[str]]]):
        self._get_urls = get_urls
        self._urls: list[str] = []
        self._urls_fetched = float("-inf")
        self._pending: list[str] = []
        self._task: asyncio.Task | None = None
        self.requested = 0
        self.failed_posts = 0

    def abort(self, request_id: str) -> None:
        self.requested += 1
        self._pending.append(request_id)
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())

    async def wait_idle(self) -> None:
        """Wait for queued aborts to be sent; used by tests and orderly teardown."""
        while self._task is not None and not self._task.done():
            await asyncio.shield(self._task)

    async def _drain(self) -> None:
        async with httpx.AsyncClient(timeout=_ABORT_TIMEOUT, limits=_ABORT_LIMITS) as client:
            while self._pending:
                request_ids, self._pending = self._pending, []
                urls = await self._worker_urls()
                if not urls:
                    logger.warning("Could not abort %d abandoned generation request(s): no engines", len(request_ids))
                    continue
                results = await asyncio.gather(
                    *(self._post(client, url, request_id) for url in urls for request_id in request_ids)
                )
                failed = results.count(False)
                if failed:
                    self.failed_posts += failed
                    logger.warning(
                        "Abort of %d abandoned generation request(s): %d of %d engine posts failed",
                        len(request_ids),
                        failed,
                        len(results),
                    )

    async def _worker_urls(self) -> list[str]:
        if time.monotonic() - self._urls_fetched >= _URL_CACHE_SECONDS or not self._urls:
            try:
                self._urls = list(await self._get_urls())
                self._urls_fetched = time.monotonic()
            except Exception as error:
                logger.warning("Engine discovery for request aborts failed: %s", error)
        return self._urls

    @staticmethod
    async def _post(client: httpx.AsyncClient, url: str, request_id: str) -> bool:
        try:
            response = await client.post(f"{url}/abort_request", json={"rid": request_id})
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False
