"""Abort serving requests the producer has stopped waiting for."""

import asyncio
import contextlib
import importlib.metadata
import os
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
# Discovery shares the rollout client, which has no timeout; one stalled router
# call must not hold the only drain task and strand every later abort.
_DISCOVERY_TIMEOUT_SECONDS = 10.0
# Teardown waits this long for queued aborts, then cancels the rest.
_CLOSE_TIMEOUT_SECONDS = 30.0

# olmo-sglang's early-abort plugin (allenai/olmo-sglang#35), as SGLang discovers it:
# an entry point in this group, optionally filtered by SGLANG_PLUGINS, and turned off
# by a non-positive TTL.
EARLY_ABORT_PLUGIN = "olmo_sglang_early_abort"
_PLUGIN_GROUP = "sglang.srt.plugins"
_EARLY_ABORT_TTL_ENV = "OLMO_SGLANG_EARLY_ABORT_TTL_SECONDS"
_EARLY_ABORT_DEFAULT_TTL = "1800"
# Set to 1 to run refresh against engines without the plugin (warning only).
ALLOW_WITHOUT_EARLY_ABORT_ENV = "MILES_ALLOW_REFRESH_WITHOUT_EARLY_ABORT"


def early_abort_problem(env=None) -> str | None:
    """Return why SGLang servers started from this environment would drop an abort
    that arrives before its request, or None when the early-abort plugin is active.

    Engines are started from the producer's image and environment, so this mirrors
    SGLang's own plugin discovery (``sglang.srt.plugins.load_plugins_by_group``).
    """
    env = os.environ if env is None else env
    names = {entry.name for entry in importlib.metadata.entry_points(group=_PLUGIN_GROUP)}
    if EARLY_ABORT_PLUGIN not in names:
        return f"the {_PLUGIN_GROUP} entry point {EARLY_ABORT_PLUGIN} (allenai/olmo-sglang#35) is not installed"
    allowed = {name.strip() for name in env.get("SGLANG_PLUGINS", "").split(",") if name.strip()}
    if allowed and EARLY_ABORT_PLUGIN not in allowed:
        return f"SGLANG_PLUGINS={env['SGLANG_PLUGINS']!r} does not load {EARLY_ABORT_PLUGIN}"
    ttl = env.get(_EARLY_ABORT_TTL_ENV, _EARLY_ABORT_DEFAULT_TTL)
    try:
        enabled = float(ttl) > 0
    except ValueError:
        return f"{_EARLY_ABORT_TTL_ENV}={ttl!r} is not a number"
    if not enabled:
        return f"{_EARLY_ABORT_TTL_ENV}={ttl} turns the plugin off"
    return None


def check_early_abort(env=None) -> bool:
    """Refuse a refresh deployment whose engines cannot abort a request whose abort
    arrived first, unless ``MILES_ALLOW_REFRESH_WITHOUT_EARLY_ABORT=1``.

    Returns True when the plugin is active, False when allowed to run without it.
    """
    env = os.environ if env is None else env
    problem = early_abort_problem(env)
    if problem is None:
        return True
    message = (
        "Core refresh aborts the generations it abandons, but SGLang engines started here cannot abort a request "
        f"whose abort arrives before the request: {problem}. Such a request decodes to its token limit for nobody. "
        f"Install olmo-sglang with its early-abort plugin, or set {ALLOW_WITHOUT_EARLY_ABORT_ENV}=1 to run without it."
    )
    if env.get(ALLOW_WITHOUT_EARLY_ABORT_ENV) == "1":
        logger.warning(message)
        return False
    raise RuntimeError(message)


class RequestAborter:
    """Send SGLang a per-request abort for generations the producer abandoned.

    A request deadline or a failed sibling cancels the producer's await. ``post``
    then cancels a distributed HTTP worker's call, which closes its connection to
    the router, but the router's own connection to the engine may stay open, so
    SGLang can decode the response to the end for nobody. This abort covers that.
    The router does not route aborts by request id and the producer does not know
    which engine received the request, so every engine gets the abort, once.
    The abort can reach the engine before the router delivers the generation.
    Pinned SGLang forgets an abort for a request id it has not registered, so
    olmo-sglang's early-abort plugin keeps such ids for a time to live and aborts
    the request when it arrives (allenai/olmo-sglang#35). ``check_early_abort``
    refuses refresh deployments without the plugin unless explicitly allowed.

    Posts carry the engines' API key when one is configured. Each engine gets at
    most its share of the concurrent posts, so one stalled engine cannot hold the
    slots that aborts to healthy engines need.

    ``abort`` is synchronous and never raises, so cancellation and timeout paths
    can call it before re-raising. Posts run in one background task per batch.
    """

    def __init__(self, get_urls: Callable[[], Awaitable[list[str]]], api_key: str | None = None):
        self._get_urls = get_urls
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._urls: list[str] = []
        self._urls_fetched = float("-inf")
        self._pending: list[str] = []
        self._task: asyncio.Task | None = None
        self._closed = False
        self.requested = 0
        self.failed_posts = 0

    def abort(self, request_id: str) -> None:
        self.requested += 1
        if self._closed:
            # Teardown has started; closing the request's connection still aborts it.
            return
        self._pending.append(request_id)
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._drain())

    async def wait_idle(self) -> None:
        """Wait for queued aborts to be sent; used by tests and orderly teardown."""
        while self._task is not None and not self._task.done():
            await asyncio.shield(self._task)

    async def close(self, timeout: float = _CLOSE_TIMEOUT_SECONDS) -> None:
        """Stop taking aborts; send queued ones within ``timeout``, then cancel the rest."""
        self._closed = True
        task = self._task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except TimeoutError:
            logger.warning(
                "Abort sender did not finish within %gs; cancelling %d queued abort(s)", timeout, len(self._pending)
            )
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _drain(self) -> None:
        async with httpx.AsyncClient(timeout=_ABORT_TIMEOUT, limits=_ABORT_LIMITS, headers=self._headers) as client:
            while self._pending:
                request_ids, self._pending = self._pending, []
                urls = await self._worker_urls()
                if not urls:
                    logger.warning("Could not abort %d abandoned generation request(s): no engines", len(request_ids))
                    continue
                # Keep at most one connection pool's worth of posts in flight. httpcore
                # rescans its whole wait queue on every connection hand-off, so queueing
                # every (engine, rid) post at once is quadratic and blocks the event loop:
                # 575 groups x 8 rids x 23 engines held the producer loop for the whole
                # 2,700 s final drain (hardening-01).
                # A post waits for its engine's share first, so a stalled engine holds
                # at most that share of the global slots and the rest keep flowing.
                slots = asyncio.Semaphore(_ABORT_LIMITS.max_connections)
                share = max(1, _ABORT_LIMITS.max_connections // len(urls))
                engine_slots = {url: asyncio.Semaphore(share) for url in urls}

                async def bounded(url, request_id, slots=slots, engine_slots=engine_slots):
                    async with engine_slots[url], slots:
                        return await self._post(client, url, request_id)

                results = await asyncio.gather(
                    *(bounded(url, request_id) for url in urls for request_id in request_ids)
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
                self._urls = list(await asyncio.wait_for(self._get_urls(), _DISCOVERY_TIMEOUT_SECONDS))
                self._urls_fetched = time.monotonic()
            except Exception as error:
                logger.warning("Engine discovery for request aborts failed: %r", error)
        return self._urls

    @staticmethod
    async def _post(client: httpx.AsyncClient, url: str, request_id: str) -> bool:
        try:
            response = await client.post(f"{url}/abort_request", json={"rid": request_id})
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False
