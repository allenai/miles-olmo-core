"""Controller-owned membership for Core's Rust router.

Registration is asynchronous in router 0.3.x. HTTP acceptance is not readiness;
only a visible, healthy worker may serve confirmed weights. Generation traffic
never passes through this control-plane object.
"""

import asyncio
import time
from dataclasses import dataclass, field

import httpx

from miles.utils import logger_utils
from miles.utils.http_utils import GeneralHttpClientProvider

logger = logger_utils.setup_logger(__name__)

MEMBERSHIP_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.1
REQUEST_TIMEOUT = httpx.Timeout(10.0)


def controller_manages_router(args) -> bool:
    return getattr(args, "train_backend", None) == "olmo_core" and not getattr(args, "use_miles_router", False)


@dataclass
class RustRouterMembership:
    router_url: str
    worker_url: str
    worker_type: str = "regular"
    bootstrap_port: int | None = None
    # POST reserves the unsuffixed parent identity; DP rank IDs are distinct.
    worker_id: str | None = None
    dp_size: int | None = None
    worker_ids: dict[str, str] = field(default_factory=dict, init=False)
    _expected_urls: frozenset[str] = field(init=False, repr=False)
    _registration_pending: bool = False
    _retired: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        if self.dp_size is not None and (type(self.dp_size) is not int or self.dp_size < 1):
            raise ValueError("Router DP size must be a positive integer")
        self._expected_urls = frozenset(
            {self.worker_url}
            if self.dp_size is None
            else {f"{self.worker_url}@{rank}" for rank in range(self.dp_size)}
        )

    async def _request(self, method, path, **kwargs):
        response = await GeneralHttpClientProvider.client().request(
            method, f"{self.router_url}{path}", timeout=REQUEST_TIMEOUT, **kwargs
        )
        response.raise_for_status()
        return response

    async def _find(self):
        response = await self._request("GET", "/workers")
        workers = response.json()["workers"]
        matches = {}
        for worker in workers:
            url = worker["url"]
            if url != self.worker_url and not url.startswith(f"{self.worker_url}@"):
                continue
            if url not in self._expected_urls:
                raise RuntimeError(f"Unexpected router worker topology for {self.worker_url}: {url}")
            if url in matches:
                raise RuntimeError(f"Duplicate router workers for {url}")
            identity = worker.get("id")
            if not isinstance(identity, str) or not identity:
                raise RuntimeError(f"Router returned no worker identity for {url}")
            expected_id = self.worker_ids.get(url)
            if self.dp_size is None:
                expected_id = self.worker_id or expected_id
            if expected_id is not None and identity != expected_id:
                raise RuntimeError(f"Router worker identity changed for {url}; refusing endpoint reuse")
            if expected_id is None and not self._registration_pending:
                raise RuntimeError(f"Unowned router worker already exists for {url}; cannot remove unowned workers")
            matches[url] = worker
        if len({worker["id"] for worker in matches.values()}) != len(matches):
            raise RuntimeError(f"Duplicate router worker identities for {self.worker_url}")
        self.worker_ids.update({url: worker["id"] for url, worker in matches.items()})
        return matches

    async def ensure_registered(self):
        """Call only after successful publication (or for a frozen policy)."""
        async with self._lock, asyncio.timeout(MEMBERSHIP_TIMEOUT_SECONDS):
            if self._retired:
                raise RuntimeError(f"Retired router membership cannot be registered: {self.worker_url}")
            started = time.monotonic()
            workers = await self._find()
            if not workers and not self._registration_pending:
                # An empty registry after router restart is reconciled only here,
                # at a successful publication boundary, never by a health probe.
                self.worker_id = None
                self.worker_ids.clear()
                self._registration_pending = True
                payload = {"url": self.worker_url, "worker_type": self.worker_type}
                if self.worker_type == "prefill":
                    payload["bootstrap_port"] = self.bootstrap_port
                try:
                    response = await self._request("POST", "/workers", json=payload)
                except httpx.HTTPStatusError as error:
                    # These rejections cannot enqueue a registration. Server
                    # errors remain uncertain and must be reconciled.
                    if error.response.status_code in (400, 401, 403, 404, 422):
                        self._registration_pending = False
                    raise
                except (httpx.TimeoutException, httpx.NetworkError):
                    # The POST may have succeeded. Observe it; never enqueue a
                    # second registration merely because its response was lost.
                    logger.warning("Router registration acknowledgement lost: worker=%s", self.worker_url)
                else:
                    self.worker_id = response.json()["worker_id"]
            while True:
                workers = await self._find()
                if workers.keys() == self._expected_urls and all(
                    worker.get("is_healthy") is True for worker in workers.values()
                ):
                    if self.dp_size is None:
                        self.worker_id = workers[self.worker_url]["id"]
                    self._require_parent_identity()
                    self._registration_pending = False
                    logger.info(
                        "Router membership ready: worker=%s id=%s seconds=%.3f",
                        self.worker_url,
                        self.worker_id,
                        time.monotonic() - started,
                    )
                    return
                await asyncio.sleep(POLL_SECONDS)

    def _require_parent_identity(self):
        if self.worker_id is None:
            # /workers omits the DP parent reservation. Rank IDs cannot replace
            # it: DELETE on a rank searches for engine@rank@*. Never repeat POST
            # to recover the ID, since that can enqueue duplicate registrations.
            raise RuntimeError(
                f"Router registration identity unknown for {self.worker_url}; "
                "DP registration acknowledgement was lost; router restart is required"
            )

    async def remove(self, *, retire=False):
        """Confirm every owned rank is gone before allowing endpoint replacement."""
        async with self._lock, asyncio.timeout(MEMBERSHIP_TIMEOUT_SECONDS):
            self._retired |= retire
            workers = await self._find()
            # A cancelled POST can still be registering ranks. Observe the full
            # set before deleting, so late siblings cannot survive disposal.
            while self._registration_pending and self.worker_ids.keys() != self._expected_urls:
                await asyncio.sleep(POLL_SECONDS)
                workers = await self._find()
            if not workers:
                self.worker_id = None
                self.worker_ids.clear()
                self._registration_pending = False
                return
            if self.dp_size is None:
                self.worker_id = workers[self.worker_url]["id"]
            self._require_parent_identity()
            self._registration_pending = False
            try:
                # With DP awareness the parent ID resolves to the base URL;
                # the pinned router removes all base@rank workers together.
                await self._request("DELETE", f"/workers/{self.worker_id}")
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 404:
                    raise
            except (httpx.TimeoutException, httpx.NetworkError):
                logger.warning("Router removal acknowledgement lost: worker=%s", self.worker_url)
            while await self._find():
                await asyncio.sleep(POLL_SECONDS)
            logger.info("Router membership removed: worker=%s id=%s", self.worker_url, self.worker_id)
            self.worker_id = None
            self.worker_ids.clear()


async def engine_healthy(worker_url: str, *, api_key: str | None, timeout: float) -> bool:
    """Cheap liveness probe, independent of queued generation or router data sockets."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        response = await GeneralHttpClientProvider.client().get(
            f"{worker_url}/health", headers=headers, timeout=timeout
        )
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False
