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
    worker_id: str | None = None
    _registration_pending: bool = False
    _retired: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def _request(self, method, path, **kwargs):
        response = await GeneralHttpClientProvider.client().request(
            method, f"{self.router_url}{path}", timeout=REQUEST_TIMEOUT, **kwargs
        )
        response.raise_for_status()
        return response

    async def _find(self):
        response = await self._request("GET", "/workers")
        workers = response.json()["workers"]
        matches = [worker for worker in workers if worker["url"] == self.worker_url]
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate router workers for {self.worker_url}")
        if not matches:
            return None
        worker = matches[0]
        if not isinstance(worker.get("id"), str) or not worker["id"]:
            raise RuntimeError(f"Router returned no worker identity for {self.worker_url}")
        if self.worker_id is not None and worker["id"] != self.worker_id:
            raise RuntimeError(f"Router worker identity changed for {self.worker_url}; refusing endpoint reuse")
        return worker

    async def ensure_registered(self):
        """Call only after successful publication (or for a frozen policy)."""
        async with self._lock, asyncio.timeout(MEMBERSHIP_TIMEOUT_SECONDS):
            if self._retired:
                raise RuntimeError(f"Retired router membership cannot be registered: {self.worker_url}")
            started = time.monotonic()
            worker = await self._find()
            if worker is not None and self.worker_id is None and not self._registration_pending:
                raise RuntimeError(f"Unowned router worker already exists for {self.worker_url}")
            if worker is None and not self._registration_pending:
                # An empty registry after router restart is reconciled only here,
                # at a successful publication boundary, never by a health probe.
                self.worker_id = None
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
                worker = await self._find()
                if worker is not None and worker.get("is_healthy") is True:
                    self.worker_id = worker["id"]
                    self._registration_pending = False
                    logger.info(
                        "Router membership ready: worker=%s id=%s seconds=%.3f",
                        self.worker_url,
                        self.worker_id,
                        time.monotonic() - started,
                    )
                    return
                await asyncio.sleep(POLL_SECONDS)

    async def remove(self, *, retire=False):
        """Confirm removal before endpoint replacement; keep unknown outcomes closed."""
        async with self._lock, asyncio.timeout(MEMBERSHIP_TIMEOUT_SECONDS):
            self._retired |= retire
            # A cancelled/lost registration may still be queued. Resolve its
            # identity before deletion; an initially empty list is not proof
            # that no future worker can appear.
            worker = await self._find()
            while worker is None and self._registration_pending:
                await asyncio.sleep(POLL_SECONDS)
                worker = await self._find()
            if worker is None:
                self.worker_id = None
                return
            if self.worker_id is None and not self._registration_pending:
                raise RuntimeError(f"Cannot remove an unowned router worker for {self.worker_url}")
            self.worker_id = worker["id"]
            try:
                await self._request("DELETE", f"/workers/{self.worker_id}")
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 404:
                    raise
                # Removal is idempotent, but still confirm registry absence.
            except (httpx.TimeoutException, httpx.NetworkError):
                logger.warning("Router removal acknowledgement lost: worker=%s", self.worker_url)
            while await self._find() is not None:
                await asyncio.sleep(POLL_SECONDS)
            logger.info("Router membership removed: worker=%s id=%s", self.worker_url, self.worker_id)
            self.worker_id = None
            self._registration_pending = False


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
