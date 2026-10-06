import asyncio
import dataclasses
import logging
import time
from urllib.parse import quote

import httpx
import sglang_router
from packaging.version import parse

from miles.utils import infra_timeouts
from miles.utils.http_utils import GeneralHttpClientProvider

logger = logging.getLogger(__name__)

_retry_sleep = asyncio.sleep

ROUTER_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


def use_legacy_router_api(args) -> bool:
    return parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router


@dataclasses.dataclass(frozen=True)
class SGLangRouterApiClient:
    router_url: str

    async def add_worker(
        self,
        worker_url: str,
        worker_type: str,
        use_legacy_api: bool,
        bootstrap_port: int | None = None,
        weights_ready: bool = False,
    ):
        if use_legacy_api:
            assert worker_type == "regular", "pd disaggregation is not supported in old router or miles router."
            if weights_ready:
                return await self._confirm_weights_ready(worker_url)
            response = await infra_timeouts.request(
                GeneralHttpClientProvider.client(),
                "post",
                f"{self.router_url}/add_worker?url={worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        else:
            payload = {
                "url": worker_url,
                "worker_type": worker_type,
            }
            if worker_type == "prefill":
                payload["bootstrap_port"] = bootstrap_port
            response = await infra_timeouts.request(
                GeneralHttpClientProvider.client(),
                "post",
                f"{self.router_url}/workers",
                json=payload,
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        response.raise_for_status()

    async def _confirm_weights_ready(self, worker_url: str):
        # Native MILES readiness confirmation is idempotent: repeating it keeps
        # the worker's active-request count and only renews its health epoch.
        # A lost response after a successful registration is safe to retry.
        url = f"{self.router_url}/add_worker?url={worker_url}&weights_ready=true"
        for attempt in range(1, 4):
            started = time.monotonic()
            logger.info("Router readiness start: router=%s worker=%s attempt=%s", self.router_url, worker_url, attempt)
            try:
                response = await infra_timeouts.request(
                    GeneralHttpClientProvider.client(), "post", url, timeout=ROUTER_REQUEST_TIMEOUT
                )
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                logger.warning(
                    "Router readiness failed: router=%s worker=%s attempt=%s seconds=%.3f error=%s",
                    self.router_url,
                    worker_url,
                    attempt,
                    time.monotonic() - started,
                    type(error).__name__,
                )
                if attempt == 3:
                    raise
                await _retry_sleep(attempt)
            else:
                logger.info(
                    "Router readiness complete: router=%s worker=%s attempt=%s seconds=%.3f",
                    self.router_url,
                    worker_url,
                    attempt,
                    time.monotonic() - started,
                )
                return

    async def remove_worker(self, worker_url: str, use_legacy_api: bool):
        response = None
        if use_legacy_api:
            response = await infra_timeouts.request(
                GeneralHttpClientProvider.client(),
                "post",
                f"{self.router_url}/remove_worker?url={worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        elif parse(sglang_router.__version__) < parse("0.3.0"):
            quoted_worker_url = quote(worker_url, safe="")
            response = await infra_timeouts.request(
                GeneralHttpClientProvider.client(),
                "delete",
                f"{self.router_url}/workers/{quoted_worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        else:
            try:
                all_workers = (
                    await infra_timeouts.request(
                        GeneralHttpClientProvider.client(),
                        "get",
                        f"{self.router_url}/workers",
                        timeout=ROUTER_REQUEST_TIMEOUT,
                    )
                ).json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        worker_id = worker["id"]
                        response = await infra_timeouts.request(
                            GeneralHttpClientProvider.client(),
                            "delete",
                            f"{self.router_url}/workers/{worker_id}",
                            timeout=ROUTER_REQUEST_TIMEOUT,
                        )
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in router during shutdown.")
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

        if response is not None:
            response.raise_for_status()
