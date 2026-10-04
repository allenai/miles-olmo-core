import asyncio
import json
import logging
import sys
from contextlib import suppress

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.router.config import MilesRouterConfig
from miles.utils import infra_timeouts
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)


def run_router(config: MilesRouterConfig):
    """
    Run the Miles router with the specified configuration.
    """
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("miles_router")
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-router")

    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(config, verbose=False)

    # HTTPX reuses idle connections for five seconds. Uvicorn's matching default
    # can close a selected connection before a delayed client writes its request.
    # Leave margin on the server; this does not limit active generation requests.
    uvicorn.run(miles_router.app, host=config.host, port=config.port, log_level="info", timeout_keep_alive=60)


class MilesRouter:
    def __init__(self, config: MilesRouterConfig, verbose=False):
        """Initialize the miles-router with SGLang router address"""
        self.config = config
        self.verbose = verbose

        self.app = FastAPI()
        self.app.router.on_startup.append(self._start_background_health_check)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()
        self._retired: set[str] = set()
        self._worker_epochs: dict[str, int] = {}
        self._health_task: asyncio.Task | None = None

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=config.max_connections),
            timeout=infra_timeouts.http_timeout(config.timeout),
        )

        # Generation may occupy every data connection for minutes. Health must
        # have its own transport, otherwise pool starvation looks like engine death.
        self.health_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=config.health_check_max_connections),
            timeout=infra_timeouts.http_timeout(config.health_check_timeout),
        )
        self.app.router.on_shutdown.append(self.close)
        self._setup_routes()

    def _setup_routes(self):
        """Setup all the HTTP routes except catch-all proxy"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/list_workers")(self.list_workers)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            async with infra_timeouts.watch(
                f"router health {url}",
                previous_timeout=self.config.health_check_timeout,
                deadline=infra_timeouts.seconds(self.config.health_check_timeout),
            ):
                response = await self.health_client.get(f"{url}/health")
            if response.status_code == 200:
                return url, True
            logger.debug(f"[miles-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except httpx.HTTPError as e:
            logger.warning("[miles-router] Worker %s health check failed: %s", url, type(e).__name__)
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.config.health_check_interval
        threshold = self.config.health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                epochs = {
                    u: self._worker_epochs.get(u, 0)
                    for u in self.worker_request_counts
                    if u not in self.dead_workers and u not in self._retired
                }
                if not epochs:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in epochs))

                for url, is_healthy in results:
                    if (
                        url not in self.worker_request_counts
                        or url in self._retired
                        or self._worker_epochs.get(url, 0) != epochs[url]
                    ):
                        continue
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[miles-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[miles-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        result = await self.do_proxy(request, path)
        return self.build_proxy_response(result)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Core proxy logic. Returns dict with request_body, response_body, status_code, headers."""
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        try:
            if body is None:
                body = await request.body()
            if headers is None:
                headers = dict(request.headers)
            if body is not None:
                headers = {
                    k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")
                }

            response = await self.client.request(request.method, url, content=body, headers=headers)
            content = await response.aread()
            return {
                "request_body": body,
                "response_body": content,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
        except httpx.RequestError as error:
            logger.warning(
                "[miles-router] Proxy transport failed: %s worker=%s path=%s active=%s",
                type(error).__name__,
                worker_url,
                path,
                self.worker_request_counts.get(worker_url),
                exc_info=True,
            )
            raise HTTPException(503, "Rollout worker unavailable", headers={"Retry-After": "1"}) from error
        finally:
            self._finish_url(worker_url)

    def build_proxy_response(self, result: dict) -> Response:
        """Build HTTP response from proxy result."""
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "content-encoding", "server", "date")
        }
        # HTTPX has already decoded Content-Encoding. Preserve the response bytes
        # without parsing and serializing large routing-replay payloads on the
        # event loop shared with weight-publication control requests.
        return Response(content=content, status_code=status_code, headers=headers)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        worker_url = await self._parse_worker_url(request)
        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        if worker_url in self._retired and self.worker_request_counts.get(worker_url, 0):
            raise HTTPException(409, "retired worker still has active requests; register a new endpoint")
        # Duplicate discovery cannot readmit a quarantined engine. Only a completed
        # weight update explicitly confirms readiness, including for existing cells.
        weights_ready = request.query_params.get("weights_ready") == "true"
        if worker_url not in self.worker_request_counts or worker_url in self._retired or weights_ready:
            self._worker_epochs[worker_url] = self._worker_epochs.get(worker_url, 0) + 1
            self._retired.discard(worker_url)
            self.dead_workers.discard(worker_url)
            self.worker_failure_counts[worker_url] = 0

        # Add if new, keep a simple request count per worker
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_failure_counts[worker_url] = 0
            self.dead_workers.discard(worker_url)
            if self.verbose:
                print(f"[miles-router] Added new worker: {worker_url}")

        logger.info(
            "Router worker registered: worker=%s weights_ready=%s active_requests=%s fleet_requests=%s epoch=%s",
            worker_url,
            weights_ready,
            self.worker_request_counts[worker_url],
            sum(self.worker_request_counts.values()),
            self._worker_epochs.get(worker_url),
        )
        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def remove_worker(self, request: Request):
        """Remove a worker from the router, using the same URL conventions as add_worker."""
        worker_url = await self._parse_worker_url(request)
        if worker_url is None:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        self._worker_epochs[worker_url] = self._worker_epochs.get(worker_url, 0) + 1
        if self.worker_request_counts.get(worker_url, 0):
            self._retired.add(worker_url)
            self.dead_workers.add(worker_url)
        else:
            self.worker_request_counts.pop(worker_url, None)
            self.worker_failure_counts.pop(worker_url, None)
            self.dead_workers.discard(worker_url)
            self._retired.discard(worker_url)
        logger.info(f"[miles-router] Removed worker: {worker_url}")

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def _parse_worker_url(self, request: Request) -> str | None:
        if worker_url := request.query_params.get("url") or request.query_params.get("worker_url"):
            return worker_url

        body = await request.body()
        payload = json.loads(body) if body else {}
        return payload.get("url") or payload.get("worker_url")

    async def list_workers(self, request: Request):
        """Include quarantined endpoints for aborts, but exclude retired ones."""
        return {"urls": [u for u in self.worker_request_counts if u not in self._retired]}

    def _use_url(self):
        """Select the healthy, registered worker with the fewest active requests."""
        workers = [u for u in self.worker_request_counts if u not in self.dead_workers and u not in self._retired]
        if not workers:
            raise HTTPException(503, "No healthy rollout workers available", headers={"Retry-After": "1"})
        url = min(workers, key=self.worker_request_counts.get)
        self.worker_request_counts[url] += 1
        return url

    async def close(self):
        """Join health monitoring before closing its transport and the data pool."""
        if self._health_task is not None:
            self._health_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._health_task
            self._health_task = None
        await self.health_client.aclose()
        await self.client.aclose()

    def _finish_url(self, url):
        """Mark the request to the given URL as finished"""
        if (count := self.worker_request_counts.get(url)) is None or count == 0:
            logger.info(f"[miles-router] Request to {url} finished after the worker was deregistered; ignoring")
            return
        self.worker_request_counts[url] = count - 1
        if count == 1 and url in self._retired:
            self.worker_request_counts.pop(url)
            self.worker_failure_counts.pop(url, None)
            self.dead_workers.discard(url)
            self._retired.discard(url)


if __name__ == "__main__":
    run_router(parse_config_argv(MilesRouterConfig, sys.argv[1:]))
