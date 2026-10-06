"""Real HTTP saturation, quarantine, and shutdown checks in the pinned runtime."""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter


def config(**kwargs):
    aliases = {
        "miles_router_max_connections": "max_connections",
        "miles_router_timeout": "timeout",
        "rollout_health_check_timeout": "health_check_timeout",
        "rollout_health_check_interval": "health_check_interval",
        "miles_router_health_check_failure_threshold": "health_check_failure_threshold",
    }
    values = dict(
        host="127.0.0.1",
        port=8000,
        max_connections=1,
        timeout=1,
        health_check_interval=1,
        health_check_failure_threshold=3,
    )
    values.update({aliases[k]: v for k, v in kwargs.items() if k in aliases})
    return MilesRouterConfig(**values)


def test_busy_generation_cannot_starve_health_and_shutdown_joins_probe():
    entered, release = threading.Event(), threading.Event()
    health = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            entered.set()
            release.wait(15)
            self.reply()

        def do_GET(self):
            health.append(self.path)
            self.reply()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def exercise():
        router = MilesRouter(
            config(
                miles_router_max_connections=1,
                miles_router_timeout=10,
                rollout_num_gpus=1,
                rollout_num_gpus_per_engine=1,
                rollout_health_check_timeout=0.2,
                rollout_health_check_interval=0.01,
                miles_router_health_check_failure_threshold=2,
            )
        )
        url = f"http://127.0.0.1:{server.server_port}"
        router.worker_request_counts[url] = 0
        generation = asyncio.create_task(router.client.post(url + "/generate", json={}))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert await router._check_worker_health(url) == (url, True)
            assert health == ["/health"] and not generation.done()
            await router._start_background_health_check()
            task = router._health_task
            await asyncio.sleep(0.1)
            assert len(health) > 2 and not router.dead_workers
            release.set()
            await generation
            await router.close()
            assert task.done() and task.cancelled()
            assert router.client.is_closed and router.health_client.is_closed
        finally:
            release.set()
            await asyncio.gather(generation, return_exceptions=True)
            await router.close()

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_retired_worker_has_503_until_explicit_registration():
    async def exercise():
        router = MilesRouter(
            config(
                miles_router_max_connections=1,
                miles_router_timeout=1,
                rollout_num_gpus=1,
                rollout_num_gpus_per_engine=1,
            )
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=router.app), base_url="http://router"
            ) as client:
                assert (await client.post("/generate", json={})).status_code == 503
                assert (await client.post("/add_worker", json={"url": "http://engine"})).status_code == 200
                router._use_url()
                assert (await client.post("/remove_worker", params={"url": "http://engine"})).status_code == 200
                assert (await client.get("/list_workers")).json() == {"urls": []}
                assert (await client.post("/add_worker", json={"url": "http://engine"})).status_code == 409
                router._finish_url("http://engine")
                assert (await client.post("/generate", json={})).status_code == 503
                assert (await client.post("/add_worker", json={"url": "http://engine"})).status_code == 200
                assert router._use_url() == "http://engine"
                router._finish_url("http://engine")
                assert router.worker_request_counts == {"http://engine": 0}
        finally:
            await router.close()

    asyncio.run(exercise())


def test_engine_transport_failure_is_retryable_and_releases_counter():
    async def exercise():
        router = MilesRouter(
            config(
                miles_router_max_connections=1,
                miles_router_timeout=1,
                rollout_num_gpus=1,
                rollout_num_gpus_per_engine=1,
            )
        )

        async def disconnected(request):
            raise httpx.RemoteProtocolError("engine disappeared", request=request)

        await router.client.aclose()
        router.client = httpx.AsyncClient(transport=httpx.MockTransport(disconnected))
        router.worker_request_counts["http://engine"] = 0
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=router.app), base_url="http://router"
            ) as client:
                response = await client.post("/generate", json={})
                assert response.status_code == 503 and response.headers["Retry-After"] == "1"
                assert router.worker_request_counts == {"http://engine": 0}
        finally:
            await router.close()

    asyncio.run(exercise())


def test_late_probe_cannot_quarantine_replacement_and_failure_stays_quarantined():
    async def exercise():
        router = MilesRouter(
            config(
                miles_router_max_connections=1,
                miles_router_timeout=1,
                rollout_health_check_interval=0.001,
                miles_router_health_check_failure_threshold=1,
            )
        )
        entered, release, applied = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = 0
        second_release = asyncio.Event()

        async def probe(url):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            else:
                applied.set()
                await second_release.wait()
            return url, False

        router._check_worker_health = probe
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=router.app), base_url="http://router"
            ) as client:
                await client.post("/add_worker", json={"url": "http://engine"})
                await router._start_background_health_check()
                await asyncio.wait_for(entered.wait(), 1)
                await client.post("/remove_worker", json={"url": "http://engine"})
                await client.post("/add_worker", json={"url": "http://engine"})
                release.set()
                # A second probe starts only after the stale first result is processed.
                await asyncio.wait_for(applied.wait(), 1)
                assert not router.dead_workers
                second_release.set()

                async def wait_for_quarantine():
                    while not router.dead_workers:
                        await asyncio.sleep(0)

                await asyncio.wait_for(wait_for_quarantine(), 1)
                assert router.dead_workers == {"http://engine"}
                assert (await client.get("/list_workers")).json() == {"urls": ["http://engine"]}
                probes = calls
                await asyncio.sleep(0.01)
                assert calls == probes  # Quarantine requires registration, never automatic readmission.
                await client.post("/add_worker", json={"url": "http://engine"})
                assert router.dead_workers == {"http://engine"}
                router.worker_request_counts["http://engine"] = 1
                await client.post("/add_worker?weights_ready=true", json={"url": "http://engine"})
                assert not router.dead_workers and router.worker_failure_counts["http://engine"] == 0
                assert router.worker_request_counts["http://engine"] == 1
        finally:
            release.set()
            await router.close()

    asyncio.run(exercise())


def test_cancelled_request_body_releases_counter():
    async def exercise():
        router = MilesRouter(config(miles_router_max_connections=1, miles_router_timeout=1))
        router.worker_request_counts["http://engine"] = 0

        class CancelledRequest:
            async def body(self):
                raise asyncio.CancelledError

        try:
            try:
                await router.do_proxy(CancelledRequest(), "generate")
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("request cancellation was swallowed")
            assert router.worker_request_counts["http://engine"] == 0
        finally:
            await router.close()

    asyncio.run(exercise())
