import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
from prometheus_client import parser

from miles.router import engine_metrics
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter

TEXT = """# HELP sglang:num_running_reqs Running requests
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{engine_type="unified",dp_rank="0"} 224
# HELP sglang:requests Completed requests
# TYPE sglang:requests counter
sglang:requests_total 12
# HELP latency Latency
# TYPE latency histogram
latency_bucket{le="1"} 2
latency_bucket{le="+Inf"} 3
latency_sum 4
latency_count 3
"""


def samples(body):
    return [sample for family in parser.text_string_to_metric_families(body) for sample in family.samples]


def config():
    return MilesRouterConfig(
        host="127.0.0.1",
        port=8000,
        max_connections=1,
        timeout=10,
        health_check_interval=1,
        health_check_failure_threshold=3,
    )


def test_aggregate_preserves_worker_identity_histograms_and_missing_scrapes():
    result = engine_metrics.render_engine_metrics(
        [
            ("http://first", TEXT),
            ("http://second", TEXT.replace("224", "256")),
            ("http://failed", None),
            ("http://malformed", "bad_metric not-a-number"),
        ]
    ).decode()
    observed = samples(result)
    running = [s for s in observed if s.name == "sglang:num_running_reqs"]
    assert {(s.labels["worker_addr"], s.value) for s in running} == {("http://first", 224), ("http://second", 256)}
    assert all(s.labels["dp_rank"] == "0" for s in running)
    assert len([s for s in observed if s.name == "latency_bucket"]) == 4
    up = {s.labels["worker_addr"]: s.value for s in observed if s.name == "miles_router_engine_metrics_up"}
    assert up == {"http://first": 1, "http://second": 1, "http://failed": 0, "http://malformed": 0}
    assert result.count("# TYPE latency histogram") == 1


def test_metrics_endpoint_uses_current_workers_without_changing_routing_load():
    async def exercise():
        router = MilesRouter(config())
        seen = []

        async def respond(request):
            seen.append(str(request.url))
            return httpx.Response(200, text=TEXT)

        await router.engine_metrics.client.aclose()
        router.engine_metrics.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        router.worker_request_counts.update({"http://active": 2, "http://quarantined": 0, "http://retired": 1})
        router.dead_workers.add("http://quarantined")
        router._retired.add("http://retired")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=router.app), base_url="http://router") as client:
                response = await client.get("/engine_metrics")
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
                assert set(seen) == {"http://active/metrics", "http://quarantined/metrics"}
                assert router.worker_request_counts == {"http://active": 2, "http://quarantined": 0, "http://retired": 1}
                router.worker_request_counts.clear()
                seen.clear()
                assert (await client.get("/engine_metrics")).status_code == 200
                assert seen == []
        finally:
            await router.close()
        assert router.engine_metrics.client.is_closed

    asyncio.run(exercise())


def test_worker_http_error_and_deadline_leave_explicit_gaps(monkeypatch):
    async def exercise():
        async def respond(request):
            if request.url.host == "slow":
                await asyncio.sleep(10)
            return httpx.Response(503)

        monkeypatch.setattr(engine_metrics, "SCRAPE_TIMEOUT_SECONDS", 0.02)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            results = await asyncio.gather(*(engine_metrics.fetch_engine_metrics(client, url) for url in ["http://slow", "http://failed"]))
        assert results == [("http://slow", None), ("http://failed", None)]

    asyncio.run(exercise())


def test_busy_generation_does_not_block_metrics_or_health():
    entered, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, body):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            entered.set()
            release.wait(10)
            self.reply(b"{}")

        def do_GET(self):
            self.reply(TEXT.encode() if self.path == "/metrics" else b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def exercise():
        router = MilesRouter(config())
        url = f"http://127.0.0.1:{server.server_port}"
        router.worker_request_counts[url] = 1
        generation = asyncio.create_task(router.client.post(url + "/generate", json={}))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            metrics, health = await asyncio.wait_for(asyncio.gather(router.get_engine_metrics(), router._check_worker_health(url)), timeout=2)
            assert health == (url, True)
            assert any(s.value == 224 for s in samples(metrics.body.decode()))
            assert not generation.done() and router.worker_request_counts[url] == 1
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
