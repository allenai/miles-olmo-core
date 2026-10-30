"""Exercise real Ray serialization, including the 503 seen in three-node rollout."""

import asyncio
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
import ray

from miles.backends.core_utils.rollout.rollout_errors import recovery_category
from miles.utils import http_utils
from miles.utils.http_post_errors import HTTPPostFailure


@pytest.mark.parametrize("error_type, category", [(httpx.ReadError, "transport"), (httpx.ReadTimeout, "timeout")])
def test_transport_failure_roundtrip(error_type, category):
    failure = HTTPPostFailure.capture(error_type("lost response"), method="post", url="http://test/generate")
    restored = pickle.loads(pickle.dumps(failure)).exception()
    assert recovery_category(restored) == category
    assert isinstance(restored, error_type)


def test_real_ray_post_preserves_errors_without_local_resend(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            code, body = {
                "/generate": (503, b'{"detail":"Rollout worker unavailable"}'),
                "/bad": (400, b'{"detail":"invalid request"}'),
                "/ok": (200, b'{"text":"ready"}'),
            }[self.path]
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_client_concurrency", 2)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actor_idx", 0)
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        _memory=512 * 1024**2,
    )
    try:
        http_utils._init_ray_distributed_post(
            SimpleNamespace(num_gpus_per_node=1, http_post_actors_per_node=1, http_post_num_cpus=1)
        )

        async def exercise():
            url = f"http://127.0.0.1:{server.server_port}"
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await http_utils.post(url + "/generate", {"rid": "retry-group"}, max_retries=1, return_bytes=True)
            assert recovery_category(caught.value) == "transport"
            # Fatal errors can also cross a second actor boundary on the way to
            # the driver. Preserve diagnostics instead of another unpickle error.
            assert recovery_category(pickle.loads(pickle.dumps(caught.value))) == "transport"
            with pytest.raises(httpx.HTTPStatusError) as fatal:
                await http_utils.post(url + "/bad", {}, max_retries=1)
            assert recovery_category(fatal.value) is None
            assert pickle.loads(pickle.dumps(fatal.value)).response.status_code == 400
            assert await http_utils.post(url + "/ok", {}, max_retries=1, return_bytes=True) == b'{"text":"ready"}'

        asyncio.run(exercise())
        assert requests == ["/generate", "/bad", "/ok"]
    finally:
        for actor in http_utils._post_actors:
            ray.kill(actor)
        ray.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
