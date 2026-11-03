"""Large response traffic must not reuse idle engine sockets."""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter


def test_generation_uses_fresh_connections_while_health_keeps_its_pool():
    connections = {"/generate": [], "/health": []}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            connections[self.path].append(self.client_address)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def exercise():
        router = MilesRouter(
            MilesRouterConfig(
                host="127.0.0.1",
                port=8000,
                max_connections=4,
                timeout=10,
                health_check_interval=1,
                health_check_failure_threshold=3,
            )
        )
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            for _ in range(2):
                assert (await router.client.get(url + "/generate")).status_code == 200
                assert (await router.health_client.get(url + "/health")).status_code == 200
            assert connections["/generate"][0] != connections["/generate"][1]
            assert connections["/health"][0] == connections["/health"][1]
        finally:
            await router.close()

    try:
        asyncio.run(exercise())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
