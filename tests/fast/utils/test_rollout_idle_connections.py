"""Rollout requests survive a client stall past the server's idle timeout."""

import multiprocessing
import socket
import time
from types import SimpleNamespace

import httpx
import uvicorn
import uvloop

from miles.utils import http_utils


async def app(scope, receive, send):
    while (await receive()).get("more_body", False):
        pass
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
    await send({"type": "http.response.body", "body": b"ok"})


def serve(pipe):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    pipe.send(sock.getsockname()[1])
    config = uvicorn.Config(app, log_level="error", lifespan="off", timeout_keep_alive=0.1, loop="uvloop")
    uvicorn.Server(config).run(sockets=[sock])


def test_rollout_client_does_not_reuse_an_idle_connection_after_a_stall(monkeypatch):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    server = context.Process(target=serve, args=(child,))
    server.start()
    assert parent.poll(10), "HTTP fixture did not bind"
    url = f"http://127.0.0.1:{parent.recv()}/generate"
    monkeypatch.setattr(http_utils, "_http_client", None)
    monkeypatch.setattr(http_utils, "_client_concurrency", 1)
    args = SimpleNamespace(
        rollout_num_gpus=1,
        sglang_server_concurrency=1,
        rollout_num_gpus_per_engine=1,
        eval_num_gpus=0,
        use_distributed_post=False,
    )

    async def scenario():
        async def trace(name, info):
            if name == "http11.send_request_headers.started":
                # The pool has selected its connection. Another process serves
                # HTTP, so its idle timer continues while this client is blocked.
                time.sleep(0.3)

        http_utils.init_http_client(args)
        async with http_utils._http_client as client:
            assert (await client.post(url, content=b"warm")).text == "ok"
            assert (await client.post(url, content=b"next", extensions={"trace": trace})).text == "ok"

    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                if httpx.post(url, timeout=0.1).status_code == 200:
                    break
            except httpx.TransportError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        uvloop.run(scenario())
    finally:
        server.terminate()
        server.join(timeout=5)
