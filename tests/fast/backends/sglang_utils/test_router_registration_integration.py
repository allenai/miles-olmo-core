"""Exercise the installed Rust router, not a mock of its registration API.

The worker alone is fake. These CPU tests qualify the asynchronous 0.3.2
registration receipt, metadata failure, URL deduplication and registry schema.
"""

import asyncio
import json
import socket
import subprocess
import sys
import threading
from http import server

import httpx
import pytest
import sglang_router

from tests.fast.ray.rollout.test_server_cell_dispose import _make_cell

from miles.backends.sglang_utils import router_registration
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StatePendingWeights
from miles.utils.http_utils import GeneralHttpClientProvider


class Worker:
    def __init__(self):
        self.metadata_requested = threading.Event()
        self.release_metadata = threading.Event()
        self.release_metadata.set()
        owner = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/server_info", "/model_info"):
                    owner.metadata_requested.set()
                    owner.release_metadata.wait(90)
                body = json.dumps(
                    {
                        "model_path": "admission-test",
                        "served_model_name": "admission-test",
                        "is_generation": True,
                        "model_type": "llama",
                        "max_total_num_tokens": 4096,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass

            def log_message(self, *_):
                pass

        self.server = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.release_metadata.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.fixture
def worker():
    instance = Worker()
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
async def router(tmp_path):
    if sglang_router.__version__ != "0.3.2":
        pytest.skip("Contract qualification for the pinned 0.3.2 router")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (tmp_path / "router.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_router.launch_router",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warn",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                async with asyncio.timeout(20):
                    while True:
                        assert process.poll() is None, (tmp_path / "router.log").read_text()
                        try:
                            if (await client.get(url + "/health")).is_success:
                                break
                        except httpx.NetworkError:
                            pass
                        await asyncio.sleep(0.1)
                yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class LostReceiptTransport(httpx.AsyncHTTPTransport):
    def __init__(self, *, lose_first=True):
        super().__init__()
        self.lose_first = lose_first
        self.worker_ids = []

    async def handle_async_request(self, request):
        response = await super().handle_async_request(request)
        if request.method == "POST" and request.url.path == "/workers":
            await response.aread()
            self.worker_ids.append(response.json()["worker_id"])
            if self.lose_first and len(self.worker_ids) == 1:
                await response.aclose()
                raise httpx.ReadTimeout("injected lost acceptance response", request=request)
        return response


@pytest.mark.asyncio
async def test_real_router_recovers_lost_receipt_without_duplicate_workers(router, worker):
    transport = LostReceiptTransport()
    async with httpx.AsyncClient(transport=transport) as client:
        await router_registration.register_worker(
            client, router, {"url": worker.url, "worker_type": "regular"}, state=router_registration.Registration()
        )
        registered = (await client.get(router + "/workers")).json()["workers"]
    assert len(transport.worker_ids) == 1
    assert len(set(transport.worker_ids)) == 1
    assert len(registered) == 1 and registered[0]["is_healthy"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_first", [False, True])
async def test_real_metadata_delay_blocks_admission_and_failed_job_recovers(router, worker, fail_first):
    worker.release_metadata.clear()
    transport = LostReceiptTransport(lose_first=False)
    async with httpx.AsyncClient(transport=transport) as client:
        task = asyncio.create_task(
            router_registration.register_worker(
                client, router, {"url": worker.url, "worker_type": "regular"}, state=router_registration.Registration()
            )
        )
        try:
            assert await asyncio.to_thread(worker.metadata_requested.wait, 20)
            assert not task.done()
            assert (await client.get(router + "/workers")).json()["workers"] == []
            if fail_first:
                # Let the actual router's ten-second metadata deadline fire.
                async with asyncio.timeout(45):
                    while True:
                        status = (await client.get(router + "/workers/" + transport.worker_ids[0])).json()
                        if status.get("job_status", {}).get("status") == "failed":
                            assert "discover_metadata" in status["job_status"]["message"]
                            break
                        assert not task.done()
                        await asyncio.sleep(0.1)
            worker.release_metadata.set()
            await asyncio.wait_for(task, 30)
            registered = (await client.get(router + "/workers")).json()["workers"]
            assert len(registered) == 1 and registered[0]["is_healthy"]
        finally:
            worker.release_metadata.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_receipt", [False, True])
async def test_cancelled_admission_removes_a_late_worker_before_disposal(router, worker, monkeypatch, lost_receipt):
    worker.release_metadata.clear()
    transport = LostReceiptTransport(lose_first=lost_receipt)
    async with httpx.AsyncClient(transport=transport) as client:
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: client)
        api = SGLangRouterApiClient(router)
        cell = _make_cell(router_api_client=api)
        cell._state = StatePendingWeights(
            addr_info=CellAddrInfo(server_url=worker.url, bootstrap_port=None, gate_url=None)
        )
        task = asyncio.create_task(cell.mark_weights_ready())
        try:
            assert await asyncio.to_thread(worker.metadata_requested.wait, 20)
            task.cancel()
            async with asyncio.timeout(5):
                while not api._registrations[worker.url].retiring:
                    await asyncio.sleep(0)
            assert not task.done()
            assert (await client.get(router + "/workers")).json()["workers"] == []
            worker.release_metadata.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 15)
            assert not cell.is_serving
            await cell.dispose()
            assert isinstance(cell._state, StateDisposed)
            assert (await client.get(router + "/workers")).json()["workers"] == []
            assert worker.url not in api._registrations
            assert len(transport.worker_ids) == 1
        finally:
            worker.release_metadata.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await cell.dispose()
