"""CPU integration with the pinned Rust binary and real Core membership calls.

Run inside the MILES image: python -m pytest -q tests/manual/runtime/test_core_rust_router.py
No model, GPU or Ray cluster is required. Only the engine is synthetic.
"""

import asyncio
import base64
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest import mock

import httpx
import numpy as np
import pytest

from miles.backends.core_utils.publication import policy_refresh
from miles.backends.core_utils.rollout import rollout_errors
from miles.backends.sglang_utils import router_args_utils
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StatePendingWeights
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.ft_utils.health_checker import ActiveAndEpoch
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.types import Sample


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def runtime(tmp_path, request):
    dp_size = getattr(request, "param", None)
    state = dict(healthy=True, version=0, requests=[], entered=threading.Event(), release=threading.Event())
    routes = np.arange(512, dtype=np.int32).reshape(2, 16, 16)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, body, status=200):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(
                dict(model_path="synthetic", is_generation=True, max_req_input_len=34816, dp_size=dp_size or 1),
                200 if state["healthy"] else 503,
            )

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(data)
            initial = state["version"]
            if data["rid"] == "held":
                state["entered"].set()
                assert state["release"].wait(15)
            if data["rid"] == "disconnect":
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if data["rid"] == "503":
                self.reply({"message": "The request queue is full."}, 503)
                return
            current = state["version"]
            spans = [dict(version=initial, start=0, end=1), dict(version=current, start=1, end=2)]
            self.reply(
                dict(
                    text="synthetic",
                    output_ids=[17, 18],
                    meta_info=dict(
                        finish_reason=dict(type="stop"),
                        weight_version=current,
                        weight_versions=spans,
                        output_token_logprobs=[[-0.1, 17], [-0.2, 18]],
                        prompt_tokens=1,
                        routed_experts=base64.b64encode(routes.tobytes()).decode(),
                        return_probe="metadata must survive",
                    ),
                )
            )

    engine = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=engine.serve_forever, daemon=True)
    thread.start()
    port = free_port()
    args = router_args_utils._make_prefixed_cli_parser().parse_args([])
    args.train_backend = "olmo_core"
    args.use_miles_router = False
    args.router_dp_aware = dp_size is not None
    args.sglang_router_policy = "round_robin"
    args.sglang_router_request_timeout_secs = 1800
    options = router_args_utils.compute_sglang_router_args(
        args, host="127.0.0.1", port=port, prometheus_port=free_port(), has_pd_disaggregation=False
    )
    log = (tmp_path / "router.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "sglang_router.launch_router", *router_args_utils.router_args_to_argv(options)],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        with httpx.Client(timeout=0.2) as client:
            for _ in range(150):
                if proc.poll() is not None:
                    pytest.fail((tmp_path / "router.log").read_text())
                try:
                    response = client.get(f"http://127.0.0.1:{port}/health")
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("Rust startup")
        yield SimpleNamespace(
            router=f"http://127.0.0.1:{port}",
            engine=f"http://127.0.0.1:{engine.server_port}",
            state=state,
            routes=routes,
            options=options,
            dp_size=dp_size,
        )
    finally:
        state["release"].set()
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        engine.shutdown()
        engine.server_close()
        thread.join(5)


def make_cell(runtime):
    cell = ServerCell.__new__(ServerCell)
    cell.args = SimpleNamespace(
        train_backend="olmo_core",
        router_dp_aware=runtime.dp_size is not None,
        sglang_dp_size=runtime.dp_size or 1,
        use_miles_router=False,
        debug_rollout_only=False,
        rollout_health_check_interval=0,
        rollout_health_check_timeout=1,
        miles_router_health_check_failure_threshold=2,
    )
    cell.meta = SimpleNamespace(update_weights=True, cell_id="cpu-cell", sglang_api_key=None, worker_type="regular")
    cell.router_api_client = SGLangRouterApiClient(runtime.router)
    cell._state = StatePendingWeights(
        addr_info=CellAddrInfo(server_url=runtime.engine, bootstrap_port=None, gate_url=None)
    )
    cell._health_checker = mock.Mock()
    cell.health_checker_activeness = lambda: ActiveAndEpoch(active=True, epoch=0)
    return cell


def payload(rid):
    return dict(
        rid=rid,
        input_ids=[9],
        sampling_params=dict(max_new_tokens=2),
        return_logprob=True,
        return_routed_experts=True,
        routed_experts_start_len=1,
        return_sampling_mask=True,
    )


async def test_refresh_keeps_request_and_replay_then_quarantines_until_publication(runtime, monkeypatch):
    async with httpx.AsyncClient(timeout=20) as client:
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: client)
        cell = make_cell(runtime)
        try:
            assert not (await client.get(runtime.router + "/workers")).json()["workers"]
            await cell.mark_weights_ready()
            worker_id = cell._rust_membership.worker_id
            held = asyncio.create_task(client.post(runtime.router + "/generate", json=payload("held")))
            assert await asyncio.to_thread(runtime.state["entered"].wait, 5)
            runtime.state["version"] = 1
            await cell.mark_weights_ready()
            assert cell._rust_membership.worker_id == worker_id  # normal refresh does not churn registration
            runtime.state["release"].set()
            response = await held
            response.raise_for_status()
            meta = response.json()["meta_info"]
            sample = Sample(tokens=[9, 17, 18], response_length=2, rollout_log_probs=[-0.1, -0.2])
            policy_refresh.record_response(sample, meta)
            policy_refresh.validate_batch(
                dict(metadata=[sample.train_metadata], response_lengths=[2], weight_versions=[sample.weight_versions])
            )
            assert sample.train_metadata["policy_refresh"]["spans"] == [
                dict(version=0, start=0, end=1),
                dict(version=1, start=1, end=2),
            ]
            np.testing.assert_array_equal(
                np.frombuffer(base64.b64decode(meta["routed_experts"]), dtype=np.int32).reshape(2, 16, 16),
                runtime.routes,
            )
            assert runtime.state["requests"][0]["return_routed_experts"] is True
            assert runtime.state["requests"][0]["routed_experts_start_len"] == 1
            assert runtime.state["requests"][0]["return_sampling_mask"] is True
            runtime.state["healthy"] = False
            await cell._check_router_membership_health()
            await cell._check_router_membership_health()
            assert cell.is_pending_weights
            runtime.state["healthy"] = True
            await cell._check_router_membership_health()
            denied = await client.post(runtime.router + "/generate", json=payload("denied"))
            assert denied.status_code == 503
            with pytest.raises(httpx.HTTPStatusError) as failure:
                denied.raise_for_status()
            assert rollout_errors.recovery_category(failure.value) == "unavailable", denied.text
            assert len(runtime.state["requests"]) == 1
            runtime.state["version"] = 2
            await cell.mark_weights_ready()
            assert cell.is_serving and cell._rust_membership.worker_id != worker_id
            resumed = await client.post(runtime.router + "/generate", json=payload("resumed"))
            assert resumed.json()["meta_info"]["weight_version"] == 2
        finally:
            runtime.state["release"].set()
            await cell.dispose()
        assert not (await client.get(runtime.router + "/workers")).json()["workers"]


async def test_rust_launch_contract_disables_all_independent_recovery(runtime, monkeypatch):
    for key in ("disable_retries", "disable_health_check", "disable_circuit_breaker"):
        assert runtime.options[key] is True
    async with httpx.AsyncClient(timeout=20) as client:
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: client)
        cell = make_cell(runtime)
        try:
            await cell.mark_weights_ready()
            response = await client.post(runtime.router + "/generate", json=payload("503"))
            assert response.status_code == 503
            assert len(runtime.state["requests"]) == 1
        finally:
            await cell.dispose()


async def test_disconnected_engine_is_reported_to_group_retry(runtime, monkeypatch):
    async with httpx.AsyncClient(timeout=20) as client:
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: client)
        cell = make_cell(runtime)
        try:
            await cell.mark_weights_ready()
            response = await client.post(runtime.router + "/generate", json=payload("disconnect"))
            with pytest.raises(httpx.HTTPStatusError) as failure:
                response.raise_for_status()
            assert rollout_errors.recovery_category(failure.value) == "transport", response.text
            assert len(runtime.state["requests"]) == 1
        finally:
            await cell.dispose()


@pytest.mark.parametrize("runtime", [1, 2], indirect=True)
async def test_dp_registration_and_removal_against_pinned_rust(runtime):
    cell = make_cell(runtime)
    try:
        async with httpx.AsyncClient() as client:
            with mock.patch.object(GeneralHttpClientProvider, "client", return_value=client):
                await cell.mark_weights_ready()
                assert cell.is_serving
                workers = (await client.get(f"{runtime.router}/workers")).json()["workers"]
                assert {w["url"] for w in workers} == {f"{runtime.engine}@{rank}" for rank in range(runtime.dp_size)}
                assert cell._rust_membership.worker_id not in {w["id"] for w in workers}
                await cell.mark_weights_ready()
                await cell.dispose()
                assert (await client.get(f"{runtime.router}/workers")).json()["workers"] == []
    finally:
        if not isinstance(cell._state, StateDisposed):
            # Keep the router/client alive for cleanup if an assertion fails.
            async with httpx.AsyncClient() as client:
                with mock.patch.object(GeneralHttpClientProvider, "client", return_value=client):
                    await cell.dispose()
