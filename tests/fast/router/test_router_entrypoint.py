import asyncio
import subprocess
import sys
import time

import httpx
import pytest

from miles.router import router as router_module
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter, run_router
from miles.utils.http_utils import find_available_port


def test_miles_router_module_help_exits_successfully() -> None:
    """The Miles router module exposes its CLI help without starting the server."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-m", "miles.router.router", "--help"],
        capture_output=True,
        text=True,
        # Generous because this imports the whole miles stack, and a loaded runner takes longer
        # than the few seconds it costs idle; the bound is here to catch a hang, not to time it.
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config-json" in result.stdout


class RouterRecorder:
    def __init__(self) -> None:
        self.routers: list[MilesRouter] = []

    def create(self, config: MilesRouterConfig, verbose: bool = False) -> MilesRouter:
        router = MilesRouter(config, verbose=verbose)
        self.routers.append(router)
        return router


class UvicornRunRecorder:
    def __init__(self) -> None:
        self.apps: list[object] = []
        self.bind_addresses: list[tuple[str, int]] = []

    def run(self, app: object, *, host: str, port: int, **kwargs: object) -> None:
        self.apps.append(app)
        self.bind_addresses.append((host, port))


class TestRunRouter:
    def test_config_controls_router_construction_and_uvicorn_bind_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`run_router` builds the router from the given config and serves its app on the configured host and port."""
        config = MilesRouterConfig(
            host="192.0.2.7",
            port=21234,
            max_connections=11,
            timeout=None,
            health_check_interval=1.0,
            health_check_failure_threshold=2,
        )
        router_recorder = RouterRecorder()
        uvicorn_recorder = UvicornRunRecorder()
        monkeypatch.setattr(router_module, "configure_logger_raw", lambda name: None)
        monkeypatch.setattr(router_module.setproctitle, "setproctitle", lambda title: None)
        monkeypatch.setattr(router_module, "MilesRouter", router_recorder.create)
        monkeypatch.setattr(router_module.uvicorn, "run", uvicorn_recorder.run)

        run_router(config)

        assert len(router_recorder.routers) == 1
        assert router_recorder.routers[0].config is config
        assert uvicorn_recorder.apps == [router_recorder.routers[0].app]
        assert uvicorn_recorder.bind_addresses == [("192.0.2.7", 21234)]


@pytest.fixture
def running_router(tmp_path):
    config = MilesRouterConfig(
        host="127.0.0.1",
        port=find_available_port(21234),
        max_connections=4,
        timeout=None,
        health_check_interval=60,
        health_check_failure_threshold=3,
    )
    url = f"http://{config.host}:{config.port}/list_workers"
    log_path = tmp_path / "router.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "miles.router.router", "--config-json", config.model_dump_json()],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            with httpx.Client(timeout=1, trust_env=False) as client:
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        client.get(url).raise_for_status()
                        break
                    except httpx.HTTPError:
                        time.sleep(0.05)
                else:
                    pytest.fail(f"Router did not start: {log_path.read_text()}")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_router_keeps_idle_connection_beyond_client_reuse_window(running_router):
    """Real entrypoint must leave margin beyond HTTPX's normal five-second expiry.

    Extend this test client's expiry to observe the server's policy directly.
    HTTPX transparently reconnects when it sees a closed socket, so success alone
    would miss the regression: also assert that it reused the original socket.
    """
    connects = []

    async def trace(event, info):
        if event == "connection.connect_tcp.started":
            connects.append(event)

    async def check_connection():
        async with httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1, keepalive_expiry=15), timeout=10, trust_env=False
        ) as client:
            first = await client.get(running_router, extensions={"trace": trace})
            first.raise_for_status()
            await asyncio.sleep(5.25)
            second = await client.get(running_router, extensions={"trace": trace})
            second.raise_for_status()

        assert first.json() == second.json() == {"urls": []}

    asyncio.run(check_connection())
    assert len(connects) == 1, "Router closed the connection at the old five-second boundary"
