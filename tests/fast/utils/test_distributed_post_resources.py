"""CPU-only coverage for response offload resources and ambiguous failures."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import ray

from miles.ray.specs.rollout import spec_rollout_executor
from miles.utils import http_utils
from miles.utils.http_post_observer import PostObserver


def test_worker_resources_and_total_connections(monkeypatch):
    created = []

    class Actor:
        def options(self, **options):
            self.options_value = options
            return self

        def remote(self, **kwargs):
            created.append((self.options_value, kwargs))
            return object()

    monkeypatch.setattr(ray, "remote", lambda cls: Actor())
    monkeypatch.setattr(ray, "nodes", lambda: [dict(Alive=True, NodeID=f"{n:056x}") for n in range(3)])
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_client_concurrency", 9728)
    args = SimpleNamespace(num_gpus_per_node=8, http_post_actors_per_node=16, http_post_num_cpus=1)
    http_utils._init_ray_distributed_post(args)
    assert len(created) == 48
    assert all(options["num_cpus"] == 1 and options["max_concurrency"] == 203 for options, _ in created)
    assert all(kwargs["concurrency"] == 203 for _, kwargs in created)


def test_remote_failure_does_not_resubmit_locally(monkeypatch):
    remote = AsyncMock(side_effect=RuntimeError("response lost after delivery"))
    local = AsyncMock()
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", [SimpleNamespace(do_post=SimpleNamespace(remote=remote))])
    monkeypatch.setattr(http_utils, "_post", local)
    with pytest.raises(RuntimeError, match="after delivery"):
        asyncio.run(http_utils.post("http://example/generate", {"rid": "test"}, max_retries=1, return_bytes=True))
    local.assert_not_called()
    assert remote.call_args.kwargs["return_bytes"] is True


def test_worker_measurement_preserves_privacy_and_failure(tmp_path):
    observer = PostObserver(tmp_path)
    successful = observer.start({"rid": "one", "text": "do not retain this"})
    failed = observer.start({"rid": "two"})
    observer.finish(successful, b"abcdef")
    observer.finish(failed, None)
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert rows[0]["response_bytes"] == 6
    assert rows[0]["worker_active_requests"] == 1
    assert rows[1]["succeeded"] is False and rows[1]["worker_active_requests"] == 0
    assert "do not retain" not in str(rows)


def test_rollout_executor_reserves_configured_cpus():
    args = SimpleNamespace(rollout_executor_num_cpus=2, pin_rollout_manager_to_head=False)
    assert spec_rollout_executor(args).scheduling.num_cpus_per_worker == 2
    args.rollout_executor_num_cpus = 0
    with pytest.raises(ValueError, match="positive"):
        spec_rollout_executor(args)
