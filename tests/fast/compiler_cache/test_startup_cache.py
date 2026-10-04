"""Worker isolation, cold fallback and successful-teardown publication contract."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from ray._private.ray_constants import WORKER_PROCESS_SETUP_HOOK_ENV_VAR

from miles.utils.compiler_cache import startup_cache as startup


def policy(root, *, restore=True):
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    return dict(
        shared=str(root / "shared"),
        report_dir=str(report_dir),
        slot="train-actor-cell0-rank0",
        image="sha256:" + "a" * 64,
        runtime_lock={},
        sources={k: {"sha256": "same"} for k in ("olmo-core", "miles", "open-instruct", "olmo-sglang")},
        model_config={},
        run_config={"core": {}, "miles": {}},
        restore=restore,
        observe=False,
    )


def test_explicit_worker_environment_preserves_existing_flags_and_separates_roles(tmp_path):
    args = SimpleNamespace(olmo_core_startup_cache=policy(tmp_path))
    train = startup.worker_runtime_env(args, "train-rank0", {"NCCL_CUMEM_ENABLE": "1"})
    serve = startup.worker_runtime_env(args, "serve-rank0", {})
    assert train["env_vars"]["NCCL_CUMEM_ENABLE"] == "1"
    assert train["worker_process_setup_hook"].endswith(".setup_worker")
    assert train["env_vars"][WORKER_PROCESS_SETUP_HOOK_ENV_VAR] == train["worker_process_setup_hook"]
    assert json.loads(train["env_vars"][startup.ENV])["slot"] != json.loads(serve["env_vars"][startup.ENV])["slot"]
    assert train["env_vars"]["HF_MODULES_CACHE"] != serve["env_vars"]["HF_MODULES_CACHE"]
    args.olmo_core_startup_cache = None
    uncached = startup.worker_runtime_env(args, "unused", {"X": "1"})
    assert uncached["env_vars"]["X"] == "1"
    assert uncached["env_vars"]["HF_MODULES_CACHE"].startswith("/tmp/miles-hf-modules-")


def test_serving_module_cache_is_private_even_without_compiler_cache():
    args = SimpleNamespace(olmo_core_startup_cache=None)
    original = {"HF_HOME": "/tmp/shared-models", "HF_MODULES_CACHE": "/tmp/old-shared-modules"}
    spec = SimpleNamespace(env_var=lambda context: original)
    first = startup._serving_environment(args, spec, None)
    second = startup._serving_environment(args, spec, None)
    assert first["HF_MODULES_CACHE"] != second["HF_MODULES_CACHE"]
    assert first["HF_HOME"] == second["HF_HOME"] == original["HF_HOME"]
    assert original["HF_MODULES_CACHE"] == "/tmp/old-shared-modules"


def test_worker_restore_publish_and_rank_isolation(tmp_path, monkeypatch):
    descriptor = policy(tmp_path, restore=False)
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    fake_ray = SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "node"))
    original_import = startup.importlib.import_module
    monkeypatch.setattr(
        startup.importlib, "import_module", lambda name: fake_ray if name == "ray" else original_import(name)
    )
    monkeypatch.setattr(startup.probes, "toolchain", lambda env: {"gpu": "B300"})
    monkeypatch.setattr(startup.probes, "compiler_environment", lambda env: {})
    startup.setup_worker()
    cold = json.loads(next((tmp_path / "reports").glob("*.json")).read_text())
    local = Path(cold["local"])
    (local / "triton/kernel").write_bytes(b"compiled")
    assert startup.publish_worker(cold)["publish"]["status"] == "published"
    assert not local.exists()
    descriptor["restore"] = True
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    startup.setup_worker()
    reports = [json.loads(p.read_text()) for p in (tmp_path / "reports").glob("*.json")]
    warm = next(r for r in reports if r["restore"]["status"] == "hit")
    assert warm["fingerprint"] == cold["fingerprint"] and warm["local"] != cold["local"]
    assert (Path(warm["local"]) / "triton/kernel").read_bytes() == b"compiled"
    startup.publish_worker(warm)
    descriptor["slot"] = "train-actor-cell0-rank1"
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    startup.setup_worker()
    other = next(
        json.loads(p.read_text())
        for p in (tmp_path / "reports").glob("*.json")
        if json.loads(p.read_text())["slot"] == descriptor["slot"]
    )
    assert other["fingerprint"] != cold["fingerprint"] and other["restore"]["status"] == "miss"
    startup.publish_worker(other)


def test_failed_run_never_publishes(tmp_path):
    args = SimpleNamespace(olmo_core_startup_cache=policy(tmp_path), save=str(tmp_path))
    with mock.patch.object(startup, "publish_worker", side_effect=AssertionError("published failed run")):
        asyncio.run(startup.finish(args, success=False))
    assert json.loads((tmp_path / "compiler-cache.json").read_text())["success"] is False


def test_disabled_cache_needs_no_sources_or_filesystem(tmp_path):
    args = SimpleNamespace(olmo_core=SimpleNamespace(compiler_cache=False))
    startup.prepare(args, application_root="/unused")
    assert args.olmo_core_startup_cache is None


def test_invalid_retention_path_cannot_create_reports():
    args = SimpleNamespace(olmo_core=SimpleNamespace(compiler_cache=True, compiler_cache_root="/weka/no-ttl/cache"))
    with (
        mock.patch.object(Path, "mkdir", side_effect=AssertionError("created before validation")),
        pytest.raises(ValueError, match="expiry"),
    ):
        startup.prepare(args, application_root="/unused")


def test_publication_has_one_budget_and_cancels_ray_tasks(tmp_path, monkeypatch):
    descriptor = policy(tmp_path)
    workers = [dict(slot=f"rank-{n}", node_id="0" * 56, fingerprint="a" * 64) for n in range(4)]
    for n, worker in enumerate(workers):
        (Path(descriptor["report_dir"]) / f"{n}.json").write_text(json.dumps(worker))
    submitted = []
    cancelled = []

    class Remote:
        def options(self, **kwargs):
            return self

        def remote(self, worker):
            future = asyncio.get_running_loop().create_future()
            submitted.append(future)
            return future

    fake_ray = SimpleNamespace(
        remote=lambda **kwargs: lambda fn: Remote(), cancel=lambda task, **kwargs: cancelled.append((task, kwargs))
    )
    original_import = startup.importlib.import_module
    monkeypatch.setattr(
        startup.importlib, "import_module", lambda name: fake_ray if name == "ray" else original_import(name)
    )
    monkeypatch.setattr(startup, "PUBLICATION_TIMEOUT_SECONDS", 0.02)
    args = SimpleNamespace(olmo_core_startup_cache=descriptor, save=str(tmp_path))
    asyncio.run(startup.finish(args, success=True))
    report = json.loads((tmp_path / "compiler-cache.json").read_text())
    assert report["success"] is True
    assert len(submitted) == startup.PUBLISHERS_PER_NODE
    assert cancelled == [(task, {"force": True}) for task in submitted]
    assert len(report["workers"]) == 4
    assert all("TimeoutError: shared 0.02s" in w["publish"]["reason"] for w in report["workers"])


def test_publication_handles_mixed_results_and_keeps_training_success(tmp_path, monkeypatch):
    descriptor = policy(tmp_path)
    for n in range(3):
        (Path(descriptor["report_dir"]) / f"{n}.json").write_text(
            json.dumps(dict(slot=str(n), node_id="0" * 56, fingerprint="a" * 64))
        )

    class Remote:
        def options(self, **kwargs):
            return self

        def remote(self, worker):
            future = asyncio.get_running_loop().create_future()
            if worker["slot"] == "1":
                future.set_exception(OSError("node disappeared"))
            else:
                future.set_result({"slot": worker["slot"], "publish": {"status": "published"}})
            return future

    fake_ray = SimpleNamespace(remote=lambda **kwargs: lambda fn: Remote())
    original_import = startup.importlib.import_module
    monkeypatch.setattr(
        startup.importlib, "import_module", lambda name: fake_ray if name == "ray" else original_import(name)
    )
    args = SimpleNamespace(olmo_core_startup_cache=descriptor, save=str(tmp_path))
    asyncio.run(startup.finish(args, success=True))
    report = json.loads((tmp_path / "compiler-cache.json").read_text())
    assert report["success"] is True
    assert [w["publish"]["status"] for w in report["workers"]] == ["published", "unavailable", "published"]
    assert report["workers"][1]["publish"]["reason"] == "OSError: node disappeared"


def test_optional_cache_report_failure_cannot_fail_completed_training(tmp_path, monkeypatch):
    args = SimpleNamespace(olmo_core_startup_cache=policy(tmp_path), save=str(tmp_path))
    monkeypatch.setattr(Path, "write_bytes", mock.Mock(side_effect=OSError("shared filesystem unavailable")))
    asyncio.run(startup.finish(args, success=False))
