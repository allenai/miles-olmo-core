"""Worker isolation, cold fallback and successful-teardown publication contract."""

import asyncio
import dataclasses
import errno
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
        kernel_runtime={"triton": {"sha256": "compiler"}, "native_sources": {}},
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


@pytest.mark.parametrize(
    "slot,other_slot,shared",
    [
        ("train-actor-cell0-rank0", "train-actor-cell0-rank1", False),
        ("inference-engine-all-0-0-0-0", "inference-engine-all-0-0-12-0", True),
    ],
)
def test_worker_restore_publish_and_rank_isolation(tmp_path, monkeypatch, slot, other_slot, shared):
    descriptor = policy(tmp_path, restore=False)
    descriptor["slot"] = slot
    descriptor["run_config"]["miles"]["rollout_num_gpus_per_engine"] = 1
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    fake_ray = SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "node"))
    original_import = startup.importlib.import_module
    monkeypatch.setattr(
        startup.importlib, "import_module", lambda name: fake_ray if name == "ray" else original_import(name)
    )
    monkeypatch.setattr(startup.probes, "toolchain", lambda env: {"torch_hardware": {"gpus": [{"name": "B300"}] * 8}})
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
    descriptor["slot"] = other_slot
    if shared:
        monkeypatch.setattr(
            startup.probes, "toolchain", lambda env: {"torch_hardware": {"gpus": [{"name": "B300"}] * 7}}
        )
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    startup.setup_worker()
    other = next(
        json.loads(p.read_text())
        for p in (tmp_path / "reports").glob("*.json")
        if json.loads(p.read_text())["slot"] == descriptor["slot"]
    )
    assert (other["fingerprint"] == cold["fingerprint"]) is shared
    assert other["restore"]["status"] == ("hit" if shared else "miss")
    assert other["local"] not in {cold["local"], warm["local"]}
    if shared:
        assert (Path(other["local"]) / "triton/kernel").read_bytes() == b"compiled"
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


@pytest.fixture
def driver_sources(tmp_path, monkeypatch):
    application = tmp_path / "application"
    lock = application / "runtime/miles/runtime.lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({"base_image": {"docker_id": "sha256:" + "a" * 64}}))
    packages = {}
    for name in ("open_instruct", "olmo_core", "miles", "olmo_sglang", "sglang", "triton"):
        root = (application if name == "open_instruct" else tmp_path / "installed") / name
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("# package\n")
        packages[name] = root
    (packages["sglang"] / "kernel.so").write_bytes(b"native-library")
    (packages["triton"] / "ptxas").write_bytes(b"extensionless-compiler-tool")
    specs = {name: SimpleNamespace(submodule_search_locations=[str(root)]) for name, root in packages.items()}
    monkeypatch.setattr(startup.util, "find_spec", specs.get)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"hidden_size": 128}))
    values = dict(
        compiler_cache=True,
        compiler_cache_root=str(tmp_path / "shared"),
        compiler_cache_restore=True,
        compiler_cache_diagnostics=False,
        compiler_cache_max_storage_bytes=1024,
        compiler_cache_publish_interval_seconds=600,
        precision="bf16",
    )
    config = dataclasses.make_dataclass("CoreConfig", [(key, type(value)) for key, value in values.items()])
    args = SimpleNamespace(olmo_core=config(**values), hf_checkpoint=str(checkpoint), global_batch_size=128)
    return args, application, packages, specs


def test_enabled_prepare_fingerprints_installed_compiler_and_native_packages(driver_sources):
    args, application, packages, _ = driver_sources
    startup.prepare(args, application_root=application)
    before = args.olmo_core_startup_cache
    assert set(before["sources"]) == {"open-instruct", "olmo-core", "miles", "olmo-sglang", "sglang"}
    assert before["kernel_runtime"]["triton"]["files"] == 2
    assert before["kernel_runtime"]["native_sources"]["sglang"]["files"] == 1
    assert before["model_config"] == {"hidden_size": 128}
    assert before["run_config"] == {"core": {"precision": "bf16"}, "miles": {"global_batch_size": 128}}
    assert Path(before["report_dir"]).is_dir()
    assert "tilelang_runtime" not in before
    (packages["triton"] / "ptxas").write_bytes(b"changed-compiler-tool")
    startup.prepare(args, application_root=application)
    assert args.olmo_core_startup_cache["kernel_runtime"]["triton"] != before["kernel_runtime"]["triton"]
    assert args.olmo_core_startup_cache["sources"] == before["sources"]


@pytest.mark.parametrize("package", ["sglang", "triton"])
@pytest.mark.parametrize("spec", [None, SimpleNamespace(submodule_search_locations=[])])
def test_enabled_prepare_rejects_missing_required_package_before_creating_reports(driver_sources, package, spec):
    args, application, _, specs = driver_sources
    specs[package] = spec
    with pytest.raises(ValueError, match="sglang" if package == "sglang" else "Triton compiler"):
        startup.prepare(args, application_root=application)
    assert args.olmo_core_startup_cache is None
    assert not Path(args.olmo_core.compiler_cache_root).exists()


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EROFS, errno.EACCES])
def test_driver_shared_report_failure_uses_ordinary_local_caches(driver_sources, monkeypatch, caplog, code):
    args, application, _, _ = driver_sources
    shared = Path(args.olmo_core.compiler_cache_root)
    original = Path.mkdir

    def unavailable(path, *args, **kwargs):
        if path.is_relative_to(shared):
            raise OSError(code, "shared cache unavailable", str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", unavailable)
    startup.prepare(args, application_root=application)
    assert args.olmo_core_startup_cache is None
    assert f"errno={code}" in caplog.text
    assert str(shared) in caplog.text
    assert "without shared restore or publication" in caplog.text
    assert any(r.levelname == "WARNING" and "SHARED COMPILER CACHE DISABLED" in r.message for r in caplog.records)
    environment = startup.worker_runtime_env(args, "train-rank0", {"KEEP": "value"})
    assert environment["env_vars"]["KEEP"] == "value"
    assert startup.ENV not in environment["env_vars"]
    assert "worker_process_setup_hook" not in environment


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EROFS, errno.EACCES])
@pytest.mark.parametrize("restore_fails", [False, True])
def test_worker_report_failure_preserves_local_compilation(tmp_path, monkeypatch, caplog, code, restore_fails):
    descriptor = policy(tmp_path)
    descriptor["run_config"]["miles"]["rollout_num_gpus_per_engine"] = 1
    local = tmp_path / "core-triton-worker"
    local.mkdir()
    monkeypatch.setattr(startup.tempfile, "mkdtemp", lambda **kwargs: str(local))
    monkeypatch.setenv(startup.ENV, json.dumps(descriptor))
    monkeypatch.setenv("MILES_WORKER_NODE_ID", "node")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/unused")
    monkeypatch.setattr(startup.probes, "toolchain", lambda env: {"torch_hardware": {"gpus": [{"name": "B300"}]}})
    monkeypatch.setattr(startup.probes, "compiler_environment", lambda env: {})

    def restore(shared, destination, key, family):
        if restore_fails:
            raise OSError(code, "shared restore unavailable")
        (destination / "triton" / "restored.bin").write_bytes(b"restored")
        return {"family": family, "status": "hit"}

    original = Path.write_bytes

    def unavailable(path, data):
        if path.is_relative_to(descriptor["report_dir"]):
            raise OSError(code, "shared report unavailable", str(path))
        return original(path, data)

    monkeypatch.setattr(startup.cache, "restore", restore)
    monkeypatch.setattr(Path, "write_bytes", unavailable)
    startup.setup_worker()
    assert startup.os.environ["TRITON_CACHE_DIR"] == str(local / "triton")
    if not restore_fails:
        assert (local / "triton" / "restored.bin").read_bytes() == b"restored"
    # New compiler output remains writable even when both restore and reporting fail.
    (local / "triton" / "new.bin").write_bytes(b"compiled locally")
    assert (local / "triton" / "new.bin").read_bytes() == b"compiled locally"
    assert f"errno={code}" in caplog.text
    assert descriptor["report_dir"] in caplog.text
    assert "continuing with local compiler cache" in caplog.text
    assert "cannot be published or cleaned up by the driver" in caplog.text
    assert any(r.levelname == "WARNING" and "SHARED COMPILER CACHE REPORT FAILED" in r.message for r in caplog.records)


@pytest.mark.parametrize("failure_stage", ["write", "rename"])
def test_worker_report_commit_never_exposes_partial_json(tmp_path, monkeypatch, failure_stage):
    healthy = {"slot": "healthy", "fingerprint": "a" * 64}
    startup._write_worker_report(tmp_path / "healthy.json", healthy)
    target = tmp_path / "new.json"
    incoming = {"slot": "new", "fingerprint": "b" * 64}
    original_write = Path.write_bytes

    def partial_write(path, data):
        original_write(path, data[:24])
        assert startup._worker_reports(tmp_path) == ([healthy], [])
        raise OSError(errno.ENOSPC, "injected mid-write failure", str(path))

    def failed_rename(path, destination):
        assert startup._worker_reports(tmp_path) == ([healthy], [])
        raise OSError(errno.EACCES, "injected rename failure", str(destination))

    with monkeypatch.context() as patch:
        if failure_stage == "write":
            patch.setattr(Path, "write_bytes", partial_write)
        else:
            patch.setattr(Path, "replace", failed_rename)
        with pytest.raises(OSError):
            startup._write_worker_report(target, incoming)
    assert not target.exists()
    assert target.with_suffix(".json.pending").is_file()
    assert startup._worker_reports(tmp_path) == ([healthy], [])
    startup._write_worker_report(target, incoming)
    assert json.loads(target.read_text()) == incoming
    assert not target.with_suffix(".json.pending").exists()
    assert startup._worker_reports(tmp_path) == ([healthy, incoming], [])


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
