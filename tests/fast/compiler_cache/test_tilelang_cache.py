"""TileLang identity, staged-export rejection and independent publication."""

import tempfile
from pathlib import Path

import pytest

from miles.utils.compiler_cache import compiler_cache as cache
from miles.utils.compiler_cache import startup_cache, tilelang_cache


def descriptor(tmp_path):
    return {
        "tilelang_runtime": {"sha256": "compiler"},
        "restore": True,
        "observe": True,
        "shared": str(tmp_path / "shared"),
    }


def worker(tmp_path):
    return {
        "identity": {"image": "binary-base", "kernel_runtime": {"native_sources": {}}},
        "slot": "trainer-0",
        "shared": str(tmp_path / "shared"),
    }


def test_round_trip_separate_key_and_fresh_private_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(tilelang_cache.os, "environ", {})
    policy = descriptor(tmp_path)
    first = tmp_path / "first"
    first.mkdir()
    report = worker(tmp_path)
    tilelang_cache.configure(policy, first)
    assert tilelang_cache.os.environ["TILELANG_TMP_DIR"] == str(first / "tilelang-tmp")
    assert tilelang_cache.os.environ["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] == "1"
    tilelang_cache.restore(policy, report, first)
    assert report["tilelang"]["restore"]["status"] == "miss"
    assert report["tilelang"]["identity"]["compatibility_scope"] == "core-tilelang-v1"
    (first / "tilelang/executable.so").write_bytes(b"compiled")
    assert tilelang_cache.publish(report, first)["publish"]["status"] == "published"
    assert tilelang_cache.publish(report, first)["publish"]["status"] == "unchanged"
    second = tmp_path / "second"
    second.mkdir()
    warm = worker(tmp_path)
    tilelang_cache.configure(policy, second)
    tilelang_cache.restore(policy, warm, second)
    assert warm["tilelang"]["restore"]["status"] == "hit"
    assert (second / "tilelang/executable.so").read_bytes() == b"compiled"
    policy["tilelang_runtime"]["sha256"] = "changed-compiler"
    third = tmp_path / "third"
    third.mkdir()
    changed = worker(tmp_path)
    tilelang_cache.configure(policy, third)
    tilelang_cache.restore(policy, changed, third)
    assert changed["tilelang"]["restore"]["status"] == "miss"
    assert changed["tilelang"]["fingerprint"] != warm["tilelang"]["fingerprint"]


def test_partial_kernel_export_cannot_be_published(tmp_path):
    local = tmp_path / "local"
    staging = local / "tilelang/version/platform/.staging/in-progress"
    staging.mkdir(parents=True)
    (staging / "executable.so").write_bytes(b"partial")
    with pytest.raises(ValueError, match="export is in progress"):
        cache.publish(tmp_path / "shared", local, "a" * 64, "tilelang")


def test_absent_opt_in_leaves_environment_and_report_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(tilelang_cache.os, "environ", {})
    report = worker(tmp_path)
    tilelang_cache.configure({}, tmp_path)
    tilelang_cache.restore({}, report, tmp_path)
    assert tilelang_cache.publish(report, tmp_path) is None
    assert tilelang_cache.os.environ == {}
    assert "tilelang" not in report


def test_publication_failure_is_nonfatal_and_retryable(tmp_path, monkeypatch):
    monkeypatch.setattr(tilelang_cache.os, "environ", {})
    report = worker(tmp_path)
    tilelang_cache.configure(descriptor(tmp_path), tmp_path)
    tilelang_cache.restore(descriptor(tmp_path), report, tmp_path)
    (tmp_path / "tilelang/executable.so").write_bytes(b"compiled")

    def fail(*args, **kwargs):
        raise ValueError("changed during snapshot")

    monkeypatch.setattr(cache, "publish", fail)
    assert tilelang_cache.publish(report, tmp_path)["publish"]["status"] == "rejected"
    assert not (tmp_path / "tilelang-publication-state.json").exists()


def test_tilelang_publication_runs_even_when_triton_has_not_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(tilelang_cache.os, "environ", {})
    with tempfile.TemporaryDirectory(prefix="core-triton-", dir="/tmp") as directory:
        local = Path(directory)
        (local / "triton").mkdir()
        (local / "triton/kernel.cubin").write_bytes(b"triton")
        report = {**worker(tmp_path), "local": directory, "fingerprint": "b" * 64, "restore": {}, "setup_seconds": 0}
        tilelang_cache.configure(descriptor(tmp_path), local)
        tilelang_cache.restore(descriptor(tmp_path), report, local)
        assert startup_cache.publish_worker(report, retain_local=True)["publish"]["status"] == "published"
        (local / "tilelang/executable.so").write_bytes(b"tilelang")
        second = startup_cache.publish_worker(report, retain_local=True)
        assert second["publish"]["status"] == "unchanged"
        assert second["tilelang"]["publish"]["status"] == "published"
