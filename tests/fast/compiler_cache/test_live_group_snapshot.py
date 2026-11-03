"""A concurrent kernel compile must not invalidate independent completed groups."""

import json
from pathlib import Path

import pytest

from miles.utils.compiler_cache import compiler_cache as cache

KEY = "b" * 64


def group(local, name, value=b"compiled"):
    root = local / "triton" / name
    root.mkdir(parents=True)
    (root / "kernel.cubin").write_bytes(value)
    (root / "__grp__kernel.json").write_text(json.dumps({"child_paths": {"kernel.cubin": str(root / "kernel.cubin")}}))
    return root


def test_completed_group_survives_another_groups_concurrent_write(tmp_path, monkeypatch):
    local, shared = tmp_path / "local", tmp_path / "shared"
    group(local, "stable")
    changing = group(local, "changing")
    original = cache.shutil.copytree

    def copy_then_mutate(source, destination, *args, **kwargs):
        result = original(source, destination, *args, **kwargs)
        if Path(source) == changing:
            (changing / "kernel.cubin").write_bytes(b"changed while copying")
        return result

    monkeypatch.setattr(cache.shutil, "copytree", copy_then_mutate)
    result = cache.publish(shared, local, KEY, "triton", partial_triton=True)
    assert result["status"] == "published"
    assert result["skipped_live_groups"] == ["changing"]
    restored = tmp_path / "restored"
    (restored / "triton").mkdir(parents=True)
    assert cache.restore(shared, restored, KEY, "triton")["status"] == "hit"
    assert (restored / "triton/stable/kernel.cubin").read_bytes() == b"compiled"
    assert not (restored / "triton/changing").exists()


@pytest.mark.parametrize("operation", ["stat", "hash"])
def test_disappearing_file_during_initial_inventory_skips_only_its_group(tmp_path, monkeypatch, operation):
    local, shared = tmp_path / "local", tmp_path / "shared"
    group(local, "a-stable")
    changing = group(local, "z-changing")
    temporary = changing / "tmp.pid_1_uuid"
    temporary.mkdir()
    disappearing = temporary / "kernel.cubin"
    disappearing.write_bytes(b"partial")
    original = Path.lstat if operation == "stat" else cache.sha256
    removed = False

    def remove_then_read(path, *args, **kwargs):
        nonlocal removed
        if Path(path) == disappearing and not removed:
            disappearing.unlink()
            removed = True
        return original(path, *args, **kwargs)

    if operation == "stat":
        monkeypatch.setattr(Path, "lstat", remove_then_read)
    else:
        monkeypatch.setattr(cache, "sha256", remove_then_read)
    result = cache.publish(shared, local, KEY, "triton", partial_triton=True)
    assert removed
    assert result["status"] == "published"
    assert result["skipped_live_groups"] == ["z-changing"]
    restored = tmp_path / "restored"
    (restored / "triton").mkdir(parents=True)
    assert cache.restore(shared, restored, KEY, "triton")["status"] == "hit"
    assert (restored / "triton/a-stable/kernel.cubin").read_bytes() == b"compiled"
    assert not (restored / "triton/z-changing").exists()


def test_uncommitted_kernel_is_skipped_until_manifest_is_present(tmp_path):
    local = tmp_path / "local"
    incomplete = local / "triton/new"
    incomplete.mkdir(parents=True)
    (incomplete / "kernel.cubin").write_bytes(b"compiled")
    first = cache.publish(tmp_path / "shared", local, KEY, "triton", partial_triton=True)
    assert first["status"] == "empty"
    assert first["skipped_live_groups"] == ["new"]
    (incomplete / "__grp__kernel.json").write_text(
        json.dumps({"child_paths": {"kernel.cubin": str(incomplete / "kernel.cubin")}})
    )
    assert cache.publish(tmp_path / "shared", local, KEY, "triton", partial_triton=True)["status"] == "published"


def test_copytree_aggregated_disappearance_skips_only_its_group(tmp_path, monkeypatch):
    local, shared = tmp_path / "local", tmp_path / "shared"
    group(local, "a-stable")
    changing = group(local, "z-changing")
    original = cache.shutil.copyfile
    removed = False

    def remove_before_copy(source, destination, **kwargs):
        nonlocal removed
        if Path(source) == changing / "kernel.cubin":
            Path(source).unlink()
            removed = True
        return original(source, destination, **kwargs)

    # Real copytree collects this per-file FileNotFoundError into shutil.Error.
    monkeypatch.setattr(cache.shutil, "copyfile", remove_before_copy)
    result = cache.publish(shared, local, KEY, "triton", partial_triton=True)
    assert removed
    assert result["status"] == "published"
    assert result["skipped_live_groups"] == ["z-changing"]
    restored = tmp_path / "restored"
    (restored / "triton").mkdir(parents=True)
    assert cache.restore(shared, restored, KEY, "triton")["status"] == "hit"
    assert (restored / "triton/a-stable/kernel.cubin").read_bytes() == b"compiled"
    assert not (restored / "triton/z-changing").exists()


def test_standalone_runtime_library_and_autotuning_are_preserved(tmp_path):
    local = tmp_path / "local"
    for name, filename in (("runtime", "cuda_utils.so"), ("tuning", "kernel.autotune.json")):
        root = local / "triton" / name
        root.mkdir(parents=True)
        (root / filename).write_bytes(b"atomic-file")
    result = cache.publish(tmp_path / "shared", local, KEY, "triton", partial_triton=True)
    assert result["status"] == "published" and result["files"] == 2
    assert not result["skipped_live_groups"]


def test_cross_group_references_remain_fail_closed(tmp_path):
    local = tmp_path / "local"
    one, two = group(local, "one"), group(local, "two")
    (one / "__grp__kernel.json").write_text(json.dumps({"child_paths": {"kernel.cubin": str(two / "kernel.cubin")}}))
    with pytest.raises(ValueError, match="cross-group"):
        cache.publish(tmp_path / "shared", local, KEY, "triton", partial_triton=True)


def test_transient_triton_write_is_not_archived(tmp_path):
    local = tmp_path / "local"
    root = group(local, "kernel")
    temporary = root / "tmp.pid_1_uuid"
    temporary.mkdir()
    (temporary / "kernel.cubin").write_bytes(b"partial")
    result = cache.publish(tmp_path / "shared", local, KEY, "triton", partial_triton=True)
    assert result["status"] == "empty"
    assert result["skipped_live_groups"] == ["kernel"]
