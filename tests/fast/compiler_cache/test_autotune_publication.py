"""Sharing compiled entries must tolerate repeated measurements, not collisions."""

import copy
import json

import pytest
from tests.fast.compiler_cache import test_compiler_cache as helpers

from miles.utils.compiler_cache import compiler_cache as cache


def record():
    return {
        "key": [8192, "torch.bfloat16"],
        "configs_timings": [
            [{"kwargs": {"BLOCK_SIZE": 128}, "num_warps": 4, "num_stages": 3}, [1.2, 1.1, 1.3]],
            [{"kwargs": {"BLOCK_SIZE": 256}, "num_warps": 8, "num_stages": 3}, [1.4, 1.3, 1.5]],
        ],
    }


def publish_first(tmp_path):
    shared = tmp_path / "shared"
    source = helpers.private(tmp_path, "first")
    helpers.write(source / "triton", "hash/kernel.cubin")
    helpers.write(source / "triton", "hash/kernel.autotune.json", json.dumps(record()).encode())
    cache.publish(shared, source, helpers.KEY, "triton")
    return shared


def test_repeated_autotuning_preserves_first_timings_and_adds_new_kernels(tmp_path):
    shared = publish_first(tmp_path)
    source = helpers.private(tmp_path, "second")
    measured = record()
    measured["configs_timings"][0][1] = [1.5, 1.4, 1.6]
    helpers.write(source / "triton", "hash/kernel.autotune.json", json.dumps(measured).encode())
    helpers.write(source / "triton", "new-shape/kernel.cubin", b"new kernel")
    report = cache.publish(shared, source, helpers.KEY, "triton")
    assert report["status"] == "published"
    assert report["reused_autotune_records"] == 1
    restored = helpers.private(tmp_path, "restored")
    assert cache.restore(shared, restored, helpers.KEY, "triton")["status"] == "hit"
    assert json.loads((restored / "triton/hash/kernel.autotune.json").read_text()) == record()
    assert (restored / "triton/new-shape/kernel.cubin").read_bytes() == b"new kernel"


@pytest.mark.parametrize("conflict", ["key", "candidates", "format", "timing", "binary"])
def test_true_conflicts_leave_published_generation_unchanged(tmp_path, conflict):
    shared = publish_first(tmp_path)
    pointer = cache.artifact_root(shared, helpers.KEY, "triton") / "CURRENT"
    before = pointer.read_bytes()
    source = helpers.private(tmp_path, "second")
    changed = copy.deepcopy(record())
    if conflict == "key":
        changed["key"][0] = 16384
    elif conflict == "candidates":
        changed["configs_timings"][0][0]["num_warps"] = 8
    elif conflict == "format":
        changed["unknown"] = True
    elif conflict == "timing":
        changed["configs_timings"][0][1] = "invalid"
    if conflict == "binary":
        helpers.write(source / "triton", "hash/kernel.cubin", b"different binary")
    else:
        helpers.write(source / "triton", "hash/kernel.autotune.json", json.dumps(changed).encode())
    with pytest.raises(ValueError, match="Conflicting compiler-cache key"):
        cache.publish(shared, source, helpers.KEY, "triton")
    assert pointer.read_bytes() == before
