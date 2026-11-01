"""One binary collision must not discard independent new kernels or mix groups."""

import json

import pytest
from tests.fast.compiler_cache import test_compiler_cache as helpers

from miles.utils.compiler_cache import compiler_cache as cache


def test_reject_entire_conflicting_group_and_publish_independent_kernel(tmp_path):
    shared = tmp_path / "shared"
    first = helpers.private(tmp_path, "first")
    helpers.write(first / "triton", "old/kernel.cubin", b"original")
    cache.publish(shared, first, helpers.KEY, "triton")
    second = helpers.private(tmp_path, "second")
    helpers.write(second / "triton", "old/kernel.cubin", b"conflicting")
    helpers.write(second / "triton", "old/new_metadata.json", b"must-not-be-mixed")
    helpers.write(second / "triton", "new/kernel.cubin", b"independent")
    result = cache.publish(shared, second, helpers.KEY, "triton", isolate_triton_conflicts=True)
    assert result["status"] == "published"
    assert result["rejected_incoming_groups"] == ["old"]
    restored = helpers.private(tmp_path, "restored")
    assert cache.restore(shared, restored, helpers.KEY, "triton")["status"] == "hit"
    assert (restored / "triton/old/kernel.cubin").read_bytes() == b"original"
    assert not (restored / "triton/old/new_metadata.json").exists()
    assert (restored / "triton/new/kernel.cubin").read_bytes() == b"independent"


def test_cross_group_metadata_remains_a_strict_rejection(tmp_path):
    shared = tmp_path / "shared"
    first = helpers.private(tmp_path, "first")
    helpers.write(first / "triton", "old/kernel.cubin", b"original")
    cache.publish(shared, first, helpers.KEY, "triton")
    second = helpers.private(tmp_path, "second")
    helpers.write(second / "triton", "old/kernel.cubin", b"conflicting")
    helpers.write(
        second / "triton",
        "other/__grp__kernel.json",
        json.dumps({"child_paths": {"kernel.cubin": str(second / "triton/old/kernel.cubin")}}).encode(),
    )
    with pytest.raises(ValueError, match="cross-group"):
        cache.publish(shared, second, helpers.KEY, "triton", isolate_triton_conflicts=True)


def test_other_compiler_families_cannot_opt_in(tmp_path):
    with pytest.raises(ValueError, match="only for Triton"):
        cache.publish(tmp_path, tmp_path, helpers.KEY, "fa4", isolate_triton_conflicts=True)
