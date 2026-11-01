"""Reuse Python-only revisions without weakening compiler/native compatibility."""

import copy
import json

import pytest
from tests.fast.compiler_cache import test_startup_cache

from miles.utils.compiler_cache import triton_compatibility as compatibility


def policy(tmp_path):
    result = test_startup_cache.policy(tmp_path)
    result["kernel_runtime"] = {"triton": {"sha256": "compiler"}, "native_sources": {"sha256": "extensions"}}
    return result


def key(policy):
    return compatibility.fingerprint(policy, {"gpu": "B300"}, {})[0]


def test_python_revision_and_orchestration_keep_artifacts(tmp_path):
    before = policy(tmp_path)
    after = copy.deepcopy(before)
    after["sources"]["miles"]["sha256"] = "new-python"
    after["sources"]["open-instruct"]["sha256"] = "new-python"
    after["runtime_lock"] = {"sources": {"miles": {"revision": "new-commit"}}}
    after["run_config"]["core"]["prepared_batch_overlap"] = True
    first = compatibility.fingerprint(before, {}, {})
    second = compatibility.fingerprint(after, {}, {})
    assert first[:2] == second[:2]
    assert first[2] != second[2]
    assert after["run_config"]["core"]["prepared_batch_overlap"] is True


@pytest.mark.parametrize(
    "change", ["compiler", "native", "image", "model", "unknown_setting", "hardware", "environment"]
)
def test_material_differences_still_miss(tmp_path, change):
    before = policy(tmp_path)
    after = copy.deepcopy(before)
    hardware, environment = {}, {}
    if change in {"compiler", "native"}:
        after["kernel_runtime"]["triton" if change == "compiler" else "native_sources"] = {"sha256": "changed"}
    elif change == "image":
        after["image"] = "sha256:" + "b" * 64
    elif change == "model":
        after["model_config"]["hidden_size"] = 4096
    elif change == "unknown_setting":
        after["run_config"]["core"]["new_kernel_switch"] = True
    elif change == "hardware":
        hardware["gpu"] = "H100"
    else:
        environment["TRITON_DEBUG"] = "hashed"
    assert compatibility.fingerprint(before, {}, {})[0] != compatibility.fingerprint(after, hardware, environment)[0]


def test_legacy_policy_cannot_silently_enter_new_scope(tmp_path):
    legacy = test_startup_cache.policy(tmp_path)
    del legacy["kernel_runtime"]
    with pytest.raises(KeyError, match="kernel_runtime"):
        compatibility.fingerprint(legacy, {}, {})


def test_native_digest_ignores_python_but_tracks_extension(tmp_path):
    (tmp_path / "driver.py").write_text("pass")
    before = compatibility.native_source(tmp_path)
    (tmp_path / "driver.py").write_text("changed")
    assert compatibility.native_source(tmp_path) == before
    (tmp_path / "kernel.so").write_bytes(b"native")
    assert compatibility.native_source(tmp_path) != before


def test_diagnostic_explains_changes_without_selecting_cache(tmp_path):
    descriptor = policy(tmp_path)
    fingerprint, identity, provenance = compatibility.fingerprint(descriptor, {}, {})
    report = dict(slot=descriptor["slot"], fingerprint=fingerprint, identity=identity, provenance=provenance)
    assert compatibility.explain_previous(tmp_path, report)["status"] == "no_previous_observation"
    assert compatibility.explain_previous(tmp_path, report)["status"] == "same_compatibility"
    changed = copy.deepcopy(report)
    changed["identity"]["compiler_env"]["TRITON_DEBUG"] = "secret-value-hash"
    result = compatibility.explain_previous(tmp_path, changed)
    assert result["changed_fields"] == ["compiler_env.TRITON_DEBUG"]
    assert "secret-value-hash" not in json.dumps(result)
