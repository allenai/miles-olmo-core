"""Triton artifact compatibility, separate from experiment provenance.

Triton's per-kernel key includes source/dependencies, signature, specialization,
backend and compiler options. Restoring an artifact directory does not select a
kernel. Keep the outer envelope strict about compiler/native code and hardware,
without invalidating every artifact for a Python orchestration edit.
"""

import json
import os
import uuid
from pathlib import Path

from miles.utils.compiler_cache import compiler_cache as cache

SCOPE = "core-triton-v3"
NATIVE_SUFFIXES = {".so", ".c", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
NON_KERNEL_CORE = {
    "prepared_batch_overlap",
    "layer_profile_interval",
    "reward_zero_truncated",
}


def native_source(root):
    """Native extensions are not covered by Triton's Python dependency hash."""
    files = {}
    for path in sorted(root.rglob("*")):
        if any(part in {".git", "__pycache__", ".venv"} for part in path.relative_to(root).parts):
            continue
        if path.suffix in NATIVE_SUFFIXES and path.is_file():
            if path.is_symlink():
                raise ValueError(f"Native source symlink requires a resolved source root: {path}")
            files[path.relative_to(root).as_posix()] = cache.sha256(path)
    return {"sha256": cache.digest(files), "files": len(files)}


def fingerprint(policy, hardware, compiler_env):
    """Retain complete input provenance while hashing only artifact compatibility."""
    _, identity = cache.fingerprint(
        image=policy["image"],
        runtime_lock=policy["runtime_lock"],
        sources=policy["sources"],
        model_config=policy["model_config"],
        run_config={**policy["run_config"], "worker_slot": policy["slot"]},
        toolchain=hardware,
        compiler_env=compiler_env,
        scope=cache.CORE_TRITON_SCOPE,
    )
    provenance = {key: identity.pop(key) for key in ("runtime_lock", "sources")}
    # Required, not optional: old reports lack this evidence and must not seed v3.
    identity["kernel_runtime"] = policy["kernel_runtime"]
    identity["compatibility_scope"] = SCOPE
    identity["cache_families"] = {"triton": cache.FAMILIES["triton"]}
    for key in NON_KERNEL_CORE:
        identity["compile_settings"].get("core", {}).pop(key, None)
    return cache.digest(identity), identity, provenance


def changed_fields(before, after, prefix=""):
    """Return field names, never environment values or source contents."""
    if isinstance(before, dict) and isinstance(after, dict):
        return [
            field
            for key in sorted(before.keys() | after.keys())
            for field in changed_fields(before.get(key), after.get(key), f"{prefix}.{key}" if prefix else key)
        ]
    return [prefix] if before != after else []


def explain_previous(shared, report):
    """One bounded diagnostic pointer per model/worker; never used for restore.

    Concurrent startups may replace this pointer in either order. It describes
    the last observed startup, not the last successful publication. Only exact
    fingerprint lookup is allowed to select artifacts.
    """
    scope = cache.digest({"model": report["identity"]["model_config"], "slot": report["slot"]})
    directory = Path(shared) / "diagnostics" / SCOPE
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{scope}.json"
    previous = None
    try:
        previous = json.loads(target.read_text())
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        return {"status": "previous_diagnostic_unreadable"}
    result = {"status": "no_previous_observation"}
    if previous is not None and not (
        isinstance(previous, dict) and isinstance(previous.get("identity"), dict) and "fingerprint" in previous
    ):
        return {"status": "previous_diagnostic_unreadable"}
    if previous is not None:
        differences = changed_fields(previous["identity"], report["identity"])
        result = {
            "status": "compatibility_changed" if differences else "same_compatibility",
            "previous_fingerprint": previous["fingerprint"],
            "changed_fields": differences[:32],
            "changed_field_count": len(differences),
            "provenance_changed": previous.get("provenance") != report.get("provenance"),
        }
    temporary = directory / f"{scope}-{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(cache.encoded({key: report[key] for key in ("identity", "fingerprint", "provenance")}))
    os.replace(temporary, target)
    return result
