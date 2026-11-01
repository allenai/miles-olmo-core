"""Verified immutable compiler-cache generations and private mutable restores."""

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

SCHEMA = 1
DEFAULT_MAX_STORAGE_BYTES = 8 * 1024**3
FAMILIES = {
    "triton": "TRITON_CACHE_DIR",
    "tilelang": "TILELANG_CACHE_DIR",
    "torchinductor": "TORCHINDUCTOR_CACHE_DIR",
    "fa4": "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
    "deepep": "EP_JIT_CACHE_DIR",
    "deepgemm": "DG_JIT_CACHE_DIR",
}
HEX = re.compile(r"[0-9a-f]{64}")
SOURCE_SUFFIXES = {".py", ".c", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".so"}
CORE_TRITON_SCOPE = "core-triton-v2"
NON_COMPILE_MILES = {
    "hf_checkpoint",
    "save",
    "load",
    "prompt_data",
    "eval_prompt_data",
    "save_debug_rollout_data",
    "rollout_sample_rate",
    "save_debug_train_data",
    "wandb_project",
    "wandb_group",
    "wandb_entity",
    "wandb_run_name",
}

# Explicit exclusions: new/unknown settings continue to invalidate by default.
# Keep this catalogue aligned with docs/miles/compiler-cache.md.
NON_COMPILE_CORE = {
    "compiler_cache",
    "compiler_cache_root",
    "compiler_cache_restore",
    "compiler_cache_diagnostics",
    "compiler_cache_max_storage_bytes",
    "compiler_cache_publish_interval_seconds",
    "checkpoint_profile",
    "checkpoint_thread_count",
    "checkpoint_process_count",
    "checkpoint_compact_storage",
    "checkpoint_dedup_save_to_lowest_rank",
    "checkpoint_constant_memory_planning",
    "checkpoint_keep_last",
    "checkpoint_keep_every",
    "diagnostic_interval",
    "pipeline_observation_interval",
    "replay_diagnostics",
    "max_train_rollout_logprob_abs_diff",
    "engine_drain_timeout",
    "engine_update_timeout",
    "refresh_request_timeout",
    "scoring_check_interval",
    "scoring_check_tolerance",
    "records_root",
    "records_responses",
    "records_response_sample_rate",
    "selection_table",
    "selection_sha256",
    "reward_config",
}
NON_COMPILE_MILES |= {
    "sglang_host",
    "sglang_port",
    "sglang_log_level",
    "sglang_log_level_http",
    "sglang_log_requests",
    "sglang_log_requests_level",
    "sglang_log_requests_format",
    "sglang_show_time_cost",
    "sglang_enable_metrics",
    "sglang_enable_metrics_for_all_schedulers",
    "sglang_collect_traces",
    "sglang_otlp_traces_endpoint",
    "sglang_watchdog_timeout",
    "sglang_dist_timeout",
    "sglang_download_dir",
    "sglang_file_storage_path",
}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def source_identity(root, *, excluded_paths=(), exclude_tests=False):
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".venv"} for part in relative.parts):
            continue
        if any(relative.is_relative_to(excluded) for excluded in excluded_paths):
            continue
        if exclude_tests and (
            "tests" in relative.parts or path.name.startswith("test_") or path.name.endswith("_test.py")
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"Source code symlink requires an explicit resolved source root: {path}")
        if path.suffix in SOURCE_SUFFIXES and path.is_file():
            files[path.relative_to(root).as_posix()] = sha256(path)
    if not files:
        raise ValueError(f"No runtime source files under {root}")
    return {"sha256": digest(files), "files": len(files)}


def core_application_source(root):
    """Keep Core's application code; omit standalone evaluation, OPD and tests."""
    return source_identity(root, excluded_paths=("miles/evaluation", "miles/distillation"), exclude_tests=True)


def _core_triton_inputs(runtime_lock, settings, toolchain):
    """Normalize proven non-kernel differences only for the Core Triton path.

    Triton's own entry keys still distinguish compiled specializations. Preserve
    unknown fields, trainer ranks, parallel geometry and compiler/environment
    controls; this is not a compatibility policy for the experimental families.
    """
    lock = json.loads(encoded(runtime_lock))
    hardware = json.loads(encoded(toolchain))
    unused_sources = {"megatron-lm", "megatron-bridge", "olmo-megatron"}
    if "source_patches" in lock:
        lock["source_patches"] = [
            patch
            for patch in lock["source_patches"]
            if not (patch.get("source") in unused_sources and patch.get("target") in unused_sources)
        ]
        if not lock["source_patches"]:
            del lock["source_patches"]
    referenced_sources = {
        patch.get(field) for patch in lock.get("source_patches", []) for field in ("source", "target")
    }
    for name in unused_sources - referenced_sources:
        lock.get("sources", {}).pop(name, None)
    for key in ("max_policy_lag", "max_run_seconds"):
        settings.get("core", {}).pop(key, None)
    miles = settings.get("miles", {})
    for key in (
        "global_batch_size",
        "rollout_batch_size",
        "n_samples_per_prompt",
        "rollout_num_gpus",
        "sglang_server_concurrency",
        "async_max_concurrent_samples",
    ):
        miles.pop(key, None)
    # Only known single-GPU serving workers share a role key. Physical slot IDs
    # and other visible identical GPUs do not change that engine's kernels.
    slot = settings.get("worker_slot", "")
    if miles.get("rollout_num_gpus_per_engine") == 1 and re.fullmatch(
        r"inference-engine-(?:all-)?\d+-\d+-\d+-\d+", slot
    ):
        devices = hardware.get("torch_hardware", {}).get("gpus", [])
        if devices and all(device == devices[0] for device in devices):
            settings["worker_slot"] = "inference-engine-tp1"
            hardware["torch_hardware"]["gpus"] = devices[:1]
    return lock, settings, hardware


def fingerprint(*, image, runtime_lock, sources, model_config, run_config, toolchain, compiler_env, scope="strict"):
    if not re.fullmatch(r"(?:sha256:[0-9a-f]{64}|[0-9A-HJKMNP-TV-Z]{26})", image):
        raise ValueError("Use an immutable Docker SHA256 or Beaker image ID")
    required = {"olmo-core", "miles", "open-instruct", "olmo-sglang"}
    if not required <= sources.keys():
        raise ValueError(f"Fingerprint requires runtime source trees: {sorted(required)}")
    settings = json.loads(encoded(run_config))
    if settings.get("core", {}).get("model_config"):
        raise ValueError("External Core model-config files are not qualified by this wrapper")
    for key in NON_COMPILE_MILES:
        settings.get("miles", {}).pop(key, None)
    for key in NON_COMPILE_CORE:
        settings.get("core", {}).pop(key, None)
    if scope == CORE_TRITON_SCOPE:
        runtime_lock, settings, toolchain = _core_triton_inputs(runtime_lock, settings, toolchain)
    elif scope != "strict":
        raise ValueError(f"Unknown compiler-cache compatibility scope: {scope}")
    identity = {
        "schema_version": SCHEMA,
        "backend": "open-instruct-miles-olmo-core",
        "image": image,
        "runtime_lock": runtime_lock,
        "sources": sources,
        "model_config": model_config,
        "compile_settings": settings,
        "toolchain": toolchain,
        "compiler_env": compiler_env,
        "cache_families": FAMILIES,
    }
    if scope != "strict":
        identity["compatibility_scope"] = scope
    return digest(identity), identity


def local_environment(root):
    root = root.resolve()
    if str(root).startswith(("/weka/", "/net/")):
        raise ValueError("Mutable compiler cache must be node-local")
    mount = None
    mount_length = -1
    mountinfo = Path("/proc/self/mountinfo")
    if mountinfo.exists():
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            path = Path(fields[4].replace("\\040", " "))
            if root.is_relative_to(path) and len(str(path)) > mount_length:
                mount, mount_length = fields[fields.index("-") + 1], len(str(path))
        if mount in {"weka", "wekafs", "nfs", "nfs4", "cifs", "lustre", "ceph", "gpfs"}:
            raise ValueError(f"Mutable compiler cache cannot use shared filesystem {mount}")
    env = {}
    for family, variable in FAMILIES.items():
        directory = root / family
        directory.mkdir(parents=True, exist_ok=False)
        env[variable] = str(directory)
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    return env


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Cache contains a link or special file: {path}")
        result[path.relative_to(root).as_posix()] = {
            "sha256": sha256(path),
            "size": info.st_size,
            "executable": bool(info.st_mode & 0o111),
        }
    return result


def relocate_triton(root, destination=None):
    """Canonical relative group paths on publish; private absolute paths on restore."""
    count = 0
    for path in root.rglob("__grp__*"):
        data = json.loads(path.read_text())
        children = data.get("child_paths")
        if set(data) != {"child_paths"} or not isinstance(children, dict):
            raise ValueError("Unknown Triton group metadata format")
        converted = {}
        for name, value in children.items():
            child = Path(value)
            if destination is None:
                if not child.is_absolute() or not child.is_relative_to(root):
                    raise ValueError("Triton group references another cache root")
                relative = child.relative_to(root)
            else:
                if child.is_absolute() or ".." in child.parts:
                    raise ValueError("Unsafe relative Triton cache group")
                relative = child
            if not (root / relative).is_file():
                raise ValueError("Triton group references a missing artifact")
            converted[name] = str(destination / relative) if destination else relative.as_posix()
        path.write_bytes(encoded({"child_paths": converted}))
        count += 1
    return count


def _snapshot_completed_triton_groups(source, destination):
    """Copy independently complete groups while another kernel is compiling.

    Triton commits a kernel's child-path manifest last. Standalone runtime shared
    libraries and autotuning records are single-file atomic cache writes. Unknown
    formats remain cold; cross-group references are not qualified here.
    """
    skipped = []
    destination.mkdir(parents=True, exist_ok=True)
    for group in sorted(source.iterdir()):
        if group.is_symlink() or not group.is_dir():
            raise ValueError("Partial Triton snapshots require directory-isolated kernel groups")
        before = inventory(group)
        manifests = [name for name in before if PurePosixPath(name).name.startswith("__grp__")]
        standalone = len(before) == 1 and next(iter(before)).endswith((".so", ".autotune.json"))
        transient = any(any(part.startswith("tmp.") for part in PurePosixPath(name).parts) for name in before)
        if not before or transient or (not manifests and not standalone):
            skipped.append(group.name)
            continue
        with tempfile.TemporaryDirectory(prefix=".kernel-", dir=destination.parent) as temporary:
            staged = Path(temporary) / group.name
            try:
                shutil.copytree(group, staged)
                if inventory(staged) != before or inventory(group) != before:
                    skipped.append(group.name)
                    continue
            except FileNotFoundError:
                skipped.append(group.name)
                continue
            for name in manifests:
                data = json.loads((staged / name).read_text())
                if set(data) != {"child_paths"} or not isinstance(data["child_paths"], dict):
                    raise ValueError("Unknown Triton group metadata format")
                for child in data["child_paths"].values():
                    original = Path(child)
                    if not original.is_absolute() or not original.is_relative_to(group):
                        raise ValueError("Partial Triton snapshot cannot include cross-group metadata")
                    if original.relative_to(group).as_posix() not in before:
                        raise ValueError("Triton group references a missing artifact")
            os.rename(staged, destination / group.name)
    return skipped


def snapshot(source, destination, family, *, partial_triton=False):
    if family == "tilelang" and any(path.is_file() for path in source.glob("**/.staging/**/*")):
        raise ValueError("TileLang publication deferred: an atomic kernel export is in progress")
    skipped = []
    if partial_triton and family == "triton" and all(p.is_dir() and not p.is_symlink() for p in source.iterdir()):
        skipped = _snapshot_completed_triton_groups(source, destination)
    else:
        before = inventory(source)
        shutil.copytree(source, destination, dirs_exist_ok=True)
        if inventory(destination) != before or inventory(source) != before:
            raise ValueError("Compiler cache changed during snapshot")
    if family == "tilelang" and any(path.is_file() for path in source.glob("**/.staging/**/*")):
        raise ValueError("TileLang publication deferred: an atomic kernel export is in progress")
    if family == "triton":
        # Relocate copied absolute paths through the original namespace before
        # canonicalizing them; only the documented group JSON is rewritten.
        for path in destination.rglob("__grp__*"):
            data = json.loads(path.read_text())
            if set(data) != {"child_paths"} or not isinstance(data["child_paths"], dict):
                raise ValueError("Unknown Triton group metadata format")
            changed = {}
            for name, value in data["child_paths"].items():
                original = Path(value)
                if not original.is_absolute() or not original.is_relative_to(source):
                    raise ValueError("Triton group references another cache root")
                changed[name] = str(destination / original.relative_to(source))
            path.write_bytes(encoded({"child_paths": changed}))
        relocate_triton(destination)
    marker = str(source.parent).encode()
    for path in destination.rglob("*"):
        if not path.is_file():
            continue
        tail = b""
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                data = tail + chunk
                if marker in data:
                    raise ValueError(f"Unqualified embedded private-cache path: {family}/{path.name}")
                tail = data[-len(marker) :]
    return skipped


def artifact_root(shared, key, family):
    if not HEX.fullmatch(key) or family not in FAMILIES:
        raise ValueError("Invalid fingerprint or cache family")
    validate_shared_root(shared)
    return shared / f"core-v{SCHEMA}" / key / family


def validate_shared_root(shared):
    """Validate retention policy before creating reports or cache artifacts."""
    if not shared.is_absolute():
        raise ValueError("Compiler cache root must be an absolute path")
    resolved = shared.resolve()
    if resolved.is_relative_to("/weka") and not any(
        re.fullmatch(r"tmp-[1-9][0-9]*[hdwmy]", part) for part in resolved.parts
    ):
        raise ValueError("WEKA cache root requires an expiry component such as tmp-30d")


def extract_verified(base, key, family, destination, generation=None):
    pointer = base / "CURRENT"
    if generation is None:
        if pointer.is_symlink():
            raise ValueError("Cache pointer must not be a symlink")
        if not pointer.exists():
            return None
        generation = pointer.read_text().strip()
    if not HEX.fullmatch(generation):
        raise ValueError("Invalid cache generation pointer")
    selected = base / "generations" / generation
    if any(path.is_symlink() for path in (selected, selected / "manifest.json", selected / "cache.tar.gz")):
        raise ValueError("Cache generation must not contain symlinks")
    manifest = json.loads((selected / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Malformed cache manifest")
    for name, record in manifest["files"].items():
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("Unsafe cache manifest path")
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("sha256"), str)
            or not HEX.fullmatch(record["sha256"])
            or type(record.get("size")) is not int
            or record["size"] < 0
            or type(record.get("executable")) is not bool
        ):
            raise ValueError("Malformed cache file inventory")
    expected = {"schema_version": SCHEMA, "fingerprint": key, "family": family, "generation": generation}
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError("Incompatible cache generation metadata")
    if digest(manifest["files"]) != generation:
        raise ValueError("Cache file inventory digest changed")
    archive = selected / "cache.tar.gz"
    if sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("Cache archive checksum changed")
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or str(path) != member.name or not member.isfile():
                raise ValueError("Unsafe cache archive member")
            if member.name in seen or member.name not in manifest["files"]:
                raise ValueError("Duplicate or unlisted cache archive member")
            seen.add(member.name)
            if member.size != manifest["files"][member.name]["size"]:
                raise ValueError("Cache archive member size changed")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError("Missing cache archive member contents")
            with stream, target.open("xb") as output:
                shutil.copyfileobj(stream, output)
            target.chmod(0o700 if manifest["files"][member.name]["executable"] else 0o600)
    if inventory(destination) != manifest["files"]:
        raise ValueError("Restored cache file inventory changed")
    return {
        "generation": generation,
        "archive_bytes": archive.stat().st_size,
        "local_bytes": sum(item["size"] for item in manifest["files"].values()),
    }


def record_use(base, generation):
    """Best-effort recency for future generation-level eviction, not per kernel.

    A zero-byte marker keeps usage metadata out of immutable manifests/archives
    and does not consume the payload budget. It is not an eviction/TTL policy.
    """
    marker = base / "generations" / generation / ".last_used"
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.utime(fd, None)
        finally:
            os.close(fd)
        return True
    except OSError:
        # Read-only caches must remain usable even if recency cannot be recorded.
        return False


def restore(shared, local, key, family):
    started = time.monotonic()
    destination = local / family
    if any(destination.iterdir()):
        raise ValueError("Restore requires a fresh private cache directory")
    try:
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=local) as directory:
            stage = Path(directory)
            evidence = extract_verified(artifact_root(shared, key, family), key, family, stage)
            if evidence is None:
                return {
                    "family": family,
                    "status": "miss",
                    "reason": "no_published_generation",
                    "seconds": time.monotonic() - started,
                }
            relocated = relocate_triton(stage, destination) if family == "triton" else 0
            evidence["relocated_groups"] = relocated
            destination.rmdir()
            os.replace(stage, destination)
        evidence["recency_recorded"] = record_use(artifact_root(shared, key, family), evidence["generation"])
        return {"family": family, "status": "hit", "seconds": time.monotonic() - started, **evidence}
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        return {"family": family, "status": "rejected", "seconds": time.monotonic() - started, "reason": str(error)}


def storage_bytes(root):
    """Logical file bytes, including old generations and interrupted uploads."""
    total = 0
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Cache storage contains a link or special file")
        total += info.st_size
    return total


def metadata_signature(root):
    """Cheap change detection; publication still verifies actual file contents."""
    files = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Cache contains a link or special file")
        files[path.relative_to(root).as_posix()] = (info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return digest(files)


def _same_autotune_candidates(existing, incoming):
    """Allow timing noise for identical Triton tuning keys and candidates.

    Keep the first published measurements. A different candidate set, unknown
    format or malformed timing remains a conflict rather than an implicit repair.
    """

    def candidates(path):
        data = json.loads(path.read_text())
        if set(data) != {"key", "configs_timings"} or not isinstance(data["key"], list) or not data["key"]:
            raise ValueError("Unknown autotune record")
        if not isinstance(data["configs_timings"], list) or not data["configs_timings"]:
            raise ValueError("Missing autotune candidates")
        configs = []
        for config, timing in data["configs_timings"]:
            if not isinstance(config, dict) or not config:
                raise ValueError("Invalid autotune candidate")
            values = timing if isinstance(timing, list) else [timing]
            if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Invalid autotune timing")
            configs.append(config)
        return data["key"], configs

    try:
        return candidates(existing) == candidates(incoming)
    except (ValueError, TypeError, KeyError, OSError):
        return False


def _conflicting_triton_groups(existing, incoming, stage, incoming_root):
    """Reject whole incoming kernel groups; never mix their files with old bytes."""
    conflicts = set()
    for name in existing.keys() & incoming.keys():
        if existing[name] == incoming[name]:
            continue
        if (
            name.endswith(".autotune.json")
            and existing[name]["executable"] == incoming[name]["executable"]
            and _same_autotune_candidates(stage / name, incoming_root / name)
        ):
            continue
        conflicts.add(PurePosixPath(name).parts[0])
    # Per-group rejection is safe only for self-contained groups. Preserve the
    # strict failure for unknown metadata that references another kernel group.
    if conflicts:
        for root in (stage, incoming_root):
            for path in root.rglob("__grp__*"):
                group = path.relative_to(root).parts[0]
                children = json.loads(path.read_text())["child_paths"].values()
                if any(PurePosixPath(child).parts[0] != group for child in children):
                    raise ValueError("Cannot isolate conflicting cross-group Triton metadata")
    return conflicts


def publish(
    shared,
    local,
    key,
    family,
    *,
    progress=None,
    max_storage_bytes=DEFAULT_MAX_STORAGE_BYTES,
    isolate_triton_conflicts=False,
    partial_triton=False,
):
    """Publish a verified snapshot, bounded across all families of this key.

    Live Triton writers are allowed: validation rejects concurrent changes, or
    skips individual changing groups with the qualified partial snapshot policy.
    Other families require their own writer synchronization before publication.
    """
    if (isolate_triton_conflicts or partial_triton) and family != "triton":
        raise ValueError("Partial snapshots and conflict isolation are qualified only for Triton")
    if type(max_storage_bytes) is not int or max_storage_bytes < 0:
        raise ValueError("max_storage_bytes must be a nonnegative integer")
    started = time.monotonic()
    phases = {}
    reused_autotune_records = 0
    phase = "prepare"
    phase_started = started

    def mark(name):
        nonlocal phase, phase_started
        now = time.monotonic()
        phases[phase] = phases.get(phase, 0) + now - phase_started
        phase, phase_started = name, now
        if progress is not None:
            progress({"phase": name, "seconds": now - started, "phase_seconds": dict(phases)})

    base = artifact_root(shared, key, family)
    base.mkdir(parents=True, exist_ok=True)
    # One lock for the entire key: concurrent runs and different families share
    # the same allowance. Never budget from a run-local upload counter.
    key_root = base.parent
    with (key_root / ".publish.lock").open("a+b") as lock:
        mark("lock_wait")
        fcntl.flock(lock, fcntl.LOCK_EX)
        used_bytes = storage_bytes(key_root)

        def limited(required_bytes=0):
            return {
                "family": family,
                "status": "storage_limit",
                "reason": "cache_key_storage_cap",
                "storage_bytes": used_bytes,
                "required_bytes": required_bytes,
                "max_storage_bytes": max_storage_bytes,
                "seconds": time.monotonic() - started,
            }

        if used_bytes >= max_storage_bytes:
            return limited()
        mark("merge")
        with tempfile.TemporaryDirectory(prefix=".merge-", dir=local) as directory:
            stage = Path(directory)
            # Corrupt current generations are never overwritten/repaired implicitly.
            extract_verified(base, key, family, stage)
            existing = inventory(stage)
            with tempfile.TemporaryDirectory(prefix=".incoming-", dir=local) as incoming_directory:
                incoming_root = Path(incoming_directory)
                skipped = snapshot(local / family, incoming_root, family, partial_triton=partial_triton)
                incoming = inventory(incoming_root)
                conflicts = (
                    _conflicting_triton_groups(existing, incoming, stage, incoming_root)
                    if isolate_triton_conflicts
                    else set()
                )
                for name, record in incoming.items():
                    if PurePosixPath(name).parts[0] in conflicts:
                        continue
                    if name in existing and existing[name] != record:
                        if (
                            family == "triton"
                            and name.endswith(".autotune.json")
                            and existing[name]["executable"] == record["executable"]
                            and _same_autotune_candidates(stage / name, incoming_root / name)
                        ):
                            reused_autotune_records += 1
                            continue
                        raise ValueError(f"Conflicting compiler-cache key: {family}/{name}")
                    if name not in existing:
                        target = stage / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(incoming_root / name, target)
            files = inventory(stage)
            if not files:
                return {
                    "family": family,
                    "status": "empty",
                    "skipped_live_groups": skipped,
                    "seconds": time.monotonic() - started,
                }
            generation = digest(files)
            generations = base / "generations"
            generations.mkdir(exist_ok=True)
            target = generations / generation
            if target.exists():
                mark("verify_existing")
                with tempfile.TemporaryDirectory(prefix=".verify-", dir=local) as verification:
                    extract_verified(base, key, family, Path(verification), generation=generation)
            else:
                mark("archive_local")
                with tempfile.TemporaryDirectory(prefix=".archive-", dir=local) as temporary:
                    prepared = Path(temporary)
                    archive = prepared / "cache.tar.gz"
                    with tarfile.open(archive, "w:gz", compresslevel=1) as output:
                        for name in files:
                            output.add(stage / name, arcname=name, recursive=False)
                    manifest = {
                        "schema_version": SCHEMA,
                        "fingerprint": key,
                        "family": family,
                        "generation": generation,
                        "files": files,
                        "archive_sha256": sha256(archive),
                        "created_unix": time.time(),
                    }
                    (prepared / "manifest.json").write_bytes(encoded(manifest))
                    # Account for the archive, manifest, and temporary CURRENT
                    # pointer before writing any new payload to shared storage.
                    required_bytes = storage_bytes(prepared) + len(generation) + 1
                    if used_bytes + required_bytes > max_storage_bytes:
                        return limited(required_bytes)
                    mark("upload")
                    # Only two bulk files cross the shared filesystem. All tree
                    # extraction, relocation, hashing and compression stay local.
                    with tempfile.TemporaryDirectory(prefix=".publish-", dir=generations) as upload_directory:
                        upload = Path(upload_directory)
                        shutil.copy2(archive, upload / archive.name)
                        shutil.copy2(prepared / "manifest.json", upload / "manifest.json")
                        os.rename(upload, target)
            if storage_bytes(key_root) + len(generation) + 1 > max_storage_bytes:
                return limited(len(generation) + 1)
            mark("pointer")
            current = base / "CURRENT"
            old = current.read_text().strip() if current.exists() else None
            with tempfile.NamedTemporaryFile("w", dir=base, prefix=".CURRENT-", delete=False) as stream:
                stream.write(generation + "\n")
                pointer = Path(stream.name)
            os.replace(pointer, current)
            recency_recorded = record_use(base, generation)
            mark("complete")
            return {
                "family": family,
                "status": "unchanged" if old == generation else "published",
                "generation": generation,
                "local_bytes": sum(item["size"] for item in files.values()),
                "files": len(files),
                "archive_bytes": (target / "cache.tar.gz").stat().st_size,
                "storage_bytes": storage_bytes(key_root),
                "max_storage_bytes": max_storage_bytes,
                "recency_recorded": recency_recorded,
                "reused_autotune_records": reused_autotune_records,
                "rejected_incoming_groups": sorted(conflicts),
                "skipped_live_groups": skipped,
                "phase_seconds": phases,
                "seconds": time.monotonic() - started,
            }
