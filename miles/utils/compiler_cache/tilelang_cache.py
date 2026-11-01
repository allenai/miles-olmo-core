"""Opt-in TileLang persistence with a separate, compiler-qualified identity."""

import json
import os
from importlib import util
from pathlib import Path

from miles.utils import logger_utils
from miles.utils.compiler_cache import compiler_cache as cache

logger = logger_utils.setup_logger(__name__)


def compiler_identity():
    # Includes TileLang's Python lowering/export code, bundled TVM and native
    # libraries. TileLang's own version-only key does not cover all of these.
    spec = util.find_spec("tilelang")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("Cannot fingerprint the installed TileLang compiler")
    return cache.source_identity(Path(next(iter(spec.submodule_search_locations))))


def configure(policy, local):
    if not policy.get("tilelang_runtime"):
        return
    (local / "tilelang").mkdir()
    os.environ["TILELANG_CACHE_DIR"] = str(local / "tilelang")
    # Compiler scratch must not be included in a published snapshot.
    os.environ["TILELANG_TMP_DIR"] = str(local / "tilelang-tmp")
    # Persistence requires content-based native-library keys, regardless of
    # an inherited setting. This is a run-wide safety constraint.
    os.environ["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    if policy["observe"]:
        os.environ["TILELANG_JIT_DIAGNOSTICS"] = "1"


def restore(policy, report, local):
    if not policy.get("tilelang_runtime"):
        return
    identity = {
        **report["identity"],
        "compatibility_scope": "core-tilelang-v1",
        "cache_families": {"tilelang": cache.FAMILIES["tilelang"]},
        "tilelang_runtime": policy["tilelang_runtime"],
    }
    key = cache.digest(identity)
    result = {"fingerprint": key, "identity": identity}
    try:
        result["restore"] = (
            cache.restore(Path(policy["shared"]), local, key, "tilelang")
            if policy["restore"]
            else {"family": "tilelang", "status": "cold"}
        )
    except Exception as error:
        result["restore"] = {"family": "tilelang", "status": "unavailable", "reason": str(error)}
    report["tilelang"] = result
    logger.info("Core TileLang cache worker %s: %s", report["slot"], result["restore"])


def publish(report, local):
    descriptor = report.get("tilelang")
    if descriptor is None:
        return None
    result = {"restore": descriptor["restore"]}
    try:
        state_path = local / "tilelang-publication-state.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        signature = cache.metadata_signature(local / "tilelang")
        if previous.get("capped"):
            result["publish"] = previous["publish"]
        elif signature == previous.get("signature"):
            result["publish"] = {"family": "tilelang", "status": "unchanged", "reason": "local_artifacts_unchanged"}
        else:
            result["publish"] = cache.publish(
                Path(report["shared"]),
                local,
                descriptor["fingerprint"],
                "tilelang",
                max_storage_bytes=report.get("max_storage_bytes", cache.DEFAULT_MAX_STORAGE_BYTES),
            )
            status = result["publish"]["status"]
            if status in {"published", "unchanged", "empty", "storage_limit"}:
                state_path.write_bytes(
                    cache.encoded(
                        {"signature": signature, "capped": status == "storage_limit", "publish": result["publish"]}
                    )
                )
    except Exception as error:
        result["publish"] = {"family": "tilelang", "status": "rejected", "reason": f"{type(error).__name__}: {error}"}
    logger.info("Core TileLang cache publication: slot=%s %s", report["slot"], result)
    return result
