"""Optional Triton compilation timings through its supported listener API."""

import importlib
import json
import os
import time

from miles.utils import logger_utils
from miles.utils.compiler_cache import compiler_cache as cache

logger = logger_utils.setup_logger(__name__)


def install(local):
    knobs = importlib.import_module("triton.knobs")
    previous = knobs.compilation.listener
    if getattr(previous, "_miles_cache_observer", False):
        return

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        if previous is not None:
            previous(src=src, metadata=metadata, metadata_group=metadata_group, times=times, cache_hit=cache_hit)
        try:
            stages = dict(times.lowering_stages)
            record = {
                "event": "compile_hit" if cache_hit else "compile_miss",
                "timestamp": time.time(),
                "kernel": src.name,
                "signature": {str(key): str(value) for key, value in getattr(src, "signature", {}).items()},
                "specialization_digest": cache.digest(str(getattr(src, "constants", {}))),
                "compiler_microseconds": times.ir_initialization + sum(stages.values()) + times.store_results,
                "lowering_microseconds": stages,
            }
            with (local / f"activity-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        except (OSError, AttributeError, TypeError, ValueError) as error:
            logger.warning("Could not record compiler timing: %s", error)

    listener._miles_cache_observer = True
    knobs.compilation.listener = listener
