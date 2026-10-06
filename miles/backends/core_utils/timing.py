"""Core worker startup timing."""

import json
import time
from contextlib import contextmanager
from pathlib import Path

from miles.backends.core_utils import infra_timeouts


@contextmanager
def startup_stage(args, name, *, device=None):
    """Per-rank initialization intervals; synchronize only explicitly GPU stages."""
    started = time.perf_counter()
    wall = time.time()
    passed = False
    try:
        with infra_timeouts.watch(f"startup stage={name} rank={getattr(args, 'rank', 0)}"):
            yield
            if device is not None:
                device.synchronize()
        passed = True
    finally:
        if getattr(args, "save", None):
            root = Path(args.save)
            root.mkdir(parents=True, exist_ok=True)
            rank = getattr(args, "rank", 0)
            with (root / f"startup_rank{rank}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(stage=name, started_unix=wall, seconds=time.perf_counter() - started, passed=passed)
                    )
                    + "\n"
                )
