"""Request completion boundaries from HTTP workers, without response contents.

Worker CPU counters are process-wide and cumulative: overlapping request CPU
costs must not be inferred by subtracting per-request start/end counters.
"""

import json
import os
import socket
import time
from pathlib import Path

from miles.utils import logger_utils

logger = logger_utils.setup_logger(__name__)


class PostObserver:
    def __init__(self, directory):
        self._active = 0
        self._path = None
        if directory:
            self._path = Path(directory) / f"http_post_worker_{socket.gethostname()}_{os.getpid()}.jsonl"

    def start(self, payload):
        self._active += 1
        if self._path is None:
            return None
        return dict(request_id=(payload or {}).get("rid"), request_started_unix=time.time(), started=time.monotonic())

    def finish(self, record, result):
        self._active -= 1
        if record is None or self._path is None:
            return
        record.update(
            response_ready_unix=time.time(),
            request_seconds=time.monotonic() - record.pop("started"),
            response_bytes=record.get("response_bytes", len(result) if isinstance(result, bytes) else None),
            succeeded=result is not None,
            worker_active_requests=self._active,
            worker_process_cpu_seconds=time.process_time(),
        )
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception("HTTP worker observation failed; disabling this worker's observer")
            self._path = None
