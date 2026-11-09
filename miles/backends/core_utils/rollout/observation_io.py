"""Serialize small observation writes without blocking the producer event loop."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from miles.utils import logger_utils

logger = logger_utils.setup_logger(__name__)


def _append(rows):
    for path, row in rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


class ObservationWriter:
    """One submitted write at a time; callers await completion and retain ownership.

    Group callers are bounded by the producer's admission budget. Observers
    submit one sample at a time. Cancellation joins a submitted filesystem write
    before propagating; it never abandons a thread or races a later append.
    Rows must be detached snapshots, not live producer/Sample objects.
    """

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-observations")
        self._lock = asyncio.Lock()
        self._closed = False

    async def append(self, rows):
        # Shield the lock wait as well: cancelled groups still leave a record.
        pending = asyncio.create_task(self._append_serial(rows))
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            while not pending.done():
                try:
                    await asyncio.shield(pending)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not pending.cancelled() and (error := pending.exception()) is not None:
                logger.error("Observation write failed during cancellation: %r", error)
            raise

    async def _append_serial(self, rows):
        async with self._lock:
            if self._closed:
                raise RuntimeError("Observation writer is closed")
            await asyncio.wrap_future(self._pool.submit(_append, rows))

    async def close(self):
        self._closed = True
        await asyncio.to_thread(self._pool.shutdown, wait=True)


def for_producer(producer):
    # Lazy allocation also supports producer test doubles and disabled observers.
    if getattr(producer, "_observation_writer", None) is None:
        producer._observation_writer = ObservationWriter()
    return producer._observation_writer


async def close(producer):
    writer = getattr(producer, "_observation_writer", None)
    if writer is not None:
        await writer.close()
