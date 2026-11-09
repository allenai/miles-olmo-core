"""Serialize small observation writes without blocking the producer event loop."""

import asyncio
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from miles.utils import logger_utils

logger = logger_utils.setup_logger(__name__)


def _append(rows):
    for path, row in rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


async def _finish_on_cancel(awaitable):
    """Join owned admission/flush work even under repeated cancellation."""
    pending = asyncio.create_task(awaitable)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not pending.cancelled() and (error := pending.exception()) is not None:
            logger.error("Observation operation failed during cancellation: %r", error)
        raise


class ObservationWriter:
    """FIFO background writes with at most 128 outstanding batches by default.

    append() waits only for queue capacity, never for its own filesystem write.
    The cap includes the active batch. A single executor thread preserves order;
    the admission lock keeps full-queue waiters in order. Cancellation retains
    the caller's row by finishing admission, without waiting for that row's I/O.
    Rows must be detached snapshots. Stop/join submitters before close().
    """

    def __init__(self, max_pending=128):
        if max_pending < 1:
            raise ValueError("Observation queue capacity must be positive")
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-observations")
        self._lock = asyncio.Lock()
        self._pending = deque()
        self._max_pending = max_pending
        self._closed = False

    def _reap(self):
        while self._pending and self._pending[0].done():
            future = self._pending.popleft()
            if (error := future.exception()) is not None:
                logger.error("Observation write unavailable: %r", error)

    async def append(self, rows):
        if rows:
            await _finish_on_cancel(self._enqueue(tuple(rows)))

    async def _enqueue(self, rows):
        async with self._lock:
            self._reap()
            while not self._closed and len(self._pending) >= self._max_pending:
                try:
                    await asyncio.wrap_future(self._pending[0])
                except Exception:
                    pass  # Report the failed batch once in _reap; retain later rows.
                self._reap()
            if self._closed:
                raise RuntimeError("Observation writer is closed")
            self._pending.append(self._pool.submit(_append, rows))

    async def _flush(self):
        await asyncio.gather(*(asyncio.wrap_future(f) for f in tuple(self._pending)), return_exceptions=True)
        self._reap()

    async def flush(self):
        """Drain batches accepted before this call; report failures without raising."""
        await _finish_on_cancel(self._flush())

    async def close(self):
        self._closed = True
        try:
            await _finish_on_cancel(asyncio.to_thread(self._pool.shutdown, wait=True))
        finally:
            self._reap()


def for_producer(producer):
    # Lazy allocation also supports producer test doubles and disabled observers.
    if getattr(producer, "_observation_writer", None) is None:
        producer._observation_writer = ObservationWriter()
    return producer._observation_writer


async def close(producer):
    writer = getattr(producer, "_observation_writer", None)
    if writer is not None:
        await writer.close()
