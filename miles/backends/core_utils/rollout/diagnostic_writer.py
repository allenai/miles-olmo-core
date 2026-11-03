"""Bounded diagnostic I/O outside the executor and producer event loops.

Only sampled records are copied. A spawned process owns serialization and file
writing; at most one snapshot is outstanding. Backpressure and writer failures
are surfaced at the next submission (and at shutdown), never silently dropped.
"""

import asyncio
import copy
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

from miles.ray.rollout.debug_data import sample_rollout_groups, save_debug_rollout_data


def _write(args, samples, rollout_id, metadata, trainer_model_id):
    started = time.perf_counter()
    save_debug_rollout_data(
        args, samples, rollout_id, evaluation=False, metadata=metadata, trainer_model_id=trainer_model_id
    )
    return time.perf_counter() - started


class DiagnosticWriter:
    def __init__(self, args):
        self._rate = min(args.rollout_sample_rate, 1.0)
        self._args = SimpleNamespace(
            rollout_sample_rate=1.0,
            rollout_seed=getattr(args, "rollout_seed", 0),
            save_debug_rollout_data=args.save_debug_rollout_data,
            save_debug_trajectory_data=args.save_debug_trajectory_data,
        )
        self._pool = None
        self._pending = None

    async def submit(self, samples, rollout_id, metadata, trainer_model_id):
        started = time.perf_counter()
        metrics = {}
        if self._pending is not None:
            metrics["rollout/diagnostics/previous_write_seconds"] = await asyncio.wrap_future(self._pending)
            self._pending = None
        metrics["rollout/diagnostics/backpressure_seconds"] = time.perf_counter() - started
        selected = sample_rollout_groups(
            samples, self._rate, seed=self._args.rollout_seed, rollout_id=(False, rollout_id), dataset="train"
        )
        started = time.perf_counter()
        # Conversion/custom hooks can mutate masks, rewards and metadata. Never
        # serialize live Samples concurrently; copy only the selected subset.
        snapshot, metadata = await asyncio.to_thread(copy.deepcopy, (selected, metadata))
        metrics["rollout/diagnostics/snapshot_seconds"] = time.perf_counter() - started
        metrics["rollout/diagnostics/samples"] = len(snapshot)
        if snapshot:
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            self._pending = self._pool.submit(_write, self._args, snapshot, rollout_id, metadata, trainer_model_id)
        return metrics

    async def close(self):
        try:
            if self._pending is not None:
                await asyncio.wrap_future(self._pending)
        finally:
            if self._pool is not None:
                await asyncio.to_thread(self._pool.shutdown, wait=True)
                self._pool = None
            self._pending = None


def create_writer(args):
    if getattr(args, "rollout_sample_rate", 0.0) > 0 and args.save_debug_rollout_data is not None:
        return DiagnosticWriter(args)
    return None
