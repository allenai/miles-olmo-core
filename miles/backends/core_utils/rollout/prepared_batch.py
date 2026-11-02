"""One owned, future-version batch prepared while the current update trains."""

import asyncio
import time
from contextlib import suppress

from miles.utils import async_utils, logger_utils
from miles.utils.data import remove_rollout_data_refs
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.tracking_utils import tracking

logger = logger_utils.setup_logger(__name__)


def enabled(args):
    """Only the single-update, disaggregated refresh schedule predicts v + 1."""
    core = getattr(args, "olmo_core", None)
    return (
        getattr(core, "prepared_batch_overlap", False)
        and getattr(core, "publication_mode", None) == "refresh"
        and args.fully_async
        and not args.offload_rollout
        and not getattr(args, "colocate", False)
        and args.rollout_batch_size * args.n_samples_per_prompt == args.global_batch_size
        and args.update_weights_interval == 1
        and args.async_data_buffer_capacity_factor >= 1
        and not any(
            getattr(args, name, False)
            for name in (
                "debug_train_only",
                "debug_rollout_only",
                "debug_skip_weight_update",
                "load_debug_rollout_data",
            )
        )
    )


class PreparedBatch:
    """Executor-loop ownership; producer reservations stay on the producer loop."""

    def __init__(self, executor, rollout_id):
        if not enabled(executor.args):
            raise ValueError("Prepared batches require one-update, disaggregated Core refresh")
        current = executor._weight_versions_of_model_id.get(None)
        if type(current) is not int:
            raise RuntimeError("Prepared batches require a published numeric policy version")
        self.executor = executor
        self.rollout_id = rollout_id
        self.version = current + 1
        self.started = time.monotonic()
        self.finished = None
        self._release = None
        self.task = asyncio.create_task(self._prepare())

    async def _prepare(self):
        try:
            return await self.executor._prepare_rollout(self.rollout_id, weight_version=self.version, reserve=True)
        finally:
            self.finished = time.monotonic()

    async def consume(self, rollout_id):
        if rollout_id != self.rollout_id:
            raise RuntimeError(f"Prepared rollout {self.rollout_id} cannot serve rollout {rollout_id}")
        waited_at = time.monotonic()
        batch = await asyncio.shield(self.task)
        # Publication may have progressed while collection waited for generations.
        actual = self.executor._weight_versions_of_model_id.get(None)
        if actual != self.version:
            raise RuntimeError(f"Prepared batch expected policy {self.version}, published policy is {actual}")
        # Keep the transfer result even if this consumer is cancelled. Prompts
        # remain in the checkpoint ledger until the no-await commit below.
        self._release = asyncio.wrap_future(
            async_utils.submit(self.executor.generate_rollout.release_reservation(consume=True))
        )
        groups = await asyncio.shield(self._release)
        self._log_timings(rollout_id, waited_at)
        # The data source serializes ledger mutations with its cursor lock.
        # No await may separate acknowledgement from executor ownership handoff.
        self.executor.data_source.acknowledge_groups(groups)
        return batch

    def _log_timings(self, rollout_id, waited_at):
        # Lightweight timing metrics remain enabled independently of diagnostic
        # payload writing. Telemetry must never discard an otherwise ready batch.
        try:
            self._write_timings(rollout_id, waited_at)
        except Exception:
            with suppress(Exception):
                logger.warning("Could not record prepared-batch timings for rollout %s", rollout_id, exc_info=True)

    def _write_timings(self, rollout_id, waited_at):
        timings = {
            "preparation_seconds": self.finished - self.started,
            "consumer_wait_seconds": time.monotonic() - waited_at,
            "ready_before_consumer_seconds": max(0.0, waited_at - self.finished),
            "expected_policy_version": self.version,
        }
        logger.info("Prepared batch timing: rollout_id=%s %s", rollout_id, timings)
        tracking.log(
            self.executor.args,
            {
                "rollout/step": compute_rollout_step(self.executor.args, rollout_id),
                **{f"rollout/prepared_batch/{key}": value for key, value in timings.items()},
            },
            step_key="rollout/step",
        )

    async def close(self):
        """Cancel collection, join it on both loops, then recover prompts and refs."""
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        try:
            # Cancelling the cross-thread future signals cancellation; this call
            # joins the actual producer task before touching its reservation.
            if self._release is not None:
                groups = await asyncio.shield(self._release)
                self.executor.data_source.requeue_pending_groups([g[0].group_index for g in groups])
            else:
                await asyncio.wrap_future(
                    async_utils.submit(self.executor.generate_rollout.release_reservation(consume=False))
                )
        finally:
            if not self.task.cancelled() and self.task.exception() is None:
                remove_rollout_data_refs(self.executor.args, self.task.result())
