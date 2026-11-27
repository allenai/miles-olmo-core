"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import logging
import math
import time
from dataclasses import replace
from typing import Any

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    Group,
    add_data_buffer_arguments,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.jit_pacer import JitPacerConfig, JitRetainedPacer, default_bounds
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0


def _resolve_max_retained_groups(args: Any) -> int | None:
    """Resolve --async-max-retained-batches into a concrete group ceiling."""
    batches = getattr(args, "async_max_retained_batches", None)
    if batches is None:
        return None
    if not math.isfinite(batches) or batches < 1.0:
        raise ValueError(f"async_max_retained_batches ({batches}) must be a finite number >= 1.0")
    return math.floor(batches * args.rollout_batch_size)


def is_supported_retained_buffer_cls(buffer_cls: Any) -> bool:
    """Return True if a DataBuffer class supports exact single-policy retained-group accounting."""
    return bool(getattr(buffer_cls, "supports_retained_group_accounting", False))


def _is_supported_retained_buffer(output: Any) -> bool:
    return bool(getattr(output, "supports_retained_group_accounting", False)) and callable(
        getattr(output, "retained_group_count", None)
    )


def _buffered_group_count(output: Any) -> int:
    """Count completed groups currently queued inside a supported single-policy data buffer."""
    if output is None:
        return 0
    if not _is_supported_retained_buffer(output):
        raise ValueError(
            f"async_max_retained_batches requires a built-in single-policy buffer with retained-group accounting, "
            f"got {type(output).__name__}"
        )
    return int(output.retained_group_count())


def _drain_events(output: Any) -> list[asyncio.Event]:
    """Return the drain wakeup event exposed by a supported single-policy data buffer."""
    if output is None:
        return []
    drain_event = getattr(output, "drain_event", None)
    if callable(drain_event):
        event = drain_event()
        if isinstance(event, asyncio.Event):
            return [event]
    return []


class FullyAsyncRolloutFn(BaseRolloutFn):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    add_arguments = staticmethod(add_data_buffer_arguments)

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else (lambda prompt_group: None)
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None
        self._jit_pacer: JitRetainedPacer | None = None
        jit_enabled = bool(getattr(input.args, "async_auto_jit_pacing", False))
        explicit_max_groups = _resolve_max_retained_groups(input.args)
        if explicit_max_groups is not None or jit_enabled:
            flag = "async_max_retained_batches" if explicit_max_groups is not None else "async_auto_jit_pacing"
            if resolve_megatron_config(input.args).is_multi_policy:
                raise ValueError(
                    f"{flag} is only supported for single-policy runs; "
                    "multi-policy buffers can deadlock under a shared retained-group cap"
                )
            if (custom_path := getattr(input.args, "custom_async_data_buffer_path", None)) is not None:
                buffer_cls = load_function(custom_path)
                if not is_supported_retained_buffer_cls(buffer_cls):
                    raise ValueError(
                        f"{flag} is only supported with built-in single-policy buffers, "
                        f"got custom buffer {custom_path!r}"
                    )
        if jit_enabled:
            min_batches, max_batches, initial_batches = default_bounds(
                rollout_batch_size=input.args.rollout_batch_size,
                producer_group_budget=self._max_in_flight_groups(),
                buffer_capacity_factor=getattr(input.args, "async_data_buffer_capacity_factor", 2.0),
                max_retained_batches=getattr(input.args, "async_max_retained_batches", None),
                prepared_batch_overlap=bool(
                    getattr(getattr(input.args, "olmo_core", None), "prepared_batch_overlap", False)
                ),
            )
            self._jit_pacer = JitRetainedPacer(
                JitPacerConfig(
                    rollout_batch_size=input.args.rollout_batch_size,
                    min_batches=min_batches,
                    max_batches=max_batches,
                    initial_batches=initial_batches,
                )
            )

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            default_buffer_cls = (
                DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
            )
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
            if self._max_retained_groups() is not None and not is_supported_retained_buffer_cls(buffer_cls):
                raise ValueError(
                    f"async_max_retained_batches is only supported with built-in single-policy buffers, "
                    f"got {buffer_cls.__name__}"
                )
            self._output = buffer_cls(
                DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
            )
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input)

    async def dispose(self) -> None:
        if (worker := self._worker) is None:
            return
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_end_worker(worker), worker.get_loop()))

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            if (pacer := getattr(self, "_jit_pacer", None)) is not None:
                pacer.on_pause()
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _max_retained_groups(self) -> int | None:
        if (pacer := getattr(self, "_jit_pacer", None)) is not None:
            return pacer.target_groups
        return _resolve_max_retained_groups(getattr(self, "args", None))

    def _retained_group_count(self, producer_owned: int | None = None) -> int:
        owned = 0 if producer_owned is None else producer_owned
        return owned + _buffered_group_count(getattr(self, "_output", None))

    def _has_retained_capacity(self, producer_owned: int | None = None) -> bool:
        limit = self._max_retained_groups()
        return limit is None or self._retained_group_count(producer_owned) < limit

    async def _wait_for_retained_capacity(
        self,
        active: set[asyncio.Task],
        *,
        producer_owned: int | None = None,
        timeout: float | None = 0.5,
    ) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        events = _drain_events(getattr(self, "_output", None))
        for event in events:
            event.clear()
        # Re-check after arming events so a drain right before clear() is never missed.
        if self._has_retained_capacity(producer_owned):
            return set(), active
        waiters = {asyncio.create_task(event.wait()) for event in events}
        stop_event = getattr(self, "_stop_requested", None)
        if isinstance(stop_event, asyncio.Event) and not getattr(self, "_stopping", False):
            waiters.add(asyncio.create_task(stop_event.wait()))
        try:
            targets = active | waiters
            if not targets:
                await asyncio.sleep(timeout if timeout is not None else 0.5)
                return set(), active
            done, pending = await asyncio.wait(targets, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            return done - waiters, pending - waiters
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            if waiters:
                await asyncio.gather(*waiters, return_exceptions=True)

    def _submit_one_group(self) -> tuple[asyncio.Task, list[Sample]]:
        samples = self.data_source.get_samples(1)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        return asyncio.create_task(self._generate_group(prompt_group)), prompt_group

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return DataBufferInput(prompt_group=prompt_group, group=result)

    async def _worker_loop(self) -> None:
        active: dict[asyncio.Task, list[Sample]] = {}
        while True:
            await self._producer_resumed.wait()
            while self._has_retained_capacity(len(active)) and self._scheduler.has_capacity(
                pending_groups=len(active), group_budget=self._max_in_flight_groups()
            ):
                task, prompt_group = self._submit_one_group()
                active[task] = prompt_group
            if not active:
                await self._wait_for_retained_capacity(set(), producer_owned=0)
                continue
            if not self._has_retained_capacity(len(active)):
                done, _ = await self._wait_for_retained_capacity(set(active), producer_owned=len(active))
                if not done:
                    continue
            else:
                done, _ = await self._scheduler.wait_for_progress(set(active))
            for task in done:
                entry = self._collect_group_result(task, active.pop(task))
                await self._output.put(entry)

    def _collect_group_result(self, task: asyncio.Task, prompt_group: list[Sample]) -> DataBufferInput:
        if not task.cancelled():
            return task.result()

        logger.warning(
            "Rollout group was cancelled; marking samples aborted: indices=%s",
            [sample.index for sample in prompt_group],
        )
        return DataBufferInput(
            prompt_group=prompt_group,
            group=[replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group],
        )

    # -------------------------- consumer --------------------------

    async def _next_group(self, *, current_version: int | None, trainer_model_id: str | None) -> DataBufferInput:
        queue_get = asyncio.create_task(
            self._output.get(current_version=current_version, trainer_model_id=trainer_model_id)
        )
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    if self._worker.cancelled():
                        raise RuntimeError("fully-async rollout was disposed while a step waited for groups")
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        do_print = True
        drain_started = time.monotonic()

        while len(data) < target_data_size:
            entry = await self._next_group(
                current_version=input.weight_version, trainer_model_id=input.trainer_model_id
            )
            assert len(entry.group) == args.n_samples_per_prompt

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            data.append(entry.group)

        drain_finished = time.monotonic()
        pacer = getattr(self, "_jit_pacer", None)
        pacer_metrics: dict[str, float] = {}
        if pacer is not None and getattr(self, "_reserved_groups", None) is None:
            prev_groups = pacer.target_groups
            pacer.observe_step(
                trainer_wait_seconds=drain_finished - drain_started,
                queue_groups_after_drain=_buffered_group_count(self._output),
                now=drain_finished,
            )
            if pacer.target_groups > prev_groups:
                for event in _drain_events(self._output):
                    event.set()
            pacer_metrics = pacer.metrics()

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        metrics = self._output.get_metrics(input.trainer_model_id)
        if pacer_metrics:
            metrics.update(pacer_metrics)
        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])


async def _end_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    await asyncio.wait({worker})
