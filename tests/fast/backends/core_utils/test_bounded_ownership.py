"""Bounded retained group ownership for fully asynchronous rollouts."""

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest

from miles.backends.core_utils import async_capacity
from miles.backends.core_utils.rollout import async_rollout, pipeline_observer, rollout_errors
from miles.backends.core_utils.rollout.async_buffer import HomogeneousPolicyDataBuffer
from miles.rollout import fully_async_rollout
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput, DefaultDataBuffer
from miles.rollout.jit_pacer import JitPacerConfig, JitRetainedPacer, default_bounds
from miles.rollout.submission_scheduler import SampleBackfillSubmission
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall


def _make_group(group_index: int, *, group_size: int = 2, version: str = "0") -> list[Sample]:
    wv = [WeightVersionsPerCall(spans=[WeightVersionSpan(version=version, abs_start=0, abs_end=1)])]
    return [
        Sample(
            group_index=group_index,
            index=group_index * 100 + i,
            prompt=f"prompt-{group_index}",
            response="ok",
            response_length=1,
            label="ok",
            reward=1.0,
            status=Sample.Status.COMPLETED,
            weight_versions=list(wv),
        )
        for i in range(group_size)
    ]


class _FakeDataSource:
    def __init__(self, *, group_size: int = 2):
        self.group_size = group_size
        self.next_index = 0
        self.submitted: list[int] = []
        self.acknowledged: list[int] = []
        self.requeued: list[int] = []

    def get_samples(self, count: int) -> list[list[Sample]]:
        assert count == 1
        idx = self.next_index
        self.next_index += 1
        self.submitted.append(idx)
        return [_make_group(idx, group_size=self.group_size)]

    def acknowledge_groups(self, groups: list[list[Sample]]) -> None:
        self.acknowledged.extend(g[0].group_index for g in groups)

    def requeue_pending_groups(self, identities: list[int]) -> None:
        self.requeued.extend(identities)


def _patch_observers(monkeypatch) -> None:
    async def noop_observe(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(rollout_errors, "observe", noop_observe)
    monkeypatch.setattr(rollout_errors.RolloutErrors, "report", lambda *args, **kwargs: None)
    monkeypatch.setattr(async_rollout.pipeline_observer, "observe", noop_observe)
    monkeypatch.setattr(async_rollout.pipeline_observer, "observe_engines", noop_observe)


def _make_managed_producer(
    *,
    rollout_batch_size: int = 2,
    n_samples_per_prompt: int = 4,
    async_max_concurrent_samples: int = 16,
    async_max_retained_batches: float | None = 2.0,
    async_data_buffer_capacity_factor: float = 1.0,
    max_weight_staleness: int | None = 8,
    use_homogeneous_buffer: bool = False,
):
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=rollout_batch_size,
        global_batch_size=rollout_batch_size * n_samples_per_prompt,
        n_samples_per_prompt=n_samples_per_prompt,
        async_max_concurrent_samples=async_max_concurrent_samples,
        async_max_retained_batches=async_max_retained_batches,
        async_data_buffer_capacity_factor=async_data_buffer_capacity_factor,
        async_unused_samples_handler="drop",
        rollout_submission_granularity="sample",
        max_weight_staleness=max_weight_staleness,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        reward_key=None,
        sglang_server_concurrency=64,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        megatron_config=None,
        custom_async_data_buffer_path=None,
        custom_async_data_buffer_path_per_model=None,
    )
    data_source = _FakeDataSource(group_size=n_samples_per_prompt)
    producer = async_rollout.ManagedFullyAsyncRolloutFn.__new__(async_rollout.ManagedFullyAsyncRolloutFn)
    producer.args = args
    producer.data_source = data_source
    producer.state = SimpleNamespace(sampling_params={}, aborted=False, generate_fn_semaphore=asyncio.Semaphore(64))
    producer._scheduler = SampleBackfillSubmission(n_samples_per_prompt)
    producer._handle_unused = lambda group: data_source.acknowledge_groups([group])
    producer._sample_filter = None
    producer._worker = None
    producer._producer_resumed = asyncio.Event()
    producer._producer_resumed.set()
    producer._producer_idle = asyncio.Event()
    producer._producer_idle.set()
    producer._stop_requested = asyncio.Event()
    producer._stopping = False
    producer._shutdown_complete = False
    producer._publication_paused = False
    producer._pool_exposure = False
    producer._active_tasks = set()
    producer._producing_groups = {}
    producer._task_groups = {}
    producer._errors = rollout_errors.RolloutErrors()
    producer._retry_after = 0.0
    producer._ready_completion_counts = {}
    producer._shutdown_unqueued_counts = dict(groups=0, samples=0, response_tokens=0)
    producer._completed_put_wait_seconds = 0.0
    producer._completed_put_started = None
    producer._draining_groups = None
    producer._reserved_groups = None
    producer._reservation_task = None
    producer._interrupted_groups = []
    producer._drain_yield_at = float("inf")
    buffer_cls = HomogeneousPolicyDataBuffer if use_homogeneous_buffer else DefaultDataBuffer
    producer._output = buffer_cls(DataBufferConstructorInput(args=args, unused_handler_fn=producer._handle_unused))
    return producer, data_source


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition not met before timeout")
        await asyncio.sleep(0.005)


def test_bounded_ownership_caps_sample_backfill_under_fifo_backpressure(monkeypatch):
    """Sample backfill must stop admitting once retained groups hit the ceiling, even with free sample slots."""
    _patch_observers(monkeypatch)

    async def scenario():
        producer, data_source = _make_managed_producer(
            rollout_batch_size=2,
            n_samples_per_prompt=4,
            async_max_concurrent_samples=16,  # 4 groups worth of sample concurrency
            async_max_retained_batches=2.0,  # 2.0 * 2 = 4 groups
            async_data_buffer_capacity_factor=1.0,  # FIFO holds 2 groups
        )
        release_slow = asyncio.Event()
        partial_done = asyncio.Event()
        partial_groups_ready = 0

        async def generate(prompt_group):
            nonlocal partial_groups_ready
            idx = prompt_group[0].group_index
            if idx < 2:
                # Groups 0 and 1 complete immediately and fill the FIFO.
                for _ in prompt_group:
                    producer._scheduler.sample_done_callback()
                return DataBufferInput(prompt_group=prompt_group, group=prompt_group)
            # Groups 2 and 3 finish 3 of 4 siblings quickly, freeing 6 sample slots
            # while 1 straggler sibling keeps each group retained.
            for _ in range(3):
                producer._scheduler.sample_done_callback()
            partial_groups_ready += 1
            if partial_groups_ready == 2:
                partial_done.set()
            await release_slow.wait()
            producer._scheduler.sample_done_callback()
            return DataBufferInput(prompt_group=prompt_group, group=prompt_group)

        producer._generate_group = generate

        refill_calls = 0
        original_refill = producer._refill

        async def counted_refill(active):
            nonlocal refill_calls
            refill_calls += 1
            return await original_refill(active)

        producer._refill = counted_refill

        producer._worker = asyncio.create_task(producer._worker_loop())
        try:
            await asyncio.wait_for(partial_done.wait(), timeout=2.0)
            await _wait_until(lambda: len(producer._output._buffer) == 2)
            # Let the loop settle and verify it neither over-admits nor busy-spins.
            calls_before = refill_calls
            await asyncio.sleep(0.06)
            assert refill_calls - calls_before <= 2

            # Plenty of sample concurrency remains (2 + 4 <= 16), yet ownership caps at 4 groups.
            assert producer._scheduler.samples_in_flight == 2
            assert producer._scheduler.has_capacity(
                pending_groups=len(producer._active_tasks), group_budget=producer._max_in_flight_groups()
            )
            assert data_source.submitted == [0, 1, 2, 3]
            assert producer._retained_group_count() == 4

            snap = pipeline_observer.snapshot(producer)
            assert snap["producer_owned_groups"] == 2
            assert snap["completed_queue_groups"] == 2
            assert snap["producer_reserved_groups"] == 0
            assert snap["retained_groups"] == 4
            assert snap["max_retained_batches"] == 2.0
            assert snap["max_retained_groups"] == 4
            assert snap["retained_ownership_saturated"] is True
        finally:
            release_slow.set()
            await producer.shutdown()

    asyncio.run(scenario())


def test_bounded_ownership_resumes_immediately_after_consumer_drain_with_empty_active(monkeypatch):
    """When all retained groups sit in the FIFO (active empty), a consumer dequeue wakes refill immediately."""
    _patch_observers(monkeypatch)

    async def scenario():
        producer, data_source = _make_managed_producer(
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            async_max_concurrent_samples=4,
            async_max_retained_batches=1.0,  # 1.0 * 2 = 2 groups
            async_data_buffer_capacity_factor=1.0,
        )

        async def generate(prompt_group):
            for _ in prompt_group:
                producer._scheduler.sample_done_callback()
            return DataBufferInput(prompt_group=prompt_group, group=prompt_group)

        producer._generate_group = generate
        producer._worker = asyncio.create_task(producer._worker_loop())
        try:
            await _wait_until(lambda: len(producer._output._buffer) == 2 and not producer._active_tasks)
            assert data_source.submitted == [0, 1]
            assert producer._retained_group_count() == 2

            # Dequeuing one group must wake the idle worker well below the 0.5s fallback poll.
            delivered = set()
            started = time.monotonic()
            first = await producer._output.get(current_version=0)
            delivered.add(first.prompt_group[0].group_index)
            await _wait_until(lambda: data_source.submitted == [0, 1, 2], timeout=0.3)
            assert time.monotonic() - started < 0.3

            second = await producer._output.get(current_version=0)
            delivered.add(second.prompt_group[0].group_index)
            assert delivered == {0, 1}
            await _wait_until(lambda: data_source.submitted == [0, 1, 2, 3], timeout=0.3)
        finally:
            await producer.shutdown()

    asyncio.run(scenario())


def test_bounded_ownership_counts_prepared_reservation_until_released(monkeypatch):
    """Prepared-batch reservation retains ownership until release_reservation frees it."""
    _patch_observers(monkeypatch)

    async def scenario():
        producer, data_source = _make_managed_producer(
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            async_max_concurrent_samples=4,
            async_max_retained_batches=1.0,  # 1.0 * 2 = 2 groups
            async_data_buffer_capacity_factor=1.0,
            use_homogeneous_buffer=True,
        )

        async def generate(prompt_group):
            for _ in prompt_group:
                producer._scheduler.sample_done_callback()
            return DataBufferInput(prompt_group=prompt_group, group=prompt_group)

        producer._generate_group = generate
        producer._worker = asyncio.create_task(producer._worker_loop())
        try:
            await _wait_until(lambda: len(producer._output._delegate._buffer) == 2 and not producer._active_tasks)
            assert data_source.submitted == [0, 1]

            result = await producer.reserve(RolloutFnTrainInput(rollout_id=0, weight_version=0))
            assert len(result.samples) == 2
            await asyncio.sleep(0.05)

            # Groups moved from FIFO into _reserved_groups; total retained is still 2, so no new admission.
            assert data_source.submitted == [0, 1]
            snap = pipeline_observer.snapshot(producer)
            assert snap["completed_queue_groups"] == 0
            assert snap["producer_reserved_groups"] == 2
            assert snap["retained_groups"] == 2
            assert snap["max_retained_batches"] == 1.0
            assert snap["retained_ownership_saturated"] is True

            # Releasing the reservation must wake the producer promptly to admit groups 2 and 3.
            started = time.monotonic()
            reserved = await producer.release_reservation(consume=True)
            assert len(reserved) == 2
            await _wait_until(lambda: data_source.submitted == [0, 1, 2, 3], timeout=0.3)
            assert time.monotonic() - started < 0.3
        finally:
            if producer._reserved_groups is not None:
                await producer.release_reservation(consume=False)
            await producer.shutdown()

    asyncio.run(scenario())


def test_prepared_reservation_dequeue_handoff_is_atomic_under_cap(monkeypatch):
    """Queue dequeue into a prepared reservation must not expose a transient free slot to _worker_loop."""
    _patch_observers(monkeypatch)

    async def scenario():
        producer, data_source = _make_managed_producer(
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            async_max_concurrent_samples=8,
            async_max_retained_batches=1.0,  # cap = 2 groups
            async_data_buffer_capacity_factor=1.0,
            use_homogeneous_buffer=True,
        )
        # Force _next_group through the async queue_get task path (not get_ready)
        # while the worker is already waiting on _drain_event.
        producer._output.get_ready = None

        gates = {0: asyncio.Event(), 1: asyncio.Event()}

        async def generate(prompt_group):
            idx = prompt_group[0].group_index
            if idx in gates:
                await gates[idx].wait()
            for _ in prompt_group:
                producer._scheduler.sample_done_callback()
            return DataBufferInput(prompt_group=prompt_group, group=prompt_group)

        producer._generate_group = generate
        producer._worker = asyncio.create_task(producer._worker_loop())
        try:
            await _wait_until(lambda: data_source.submitted == [0, 1])
            # Start reserve() so queue_get is blocked inside _output.get() waiting for groups 0 and 1.
            reserve_task = asyncio.create_task(producer.reserve(RolloutFnTrainInput(rollout_id=0, weight_version=0)))
            await asyncio.sleep(0.02)

            # Release group 0, then group 1, letting the producer wake on every tick.
            gates[0].set()
            await asyncio.sleep(0.02)
            assert data_source.submitted == [0, 1]
            assert producer._retained_group_count() == 2

            gates[1].set()
            result = await asyncio.wait_for(reserve_task, timeout=2.0)
            await asyncio.sleep(0.05)

            assert len(result.samples) == 2
            assert data_source.submitted == [0, 1]
            assert producer._retained_group_count() == 2
            assert len(producer._reserved_groups) == 2
            assert producer._output.retained_group_count() == 0
        finally:
            if producer._reserved_groups is not None:
                await producer.release_reservation(consume=False)
            await producer.shutdown()

    asyncio.run(scenario())


class _UnsupportedCustomBuffer:
    async def put(self, item):
        return None

    async def get(self, **context):
        raise RuntimeError("unused")

    def get_metrics(self, trainer_model_id=None):
        return {}


def test_rejects_multi_policy_and_custom_buffers_when_bounded_ownership_enabled(monkeypatch):
    """Multi-policy and non-built-in custom buffers must be rejected when async_max_retained_batches is set."""

    class FakeGenerateState:
        def __init__(self, args):
            self.args = args
            self.sampling_params = {}
            self.aborted = False

    monkeypatch.setattr(fully_async_rollout, "GenerateState", FakeGenerateState)

    base_args = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        max_weight_staleness=None,
        async_max_concurrent_samples=8,
        async_max_retained_batches=2.0,
        async_data_buffer_capacity_factor=1.0,
        async_unused_samples_handler="drop",
        custom_async_data_buffer_path=None,
        custom_async_data_buffer_path_per_model=None,
        megatron_config=None,
        rollout_submission_granularity="sample",
        dynamic_sampling_filter_path=None,
        reward_key=None,
        rollout_sample_filter_path=None,
    )

    # Built-in core buffers are accepted.
    for path in (
        "miles.backends.core_utils.rollout.async_buffer.HomogeneousPolicyDataBuffer",
        "miles.backends.core_utils.rollout.async_buffer.RefreshPolicyDataBuffer",
    ):
        fn = fully_async_rollout.FullyAsyncRolloutFn(
            RolloutFnConstructorInput(
                args=SimpleNamespace(**{**base_args, "custom_async_data_buffer_path": path}),
                data_source=_FakeDataSource(),
            )
        )
        assert fn._max_retained_groups() == 4

    # Arbitrary custom buffer without retained-group accounting is rejected at startup.
    custom_path = f"{__name__}._UnsupportedCustomBuffer"
    with pytest.raises(ValueError, match="built-in single-policy buffers"):
        fully_async_rollout.FullyAsyncRolloutFn(
            RolloutFnConstructorInput(
                args=SimpleNamespace(**{**base_args, "custom_async_data_buffer_path": custom_path}),
                data_source=_FakeDataSource(),
            )
        )

    # Multi-policy config is rejected at startup to prevent cross-policy deadlock.
    monkeypatch.setattr(
        fully_async_rollout, "resolve_megatron_config", lambda args: SimpleNamespace(is_multi_policy=True)
    )
    with pytest.raises(ValueError, match="single-policy"):
        fully_async_rollout.FullyAsyncRolloutFn(
            RolloutFnConstructorInput(args=SimpleNamespace(**base_args), data_source=_FakeDataSource())
        )

    # Unset flag preserves backward compatibility for multi-policy and custom buffers.
    fn = fully_async_rollout.FullyAsyncRolloutFn(
        RolloutFnConstructorInput(
            args=SimpleNamespace(
                **{**base_args, "async_max_retained_batches": None, "custom_async_data_buffer_path": custom_path}
            ),
            data_source=_FakeDataSource(),
        )
    )
    assert fn._max_retained_groups() is None

    # async_auto_jit_pacing enforces the same single-policy and built-in-buffer constraints.
    with pytest.raises(ValueError, match="single-policy"):
        fully_async_rollout.FullyAsyncRolloutFn(
            RolloutFnConstructorInput(
                args=SimpleNamespace(
                    **{**base_args, "async_max_retained_batches": None, "async_auto_jit_pacing": True}
                ),
                data_source=_FakeDataSource(),
            )
        )
    monkeypatch.setattr(
        fully_async_rollout, "resolve_megatron_config", lambda args: SimpleNamespace(is_multi_policy=False)
    )
    with pytest.raises(ValueError, match="built-in single-policy buffers"):
        fully_async_rollout.FullyAsyncRolloutFn(
            RolloutFnConstructorInput(
                args=SimpleNamespace(
                    **{
                        **base_args,
                        "async_max_retained_batches": None,
                        "async_auto_jit_pacing": True,
                        "custom_async_data_buffer_path": custom_path,
                    }
                ),
                data_source=_FakeDataSource(),
            )
        )
    jit_fn = fully_async_rollout.FullyAsyncRolloutFn(
        RolloutFnConstructorInput(
            args=SimpleNamespace(**{**base_args, "async_max_retained_batches": None, "async_auto_jit_pacing": True}),
            data_source=_FakeDataSource(),
        )
    )
    assert jit_fn._jit_pacer is not None
    assert jit_fn._max_retained_groups() == jit_fn._jit_pacer.target_groups


def test_filter_drops_and_recoverable_failures_release_retained_ownership(monkeypatch):
    """Dynamic-filter drops and requeued serving failures must free retained ownership slots."""
    _patch_observers(monkeypatch)
    monkeypatch.setattr(rollout_errors, "RETRY_DELAY_SECONDS", 0.01)

    async def scenario():
        producer, data_source = _make_managed_producer(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            async_max_concurrent_samples=4,
            async_max_retained_batches=2.0,  # 2.0 * 1 = 2 groups
            async_data_buffer_capacity_factor=2.0,
        )

        # Group 0 succeeds into FIFO; Group 1 fails recoverably (requeued);
        # Group 2 is rejected by dynamic filter (put returns False); Group 3 succeeds into FIFO.
        original_put = producer._output.put

        async def filtering_put(item):
            if item.prompt_group[0].group_index == 2:
                return False
            await original_put(item)
            return True

        producer._output.put = filtering_put

        async def generate(prompt_group):
            idx = prompt_group[0].group_index
            for _ in prompt_group:
                producer._scheduler.sample_done_callback()
            if idx == 1:
                raise httpx.ConnectError("transient serving error")
            return DataBufferInput(prompt_group=prompt_group, group=prompt_group)

        producer._generate_group = generate
        producer._worker = asyncio.create_task(producer._worker_loop())
        try:
            await _wait_until(lambda: len(producer._output._buffer) == 2, timeout=2.0)
            buffered_ids = [entry.prompt_group[0].group_index for entry in producer._output._buffer]
            assert buffered_ids == [0, 3]
            assert data_source.requeued == [1]
            assert data_source.acknowledged == [2]
            assert data_source.submitted == [0, 1, 2, 3]
            assert producer._retained_group_count() == 2
        finally:
            await producer.shutdown()

    asyncio.run(scenario())


def test_base_fully_async_rollout_validation_and_bounded_ownership(monkeypatch):
    """Base FullyAsyncRolloutFn validates async_max_retained_batches and bounds retained groups."""

    class FakeGenerateState:
        def __init__(self, args):
            self.args = args
            self.sampling_params = {}
            self.aborted = False

    monkeypatch.setattr(fully_async_rollout, "GenerateState", FakeGenerateState)

    base_args = dict(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        max_weight_staleness=None,
        async_max_concurrent_samples=8,
        async_max_retained_batches=0.5,  # invalid: < 1.0
        async_data_buffer_capacity_factor=1.0,
        async_unused_samples_handler="drop",
        custom_async_data_buffer_path=None,
        custom_async_data_buffer_path_per_model=None,
        megatron_config=None,
        rollout_submission_granularity="sample",
        dynamic_sampling_filter_path=None,
        reward_key=None,
        rollout_sample_filter_path=None,
    )
    for invalid in (0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="async_max_retained_batches"):
            fully_async_rollout.FullyAsyncRolloutFn(
                RolloutFnConstructorInput(
                    args=SimpleNamespace(**{**base_args, "async_max_retained_batches": invalid}),
                    data_source=_FakeDataSource(),
                )
            )

    async def scenario():
        total_groups = 3  # floor(1.5 * 2) = 3 groups
        gates = {i: asyncio.Event() for i in range(total_groups)}
        done_flags = {i: asyncio.Event() for i in range(total_groups)}
        release_extra = asyncio.Event()

        async def generate(state, group, sampling_params, evaluation=False, sample_done_callback=None):
            idx = group[0].group_index
            if idx < total_groups:
                await gates[idx].wait()
            else:
                await release_extra.wait()
            for _ in group:
                if sample_done_callback is not None:
                    sample_done_callback()
            if idx < total_groups:
                done_flags[idx].set()
            return group

        monkeypatch.setattr(fully_async_rollout, "generate_and_rm_group", generate)

        args = SimpleNamespace(
            **{
                **base_args,
                "async_max_concurrent_samples": 8,
                "async_max_retained_batches": 1.5,
                "async_data_buffer_capacity_factor": 0.5,  # capacity=1 so put() blocks after group 0
            }
        )
        data_source = _FakeDataSource(group_size=2)
        fn = fully_async_rollout.FullyAsyncRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
        fn._output = DefaultDataBuffer(DataBufferConstructorInput(args=args, unused_handler_fn=fn._handle_unused))
        fn._worker = asyncio.create_task(fn._worker_loop())
        try:
            await _wait_until(lambda: len(data_source.submitted) == total_groups)
            for idx in range(total_groups):
                gates[idx].set()
                await asyncio.wait_for(done_flags[idx].wait(), timeout=2.0)

            # Bounded ownership must hold total submitted at 3 even though all samples finished.
            await asyncio.sleep(0.03)
            assert data_source.submitted == list(range(total_groups))

            received = []
            for _ in range(total_groups):
                entry = await asyncio.wait_for(fn._output.get(), timeout=2.0)
                received.append(entry.prompt_group[0].group_index)
            assert sorted(received) == list(range(total_groups))
            # Draining freed retained capacity, so the worker resumes admitting new groups.
            await _wait_until(lambda: len(data_source.submitted) > total_groups, timeout=1.0)
        finally:
            release_extra.set()
            for gate in gates.values():
                gate.set()
            fn._worker.cancel()
            await asyncio.gather(fn._worker, return_exceptions=True)

    asyncio.run(scenario())


def test_async_capacity_report_reflects_max_retained_batches():
    base = dict(
        fully_async=True,
        n_samples_per_prompt=8,
        rollout_batch_size=256,
        global_batch_size=2048,
        rollout_num_gpus=48,
        rollout_num_gpus_per_engine=1,
        sglang_server_concurrency=64,
        sglang_max_running_requests=32,
        async_max_concurrent_samples=3072,  # 384 groups = 1.5 batches
        async_data_buffer_capacity_factor=2.0,  # 512 groups = 2.0 batches
        rollout_submission_granularity="sample",
    )

    unbounded = async_capacity.report(base, max_policy_lag=12)
    assert unbounded["max_retained_batches"] is None
    assert unbounded["max_retained_groups"] is None
    assert unbounded["max_retained_samples"] is None
    assert any("--async-max-retained-batches" in warning for warning in unbounded["warnings"])

    bounded = async_capacity.report({**base, "async_max_retained_batches": 4.0}, max_policy_lag=12)
    assert bounded["max_retained_batches"] == 4.0
    assert bounded["max_retained_groups"] == 1024
    assert bounded["max_retained_samples"] == 8192
    assert not any("Sample backfill limits unfinished samples" in warning for warning in bounded["warnings"])

    oversized = async_capacity.report({**base, "async_max_retained_batches": 16.0}, max_policy_lag=12)
    assert any("spans 16 optimizer batches" in warning for warning in oversized["warnings"])

    tighter_than_concurrency = async_capacity.report({**base, "async_max_retained_batches": 1.0}, max_policy_lag=12)
    assert any("below producer_group_budget=384" in warning for warning in tighter_than_concurrency["warnings"])

    jit_only = async_capacity.report({**base, "async_auto_jit_pacing": True}, max_policy_lag=12)
    assert jit_only["auto_jit_pacing"] is True
    assert jit_only["jit_min_retained_batches"] == 1.5
    assert jit_only["jit_max_retained_batches"] == pytest.approx(3.5)
    assert jit_only["jit_initial_retained_batches"] == 2.5
    assert not any("Sample backfill limits unfinished samples" in warning for warning in jit_only["warnings"])


def test_shutdown_with_active_task_and_saturated_cap_does_not_spin(monkeypatch):
    """During managed shutdown with a saturated cap and an active task, the worker loop must not busy-spin."""
    _patch_observers(monkeypatch)

    async def scenario():
        producer, data_source = _make_managed_producer(
            rollout_batch_size=1,
            n_samples_per_prompt=2,
            async_max_concurrent_samples=4,
            async_max_retained_batches=2.0,  # cap = 2 groups
            async_data_buffer_capacity_factor=1.0,  # FIFO holds 1 group
        )
        release_active = asyncio.Event()
        active_started = asyncio.Event()

        async def generate(prompt_group):
            idx = prompt_group[0].group_index
            if idx == 0:
                for _ in prompt_group:
                    producer._scheduler.sample_done_callback()
                return DataBufferInput(prompt_group=prompt_group, group=prompt_group)
            active_started.set()
            await release_active.wait()
            for _ in prompt_group:
                producer._scheduler.sample_done_callback()
            return DataBufferInput(prompt_group=prompt_group, group=prompt_group)

        producer._generate_group = generate
        wait_calls = 0
        original_wait = asyncio.wait

        async def counted_wait(*args, **kwargs):
            nonlocal wait_calls
            wait_calls += 1
            return await original_wait(*args, **kwargs)

        producer._worker = asyncio.create_task(producer._worker_loop())
        try:
            await asyncio.wait_for(active_started.wait(), timeout=2.0)
            await _wait_until(lambda: len(producer._output._buffer) == 1 and data_source.submitted == [0, 1])

            monkeypatch.setattr(async_rollout.asyncio, "wait", counted_wait)
            shutdown_task = asyncio.create_task(producer.shutdown())
            await asyncio.sleep(0.08)
            # Waiting for group 1 to finish during shutdown must block on active, not spin on _stop_requested.
            assert not shutdown_task.done()
            assert wait_calls <= 3
            release_active.set()
            await asyncio.wait_for(shutdown_task, timeout=2.0)
        finally:
            release_active.set()
            if producer._worker is not None and not producer._worker.done():
                await producer.shutdown()

    asyncio.run(scenario())


def test_pipeline_observer_deduplicates_group_in_producing_and_buffer():
    """Transient overlap between _producing_groups and _output._buffer must not double-count retained_groups."""
    group0 = _make_group(0, group_size=2)
    group1 = _make_group(1, group_size=2)
    entry0 = DataBufferInput(prompt_group=group0, group=group0)
    resumed = asyncio.Event()
    resumed.set()
    producer = SimpleNamespace(
        args=SimpleNamespace(
            sglang_server_concurrency=64,
            rollout_num_gpus=1,
            rollout_num_gpus_per_engine=1,
            rollout_batch_size=2,
            async_max_retained_batches=1.0,
        ),
        state=SimpleNamespace(generate_fn_semaphore=asyncio.Semaphore(64)),
        _output=SimpleNamespace(_buffer=[entry0], _capacity=2),
        _ready_completion_counts={},
        _active_tasks=set(),
        # Group 0 is already appended to _buffer while still transiently present in _producing_groups.
        _producing_groups={0: group0, 1: group1},
        _reserved_groups=None,
        _producer_resumed=resumed,
        _scheduler=SimpleNamespace(samples_in_flight=2),
    )
    snap = pipeline_observer.snapshot(producer)
    assert snap["producer_owned_groups"] == 2
    assert snap["completed_queue_groups"] == 1
    assert snap["retained_groups"] == 2
    assert snap["max_retained_groups"] == 2


def test_jit_pacer_tightens_on_slack_bumps_on_moderate_wait_and_escapes_on_stall():
    """Verify reverse-AIMD control law: slow tighten on zero-wait slack, bump on >2s, snap-to-max on >10s."""
    min_b, max_b, init_b = default_bounds(
        rollout_batch_size=256,
        producer_group_budget=1092,
        buffer_capacity_factor=1.5,
        max_retained_batches=None,
        prepared_batch_overlap=True,
    )
    assert min_b == 2.0
    assert max_b == pytest.approx(1092 / 256 + 1.5)
    assert init_b == 3.0

    pacer = JitRetainedPacer(
        JitPacerConfig(
            rollout_batch_size=256,
            min_batches=2.0,
            max_batches=4.0,
            initial_batches=3.0,
        )
    )
    assert pacer.target_batches == 3.0
    assert pacer.target_groups == 768

    # Zero wait WITHOUT slack (empty queue, not early) must NOT tighten.
    t = 100.0
    for _ in range(6):
        t += 120.0
        pacer.observe_step(
            trainer_wait_seconds=0.0, queue_groups_after_drain=0, ready_before_consumer_seconds=0.0, now=t
        )
    assert pacer.target_batches == 3.0

    # Three consecutive calm steps with large slack (batch ready 60s early on 120s step) tighten by 0.5 -> 2.5.
    for _ in range(3):
        t += 120.0
        pacer.observe_step(
            trainer_wait_seconds=0.0, queue_groups_after_drain=256, ready_before_consumer_seconds=60.0, now=t
        )
    assert pacer.target_batches == 2.5

    # Three more calm steps with moderate slack tighten by 0.25 -> 2.25, then clamp at floor 2.0.
    for _ in range(3):
        t += 120.0
        pacer.observe_step(
            trainer_wait_seconds=0.2, queue_groups_after_drain=80, ready_before_consumer_seconds=0.0, now=t
        )
    assert pacer.target_batches == 2.25
    for _ in range(3):
        t += 120.0
        pacer.observe_step(
            trainer_wait_seconds=0.0, queue_groups_after_drain=300, ready_before_consumer_seconds=0.0, now=t
        )
    assert pacer.target_batches == 2.0

    # Moderate trainer wait (>2s, <=10s) bumps gently by +0.5 -> 2.5 and triggers cooldown.
    t += 124.0
    pacer.observe_step(trainer_wait_seconds=4.0, queue_groups_after_drain=0, now=t)
    assert pacer.target_batches == 2.5

    # Stall (>10s) immediately escapes to max_batches=4.0 and holds through 5-step cooldown.
    t += 132.0
    pacer.observe_step(trainer_wait_seconds=12.0, queue_groups_after_drain=0, now=t)
    assert pacer.target_batches == 4.0
    for _ in range(5):
        t += 120.0
        pacer.observe_step(
            trainer_wait_seconds=0.0, queue_groups_after_drain=256, ready_before_consumer_seconds=60.0, now=t
        )
    # Still at 4.0 through cooldown; after 3 post-cooldown calm steps it resumes tightening.
    assert pacer.target_batches == 4.0
    for _ in range(3):
        t += 120.0
        pacer.observe_step(
            trainer_wait_seconds=0.0, queue_groups_after_drain=256, ready_before_consumer_seconds=60.0, now=t
        )
    assert pacer.target_batches == 3.5

    # Pause resets wall-clock baseline so long evals don't distort rolling trainer step time.
    pacer.on_pause()
    prev_rolling = pacer.metrics()["rollout/fully_async/jit_pacer/rolling_trainer_step_seconds"]
    pacer.observe_step(trainer_wait_seconds=0.0, queue_groups_after_drain=0, now=t + 3600.0)
    assert pacer.metrics()["rollout/fully_async/jit_pacer/rolling_trainer_step_seconds"] == pytest.approx(prev_rolling)


def test_auto_jit_pacing_live_producer_tightens_and_escapes(monkeypatch):
    """Live ManagedFullyAsyncRolloutFn with JIT pacer caps startup, tightens on slack, and snaps up on >10s wait."""
    _patch_observers(monkeypatch)

    async def scenario():
        producer, data_source = _make_managed_producer(
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            async_max_concurrent_samples=16,  # 8 groups = 4 batches
            async_max_retained_batches=4.0,
            async_data_buffer_capacity_factor=4.0,
            use_homogeneous_buffer=True,
        )
        producer._jit_pacer = JitRetainedPacer(
            JitPacerConfig(rollout_batch_size=2, min_batches=2.0, max_batches=4.0, initial_batches=3.0)
        )

        async def generate(prompt_group):
            for _ in prompt_group:
                producer._scheduler.sample_done_callback()
            return DataBufferInput(prompt_group=prompt_group, group=prompt_group)

        producer._generate_group = generate
        producer._worker = asyncio.create_task(producer._worker_loop())
        try:
            # Startup compilation cap starts at initial_batches=3.0 (6 groups), not max_batches=4.0 (8 groups).
            await _wait_until(lambda: len(producer._output._delegate._buffer) == 6 and not producer._active_tasks)
            assert len(data_source.submitted) == 6
            snap = pipeline_observer.snapshot(producer)
            assert snap["jit_pacing_enabled"] is True
            assert snap["max_retained_batches"] == 3.0
            assert snap["max_retained_groups"] == 6

            # Three zero-wait prepared-batch consumptions with leftover FIFO slack tighten target 3.0 -> 2.5 (5 groups).
            for rollout_id in range(3):
                await producer.reserve(RolloutFnTrainInput(rollout_id=rollout_id, weight_version=0))
                await producer.release_reservation(
                    consume=True, trainer_wait_seconds=0.0, ready_before_consumer_seconds=30.0
                )
            assert producer._jit_pacer.target_batches == 2.5
            assert producer._max_retained_groups() == 5

            # A >10s trainer wait immediately snaps target to max_batches=4.0 (8 groups) and wakes refill.
            await producer.reserve(RolloutFnTrainInput(rollout_id=3, weight_version=0))
            await producer.release_reservation(
                consume=True, trainer_wait_seconds=12.0, ready_before_consumer_seconds=0.0
            )
            assert producer._jit_pacer.target_batches == 4.0
            assert producer._max_retained_groups() == 8
            await _wait_until(lambda: len(producer._output._delegate._buffer) == 8, timeout=0.5)
        finally:
            if producer._reserved_groups is not None:
                await producer.release_reservation(consume=False)
            await producer.shutdown()

    asyncio.run(scenario())
