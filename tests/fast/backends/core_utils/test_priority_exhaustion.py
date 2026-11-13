"""Finite priority collection drains submitted work and preserves native behavior."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.backends.core_utils import integration
from miles.backends.core_utils.rollout import errors
from miles.backends.hooks import get_backend_hooks
from miles.rollout.inference_rollout import inference_rollout_train as collector
from miles.utils.types import Sample


def test_only_finite_core_providers_enable_exhaustion():
    args = SimpleNamespace(olmo_core=SimpleNamespace(prompt_provider="priority"))
    assert isinstance(integration.prompt_exhaustion_error(args), errors.PromptPoolExhausted)
    assert "priority" in errors.FINITE_POOL_PROVIDERS
    for backend in (None, "fsdp", "megatron"):
        assert get_backend_hooks(backend).prompt_exhaustion_error(args) is None
    args.olmo_core.prompt_provider = "shuffle"
    assert integration.prompt_exhaustion_error(args) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("size,keep", [(0, True), (3, True), (4, True), (4, False)])
async def test_sync_drains_finite_tail_before_signalling_exhaustion(monkeypatch, size, keep):
    args = SimpleNamespace(
        train_backend="olmo_core",
        olmo_core=SimpleNamespace(prompt_provider="priority"),
        rollout_global_dataset=True,
        rollout_batch_size=4,
        n_samples_per_prompt=1,
        over_sampling_batch_size=2,
        dynamic_sampling_filter_path=None,
    )
    state = SimpleNamespace(args=args, reset=Mock())
    completed = []
    rows = [[Sample(prompt=f"p{i}", label="0", reward=0, index=i)] for i in range(size)]

    def select(n):
        if not rows:
            raise StopIteration
        result = rows[:n]
        del rows[:n]
        return result

    class Scheduler:
        sample_done_callback = None

        def has_capacity(self, *, pending_groups, group_budget):
            return pending_groups < min(2, group_budget)

        def on_submit(self, samples):
            pass

        async def wait_for_progress(self, pending):
            return await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

    def submit(state, groups, callback):
        async def generate(group):
            await asyncio.sleep(0)
            completed.append(group[0].index)
            return group

        return [asyncio.create_task(generate(group)) for group in groups]

    monkeypatch.setattr(collector.dumper_utils, "configure_sglang", AsyncMock())
    monkeypatch.setattr(collector, "load_function", lambda path: None)
    monkeypatch.setattr(collector, "make_submission_scheduler", lambda *a, **k: Scheduler())
    monkeypatch.setattr(collector, "submit_generate_tasks", submit)
    monkeypatch.setattr(collector, "apply_preput_filters", lambda *a: SimpleNamespace(keep=keep, reason="fixture"))
    monkeypatch.setattr(collector, "MetricGatherer", lambda: Mock())
    if size < 4 or not keep:
        with pytest.raises(errors.PromptPoolExhausted):
            await collector.generate_rollout_async(state, 0, select)
        assert sorted(completed) == list(range(size))
        state.reset.assert_called_once()
    else:
        # A complete final batch follows the existing return path, never an early
        # exhaustion error. Stop before unrelated metrics/conversion in this unit test.
        class ReachedCompleteBatch(Exception):
            pass

        async def abort(state, pending, rollout_id):
            assert not pending
            raise ReachedCompleteBatch

        monkeypatch.setattr(collector, "abort", abort)
        with pytest.raises(ReachedCompleteBatch):
            await collector.generate_rollout_async(state, 0, select)
        assert sorted(completed) == list(range(size))
