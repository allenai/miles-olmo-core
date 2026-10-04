"""CPU contracts used by the Open Instruct Core adapter, without model loading."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw
from miles.ray.train.group import TrainerController
from miles.utils.retry_utils import NonRetryableError


def _publication_pair():
    inference = InferenceController(SimpleNamespace(use_miles_router=True))
    cell = SimpleNamespace(
        meta=SimpleNamespace(workers_hash="generation-1"),
        is_pending_weights=True,
        mark_weights_ready=AsyncMock(),
    )
    server = SimpleNamespace(
        server_cells={"engine-0": cell},
        api_clients=["engine-client"],
        engine_gpu_counts=[1],
        engine_gpu_offsets=[0],
    )
    inference.servers = {"actor": server}
    inference._health_monitoring_pause = AsyncMock()
    inference._ensure_cells_ready = AsyncMock()
    inference._get_updatable_server = lambda: server
    trainer = TrainerController.__new__(TrainerController)
    trainer._inference_controller = inference
    trainer._maybe_log_inference_engine_weight_checksums = AsyncMock()
    return trainer, inference, cell


@pytest.mark.asyncio
async def test_publication_admits_engines_only_after_core_acknowledges():
    trainer, inference, cell = _publication_pair()
    entered, acknowledged = asyncio.Event(), asyncio.Event()

    async def publish(method, *, info):
        assert method == "update_weights"
        assert info.rollout_engines == ["engine-client"]
        entered.set()
        await acknowledged.wait()
        return [7]

    trainer._execute_first_alive = publish
    pending = asyncio.create_task(trainer.update_weights(rollout_id=3))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert inference.context_lock.locked
        cell.mark_weights_ready.assert_not_awaited()
        assert not pending.done()
        acknowledged.set()
        assert await asyncio.wait_for(pending, timeout=2) == 7
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    cell.mark_weights_ready.assert_awaited_once()
    assert not inference.context_lock.locked


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [NonRetryableError, asyncio.CancelledError])
async def test_failed_core_publication_releases_window_without_admission(error_type):
    trainer, inference, cell = _publication_pair()
    trainer._execute_first_alive = AsyncMock(side_effect=error_type("publication interrupted"))

    with pytest.raises(error_type, match="publication interrupted"):
        await trainer.update_weights()

    assert not inference.context_lock.locked
    cell.mark_weights_ready.assert_not_awaited()
    trainer._maybe_log_inference_engine_weight_checksums.assert_not_awaited()
    # The next publication must be able to acquire the same window.
    trainer._execute_first_alive = AsyncMock(return_value=[8])
    assert await asyncio.wait_for(trainer.update_weights(), timeout=2) == 8
    cell.mark_weights_ready.assert_awaited_once()


@pytest.mark.asyncio
async def test_checkpoint_finalization_waits_for_the_core_workers():
    trainer = TrainerController.__new__(TrainerController)
    entered, finished = asyncio.Event(), asyncio.Event()

    async def finalize(method, **kwargs):
        assert method == "finalize_checkpoint"
        assert kwargs == {"rollout_id": 4}
        entered.set()
        await finished.wait()

    trainer._cells_by_id = {
        "actor-0": SimpleNamespace(cell_index=0, is_alive=True, execute=finalize),
    }
    pending = asyncio.create_task(trainer.finalize_checkpoint(rollout_id=4))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert not pending.done()
        finished.set()
        await asyncio.wait_for(pending, timeout=2)
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_backend_extensions_forward_results_and_reject_multiple_cells():
    trainer = TrainerController.__new__(TrainerController)
    trainer._cells_by_id = {"actor-0": SimpleNamespace(cell_index=0)}
    trainer._execute_first_alive = AsyncMock(return_value=[{"optimizer_step": 5}])

    assert await trainer.execute_workers("core_status", rollout_id=4) == [{"optimizer_step": 5}]
    trainer._execute_first_alive.assert_awaited_once_with("core_status", rollout_id=4)
    trainer._cells_by_id["actor-1"] = SimpleNamespace(cell_index=1)
    with pytest.raises(ValueError, match="single trainer cell"):
        await trainer.execute_workers("core_status")
    assert trainer._execute_first_alive.await_count == 1


@pytest.mark.parametrize("balance_data", [False, True])
def test_core_replay_metadata_stays_aligned_after_dp_partitioning(balance_data):
    args = SimpleNamespace(balance_data=balance_data)
    data = {
        "tokens": [[1] * length for length in (1, 7, 3, 5)],
        "sample_indices": [0, 1, 2, 3],
        "metadata": [{"core_replay_version": version} for version in (4, 2, 3, 1)],
    }

    shards = split_train_data_by_dp_raw(args, data, dp_size=2)

    assert sorted(index for shard in shards for index in shard["sample_indices"]) == [0, 1, 2, 3]
    for shard in shards:
        for index, tokens, metadata in zip(shard["sample_indices"], shard["tokens"], shard["metadata"], strict=True):
            assert tokens == data["tokens"][index]
            assert metadata == data["metadata"][index]
