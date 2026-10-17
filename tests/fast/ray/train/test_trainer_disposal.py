from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import get_raw_actor_handles

from miles.ray.train.group import TrainerController

pytestmark = pytest.mark.asyncio


def _make_mock_args(
    *,
    indep_dp: bool = True,
    enable_witness: bool = False,
    gpus_per_cell: int = 1,
    num_cells: int = 3,
    ci_ft_test_actions: str | None = None,
) -> SimpleNamespace:
    # Use SimpleNamespace (not MagicMock) so the args object is picklable. TrainerCell.init
    # passes self.args through Ray to the remote actor; pickling a MagicMock blows the
    # recursion limit because its __getattr__ creates new sub-mocks indefinitely.
    return SimpleNamespace(
        indep_dp=indep_dp,
        enable_witness=enable_witness,
        witness_buffer_size=100,
        trainer_heartbeat_checker_interval=10.0,
        trainer_heartbeat_checker_timeout=10.0,
        trainer_heartbeat_checker_first_wait=300.0,
        trainer_heartbeat_checker_failure_threshold=3,
        ci_ft_test_actions=ci_ft_test_actions,
        debug_train_only=False,
        debug_rollout_only=False,
        # compute_megatron_world_size_except_dp(args) = TP * PP * CP. Set CP to
        # gpus_per_cell so TrainerController computes num_cells correctly.
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=gpus_per_cell,
        actor_num_nodes=1,
        actor_num_gpus_per_node=num_cells * gpus_per_cell,
    )


def _make_controller(
    *,
    num_cells: int = 3,
    actor_count_per_cell: int = 1,
    inference_controller: object | None = None,
    rollout_executor: object | None = None,
    with_ref: bool = False,
    with_opd_teacher: bool = False,
    ci_ft_test_actions: str | None = None,
) -> TrainerController:
    """Create a TrainerController and let it observe every cell, as the watcher would."""
    train_conftest.fake_worker_manager.num_cells = num_cells
    train_conftest.fake_worker_manager.actor_count_per_cell = actor_count_per_cell
    group = TrainerController(
        args=_make_mock_args(
            indep_dp=True,
            gpus_per_cell=actor_count_per_cell,
            num_cells=num_cells,
            ci_ft_test_actions=ci_ft_test_actions,
        ),
        role="actor",
        with_ref=with_ref,
        with_opd_teacher=with_opd_teacher,
        inference_controller=inference_controller,
        rollout_executor=rollout_executor,
    )
    for cell_index in range(num_cells):
        cell = group._create_cell(
            f"{group._pool_id}-{cell_index}", cell_index=cell_index, workers_hash="pseudo-hash-1"
        )
        group._cells_by_id[cell.cell_id] = cell
    return group


async def _init_controller(group: TrainerController) -> None:
    """Call init and wait for all cells to become alive."""
    await group.init()


async def _make_alive_controller(*, num_cells: int = 3, **kwargs) -> TrainerController:
    """Create a group and init all cells to alive."""
    group = _make_controller(num_cells=num_cells, **kwargs)
    await _init_controller(group)
    return group


class TestDispose:
    async def test_dispatches_dispose_to_every_alive_worker(self):
        group = await _make_alive_controller(num_cells=2, actor_count_per_cell=2)

        await group.dispose()

        for cell in group._cells:
            for handle in get_raw_actor_handles(cell):
                calls = ray.get(handle.get_calls.remote())
                assert any(call[0] == "dispose" for call in calls)
