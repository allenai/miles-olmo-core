import asyncio
import weakref
from argparse import Namespace
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.training_utils.weight_update.protocols.cuda_ipc import (
    UpdateWeightFromTensor,
    _send_to_colocated_engine,
)
from miles.utils import async_utils

_TENSOR_MODULE = "miles.backends.training_utils.weight_update.protocols.cuda_ipc"

_ENGINE_GPU_COUNTS = [2, 2, 2, 2]
_SPARSE_GPU_OFFSETS = [0, 2, 6, 8]


class _FakeFlattenedTensorBucket:
    supports_multi_dtypes = False

    def __init__(self, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        self._named_tensors = named_tensors

    def get_flattened_tensor(self) -> str:
        return self._named_tensors[0][0]

    def get_metadata(self) -> None:
        return None


class _FakeMultiprocessingSerializer:
    @staticmethod
    def serialize(data: dict, output_str: bool = False) -> str:
        return f"serialized:{data['flattened_tensor']}"


class _SerialProbeEngine:
    def __init__(self) -> None:
        self.received: list[str] = []
        self.active = 0
        self.max_active = 0

    async def update_weights_from_tensor(self, **kwargs) -> dict:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.received.append(kwargs["serialized_named_tensors"][0])
        await asyncio.sleep(0.01)
        self.active -= 1
        return {"success": True}


def _gather_object_onto_source(obj, object_gather_list=None, dst=None, group=None) -> None:
    object_gather_list[0] = obj


class TestConnect:
    @staticmethod
    def _make_protocol() -> UpdateWeightFromTensor:
        protocol = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
        protocol.args = Namespace(
            rollout_num_gpus_per_engine=2,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            check_weight_update_equal=False,
            colocate=True,
            offload_rollout=True,
        )
        protocol.rollout_engines = None
        protocol._model_update_groups = None
        protocol._pipeline_depth = 1
        protocol._ipc_gather_group = MagicMock(name="gather_group_from_init")
        protocol._ipc_gather_src = 0
        return protocol

    @staticmethod
    def _connect(
        protocol: UpdateWeightFromTensor, engines: list, *, rank: int, tp_rank: int
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        parallel_state = SimpleNamespace(
            intra_dp_cp=SimpleNamespace(rank=0),
            tp=SimpleNamespace(rank=tp_rank),
            pp=SimpleNamespace(rank=0),
        )
        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.lora_rollout_enabled", return_value=False),
            patch(f"{_TENSOR_MODULE}.connect_rollout_engines_from_distributed") as connect,
            patch(f"{_TENSOR_MODULE}.disconnect_rollout_engines_from_distributed") as disconnect,
        ):
            dist_mock.get_rank.return_value = rank
            protocol.connect(
                engines,
                engine_gpu_counts=_ENGINE_GPU_COUNTS,
                engine_gpu_offsets=_SPARSE_GPU_OFFSETS,
                parallel_state=parallel_state,
                placement=SimpleNamespace(gather_pp=True),
                selector="all",
            )
        return dist_mock, connect, disconnect

    @pytest.mark.parametrize(
        ("rank", "expected_engine_index", "expected_gather_src"),
        [(0, 0, 0), (6, 2, 6)],
        ids=["first_engine", "engine_after_sparse_gap"],
    )
    def test_connect_splits_clients_and_maps_sparse_gpu_offsets(
        self, rank: int, expected_engine_index: int, expected_gather_src: int
    ) -> None:
        """Only remote engines use NCCL, while ranks on either side of a sparse gap map to their covered colocated engine."""
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]
        protocol = self._make_protocol()
        protocol._ipc_gather_src = expected_gather_src
        gather_group_from_init = protocol._ipc_gather_group

        dist_mock, connect, disconnect = self._connect(protocol, engines, rank=rank, tp_rank=0)

        assert protocol.use_distribute is True
        assert protocol.rollout_engines == engines[:3]
        assert protocol.distributed_rollout_engines == engines[3:]
        connect.assert_called_once_with(protocol.args, "miles", engines[3:], engine_gpu_counts=[2])
        disconnect.assert_not_called()
        assert protocol._model_update_groups is connect.return_value
        assert protocol._ipc_engine is engines[expected_engine_index]
        assert protocol._ipc_gather_group is gather_group_from_init
        assert protocol._ipc_gather_src == expected_gather_src
        assert protocol.is_sender is True
        dist_mock.new_group.assert_not_called()

    def test_connect_clears_ipc_state_for_a_placeholder_rank(self) -> None:
        """A reserved GPU slot that no engine covers drops its stale gather group, or it would enter a collective nobody else joins."""
        engines = [MagicMock(name=f"engine{i}") for i in range(4)]
        protocol = self._make_protocol()

        _dist_mock, connect, _disconnect = self._connect(protocol, engines, rank=4, tp_rank=1)

        assert protocol._ipc_gather_group is None
        assert protocol._ipc_gather_src is None
        assert protocol._ipc_engine is None
        assert protocol.is_sender is False
        connect.assert_not_called()


class TestColocatedBaseSend:
    def test_multiple_colocated_buckets_are_sent_serially_in_source_order(self) -> None:
        """Two dtype buckets reach the engine one at a time and in the order they were built."""
        engine = _SerialProbeEngine()
        named_tensors = [
            ("model.layers.0.mlp.gate_proj.weight", torch.zeros(2, dtype=torch.float32)),
            ("model.layers.0.mlp.up_proj.weight", torch.zeros(2, dtype=torch.float16)),
        ]

        with (
            patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
            patch(f"{_TENSOR_MODULE}.FlattenedTensorBucket", _FakeFlattenedTensorBucket),
            patch(f"{_TENSOR_MODULE}.MultiprocessingSerializer", _FakeMultiprocessingSerializer),
        ):
            dist_mock.get_rank.return_value = 0
            dist_mock.get_world_size.return_value = 1
            dist_mock.gather_object.side_effect = _gather_object_onto_source
            futures, _long_lived_tensors = _send_to_colocated_engine(
                named_tensors,
                ipc_engine=engine,
                ipc_gather_src=0,
                ipc_gather_group=MagicMock(name="gather_group"),
            )
            async_utils.wait_futures(futures)

        assert engine.received == [
            "serialized:model.layers.0.mlp.gate_proj.weight",
            "serialized:model.layers.0.mlp.up_proj.weight",
        ]
        assert engine.max_active == 1


@pytest.mark.parametrize("depth", [1, 2])
def test_pipeline_bounds_live_buffers_and_drains_before_session_end(monkeypatch, depth):
    protocol = UpdateWeightFromTensor.__new__(UpdateWeightFromTensor)
    protocol._pending_buckets = deque()
    protocol._pipeline_depth = depth
    protocol._ipc_engine = protocol._ipc_gather_src = protocol._ipc_gather_group = None
    protocol._selector = "all"
    protocol.use_distribute = False
    pending, allocations = [], []

    def send(**kwargs):
        tensor = torch.zeros(1)
        allocations.append(weakref.ref(tensor))
        future = Future()
        pending.append(future)
        if depth == 1:
            future.set_result({"success": True})
        return [future], [tensor]

    monkeypatch.setattr(_TENSOR_MODULE + "._send_to_colocated_engine", send)
    protocol.send_bucket([])
    if depth == 1:
        assert allocations[0]() is None
    else:
        protocol.send_bucket([])
        assert len(protocol._pending_buckets) == 2
        assert all(ref() is not None for ref in allocations)
        pending[0].set_result({"success": True})
        protocol.send_bucket([])
        assert allocations[0]() is None
        assert len(protocol._pending_buckets) == 2
        for future in pending[1:]:
            future.set_result({"success": True})
    protocol.after_base_weights()
    assert not protocol._pending_buckets
    assert all(ref() is None for ref in allocations)


@pytest.mark.parametrize("colocate,lora_rank,lora_train_only", [(False, 0, False), (True, 8, False), (True, 8, True)])
def test_depth_two_rejects_unsupported_modes_before_allocating_groups(colocate, lora_rank, lora_train_only):
    with patch(f"{_TENSOR_MODULE}.dist.new_group") as allocate:
        with pytest.raises(ValueError, match="colocated full-weight"):
            UpdateWeightFromTensor(
                Namespace(
                    colocated_weight_update_pipeline_depth=2,
                    colocate=colocate,
                    lora_rank=lora_rank,
                    lora_train_only=lora_train_only,
                )
            )
        allocate.assert_not_called()


def test_depth_two_rejects_a_hybrid_engine_tail_before_connecting():
    protocol = TestConnect._make_protocol()
    protocol._pipeline_depth = 2
    with pytest.raises(ValueError, match="distributed engine tail"):
        TestConnect._connect(protocol, [MagicMock() for _ in range(4)], rank=0, tp_rank=0)


def test_failed_bucket_close_settles_remaining_receivers_and_disconnects():
    protocol = TestConnect._make_protocol()
    protocol._model_update_groups = object()
    protocol.group_name = "miles"
    protocol.distributed_rollout_engines = []
    failed, success = Future(), Future()
    failed.set_exception(RuntimeError("receiver rejected update"))
    success.set_result({"success": True})
    protocol._pending_buckets = deque([([failed], [torch.zeros(1)]), ([success], [torch.ones(1)])])
    group = protocol._ipc_gather_group
    with (
        patch(f"{_TENSOR_MODULE}.dist") as dist_mock,
        patch(f"{_TENSOR_MODULE}.disconnect_rollout_engines_from_distributed") as disconnect,
    ):
        dist_mock.get_world_size.return_value = 1
        dist_mock.all_gather_object.side_effect = lambda values, value, **kw: values.__setitem__(0, value)
        with pytest.raises(RuntimeError, match="receiver rejected"):
            protocol.close()
        assert not protocol._pending_buckets
        disconnect.assert_called_once()
        dist_mock.destroy_process_group.assert_called_once_with(group)
