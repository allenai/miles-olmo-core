"""Adapt Core's already-exported HF buckets to upstream transfer protocols."""

from miles.backends.training_utils import parallel
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocols import broadcast, cuda_ipc


class _CoreTransport:
    def __init__(self, args, model=None):
        # Core owns model export and the publication session; the transport only
        # owns connections and delivery of materialized HF buckets.
        super().__init__(args)

    def connect_rollout_engines(self, rollout_engines, engine_gpu_counts=None, engine_gpu_offsets=None):
        self.connect(
            rollout_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            parallel.get_parallel_state(),
            WeightUpdatePlacement(gather_pp=True, gather_tp=True, gather_ep=True),
            "all",
        )
        self._is_src_rank = self.is_sender
        self._group_name = self.group_name

    def update_bucket_weights(self, named_tensors, weight_version=None):
        if self.is_sender and named_tensors:
            self.send_bucket(list(named_tensors))
        # IPC may retain a bucket until the next submission. Core must await its
        # acknowledgement before publishing the version or reusing its storage.
        self.after_base_weights()


class DistributedUpdater(_CoreTransport, broadcast.UpdateWeightFromDistributed):
    pass


class TensorUpdater(_CoreTransport, cuda_ipc.UpdateWeightFromTensor):
    pass
