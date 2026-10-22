"""Core transport ownership and sender selection against the upstream protocols."""

import pytest

from miles.backends.core_utils.publication import transport
from miles.backends.training_utils.weight_update.protocols import broadcast, cuda_ipc


@pytest.mark.parametrize(
    "cls,protocol",
    [
        (transport.DistributedUpdater, broadcast.UpdateWeightFromDistributed),
        (transport.TensorUpdater, cuda_ipc.UpdateWeightFromTensor),
    ],
)
@pytest.mark.parametrize("sender", [False, True])
def test_connect_and_bucket_completion(monkeypatch, cls, protocol, sender):
    updater = object.__new__(cls)
    state = object()
    monkeypatch.setattr(transport.parallel, "get_parallel_state", lambda: state)
    engines = [object()]
    calls = []

    def connect(self, clients, counts, offsets, parallel_state, placement, selector):
        assert (clients, counts, offsets, parallel_state, selector) == (engines, [1], [0], state, "all")
        assert placement.gather_pp and placement.gather_tp and placement.gather_ep
        self.is_sender = sender
        self.group_name = "core-test"

    monkeypatch.setattr(protocol, "connect", connect)
    updater.send_bucket = lambda bucket: (calls.append("send"), bucket.clear())
    updater.after_base_weights = lambda: calls.append("ack")
    updater.connect_rollout_engines(engines, [1], [0])
    assert updater._is_src_rank == sender
    assert updater._group_name == "core-test"
    bucket = [("weight", object())]
    updater.update_bucket_weights(bucket, weight_version=7)
    assert len(bucket) == 1
    assert calls == (["send", "ack"] if sender else ["ack"])
