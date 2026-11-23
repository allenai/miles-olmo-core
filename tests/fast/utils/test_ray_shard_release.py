"""Batch shards a rollout hands to the trainer leave the Ray object store once the step consumed them.

Shape of one handoff on the ray wire: a producer actor puts one shard per trainer
DP rank (split_train_data_by_dp), every trainer rank fetches its shard, and the
driver removes every reference after the train step (remove_rollout_data_refs).
The producer also keeps a cloudpickled copy of each reference (what any
model_dump of a RolloutDataPack does), which pins the object for the worker's
lifetime. Before the fix, remove was a no-op on this wire, so every shard stayed
in the producer node's store until it spilled to disk.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pytest

from miles.utils import object_store

pytestmark = pytest.mark.skipif(object_store._ray_free is None, reason="this ray (>= 2.59) has no force-free API")

DP_SIZE = 4
SHARD_MIB = 8
HANDOFFS = 6
SETTLE_TIMEOUT_SECONDS = 30.0


def _plasma_mib() -> int:
    import ray

    summary = ray._private.internal_api.memory_summary(stats_only=True)
    match = re.search(r"Plasma memory usage (\d+) MiB", summary)
    return int(match.group(1)) if match else 0


def _wait_for(predicate) -> int:
    deadline = time.monotonic() + SETTLE_TIMEOUT_SECONDS
    while True:
        used = _plasma_mib()
        if predicate(used) or time.monotonic() > deadline:
            return used
        time.sleep(0.2)


@pytest.fixture(scope="module")
def actors(ray_local_mode):
    import ray

    @ray.remote(num_cpus=0)
    class Producer:
        def __init__(self) -> None:
            self.store = object_store.RayObjectStore(frees_objects=False)
            self.kept: list[str] = []

        def get(self, rollout_id: int) -> list:
            words = SHARD_MIB * 2**20 // 8
            refs = [self.store.put({"tokens": np.full(words, rollout_id, dtype=np.int64)}) for _ in range(DP_SIZE)]
            self.kept.extend(ref.model_dump_json() for ref in refs)  # out-of-band copies pin every shard
            return refs

    @ray.remote(num_cpus=0)
    class TrainRank:
        def __init__(self, rank: int) -> None:
            self.rank = rank
            self.store = object_store.RayObjectStore(frees_objects=False)

        def train(self, rollout_data_ref: list) -> int:
            with self.store.get(rollout_data_ref[self.rank]) as shard:
                return int(shard["tokens"][0])

    return Producer.remote(), [TrainRank.remote(rank) for rank in range(DP_SIZE)]


def _handoff(actors, rollout_id: int) -> list:
    import ray

    producer, ranks = actors
    refs = ray.get(producer.get.remote(rollout_id))
    assert ray.get([rank.train.remote(refs) for rank in ranks]) == [rollout_id] * DP_SIZE
    return refs


def test_a_pinned_shard_stays_without_remove(actors):
    """Control: the pin is real, so reference counting alone would keep the shards."""
    baseline = _wait_for(lambda used: True)
    refs = _handoff(actors, rollout_id=100)
    del refs

    time.sleep(2.0)
    assert _plasma_mib() >= baseline + DP_SIZE * SHARD_MIB


def test_the_store_returns_to_baseline_over_many_handoffs(actors):
    """Removing after the step frees every rank's shard, so N handoffs cost no more than one."""
    import ray

    store = object_store.RayObjectStore(frees_objects=False)  # the driver's store on the ray wire
    baseline = _wait_for(lambda used: True)
    removed = []
    for rollout_id in range(HANDOFFS):
        refs = _handoff(actors, rollout_id)
        for ref in refs:
            store.remove(ref)
        removed.extend(ref.payload for ref in refs)
        del refs

    used = _wait_for(lambda used: used <= baseline)
    assert used <= baseline, f"{used - baseline} MiB of the {HANDOFFS * DP_SIZE * SHARD_MIB} MiB put is still held"
    for ref in removed:
        with pytest.raises(ray.exceptions.ObjectLostError):
            ray.get(ref, timeout=SETTLE_TIMEOUT_SECONDS)
