"""Core's Ray extensions must not break RPC schemas for upstream backends."""

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.train.group import TrainerController
from miles.utils.workers.rpc.common import metadata


def test_shared_workers_keep_rpc_schemas_without_core_ray_extensions():
    for worker in (InferenceController, RolloutExecutor, TrainerController):
        specs = metadata.collect_rpc_method_specs(worker)
        assert "dispose" in specs
        assert not {"core_engine_drain", "core_publication_boundary", "execute_workers"} & specs.keys()
    assert "abort_update_weights" in metadata.collect_rpc_method_specs(InferenceController)


def test_ray_only_methods_remain_callable_and_inherited():
    class Parent:
        @metadata.rpc(enabled=False)
        def extension(self, **kwargs):
            return kwargs

        def status(self) -> str:
            return "ready"

    class Child(Parent):
        pass

    assert set(metadata.collect_rpc_method_specs(Child)) == {"status"}
    assert Child().extension(step=7) == {"step": 7}
