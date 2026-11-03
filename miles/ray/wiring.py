from __future__ import annotations

import ray
from ray.actor import ActorHandle

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend


def launch_worker_manager(args, *, transform_specs=None):
    match ClusterBackend(args.cluster_backend):
        case ClusterBackend.KUBERNETES:
            return None
        case ClusterBackend.RAY:
            if transform_specs is None:
                return _launch_ray_worker_manager(args)
            return _launch_ray_worker_manager(args, transform_specs=transform_specs)


async def shutdown_worker_manager(worker_manager_handle: ActorHandle | None) -> None:
    if worker_manager_handle is None:
        return
    try:
        await worker_manager_handle.shutdown.remote()
    finally:
        ray.kill(worker_manager_handle)


def get_backend_capability(args) -> BackendCapability:
    return factory.get_backend_capability(
        specs=compute_specs(args), cluster_backend=ClusterBackend(args.cluster_backend)
    )


def _launch_ray_worker_manager(args, *, transform_specs=None):
    from miles.ray.placement_group import create_placement_groups

    specs = compute_specs(args)
    if transform_specs is not None:
        specs = transform_specs(specs)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(args, specs, pgs, comm_backend=WorkerCommBackend(args.worker_comm_backend))
