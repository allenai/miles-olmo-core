from miles.ray.placement_group import create_placement_groups
from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.ray_worker_manager import RayWorkerManager


def launch_worker_manager(args, *, transform_specs=None):
    # TODO: after k8s native mode is created, early return when in that mode
    return _launch_ray_worker_manager(args, transform_specs=transform_specs)


def _launch_ray_worker_manager(args, *, transform_specs=None):
    specs = compute_specs(args)
    if transform_specs is not None:
        specs = transform_specs(specs)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(specs, pgs)
