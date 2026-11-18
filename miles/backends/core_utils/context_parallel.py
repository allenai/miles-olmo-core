"""Bridge Core's Ulysses mesh to MILES' existing CP response/loss helpers."""

import torch
from torch import distributed as dist

from miles.backends.training_utils import parallel
from miles.backends.training_utils.data import context_parallel as response_cp
from miles.utils import distributed_utils
from miles.utils.ft_utils.process_group_utils import GroupInfo


def validate_options(args):
    cp = getattr(args, "context_parallel_size", 1)
    if cp == 1:
        return
    if cp not in (2, 4) or dist.get_world_size() % cp:
        raise ValueError("Core Ulysses requires CP2 or CP4 dividing trainer world size")
    options = args.olmo_core
    if not options.sequence_packing or options.use_reduce_scatter:
        raise ValueError("Core Ulysses requires sequence packing and use_reduce_scatter=false")
    if options.expert_balanced_packing or getattr(options, "forced_exit_positions", 0):
        raise ValueError("Core Ulysses does not support expert-balanced packing or forced exits")
    if args.loss_type != "policy_loss" or args.advantage_estimator != "grpo":
        raise ValueError("Core Ulysses currently supports the GRPO policy loss")
    if getattr(args, "custom_loss_function_path", None) or getattr(args, "normalize_advantages", False):
        raise ValueError("Core Ulysses does not yet qualify custom loss overrides or advantage whitening")
    if options.router_aux_loss_weight or options.router_z_loss_weight:
        raise ValueError("Core Ulysses qualification requires zero router auxiliary losses")
    # Core emits contiguous token shards. MILES' existing allgather_cp path
    # maps their response scores into its loss layout; zigzag input is invalid.
    args.allgather_cp = True


def configure(module):
    world, rank = dist.get_world_size(), dist.get_rank()
    combined = GroupInfo(rank, world, dist.group.WORLD, distributed_utils.get_gloo_group())
    trivial = GroupInfo(0, 1, None)
    dp, cp = combined, trivial
    if module.cp_enabled:
        mesh = module.world_mesh["dense"]
        groups = {}
        # Every rank creates CPU groups in the same order. Their memberships
        # mirror Core's mesh; Core owns the GPU collective groups.
        for name, rows in (("cp", mesh.mesh.tolist()), ("dp", mesh.mesh.T.tolist())):
            for ranks in rows:
                group = dist.new_group(ranks=ranks, backend="gloo")
                if rank in ranks:
                    submesh = mesh[name]
                    groups[name] = GroupInfo(submesh.get_local_rank(), submesh.size(), submesh.get_group(), group)
        dp, cp = groups["dp"], groups["cp"]
    parallel.set_parallel_state(
        parallel.ParallelState(
            intra_dp=dp,
            intra_dp_cp=combined,
            cp=cp,
            tp=trivial,
            pp=trivial,
            ep=trivial,
            etp=trivial,
            indep_dp=trivial,
            cp_comm_type="a2a" if cp.size > 1 else None,
        )
    )
    return {"dp_size": dp.size}


def degree():
    return parallel.get_parallel_state().cp.size


def data_parallel_size():
    return parallel.get_parallel_state().effective_dp.size


def local_masks(batch):
    return response_cp.get_local_response_loss_masks(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"]
    )


def full_scores(values, batch):
    """Reconstruct only response vectors, never sequence-by-vocabulary logits."""
    if degree() == 1:
        return values
    with torch.no_grad():
        return [
            response_cp.all_gather_with_cp(value.detach(), total, response)
            for value, total, response in zip(values, batch["total_lengths"], batch["response_lengths"], strict=True)
        ]


def diagnostic_view(batch):
    """Keep full-response diagnostics intact and count each CP-shared sample once."""
    if degree() == 1:
        return batch
    result = dict(batch)
    for key in ("log_probs", "rollout_log_probs", "ref_log_probs", "advantages", "returns"):
        if batch.get(key) is not None:
            result[key] = full_scores(batch[key], batch)
    if parallel.get_parallel_state().cp.rank != 0:
        result["loss_masks"] = [torch.zeros_like(mask) for mask in batch["loss_masks"]]
    return result


def replay_routes(model, routes, *, packed):
    """Use the same Core load balancer and padding as model._prepare_inputs()."""
    balancer = getattr(model, "_cp_load_balancer", None)
    if balancer is None:
        return routes
    names = list(routes)
    values = balancer.batch_shard(
        inputs=[routes[name] for name in names],
        seq_dims=[1] * len(names),
        pad_values=[0] * len(names),
        length_multiple=16 if packed else balancer.cp_world_size,
    )
    return dict(zip(names, values, strict=True))


def replay_counts(model, batch):
    valid = torch.ones_like(batch["tokens"], dtype=torch.int64)
    end = 0
    for length in batch.get("total_lengths", [valid.numel()]):
        end += length
        valid[0, end - 1] = 0
    local = replay_routes(model, {"valid": valid}, packed="doc_lens" in batch)["valid"]
    return local.numel(), int(local.sum())
