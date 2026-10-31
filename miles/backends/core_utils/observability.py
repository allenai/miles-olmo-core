"""Bounded, globally reduced behavior-drift and CUDA allocator observations."""

import math

import torch
from torch import distributed as dist

AGE_LABELS = ("0", "1_2", "3_4", "5_8", "9_12", "13_plus")
LENGTH_LABELS = ("le_4096", "4097_8192", "8193_16384", "16385_plus")
ERROR_EDGES = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


def _active_tokens(rollout, current_version):
    deltas, ages, lengths, prefixes = [], [], [], []
    for score, behavior, mask, metadata in zip(
        rollout["log_probs"], rollout["rollout_log_probs"], rollout["loss_masks"], rollout["metadata"], strict=True
    ):
        active = mask.bool()
        delta = score.detach().double() - torch.as_tensor(behavior, device=score.device).double()
        age = torch.empty_like(score, dtype=torch.int64)
        prefix = torch.empty_like(active)
        record = metadata["policy_refresh"]
        # The caller validates contiguous spans and finite scores before aggregation.
        for span in record["spans"]:
            section = slice(span["start"], span["end"])
            age[section] = current_version - span["version"]
            prefix[section] = span["version"] < record["replay_version"]
        deltas.append(delta[active])
        ages.append(age[active])
        lengths.append(torch.full_like(age[active], len(score)))
        prefixes.append(prefix[active])
    return tuple(torch.cat(parts) for parts in (deltas, ages, lengths, prefixes))


def behavior_metrics(rollout, *, current_version, clip_low, clip_high):
    """Current trainer versus generating policy, token weighted over all ranks.

    ESS uses *unclamped behavior ratios*, with a global per-bucket log scale to
    avoid overflow. It is a token-weight dispersion diagnostic, not an estimate
    of the number of independent trajectories. Age zero still includes numerical
    trainer/serving mismatch. Empty buckets report zero tokens and zero summaries.
    """
    delta, age, length, prefix = _active_tokens(rollout, current_version)
    age_bucket = torch.bucketize(age, age.new_tensor([0, 2, 4, 8, 12]))
    length_bucket = torch.bucketize(length, length.new_tensor([4096, 8192, 16384]))
    selectors = [("all", torch.ones_like(prefix))]
    selectors += [(f"age/{label}", age_bucket == i) for i, label in enumerate(AGE_LABELS)]
    selectors += [(f"length/{label}", length_bucket == i) for i, label in enumerate(LENGTH_LABELS)]
    selectors += [("historical_prefix", prefix), ("latest_forward", ~prefix)]
    for i, label in enumerate(LENGTH_LABELS):
        selectors += [
            (f"length/{label}/historical_prefix", (length_bucket == i) & prefix),
            (f"length/{label}/latest_forward", (length_bucket == i) & ~prefix),
        ]
    edges = delta.new_tensor(ERROR_EDGES)
    log_low = math.log(clip_low) if clip_low > 0 else -math.inf
    log_high = math.log(clip_high)
    totals, maxima = [], []
    for _, selected in selectors:
        values = delta[selected]
        histogram = torch.bincount(torch.bucketize(values.abs(), edges), minlength=len(ERROR_EDGES) + 1)
        totals.append(
            torch.cat(
                (
                    torch.stack(
                        (
                            selected.sum(),
                            values.abs().sum(),
                            values.sum(),
                            (values < log_low).sum(),
                            (values > log_high).sum(),
                        )
                    ).double(),
                    histogram.double(),
                )
            )
        )
        maxima.append(
            torch.stack(
                (
                    torch.cat((values, delta.new_tensor([-math.inf]))).max(),
                    torch.cat((values.abs(), delta.new_zeros(1))).max(),
                )
            )
        )
    totals, maxima = torch.stack(totals), torch.stack(maxima)
    dist.all_reduce(totals)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    scaled = []
    for index, (_, selected) in enumerate(selectors):
        shift = torch.where(torch.isfinite(maxima[index, 0]), maxima[index, 0], maxima.new_zeros(()))
        weights = (delta[selected] - shift).exp()
        scaled.append(torch.stack((weights.sum(), weights.square().sum())))
    scaled = torch.stack(scaled)
    dist.all_reduce(scaled)
    result = {}
    for (label, _), sums, maximum, (sum_w, sum_w2) in zip(
        selectors, totals.tolist(), maxima.tolist(), scaled.tolist(), strict=True
    ):
        count, abs_sum, signed_sum, low, high, *histogram = sums
        cumulative, upper = 0, 0.0
        for edge, n in zip((*ERROR_EDGES, None), histogram, strict=True):
            cumulative += n
            if count and cumulative >= count * 0.95:
                upper = edge
                break
        values = {
            "tokens": count,
            "mean_abs_logratio": abs_sum / max(count, 1),
            "mean_logratio": signed_sum / max(count, 1),
            "max_abs_logratio": maximum[1],
            "tis_clip_low_fraction": low / max(count, 1),
            "tis_clip_high_fraction": high / max(count, 1),
            "tis_clip_fraction": (low + high) / max(count, 1),
            "ess_ratio": sum_w * sum_w / (count * sum_w2) if count and sum_w2 else 0.0,
            "p95_abs_logratio_overflow": int(upper is None),
        }
        if upper is not None:
            values["p95_abs_logratio_upper"] = upper
        result.update({f"behavior/{label}/{key}": value for key, value in values.items()})
    return result


def allocator_metrics(before, *, device):
    """Per-rank bytes and per-update retry/OOM deltas, reduced over trainer ranks."""
    after = torch.cuda.memory_stats(device)
    names = (
        "allocated_bytes",
        "reserved_bytes",
        "peak_reserved_bytes",
        "inactive_split_bytes",
        "alloc_retries",
        "ooms",
    )
    local = [
        after.get("allocated_bytes.all.current", 0),
        after.get("reserved_bytes.all.current", 0),
        after.get("reserved_bytes.all.peak", 0),
        after.get("inactive_split_bytes.all.current", 0),
        after.get("num_alloc_retries", 0) - before.get("num_alloc_retries", 0),
        after.get("num_ooms", 0) - before.get("num_ooms", 0),
    ]
    matrix = torch.zeros((dist.get_world_size(), len(names)), dtype=torch.float64, device=device)
    matrix[dist.get_rank()] = matrix.new_tensor(local)
    dist.all_reduce(matrix)
    rows = matrix.tolist()
    result = {
        f"allocator/rank_{rank}/{name}": value
        for rank, row in enumerate(rows)
        for name, value in zip(names, row, strict=True)
    }
    for i, name in enumerate(names):
        result[f"allocator/{name}_max"] = max(row[i] for row in rows)
        if name in ("alloc_retries", "ooms"):
            result[f"allocator/{name}_sum"] = sum(row[i] for row in rows)
    return result
