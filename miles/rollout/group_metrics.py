"""Bounded group-size statistics for selected and final training groups."""

from collections import Counter

from miles.rollout.never_give_up import NGU_BASELINE_SAMPLE_COUNT_KEY


def group_size_metrics(groups, *, attempt_size: int, stage: str) -> dict[str, float]:
    sizes = [len(group) for group in groups]
    counts = Counter()
    for size in sizes:
        attempts = size / attempt_size
        label = str(int(attempts)) if attempts in (1, 2, 3, 4) else "other"
        counts[label] += 1
    count, samples = len(sizes), sum(sizes)
    chain_sizes = [group[0].metadata.get(NGU_BASELINE_SAMPLE_COUNT_KEY, len(group)) for group in groups if group]
    metrics = {
        "groups": count,
        "samples": samples,
        "size_mean": samples / max(1, count),
        "size_min": min(sizes, default=0),
        "size_max": max(sizes, default=0),
        "merged_fraction": sum(size > attempt_size for size in sizes) / max(1, count),
        "extra_sample_fraction": sum(max(0, size - attempt_size) for size in sizes) / max(1, samples),
        "chain_retried_fraction": sum(size > attempt_size for size in chain_sizes) / max(1, count),
        "chain_attempts_mean": sum(chain_sizes) / (attempt_size * max(1, count)),
    }
    for label in ("1", "2", "3", "4", "other"):
        metrics[f"attempts_{label}/count"] = counts[label]
        metrics[f"attempts_{label}/fraction"] = counts[label] / max(1, count)
    return {f"rollout/groups/{stage}/{name}": value for name, value in metrics.items()}
