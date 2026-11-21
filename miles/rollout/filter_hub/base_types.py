import argparse
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from miles.utils.types import Sample


@dataclass
class FilterOutput:
    keep: bool
    reason: str | None = None


DynamicFilterOutput = FilterOutput


class UnusedSamplesHandler(Protocol):
    """The ``--async-unused-samples-handler`` contract: told what became of every finished group.

    ``prompt_group`` is the resubmittable prompt, ``group`` the finished samples, and ``reason``
    what became of it (``FilterReason`` in ``common_filters.py``, where ``kept`` means it trains,
    or a dynamic filter's own reason, ``None`` if it gave none). Kept groups may be edited in place.
    """

    def __call__(
        self, prompt_group: list[Sample], *, group: list[Sample | list[Sample]], reason: str | None
    ) -> None: ...

    def collect_metrics(self) -> dict[str, float]:
        return {}

    def pre_staleness_filter(
        self, group: list[Sample | list[Sample]], *, current_version: int | None
    ) -> list[Sample | list[Sample]]:
        """Return the nonempty group to check at dequeue time; by default, preserve it.

        Handlers may trim retained work here. The buffer still applies its normal
        whole-group staleness check and rejection callback to the returned group.
        """
        return group


def discard_unused(data_source, prompt_group: list[Sample]) -> None:
    """Retire unused work without crediting training; legacy sources only have acknowledgements."""
    discard = getattr(data_source, "discard_groups", None)
    if discard is None:
        discard = getattr(data_source, "acknowledge_groups", None)
    if discard is not None:
        discard([prompt_group])


def iter_samples(group: list[Sample | list[Sample]]) -> Iterator[Sample]:
    for sample in group:
        if isinstance(sample, list):
            yield from sample
        else:
            yield sample


def call_dynamic_filter(fn, args, samples: list[Sample | list[Sample]], **kwargs):
    if fn is None:
        return FilterOutput(keep=True)

    output = fn(args, samples, **kwargs)

    # compatibility for legacy version
    if not isinstance(output, FilterOutput):
        output = FilterOutput(keep=output)

    return output


class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self._unfiltered_reward_sum = 0.0
        self._unfiltered_reward_count = 0

    def on_group_before_dynamic_filter(self, args: argparse.Namespace, group: list) -> None:
        for sample in _iter_group_samples(group):
            if sample.reward is None:
                continue
            if not args.reward_key and isinstance(sample.reward, dict):
                continue
            if (value := sample.get_reward_value(args)) is None:
                continue
            self._unfiltered_reward_sum += float(value)
            self._unfiltered_reward_count += 1

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        if self._unfiltered_reward_count:
            metrics["rollout/raw_reward_unfiltered"] = self._unfiltered_reward_sum / self._unfiltered_reward_count
        return metrics


def _iter_group_samples(group: list) -> Iterator[Sample]:
    for item in group:
        if isinstance(item, list):
            yield from item
        else:
            yield item
