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


def drop_unused(prompt_group: list[Sample], *, group: list[Sample | list[Sample]], reason: str | None) -> None:
    """The ``drop`` handler: every unused group is discarded."""


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

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
