"""Unused-sample policies own retry and discard decisions, independently of the producer."""

from miles.rollout.filter_hub.base_types import UnusedSamplesHandler, discard_unused
from miles.rollout.filter_hub.common_filters import FilterReason
from miles.rollout.never_give_up import NeverGiveUp


class DropUnused(UnusedSamplesHandler):
    def __init__(self, data_source):
        self._data_source = data_source

    def __call__(self, prompt_group, *, group, reason):
        # Kept work retires after draining. Filter rejections retire when put returns False.
        if reason in (FilterReason.aborted, FilterReason.stale):
            discard_unused(self._data_source, prompt_group)


class RetryUnused(UnusedSamplesHandler):
    def __init__(self, data_source):
        self._data_source = data_source

    def __call__(self, prompt_group, *, group, reason):
        if reason not in (FilterReason.aborted, FilterReason.stale):
            return
        for sample in prompt_group:
            sample.reset_for_retry()
        self._data_source.add_samples([prompt_group])


def make_unused_samples_handler(args, data_source) -> UnusedSamplesHandler:
    name = args.async_unused_samples_handler
    if name == "never_give_up":
        return NeverGiveUp(args, data_source=data_source)
    if name == "retry":
        return RetryUnused(data_source)
    if name == "drop":
        return DropUnused(data_source)
    raise ValueError(f"Unknown unused-samples handler: {name}")
