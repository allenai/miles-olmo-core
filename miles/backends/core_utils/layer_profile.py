"""Sampled inclusive block timings; no per-layer CUDA synchronization.

Device intervals include stream waits and host enqueue gaps, not just kernel
execution. Backward intervals can include checkpoint recomputation, so summing
forward and backward is not a decomposition of step wall time.
"""

import re
import time
from collections import defaultdict
from functools import partial

import torch
from torch import distributed as dist

from miles.utils import distributed_utils


class LayerProfile:
    def __init__(self, model, enabled):
        self.model = model
        self.enabled = enabled
        self.handles = []
        self.pending = defaultdict(list)
        self.records = defaultdict(list)

    def _start(self, name, phase, module, inputs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.pending[name, phase].append((event, time.perf_counter()))

    def _end(self, name, phase, module, inputs, output):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        start, host = self.pending[name, phase].pop()
        self.records[name, phase].append((start, event, time.perf_counter() - host))

    def __enter__(self):
        if self.enabled:
            for name, module in self.model.named_modules():
                if re.search(r"(?:^|\.)blocks\.\d+$", name):
                    self.handles.extend(
                        [
                            module.register_forward_pre_hook(partial(self._start, name, "forward")),
                            module.register_forward_hook(partial(self._end, name, "forward"), always_call=True),
                            module.register_full_backward_pre_hook(partial(self._start, name, "backward")),
                            module.register_full_backward_hook(partial(self._end, name, "backward")),
                        ]
                    )
            if not self.handles:
                raise ValueError("Layer profiling requires numbered transformer blocks")
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def collect(self):
        if not self.enabled:
            return None
        torch.cuda.synchronize()
        local = {}
        for (name, phase), records in sorted(self.records.items()):
            device = [start.elapsed_time(end) / 1000 for start, end, _ in records]
            host = [seconds for _, _, seconds in records]
            local[f"{name}/{phase}"] = {
                "calls": len(records),
                "device_seconds": sum(device),
                "max_call_device_seconds": max(device),
                "host_seconds": sum(host),
                "max_call_host_seconds": max(host),
            }
        return local


def summary(rank_layers):
    """Keep rank and layer identity; sums of inclusive phases are not additive."""
    candidates = [
        (values["device_seconds"], rank, name, values)
        for rank, layers in enumerate(rank_layers)
        for name, values in layers.items()
    ]
    if not candidates:
        return {}
    seconds, rank, name, values = max(candidates, key=lambda item: item[0])
    return {
        "slowest": {"rank": rank, "layer_phase": name, **values},
        "metrics": {
            "profile/slowest_layer_device_seconds": seconds,
            "profile/slowest_layer_rank": rank,
            "profile/max_layer_call_device_seconds": max(v["max_call_device_seconds"] for _, _, _, v in candidates),
            "profile/max_layer_call_host_seconds": max(v["max_call_host_seconds"] for _, _, _, v in candidates),
        },
    }


def collect_across_ranks(profile):
    local = profile.collect()
    if local is None:
        return None
    ranks = [None] * dist.get_world_size()
    # Small per-layer summaries only; never gather token payloads here.
    dist.all_gather_object(ranks, local, group=distributed_utils.get_gloo_group())
    result = {"ranks": ranks, **summary(ranks)}
    for name in local:
        for metric in ("device_seconds", "host_seconds", "max_call_device_seconds"):
            result["metrics"][f"profile/layers/{name}/{metric}_rank_max"] = max(layer[name][metric] for layer in ranks)
    return result
