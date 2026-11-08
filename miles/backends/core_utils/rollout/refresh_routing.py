"""Stable sibling routing for the refresh transport."""

import copy

from miles.rollout.generate_utils.generate_endpoint_utils import compute_routing_headers


def request_headers(args, sample, request_id):
    routed = sample
    if args.sglang_router_policy == "consistent_hashing" and not sample.routing_key:
        if sample.group_index is None:
            raise ValueError("Consistent sibling routing requires a prompt group identity")
        routed = copy.copy(sample)
        routed.routing_key = f"prompt:{sample.group_index}"
    return {**(compute_routing_headers(args, routed) or {}), "x-miles-request-id": request_id}
