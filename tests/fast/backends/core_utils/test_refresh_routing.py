from types import SimpleNamespace

import pytest

from miles.backends.core_utils.rollout import refresh_routing
from miles.utils.types import Sample


def test_siblings_keep_affinity_without_sharing_request_ids():
    args = SimpleNamespace(sglang_router_policy="consistent_hashing")
    samples = [Sample(index=i, group_index=3) for i in range(8)]
    headers = [refresh_routing.request_headers(args, s, str(i)) for i, s in enumerate(samples)]
    assert len({h["X-SMG-Routing-Key"] for h in headers}) == 1
    assert len({h["x-miles-request-id"] for h in headers}) == 8
    assert all(s.routing_key is None for s in samples)
    other = refresh_routing.request_headers(args, Sample(group_index=4), "other")
    assert other["X-SMG-Routing-Key"] != headers[0]["X-SMG-Routing-Key"]


def test_explicit_routing_and_manual_missing_key():
    args = SimpleNamespace(sglang_router_policy="manual")
    with pytest.raises(ValueError, match="routing_key"):
        refresh_routing.request_headers(args, Sample(group_index=3), "request")
    assert (
        refresh_routing.request_headers(args, Sample(routing_key="worker-a"), "r")["X-SMG-Routing-Key"] == "worker-a"
    )
