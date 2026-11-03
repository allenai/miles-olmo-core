"""Only known Rust generation failures participate in group retry."""

import httpx
import pytest

from miles.backends.core_utils.rollout import rollout_errors


@pytest.mark.parametrize(
    "status,path,body,expected",
    [
        (503, "/generate", {"error": {"code": "no_available_workers"}}, "unavailable"),
        (500, "/generate", {"error": {"code": "call_upstream_request_error"}}, "transport"),
        (503, "/generate", {"error": {"code": "unknown"}}, None),
        (503, "/generate", {"error": "no_available_workers"}, None),
        (500, "/generate", {"error": {"code": "no_available_workers"}}, None),
        (503, "/update_weights", {"error": {"code": "no_available_workers"}}, None),
    ],
)
def test_rust_error_contract(status, path, body, expected):
    request = httpx.Request("POST", "http://router" + path)
    response = httpx.Response(status, request=request, json=body)
    error = httpx.HTTPStatusError("router", request=request, response=response)
    assert rollout_errors.recovery_category(error) == expected
