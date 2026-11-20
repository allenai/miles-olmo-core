"""TEST-ONLY fault injection for the policy-refresh producer. Not for merge.

Set MILES_TEST_FAULT_TRANSPORT_RATE (0 < rate < 1) to turn a completed engine
response into the router's ``500 call_upstream_request_error`` at that rate,
so partial-group and group-retry recovery run at a measurable frequency. The
engine already finished the request, so injection leaves no orphaned engine work.
"""

import os
import random

import httpx

ENV = "MILES_TEST_FAULT_TRANSPORT_RATE"
_rate = float(os.environ.get(ENV, "0") or 0)
_random = random.Random(int(os.environ.get("MILES_TEST_FAULT_SEED", "0")) or None)
injected = 0


def rate() -> float:
    return _rate


def maybe_fail(url: str, request_id: str) -> None:
    global injected
    if _rate <= 0 or _random.random() >= _rate:
        return
    injected += 1
    request = httpx.Request("POST", url)
    response = httpx.Response(
        500,
        request=request,
        json={
            "error": {
                "type": "Internal Server Error",
                "code": "call_upstream_request_error",
                "message": f"MILES_TEST fault injection for request {request_id}",
            }
        },
    )
    raise httpx.HTTPStatusError("injected test transport failure", request=request, response=response)
