"""Generation payload decoding preserves scores, routes, and HTTP failure semantics."""

import asyncio
import base64
import json
import struct

import httpx
import orjson
import pytest

from miles.utils import http_utils
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])


def test_generation_bytes_preserve_complete_payload_and_float_bits():
    scores = [-0.0, -0.12345678901234567, -1e-300, -1234.125]
    routes = struct.pack("<8i", 0, 3, 7, 255, 4, 2, 9, 15)
    expected = {
        "text": "Reasoning: café → ✓",
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [[score, token] for token, score in enumerate(scores)],
            "weight_versions": [{"version": "0", "start": 0, "end": 2}, {"version": "1", "start": 2, "end": 4}],
            "routed_experts": base64.b64encode(routes).decode(),
        },
    }
    encoded = json.dumps(expected, ensure_ascii=False).encode()

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=encoded))
        ) as client:
            response = await http_utils._post(client, "http://engine/generate", {}, max_retries=1, return_bytes=True)
            assert response == encoded
            actual = orjson.loads(response)
            assert actual == expected
            assert base64.b64decode(actual["meta_info"]["routed_experts"]) == routes
            for (score, _), reference in zip(actual["meta_info"]["output_token_logprobs"], scores, strict=True):
                assert struct.pack("d", score) == struct.pack("d", reference)
            assert await http_utils._post(client, "http://engine/generate", {}, max_retries=1) == expected

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["status", "read"])
def test_byte_response_mode_preserves_single_attempt_transport_errors(failure):
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "read":
            raise httpx.ReadError("connection reset", request=request)
        return httpx.Response(503, json={"detail": "Rollout worker unavailable"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(httpx.ReadError if failure == "read" else httpx.HTTPStatusError):
                await http_utils._post(client, "http://engine/generate", {}, max_retries=1, return_bytes=True)
        assert len(calls) == 1

    asyncio.run(scenario())


def test_byte_response_mode_joins_cancellation():
    async def scenario():
        started, finalized = asyncio.Event(), asyncio.Event()

        async def handler(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finalized.set()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(
                http_utils._post(client, "http://engine/generate", {}, max_retries=1, return_bytes=True)
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finalized.is_set()

    asyncio.run(scenario())
