"""The router must preserve score extensions and the packed response bytes."""

import json

import httpx
import pytest
from tests.fast.rollout.generate_utils.test_core_score_transport import args_for, response

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,transport,width",
    [("topk", "flat", 32), ("topk", "array", 32), ("mc", "array", 32), ("mc", "array", 64), ("mc", "array", 128)],
)
async def test_router_preserves_score_request_and_response(monkeypatch, mode, transport, width):
    payload, _ = compute_request_payload(args_for(mode, transport, width), [1, 2], {"sampling_seed": 17})
    output, _, _ = response(width, mode, transport)
    body = json.dumps(output).encode()

    def worker(request):
        assert json.loads(request.content) == payload
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    router = MilesRouter(
        MilesRouterConfig(
            host="127.0.0.1",
            port=30000,
            max_connections=2,
            timeout=5,
            health_check_interval=60,
            health_check_failure_threshold=3,
        )
    )
    await router.client.aclose()
    router.client = httpx.AsyncClient(transport=httpx.MockTransport(worker))
    monkeypatch.setattr(router, "_use_url", lambda: "http://worker")
    monkeypatch.setattr(router, "_finish_url", lambda url: None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=router.app), base_url="http://router"
        ) as client:
            result = await client.post("/generate", json=payload)
        assert result.status_code == 200
        assert result.content == body
    finally:
        await router.close()
