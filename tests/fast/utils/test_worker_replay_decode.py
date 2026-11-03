"""Routing arrays survive worker offload without weakening native validation."""

import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import ray

from miles.rollout.generate_utils import generate_endpoint_utils
from miles.utils import http_utils, rollout_replay


def response(routes):
    return {
        "text": "unchanged",
        "meta_info": {
            "routed_experts": base64.b64encode(routes.tobytes()).decode("ascii"),
            "output_token_logprobs": [[-0.12345678901234567, 7]],
            "weight_versions": [{"version": "1", "start": 0, "end": 1}],
        },
    }


@pytest.mark.parametrize("tokens,layers,experts", [(0, 16, 0), (1, 2, 3), (7, 3, 2), (2048, 16, 16)])
def test_decode_preserves_native_shape_values_and_metadata(tokens, layers, experts):
    routes = np.arange(tokens * layers * experts, dtype=np.int32).reshape(tokens, layers, experts)
    original = response(routes)
    observation = {}
    decoded = rollout_replay.decode_response_routes(original, observation=observation)
    # The worker must not infer route length from this turn's output-token list.
    args = SimpleNamespace(num_layers=layers)
    expected = generate_endpoint_utils.get_routed_experts_from_response(args, original, tokens)
    actual = generate_endpoint_utils.get_routed_experts_from_response(args, decoded, tokens)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int32
    assert isinstance(original["meta_info"]["routed_experts"], str)
    for key in ["output_token_logprobs", "weight_versions"]:
        assert decoded["meta_info"][key] is original["meta_info"][key]
    if tokens:
        assert np.shares_memory(actual, decoded["meta_info"]["routed_experts"])
    assert observation["replay_decoded_bytes"] == routes.nbytes


@pytest.mark.parametrize("encoded", [True, False])
def test_decoding_preserves_shape_and_all_zero_failures(encoded):
    args = SimpleNamespace(num_layers=2)
    for routes, error in [
        (np.zeros(12, dtype=np.int32), AssertionError),
        (np.ones(11, dtype=np.int32), ValueError),
    ]:
        value = response(routes)
        if not encoded:
            value = rollout_replay.decode_response_routes(value)
        with pytest.raises(error):
            generate_endpoint_utils.get_routed_experts_from_response(args, value, 3)


@pytest.mark.parametrize(
    "value", [np.ones(4, dtype=np.int64), np.ones((2, 2), dtype=np.int32), np.ones(8, dtype=np.int32)[::2]]
)
def test_reject_invalid_predecoded_arrays(value):
    with pytest.raises(ValueError, match="flat int32"):
        rollout_replay.decode_int32_buffer(value)


@pytest.mark.parametrize(
    "value", [b"raw body", {"text": "eval"}, {"meta_info": {}}, {"meta_info": {"routed_experts": None}}]
)
def test_no_replay_is_unchanged(value):
    assert rollout_replay.decode_response_routes(value) is value


@pytest.fixture
def worker_class(monkeypatch):
    captured = []

    class Remote:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            return object()

    def remote(cls):
        captured.append(cls)
        return Remote()

    monkeypatch.setattr(ray, "remote", remote)
    monkeypatch.setattr(ray, "nodes", lambda: [dict(Alive=True, NodeID="0" * 56)])
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_client_concurrency", 1)
    http_utils._init_ray_distributed_post(SimpleNamespace(num_gpus_per_node=1))
    return captured[0]


@pytest.mark.parametrize("decode,raw", [(False, False), (True, False), (True, True)])
def test_worker_opt_in_and_raw_bytes_contract(worker_class, decode, raw):
    body = response(np.arange(12, dtype=np.int32))

    async def exercise():
        worker = worker_class(concurrency=1)
        await worker._client.aclose()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as client:
            worker._client = client
            result = await worker.do_post("http://test/generate", {}, decode_routed_experts=decode, return_bytes=raw)
        if raw:
            assert json.loads(result) == body
        elif decode:
            np.testing.assert_array_equal(result["meta_info"]["routed_experts"], np.arange(12, dtype=np.int32))
        else:
            assert result == body

    asyncio.run(exercise())


def test_bad_replay_does_not_resend_http_request(worker_class, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"meta_info": {"routed_experts": "a"}})

    async def exercise():
        worker = worker_class(concurrency=1, observation_dir=tmp_path)
        await worker._client.aclose()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            worker._client = client
            with pytest.raises(ValueError):
                await worker.do_post("http://test/generate", {"rid": "bad"}, max_retries=3, decode_routed_experts=True)

    asyncio.run(exercise())
    assert len(calls) == 1
    record = json.loads(next(tmp_path.glob("http_post_worker_*.jsonl")).read_text())
    assert record["succeeded"] is False


@pytest.mark.parametrize("decode,raw", [(False, False), (True, False), (True, True)])
def test_local_post_fallback_preserves_contract(monkeypatch, decode, raw):
    body = response(np.arange(12, dtype=np.int32))
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as client:
            monkeypatch.setattr(http_utils, "_http_client", client)
            result = await http_utils.post("http://test/generate", {}, decode_routed_experts=decode, return_bytes=raw)
        if raw:
            assert json.loads(result) == body
        elif decode:
            np.testing.assert_array_equal(result["meta_info"]["routed_experts"], np.arange(12, dtype=np.int32))
        else:
            assert result == body

    asyncio.run(exercise())
