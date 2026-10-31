"""Per-token logprob records travel as flat arrays without changing what samples receive."""

import asyncio
import json
import pickle
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import ray

from miles.backends.core_utils.publication import policy_refresh
from miles.rollout.generate_utils import generate_endpoint_utils
from miles.utils import http_utils, rollout_replay
from miles.utils.rollout_replay import TokenLogprobs
from miles.utils.types import Sample


def records(length, seed=0):
    generator = np.random.default_rng(seed)
    log_probs = generator.uniform(-40.0, 0.0, size=length).tolist() + []
    token_ids = generator.integers(0, 200_000, size=length).tolist()
    return [[lp, tid, None] for lp, tid in zip(log_probs, token_ids, strict=True)]


@pytest.mark.parametrize("length", [0, 1, 5, 4096])
def test_compact_matches_legacy_readers(length):
    nested = records(length)
    compact = TokenLogprobs.from_records(nested)
    assert len(compact) == length and bool(compact) == bool(nested)
    assert [item[1] for item in compact] == [item[1] for item in nested]
    assert [item[0] for item in compact] == [item[0] for item in nested]
    assert compact.tokens() == [item[1] for item in nested]
    assert compact.logprobs() == [item[0] for item in nested]
    assert all(isinstance(item[0], float) and isinstance(item[1], int) and item[2] is None for item in compact)
    assert compact == nested and compact == TokenLogprobs.from_records(nested)
    assert compact.log_probs.dtype == np.float64 and compact.token_ids.dtype == np.int64
    if length:
        assert compact[0] == (nested[0][0], nested[0][1], None)
        assert compact[-1] == (nested[-1][0], nested[-1][1], None)
        assert compact[1:3] == nested[1:3]
    assert TokenLogprobs.from_records(compact) is compact


def test_compact_rejects_ragged_records():
    with pytest.raises(ValueError, match="one logprob per token id"):
        TokenLogprobs(np.zeros(3), np.zeros(2, dtype=np.int64))


def test_pickle_roundtrip_preserves_exact_values():
    compact = TokenLogprobs.from_records(records(1000))
    restored = pickle.loads(pickle.dumps(compact, protocol=5))
    assert restored == compact
    np.testing.assert_array_equal(restored.log_probs.view(np.int64), compact.log_probs.view(np.int64))


def response(length):
    return {
        "text": "unchanged",
        "meta_info": {
            "output_token_logprobs": records(length),
            "finish_reason": {"type": "stop"},
            "weight_version": "3",
            "weight_versions": [{"version": "3", "start": 0, "end": length}],
        },
    }


def test_compact_response_replaces_only_the_records():
    original = response(16)
    observation = {}
    compacted = rollout_replay.compact_token_logprobs(original, observation=observation)
    assert isinstance(compacted["meta_info"]["output_token_logprobs"], TokenLogprobs)
    assert isinstance(original["meta_info"]["output_token_logprobs"], list)
    for key in ("finish_reason", "weight_version", "weight_versions"):
        assert compacted["meta_info"][key] is original["meta_info"][key]
    assert compacted["text"] is original["text"]
    assert observation["token_logprobs_count"] == 16
    assert observation["token_logprobs_compact_seconds"] >= 0
    assert rollout_replay.compact_token_logprobs(compacted) is compacted


@pytest.mark.parametrize("value", [b"raw body", {"text": "eval"}, {"meta_info": {}}, {"meta_info": {"text": 1}}])
def test_no_records_is_unchanged(value):
    assert rollout_replay.compact_token_logprobs(value) is value


def converted_sample(output, length):
    args = SimpleNamespace(
        rollout_top_logprobs_num=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        num_layers=1,
        sglang_speculative_algorithm=None,
    )
    sample = Sample(index=0, group_index=0, prompt="question")
    payload = {"input_ids": [1, 2, 3]}
    asyncio.run(generate_endpoint_utils.update_sample_from_response(args, sample, payload, output))
    provenance = policy_refresh.record_response(sample, output["meta_info"])
    return sample, provenance


@pytest.mark.parametrize("length", [1, 2048])
def test_sample_conversion_and_refresh_provenance_match_legacy(length):
    nested = response(length)
    compact = rollout_replay.compact_token_logprobs(nested)
    expected, expected_provenance = converted_sample(nested, length)
    actual, provenance = converted_sample(compact, length)
    assert actual.tokens == expected.tokens == [1, 2, 3, *(item[1] for item in nested["meta_info"]["output_token_logprobs"])]
    assert actual.response_length == expected.response_length == length
    assert actual.rollout_log_probs == expected.rollout_log_probs
    assert all(isinstance(value, float) for value in actual.rollout_log_probs)
    assert provenance == expected_provenance
    assert actual.weight_versions[0].to_dicts() == expected.weight_versions[0].to_dicts()


def test_refresh_rejects_misaligned_compact_records():
    output = rollout_replay.compact_token_logprobs(response(8))
    sample, _ = converted_sample(output, 8)
    sample.tokens[-1] += 1
    with pytest.raises(ValueError, match="not aligned"):
        policy_refresh.record_response(sample, output["meta_info"])


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


@pytest.mark.parametrize("compact,raw", [(False, False), (True, False), (True, True)])
def test_worker_opt_in_and_raw_bytes_contract(worker_class, compact, raw):
    body = response(12)

    async def exercise():
        worker = worker_class(concurrency=1)
        await worker._client.aclose()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as client:
            worker._client = client
            result = await worker.do_post("http://test/generate", {}, compact_token_logprobs=compact, return_bytes=raw)
        if raw:
            assert json.loads(result) == body
        elif compact:
            assert isinstance(result["meta_info"]["output_token_logprobs"], TokenLogprobs)
            assert result["meta_info"]["output_token_logprobs"] == body["meta_info"]["output_token_logprobs"]
        else:
            assert result == body

    asyncio.run(exercise())


@pytest.mark.parametrize("compact", [False, True])
def test_local_post_fallback_preserves_contract(monkeypatch, compact):
    body = response(12)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
        ) as client:
            monkeypatch.setattr(http_utils, "_http_client", client)
            result = await http_utils.post("http://test/generate", {}, compact_token_logprobs=compact)
        if compact:
            assert isinstance(result["meta_info"]["output_token_logprobs"], TokenLogprobs)
            assert result["meta_info"]["output_token_logprobs"] == body["meta_info"]["output_token_logprobs"]
        else:
            assert result == body

    asyncio.run(exercise())
