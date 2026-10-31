"""Real Ray workers decode refresh responses before the producer converts samples."""

import asyncio
import base64
import json
import os
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import ray

from miles.backends.core_utils.rollout import refreshing_rollout, sibling_timing
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_utils import generate_endpoint_utils
from miles.utils import http_utils, rollout_replay
from miles.utils.types import Sample


def test_refresh_uses_worker_parsing_and_preserves_replay(monkeypatch, tmp_path):
    length, layers, experts = 2048, 16, 16
    score = -0.12345678901234567
    routes = np.full((length, layers, experts), 7, dtype=np.int32)
    body = json.dumps(
        {
            "text": "Answer: café → ✓",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 1,
                "output_token_logprobs": [[score, i + 10] for i in range(length)],
                "weight_version": "0",
                "weight_versions": [{"version": "0", "start": 0, "end": length}],
                "routed_experts": base64.b64encode(routes.tobytes()).decode(),
            },
        },
        ensure_ascii=False,
    ).encode()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload["rid"])
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key, value in {
        "_post_actors": [],
        "_client_concurrency": 4,
        "_distributed_post_enabled": True,
        "_post_actor_idx": 0,
    }.items():
        monkeypatch.setattr(http_utils, key, value)
    args = SimpleNamespace(
        num_gpus_per_node=1,
        http_post_actors_per_node=2,
        http_post_num_cpus=1,
        olmo_core=SimpleNamespace(pipeline_observation_interval=2, refresh_request_timeout=30),
        save=str(tmp_path),
        sglang_router_ip="127.0.0.1",
        sglang_router_port=server.server_port,
        rollout_max_response_len=length,
        rollout_max_context_len=length + 1,
        use_sampling_support_replay=False,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=False,
        rollout_top_logprobs_num=0,
        sglang_speculative_algorithm=None,
        num_layers=layers,
    )
    state = SimpleNamespace(args=args, processor=None, tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1]))
    samples = [Sample(index=i, group_index=0, prompt="question") for i in range(4)]
    sibling_timing.start_group(samples)
    native_decode = generate_endpoint_utils.decode_int32_buffer
    received_arrays = []

    def require_worker_decoded(value):
        assert isinstance(value, np.ndarray), "Producer must not receive base64 replay"
        received_arrays.append(value)
        return native_decode(value)

    monkeypatch.setattr(generate_endpoint_utils, "decode_int32_buffer", require_worker_decoded)
    native_update = refreshing_rollout.update_sample_from_response
    received_records = []

    async def require_worker_compacted(args, sample, payload, output):
        records = output["meta_info"]["output_token_logprobs"]
        assert isinstance(records, rollout_replay.TokenLogprobs), "Producer must not receive per-token lists"
        received_records.append(records)
        return await native_update(args, sample, payload, output)

    monkeypatch.setattr(refreshing_rollout, "update_sample_from_response", require_worker_compacted)
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        _memory=512 * 1024**2,
    )
    try:
        http_utils._init_ray_distributed_post(args)

        async def exercise():
            return await asyncio.gather(
                *(
                    refreshing_rollout.RefreshingRolloutFn._generate_response_impl(
                        SimpleNamespace(args=args), GenerateFnInput(state, sample, {}, evaluation=False)
                    )
                    for sample in samples
                )
            )

        results = asyncio.run(exercise())
        assert len(received_arrays) == 4
        assert len(received_records) == 4 and all(len(r) == length for r in received_records)
        assert len(set(requests)) == len(requests) == 4
        paths = list(tmp_path.glob("http_post_worker_*.jsonl"))
        assert len(paths) == 2
        assert all(not p.stem.endswith(f"_{os.getpid()}") for p in paths)
        observations = {
            r["request_id"]: r for p in paths for r in map(json.loads, p.read_text().splitlines())
        }
        for result in results:
            sample = result.samples
            np.testing.assert_array_equal(sample.rollout_routed_experts, routes)
            assert sample.tokens == [1, *range(10, length + 10)]
            assert all(struct.pack("d", v) == struct.pack("d", score) for v in sample.rollout_log_probs)
            assert sample.train_metadata["policy_refresh"]["replay_version"] == 0
            timing = sample.metadata["sibling_timing"]
            worker = observations[timing["request_id"]]
            assert worker["response_bytes"] == len(body)
            assert worker["json_decode_seconds"] >= 0
            assert worker["json_decode_thread_cpu_seconds"] >= 0
            assert worker["replay_decode_seconds"] >= 0
            assert worker["replay_decode_thread_cpu_seconds"] >= 0
            assert worker["replay_decoded_bytes"] == routes.nbytes
            assert worker["token_logprobs_count"] == length
            assert worker["token_logprobs_compact_seconds"] >= 0
            assert (
                worker["response_body_ready_unix"]
                <= worker["response_ready_unix"]
                <= timing["response_decoded_ready_unix"]
            )
            assert "json_decode_seconds" not in timing
            assert "response_bytes_ready_unix" not in timing
    finally:
        for actor in http_utils._post_actors:
            ray.kill(actor)
        ray.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
