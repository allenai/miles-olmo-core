"""core.refresh_request_timeout bounds generation, not delivery of a finished response.

kf-hero-miles-real-8node: all 747 "timed out" samples had finished at the HTTP
workers 1,416-1,738 s after admission, but reached the producer 963-2,452 s
later because the completed queue was full and the producer node's Ray object
store was spilling. They were counted as timeouts and the run aborted.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import ray

from miles.backends.core_utils.rollout import refreshing_rollout
from miles.backends.core_utils.rollout.errors import GenerationRequestTimeout
from miles.rollout.base_types import GenerateFnInput
from miles.utils import http_utils
from miles.utils.types import Sample

GENERATION_SECONDS = 0.05
DELIVERY_SECONDS = 0.6
TIMEOUT_SECONDS = 0.3

RESPONSE = {
    "text": "ok",
    "meta_info": {
        "finish_reason": {"type": "stop"},
        "prompt_tokens": 1,
        "output_token_logprobs": [[-0.5, 7], [-0.25, 8]],
        "weight_version": "0",
        "weight_versions": [{"version": "0", "start": 0, "end": 2}],
    },
}


def _args(port, timeout):
    return SimpleNamespace(
        olmo_core=SimpleNamespace(refresh_request_timeout=timeout),
        sglang_router_policy="consistent_hashing",
        sglang_router_ip="127.0.0.1",
        sglang_router_port=port,
        rollout_max_response_len=2,
        rollout_max_context_len=3,
        use_sampling_support_replay=False,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        rollout_top_logprobs_num=0,
        sglang_speculative_algorithm=None,
    )


def _generate(args):
    state = SimpleNamespace(args=args, processor=None, tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1]))
    sample = Sample(index=0, group_index=0, prompt="question")
    return refreshing_rollout.RefreshingRolloutFn._generate_response_impl(
        SimpleNamespace(args=args), GenerateFnInput(state, sample, {}, evaluation=False)
    )


class _LateDeliveryWorker:
    """An HTTP worker whose generation finishes fast but whose response reaches the caller late.

    ``do_post`` is the single call the producer awaited before the fix: Ray
    resolves it only once the result has been transferred, so delivery time is
    inside it. The held-result calls split generation from delivery.
    """

    def __init__(self, generation_seconds=GENERATION_SECONDS, delivery_seconds=DELIVERY_SECONDS):
        self.generation_seconds = generation_seconds
        self.delivery_seconds = delivery_seconds
        self.held = http_utils._HeldResults()
        self.discarded = []
        self.do_post = SimpleNamespace(remote=self._do_post)
        self.post_and_hold = SimpleNamespace(remote=self._post_and_hold)
        self.take_result = SimpleNamespace(remote=self._take_result)
        self.discard_result = SimpleNamespace(remote=self._discard)

    async def _generation(self):
        await asyncio.sleep(self.generation_seconds)
        return json.loads(json.dumps(RESPONSE))

    async def _do_post(self, *args, **kwargs):
        result = await self._generation()
        await asyncio.sleep(self.delivery_seconds)
        return result

    async def _post_and_hold(self, key, *args, timeout, **kwargs):
        return await self.held.run(key, self._generation(), timeout)

    async def _take_result(self, key):
        result = self.held.take(key)
        await asyncio.sleep(self.delivery_seconds)
        return result

    def _discard(self, key):
        self.discarded.append(key)
        self.held.discard(key)


def _use_workers(monkeypatch, workers):
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", workers)
    monkeypatch.setattr(http_utils, "_post_actor_idx", 0)


def test_finished_generation_delivered_late_is_not_a_timeout(monkeypatch):
    worker = _LateDeliveryWorker()
    _use_workers(monkeypatch, [worker])
    started = time.monotonic()
    output = asyncio.run(_generate(_args(1, TIMEOUT_SECONDS)))
    # Delivery really was slower than the deadline; the sample still counts.
    assert time.monotonic() - started >= GENERATION_SECONDS + DELIVERY_SECONDS
    assert output.samples.tokens == [1, 7, 8]
    assert output.samples.rollout_log_probs == [-0.5, -0.25]
    assert worker.held._results == {} and worker.discarded == []


def test_generation_that_never_finishes_still_times_out(monkeypatch):
    worker = _LateDeliveryWorker(generation_seconds=60)
    _use_workers(monkeypatch, [worker])
    started = time.monotonic()
    with pytest.raises(GenerationRequestTimeout, match="refresh_request_timeout"):
        asyncio.run(_generate(_args(1, TIMEOUT_SECONDS)))
    assert time.monotonic() - started < 5
    assert worker.held._results == {}


def test_unresponsive_worker_times_out_at_backstop_and_discards(monkeypatch):
    worker = _LateDeliveryWorker()

    async def hung(key, *args, **kwargs):
        await asyncio.sleep(60)

    worker.post_and_hold = SimpleNamespace(remote=hung)
    _use_workers(monkeypatch, [worker])
    monkeypatch.setattr(http_utils, "_WORKER_STATUS_GRACE_SECONDS", 0.1)
    with pytest.raises(GenerationRequestTimeout):
        asyncio.run(_generate(_args(1, TIMEOUT_SECONDS)))
    assert len(worker.discarded) == 1


def test_held_results_free_abandoned_keys():
    held = http_utils._HeldResults()

    async def finished():
        return "result"

    async def scenario():
        # Abandoned before it finished: the result must not be retained.
        held.discard("early")
        await held.run("early", finished(), 1)
        # Abandoned after it finished: the held result is freed.
        await held.run("late", finished(), 1)
        held.discard("late")
        status = await held.run("kept", finished(), 1)
        return status

    status = asyncio.run(scenario())
    assert "completed_unix" in status
    assert held._results == {"kept": "result"} and held._abandoned == set()
    assert held.take("kept") == "result"


def test_real_worker_aborts_a_stuck_request_at_the_deadline(monkeypatch, ray_local_mode):
    """Ray end to end: the worker enforces the deadline and closes the engine connection."""
    closed = threading.Event()

    class Hang(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            # Never respond; report when the client gives up on the connection.
            self.connection.settimeout(10)
            try:
                if self.connection.recv(1) == b"":
                    closed.set()
            except OSError:
                closed.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Hang)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for key, value in {"_post_actors": [], "_client_concurrency": 2, "_post_actor_idx": 0}.items():
        monkeypatch.setattr(http_utils, key, value)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    args = _args(server.server_port, 1.0)
    args.num_gpus_per_node = 1
    args.http_post_actors_per_node = 1
    args.http_post_num_cpus = 1
    args.olmo_core.pipeline_observation_interval = 0
    args.save = None
    try:
        http_utils._init_ray_distributed_post(args)
        started = time.monotonic()
        with pytest.raises(GenerationRequestTimeout):
            asyncio.run(_generate(args))
        assert time.monotonic() - started < 30
        assert closed.wait(10), "worker deadline must close the engine connection"
    finally:
        for actor in http_utils._post_actors:
            ray.kill(actor)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
