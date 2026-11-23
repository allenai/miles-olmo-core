"""core.refresh_request_timeout bounds generation, not delivery of a finished response.

kf-hero-miles-real-8node: all 747 "timed out" samples had finished at the HTTP
workers 1,416-1,738 s after admission, but reached the producer 963-2,452 s
later because the completed queue was full and the producer node's Ray object
store was spilling. They were counted as timeouts and the run aborted.

Abandoned requests (deadline or cancellation) are still aborted exactly once
(#48), and the worker never keeps a response nobody will fetch.
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
        olmo_core=SimpleNamespace(refresh_request_timeout=timeout, pipeline_observation_interval=0),
        save=None,
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


class _Aborts:
    def __init__(self):
        self.ids = []

    def abort(self, request_id):
        self.ids.append(request_id)


def _producer(args):
    fn = refreshing_rollout.RefreshingRolloutFn.__new__(refreshing_rollout.RefreshingRolloutFn)
    fn.args = args
    fn._request_aborter = _Aborts()
    return fn


def _generate(fn):
    args = fn.args
    state = SimpleNamespace(args=args, processor=None, tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1]))
    sample = Sample(index=0, group_index=0, prompt="question")
    return fn._generate_response_impl(GenerateFnInput(state, sample, {}, evaluation=False))


class _LateDeliveryWorker:
    """An HTTP worker whose generation finishes fast but whose response reaches the caller late.

    ``do_post`` is the single call the producer awaited before the fix: Ray
    resolves it only once the result has been transferred, so delivery time is
    inside it. The held-result calls split generation from delivery; the take
    waits before it runs, like a call queued behind a busy object store.
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
        await asyncio.sleep(self.delivery_seconds)
        return self.held.take(key)

    def _discard(self, key):
        self.discarded.append(key)
        self.held.discard(key)


@pytest.fixture
def no_ray_cancel(monkeypatch):
    cancelled = []
    monkeypatch.setattr(http_utils, "_cancel_remote_post", cancelled.append)
    return cancelled


def _use_workers(monkeypatch, workers):
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    monkeypatch.setattr(http_utils, "_post_actors", workers)
    monkeypatch.setattr(http_utils, "_post_actor_idx", 0)


def test_finished_generation_delivered_late_is_not_a_timeout(monkeypatch, no_ray_cancel):
    worker = _LateDeliveryWorker()
    _use_workers(monkeypatch, [worker])
    fn = _producer(_args(1, TIMEOUT_SECONDS))
    started = time.monotonic()
    output = asyncio.run(_generate(fn))
    # Delivery really was slower than the deadline; the sample still counts.
    assert time.monotonic() - started >= GENERATION_SECONDS + DELIVERY_SECONDS
    assert output.samples.tokens == [1, 7, 8]
    assert output.samples.rollout_log_probs == [-0.5, -0.25]
    assert len(worker.held) == 0 and worker.discarded == [] and no_ray_cancel == []
    assert fn._request_aborter.ids == []


def test_generation_that_never_finishes_times_out_and_is_aborted_once(monkeypatch, no_ray_cancel):
    worker = _LateDeliveryWorker(generation_seconds=60)
    _use_workers(monkeypatch, [worker])
    fn = _producer(_args(1, TIMEOUT_SECONDS))
    started = time.monotonic()
    with pytest.raises(GenerationRequestTimeout, match="refresh_request_timeout"):
        asyncio.run(_generate(fn))
    assert time.monotonic() - started < 5
    assert len(fn._request_aborter.ids) == 1
    assert len(worker.held) == 0


def test_unresponsive_worker_times_out_at_backstop_cancels_and_discards(monkeypatch, no_ray_cancel):
    worker = _LateDeliveryWorker()

    async def hung(key, *args, **kwargs):
        await asyncio.sleep(60)

    worker.post_and_hold = SimpleNamespace(remote=hung)
    _use_workers(monkeypatch, [worker])
    monkeypatch.setattr(http_utils, "_WORKER_STATUS_GRACE_SECONDS", 0.1)
    fn = _producer(_args(1, TIMEOUT_SECONDS))
    with pytest.raises(GenerationRequestTimeout):
        asyncio.run(_generate(fn))
    assert len(worker.discarded) == 1 and len(no_ray_cancel) == 1
    assert len(fn._request_aborter.ids) == 1


@pytest.mark.parametrize("cancel_after", [GENERATION_SECONDS / 2, GENERATION_SECONDS + DELIVERY_SECONDS / 2])
def test_cancel_aborts_once_and_releases_the_held_response(monkeypatch, no_ray_cancel, cancel_after):
    """A failed sibling cancels this request during generation or during delivery."""
    worker = _LateDeliveryWorker()
    _use_workers(monkeypatch, [worker])
    fn = _producer(_args(1, 30))

    async def scenario():
        task = asyncio.create_task(_generate(fn))
        await asyncio.sleep(cancel_after)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert len(fn._request_aborter.ids) == 1
    assert len(worker.discarded) == 1
    assert len(worker.held) == 0, "a cancelled request must not leave its response on the worker"
    # The worker call is cancelled only while generation is still running.
    assert len(no_ray_cancel) == (1 if cancel_after < GENERATION_SECONDS else 0)


def test_held_results_free_abandoned_keys():
    held = http_utils._HeldResults()

    async def finished():
        return "result"

    async def slow():
        await asyncio.sleep(0.05)
        return "result"

    async def scenario():
        # Abandoned while running: the result must not be retained.
        running = asyncio.create_task(held.run("running", slow(), 1))
        await asyncio.sleep(0)
        held.discard("running")
        await running
        # Abandoned after it finished: the held result is freed.
        await held.run("late", finished(), 1)
        held.discard("late")
        # Discarding a key that was already taken or never ran leaves nothing behind.
        held.discard("unknown")
        return await held.run("kept", finished(), 1)

    status = asyncio.run(scenario())
    assert "completed_unix" in status
    assert held._results == {"kept": "result"} and not held._running and not held._abandoned
    assert held.take("kept") == "result"
    assert len(held) == 0


class _Engine:
    """Hangs for ``delay`` seconds, then answers; records when the client closes the connection."""

    def __init__(self, delay):
        self.closed = threading.Event()
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.connection.settimeout(delay)
                try:
                    if self.connection.recv(1) == b"":
                        engine.closed.set()
                        return
                except TimeoutError:
                    pass
                except OSError:
                    engine.closed.set()
                    return
                body = json.dumps(RESPONSE).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def ray_worker(monkeypatch, ray_local_mode):
    for key, value in {"_post_actors": [], "_client_concurrency": 2, "_post_actor_idx": 0}.items():
        monkeypatch.setattr(http_utils, key, value)
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
    args = SimpleNamespace(num_gpus_per_node=1, http_post_actors_per_node=1, http_post_num_cpus=1)
    args.olmo_core = SimpleNamespace(pipeline_observation_interval=0)
    args.save = None
    http_utils._init_ray_distributed_post(args)
    (actor,) = http_utils._post_actors
    yield actor
    ray.kill(actor)


def _held_on(actor):
    deadline = time.monotonic() + 10
    while True:
        held = ray.get(actor.held_requests.remote())
        if held == 0 or time.monotonic() > deadline:
            return held
        time.sleep(0.05)


def test_real_worker_times_out_a_stuck_request_and_closes_it(ray_worker):
    """Ray end to end: the worker enforces the deadline and closes the engine connection."""
    with _Engine(delay=60) as engine:
        fn = _producer(_args(engine.server.server_port, 1.0))
        started = time.monotonic()
        with pytest.raises(GenerationRequestTimeout):
            asyncio.run(_generate(fn))
        assert time.monotonic() - started < 30
        assert engine.closed.wait(10), "worker deadline must close the engine connection"
    assert len(fn._request_aborter.ids) == 1
    assert _held_on(ray_worker) == 0


def test_real_worker_cancel_stops_the_call_and_holds_nothing(ray_worker):
    """Ray end to end: cancelling mid-generation ray.cancels the worker call and leaks nothing."""
    with _Engine(delay=3) as engine:
        fn = _producer(_args(engine.server.server_port, 30))

        async def scenario():
            task = asyncio.create_task(_generate(fn))
            await asyncio.sleep(1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        assert engine.closed.wait(10), "ray.cancel must close the engine connection"
    assert len(fn._request_aborter.ids) == 1
    assert _held_on(ray_worker) == 0


def test_real_worker_returns_a_response_and_releases_it(ray_worker):
    with _Engine(delay=0.2) as engine:
        output = asyncio.run(_generate(_producer(_args(engine.server.server_port, 30))))
    assert output.samples.tokens == [1, 7, 8]
    assert _held_on(ray_worker) == 0
