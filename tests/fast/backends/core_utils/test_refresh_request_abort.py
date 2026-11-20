"""Requests the producer abandons are aborted on the engines, not left decoding."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest

from miles.backends.core_utils.rollout import prompt_preparation, refreshing_rollout, request_abort
from miles.backends.core_utils.rollout.errors import GenerationRequestTimeout
from miles.backends.core_utils.rollout.generation_admission import GenerationAdmission
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout import inference_rollout_common
from miles.utils.types import Sample


class Engine:
    """A stand-in SGLang server that records /abort_request bodies."""

    def __init__(self):
        self.aborts = []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/abort_request":
                    engine.aborts.append(body)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def engines():
    started = [Engine(), Engine()]
    yield started
    for engine in started:
        engine.close()


def make_args(tmp_path, timeout):
    return SimpleNamespace(
        olmo_core=SimpleNamespace(pipeline_observation_interval=0, refresh_request_timeout=timeout),
        save=str(tmp_path),
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,
        sglang_router_policy="round_robin",
        rollout_max_response_len=10,
        rollout_max_context_len=100,
        rollout_top_logprobs_num=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        sglang_speculative_algorithm=None,
        partial_rollout=False,
        group_rm=False,
        sglang_enable_deterministic_inference=False,
        rollout_seed=0,
    )


def make_producer(args, engines):
    fn = refreshing_rollout.RefreshingRolloutFn.__new__(refreshing_rollout.RefreshingRolloutFn)
    fn.args = args

    async def urls():
        return [engine.url for engine in engines]

    fn._request_aborter = request_abort.RequestAborter(urls)
    return fn


def aborted(engines):
    rids = [sorted(body["rid"] for body in engine.aborts) for engine in engines]
    assert all(rows == rids[0] for rows in rids), "every engine gets every abort"
    assert all(set(body) == {"rid"} for engine in engines for body in engine.aborts), "never abort_all"
    return rids[0]


def test_request_deadline_aborts_the_engine_request(monkeypatch, tmp_path, engines):
    payloads = []

    async def hang(url, payload, **kwargs):
        payloads.append(payload)
        await asyncio.Event().wait()

    monkeypatch.setattr(refreshing_rollout, "post", hang)
    args = make_args(tmp_path, timeout=0.05)
    fn = make_producer(args, engines)
    state = SimpleNamespace(args=args, tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1, 2]))

    async def scenario():
        sample = Sample(index=0, group_index=3, prompt=[1, 2])
        with pytest.raises(GenerationRequestTimeout, match="refresh_request_timeout"):
            await fn._generate_response_impl(GenerateFnInput(state, sample, {}, evaluation=False))
        await fn._request_aborter.wait_idle()

    monkeypatch.setattr(refreshing_rollout, "compute_prompt_ids_from_sample", lambda state, sample: [1, 2])
    asyncio.run(scenario())
    assert aborted(engines) == [payloads[0]["rid"]]


def test_failed_sibling_aborts_cancelled_siblings_but_not_itself_or_finished(monkeypatch, tmp_path, engines):
    payloads = {}
    failing = httpx.ConnectError("upstream reset")

    async def post(url, payload, **kwargs):
        payloads[payload["rid"]] = payload
        index = len(payloads) - 1
        if index == 0:
            return {"meta_info": {"finish_reason": {"type": "stop"}}}
        if index == 1:
            await asyncio.sleep(0.01)
            raise failing
        await asyncio.Event().wait()

    async def update(args, sample, payload, output):
        sample.tokens = list(payload["input_ids"])

    async def reward(args, sample):
        return 1

    monkeypatch.setattr(refreshing_rollout, "post", post)
    monkeypatch.setattr(refreshing_rollout, "update_sample_from_response", update)
    monkeypatch.setattr(
        refreshing_rollout.policy_refresh, "record_response", lambda *a: {"spans": [], "replay_version": 0}
    )
    monkeypatch.setattr(inference_rollout_common, "async_rm", reward)
    args = make_args(tmp_path, timeout=30)

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode())

    async def scenario():
        fn = make_producer(args, engines)
        fn._worker = None
        fn._scheduler = SimpleNamespace(sample_done_callback=lambda: None)
        fn.state = SimpleNamespace(
            args=args,
            aborted=False,
            sampling_params={},
            generate_function=fn._generate_response,
            generate_fn_semaphore=GenerationAdmission(8),
        )
        fn._prompt_preparer = prompt_preparation.PromptPreparer(Tokenizer)
        fn._prepared_prompt_ids = {}
        try:
            samples = [Sample(prompt="same", index=i, group_index=0) for i in range(8)]
            with pytest.raises(httpx.ConnectError):
                await fn._generate_group(samples)
            await fn._request_aborter.wait_idle()
        finally:
            await fn._prompt_preparer.close()

    asyncio.run(scenario())
    rids = list(payloads)
    assert len(rids) == 8
    # The finished request and the failed one are not running; the six cancelled siblings are.
    assert aborted(engines) == sorted(rids[2:])


def test_completed_request_sends_no_abort(monkeypatch, tmp_path, engines):
    async def post(url, payload, **kwargs):
        return {"meta_info": {"finish_reason": {"type": "length"}}}

    async def update(args, sample, payload, output):
        sample.tokens = list(payload["input_ids"])

    monkeypatch.setattr(refreshing_rollout, "post", post)
    monkeypatch.setattr(refreshing_rollout, "update_sample_from_response", update)
    monkeypatch.setattr(refreshing_rollout, "compute_prompt_ids_from_sample", lambda state, sample: [1])
    monkeypatch.setattr(
        refreshing_rollout.policy_refresh, "record_response", lambda *a: {"spans": [], "replay_version": 0}
    )
    args = make_args(tmp_path, timeout=30)
    fn = make_producer(args, engines)

    async def scenario():
        await fn._generate_response_impl(
            GenerateFnInput(SimpleNamespace(args=args), Sample(index=0), {}, evaluation=False)
        )
        await fn._request_aborter.wait_idle()

    asyncio.run(scenario())
    assert fn._request_aborter.requested == 0
    assert aborted(engines) == []


def test_unreachable_engines_and_discovery_failure_never_raise(engines):
    calls = []

    async def urls():
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("router down")
        return [engines[0].url, "http://127.0.0.1:9"]

    async def scenario():
        aborter = request_abort.RequestAborter(urls)
        aborter.abort("a")
        await aborter.wait_idle()
        # Discovery failure drops the batch; the next abort rediscovers engines.
        aborter.abort("b")
        aborter.abort("c")
        await aborter.wait_idle()
        return aborter

    aborter = asyncio.run(scenario())
    assert len(calls) == 2
    assert sorted(body["rid"] for body in engines[0].aborts) == ["b", "c"]
    assert aborter.requested == 3 and aborter.failed_posts == 2
