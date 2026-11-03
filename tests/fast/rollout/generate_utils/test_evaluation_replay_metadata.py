"""Evaluation must not request training replay buffers from shared engines."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest

from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import single_turn
from miles.utils.types import Sample


@pytest.mark.parametrize("variant", ["single_turn", "legacy"])
@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("routing,indexer", [(False, False), (True, False), (False, True), (True, True)])
def test_generation_collects_replay_buffers_only_for_training(monkeypatch, variant, evaluation, routing, indexer):
    args = SimpleNamespace(
        ci_test=False,
        use_rollout_routing_replay=routing,
        use_rollout_indexer_replay=indexer,
        use_sampling_support_replay=False,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        rollout_temperature=1.0,
        sglang_speculative_algorithm=None,
        sglang_router_policy="round_robin",
        sglang_router_ip="unused",
        sglang_router_port=1,
        num_layers=1,
    )
    state = SimpleNamespace(args=args, processor=None, tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1, 2]))
    requests = []
    replay = np.ones((2, 1, 2), dtype=np.int32)

    async def post(url, payload, **kwargs):
        requests.append(payload)
        meta = {
            "output_token_logprobs": [[-0.2, 3, None]],
            "finish_reason": {"type": "stop"},
            "prompt_tokens": 2,
            "weight_version": "0",
        }
        if payload.get("return_routed_experts"):
            meta["routed_experts"] = pybase64.b64encode(replay.tobytes()).decode("ascii")
        if payload.get("return_indexer_topk"):
            meta["indexer_topk"] = pybase64.b64encode(replay.tobytes()).decode("ascii")
            meta["indexer_topk_num_layers"] = 1
        return {"text": "answer", "meta_info": meta}

    sample = Sample(prompt="question")
    params = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "max_new_tokens": 8}
    if variant == "single_turn":
        monkeypatch.setattr(single_turn, "post", post)
        asyncio.run(single_turn.generate(GenerateFnInput(state, sample, params, evaluation)))
    else:
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda unused: state)
        monkeypatch.setattr(sglang_rollout, "post", post)
        asyncio.run(sglang_rollout.generate(args, sample, params, evaluation=evaluation))

    assert len(requests) == 1
    assert requests[0].get("return_routed_experts", False) is (routing and not evaluation)
    assert requests[0].get("return_indexer_topk", False) is (indexer and not evaluation)
    assert requests[0]["return_logprob"] is True
    assert sample.tokens == [1, 2, 3]
    assert sample.response == "answer" and sample.rollout_log_probs == [-0.2]
    assert sample.status == Sample.Status.COMPLETED
    for enabled, buffer in ((routing, sample.rollout_routed_experts), (indexer, sample.rollout_indexer_topk)):
        if enabled and not evaluation:
            np.testing.assert_array_equal(buffer, replay)
        else:
            assert buffer is None
    # Evaluation may run beside training; never disable replay in shared launch state.
    assert args.use_rollout_routing_replay is routing
    assert args.use_rollout_indexer_replay is indexer
