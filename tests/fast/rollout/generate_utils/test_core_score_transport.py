"""Fork contracts for SCT/SCMC extraction and array-preserving Core transport."""

import asyncio
import base64
from copy import deepcopy

import numpy as np
import pytest
import torch
from tests.fast.fixtures.score_centering_fixtures import _args, _Tokenizer

from miles.backends.core_utils.data import sample_batches
from miles.backends.core_utils.packing import bind_training_fields, combine
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_raw
from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
    validate_rollout_topk_logprobs_sample,
)
from miles.rollout.generate_utils.sample_utils import merge_samples
from miles.rollout.generate_utils.score_transport import decode_score_arrays
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample, encode_samples
from miles.utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_args
from miles.utils.types import Sample


def args_for(mode="topk", transport="array", k=32, **extra):
    return _args(
        rollout_score_mode=mode,
        rollout_score_transport=transport,
        rollout_top_logprobs_num=k,
        loss_type="policy_loss" if mode == "mc" else "score_centering",
        rollout_max_response_len=20,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        sglang_speculative_algorithm=None,
        **extra,
    )


def response(k, mode="topk", transport="array"):
    # MC has intentional duplicates; every selected token has exactly the same saved logq.
    ids = np.tile(np.arange(k, dtype=np.int32) if mode == "topk" else np.arange(k, dtype=np.int32) % 8, (2, 1))
    logps = np.full((2, k), -5.0, dtype=np.float32)
    meta = {"output_token_logprobs": [(-5.0, 2, None), (-5.0, 3, None)], "finish_reason": {"type": "stop"}}
    if transport == "nested":
        meta["output_top_logprobs"] = [
            [(float(p), int(i), None) for i, p in zip(ii, pp, strict=True)] for ii, pp in zip(ids, logps, strict=True)
        ]
    else:
        meta.update(
            output_top_logprobs_shape=[2, k],
            output_top_logprobs_null_prefix=0,
            output_score_mode=f"mc{k}" if mode == "mc" else "topk",
        )
        for name, value in (("idx", ids), ("val", logps)):
            meta[f"output_top_logprobs_{name}_flat_b64"] = base64.b64encode(value.tobytes()).decode()
            meta[f"output_top_logprobs_{name}_flat_b64_dtype"] = str(value.dtype)
    return {"text": "result", "meta_info": meta}, ids, logps


@pytest.mark.parametrize("k", [32, 64, 128])
@pytest.mark.parametrize("mode,transport", [("topk", "nested"), ("topk", "flat"), ("topk", "array"), ("mc", "array")])
def test_native_wire_dp_and_core_packing(k, mode, transport):
    args = args_for(mode, transport, k, ci_test=True)
    validate_rollout_topk_logprobs_args(args)
    payload, status = compute_request_payload(args, [0, 1], {"sampling_seed": 17})
    assert status is None and payload["sampling_params"]["top_k"] == -1
    assert payload["top_logprobs_num"] == k
    if transport != "nested":
        assert payload["logprob_start_len"] == -1 and payload["stream"] is False
        assert payload["return_flat_raw_top_logprobs_b64"]
    if mode == "mc":
        assert payload["sampling_params"]["score_centering_samples"] == k
        assert payload["sampling_params"]["sampling_seed"] == 17
    output, ids, logps = response(k, mode, transport)
    first = Sample(index=0, group_index=0, reward=1.0)
    asyncio.run(update_sample_from_response(args, first, payload, output, True))
    validate_rollout_topk_logprobs_sample(first, k)
    second = deepcopy(first)
    second.tokens = first.tokens + [9] + [2, 3]
    merged = merge_samples([first, second], _Tokenizer())
    merged.strip_last_output_tokens(1, _Tokenizer())
    restored = decode_samples_and_merge_input_sample(encode_samples([merged], {}), Sample()).samples[0]
    assert restored.rollout_score_mode == mode
    assert restored.rollout_topk_token_ids.dtype == np.int32
    assert restored.rollout_topk_log_probs.dtype == np.float32
    np.testing.assert_array_equal(restored.rollout_topk_token_ids[:2], ids)
    np.testing.assert_array_equal(restored.rollout_topk_log_probs[:2], logps)
    assert (restored.rollout_topk_token_ids[2] == -1).all()
    assert np.isneginf(restored.rollout_topk_log_probs[2]).all()
    restored.index, restored.reward = 0, 1.0
    other = deepcopy(restored)
    other.index, other.reward = 1, -1.0
    data = convert_samples_to_train_data(args, [restored, other], {}, None, None)
    shard = split_train_data_by_dp_raw(args, data, dp_size=2)[1]
    assert shard["rollout_score_mode"] == [mode]
    shard["total_lengths"] = [len(restored.tokens)]
    shard["tokens"] = [torch.tensor(restored.tokens)]
    shard["loss_masks"] = [torch.tensor(restored.loss_mask)]
    batches = sample_batches(shard, 32)
    packed = combine(batches, [0])
    bind_training_fields([[packed]], shard)
    assert packed["rollout_score_mode"] == [mode]
    np.testing.assert_array_equal(packed["rollout_topk_token_ids"][0], restored.rollout_topk_token_ids)
    np.testing.assert_array_equal(packed["rollout_topk_log_probs"][0], restored.rollout_topk_log_probs)
    if mode == "mc":
        with pytest.raises(ValueError, match="cannot consume SCMC"):
            score_centering_loss_function(args, packed, None, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_top_logprobs_shape", [1, 32]),
        ("output_top_logprobs_shape", [2, 64]),
        ("output_top_logprobs_shape", [2.0, 32]),
        ("output_top_logprobs_null_prefix", 1),
        ("output_top_logprobs_val_flat_b64_dtype", "float64"),
        ("output_top_logprobs_idx_flat_b64_dtype", "int64"),
        ("output_top_logprobs_val_flat_b64", "bad!"),
        ("output_score_mode", "mc32"),
    ],
)
def test_reject_malformed_packed_scores(field, value):
    output, _, _ = response(32)
    output["meta_info"][field] = value
    with pytest.raises(ValueError):
        decode_score_arrays(output["meta_info"], rows=2, width=32, mode="topk")


def test_missing_packed_data_is_not_silently_replaced_with_nested_rows():
    output, _, _ = response(32, transport="nested")
    with pytest.raises(ValueError, match="acknowledgement"):
        append_rollout_topk_logprobs(Sample(), output["meta_info"], 32, transport="array")


def test_mc_requires_server_ack_and_preserves_repeated_draws():
    output, ids, _ = response(32, "mc")
    del output["meta_info"]["output_score_mode"]
    with pytest.raises(ValueError, match="output_score_mode"):
        decode_score_arrays(output["meta_info"], rows=2, width=32, mode="mc")
    assert np.unique(ids[0]).size < 32


@pytest.mark.parametrize(
    "extra",
    [
        {"rollout_top_logprobs_num": 0},
        {"rollout_top_logprobs_num": 16},
        {"rollout_score_transport": "nested"},
        {"rollout_score_transport": "flat"},
        {"use_session_server": True},
        {"rollout_sampling_logprobs_mode": "support"},
    ],
)
def test_invalid_mc_launch_rejected(extra):
    args = args_for("mc")
    vars(args).update(extra)
    with pytest.raises(ValueError):
        validate_rollout_topk_logprobs_args(args)


@pytest.mark.parametrize("sampling", [{"top_p": 0.9}, {"top_k": 8}, {"min_p": 0.1}, {"array_output_logprobs": False}])
def test_reject_incompatible_request(sampling):
    with pytest.raises(ValueError):
        compute_request_payload(args_for("mc"), [0, 1], sampling)


def test_seed_default_is_per_request_and_openai_is_rejected():
    args = args_for("mc")
    first, _ = compute_request_payload(args, [0, 1], {})
    second, _ = compute_request_payload(args, [0, 1], {})
    assert first["sampling_params"]["sampling_seed"] != second["sampling_params"]["sampling_seed"]
    with pytest.raises(ValueError, match="native /generate"):
        configure_rollout_topk_logprobs_request(args, {}, openai=True)


def test_sct_still_rejects_duplicate_ids():
    output, _, _ = response(32, "mc")
    output["meta_info"]["output_score_mode"] = "topk"
    sample = Sample(tokens=[0, 2, 3], response_length=2, rollout_log_probs=[-5.0, -5.0])
    append_rollout_topk_logprobs(sample, output["meta_info"], 32, transport="array")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rollout_topk_logprobs_sample(sample, 32)
