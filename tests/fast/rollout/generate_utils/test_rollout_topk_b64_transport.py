"""Base64 candidate arrays from SGLang must match the nested candidate records."""

import pickle

import numpy as np
import pybase64
import pytest
from tests.fast.fixtures.score_centering_fixtures import _args

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
)
from miles.utils.rollout_replay import compact_token_logprobs
from miles.utils.types import Sample

_LOGPS = np.array([[-0.5, -1.25, -2.0], [-0.125, -2.5, -3.0]], dtype=np.float32)
_IDS = np.array([[7, 2, 5], [3, 4, 1]], dtype=np.int32)


def _nested_meta(logps=_LOGPS, ids=_IDS) -> dict:
    return {
        "output_token_logprobs": [(float(row[0]), int(i[0]), None) for row, i in zip(logps, ids, strict=True)],
        "output_top_logprobs": [
            [(float(lp), int(t), None) for lp, t in zip(row, i, strict=True)]
            for row, i in zip(logps, ids, strict=True)
        ],
    }


def _b64_meta(logps=_LOGPS, ids=_IDS) -> dict:
    meta = _nested_meta(logps, ids)
    del meta["output_top_logprobs"]
    meta.update(
        output_top_logprobs_val_b64=pybase64.b64encode(np.ascontiguousarray(logps, "<f4").tobytes()).decode(),
        output_top_logprobs_idx_b64=pybase64.b64encode(np.ascontiguousarray(ids, "<i4").tobytes()).decode(),
        output_top_logprobs_shape=list(logps.shape),
    )
    return meta


def _append(meta: dict, k: int = 3) -> Sample:
    sample = Sample()
    append_rollout_topk_logprobs(sample, meta, k)
    return sample


def test_native_selected_requests_request_base64_candidates() -> None:
    request = {"sampling_params": {}}
    configure_rollout_topk_logprobs_request(_args(rollout_top_logprobs_num=20), request)
    assert request["top_logprobs_num"] == 20
    assert request["return_output_top_logprobs_b64"] is True


@pytest.mark.parametrize("openai", [False, True])
def test_support_mode_and_openai_keep_their_formats(openai: bool) -> None:
    request = {} if openai else {"sampling_params": {}}
    configure_rollout_topk_logprobs_request(_args(rollout_top_logprobs_num=20), request, openai=openai)
    assert ("return_output_top_logprobs_b64" in request) is not openai
    request = {"top_p": 0.9} if openai else {"sampling_params": {"top_p": 0.9}}
    configure_rollout_topk_logprobs_request(
        _args(rollout_top_logprobs_num=20, rollout_sampling_logprobs_mode="support"), request, openai=openai
    )
    assert "return_output_top_logprobs_b64" not in request


def test_nested_consumer_keeps_nested_candidates() -> None:
    # Top-k OPD sets top_logprobs_num first and reads nested output_top_logprobs.
    request = {"sampling_params": {}, "top_logprobs_num": 8}
    configure_rollout_topk_logprobs_request(_args(rollout_top_logprobs_num=20), request)
    assert "return_output_top_logprobs_b64" not in request


def test_base64_matches_nested_records() -> None:
    nested, b64 = _append(_nested_meta()), _append(_b64_meta())
    np.testing.assert_array_equal(b64.rollout_topk_token_ids, nested.rollout_topk_token_ids)
    np.testing.assert_array_equal(b64.rollout_topk_log_probs, nested.rollout_topk_log_probs)
    assert b64.rollout_topk_token_ids.dtype == np.int32
    assert b64.rollout_topk_log_probs.dtype == np.float32
    assert b64.rollout_topk_token_ids.flags.writeable


def test_base64_appends_across_turns() -> None:
    sample = _append(_b64_meta())
    append_rollout_topk_logprobs(sample, _b64_meta(_LOGPS[:1], _IDS[:1]), 3)
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, np.concatenate((_IDS, _IDS[:1])))


def test_narrower_server_width_is_padded() -> None:
    sample = _append(_b64_meta(_LOGPS[:, :2].copy(), _IDS[:, :2].copy()), k=3)
    assert sample.rollout_topk_token_ids[:, 2].tolist() == [-1, -1]
    assert np.isneginf(sample.rollout_topk_log_probs[:, 2]).all()


def test_row_count_mismatch_is_rejected() -> None:
    meta = _b64_meta()
    meta["output_token_logprobs"] = meta["output_token_logprobs"][:1]
    with pytest.raises(ValueError, match="every generated token"):
        _append(meta)


def test_buffer_size_mismatch_is_rejected() -> None:
    meta = _b64_meta()
    meta["output_top_logprobs_shape"] = [2, 4]
    with pytest.raises(ValueError, match="do not match"):
        _append(meta, k=4)


def test_http_worker_decodes_buffers_once() -> None:
    response = compact_token_logprobs({"meta_info": _b64_meta(), "text": ""})
    meta = response["meta_info"]
    assert isinstance(meta["output_top_logprobs_val_b64"], np.ndarray)
    assert isinstance(meta["output_top_logprobs_idx_b64"], np.ndarray)
    meta = pickle.loads(pickle.dumps(meta))
    sample = _append(meta)
    np.testing.assert_array_equal(sample.rollout_topk_token_ids, _IDS)
    np.testing.assert_array_equal(sample.rollout_topk_log_probs, _LOGPS)
    # Already compact responses are unchanged.
    assert compact_token_logprobs(response) is response
