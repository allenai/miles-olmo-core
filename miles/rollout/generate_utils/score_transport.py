"""Native SGLang score-candidate transport; arrays stay arrays after decoding."""

import base64
import binascii
import secrets
from collections.abc import Mapping
from typing import Any

import numpy as np


def configure_score_transport(args, request: dict[str, Any], *, openai: bool) -> None:
    mode = getattr(args, "rollout_score_mode", "topk")
    transport = getattr(args, "rollout_score_transport", "nested")
    if mode == "topk" and transport == "nested":
        return
    if openai or getattr(args, "use_session_server", False):
        raise ValueError("Packed score transport currently requires native /generate, not the OpenAI/session API")
    sampling = request["sampling_params"]
    if sampling["top_k"] != -1 or sampling["top_p"] != 1.0 or sampling["min_p"] != 0.0:
        raise ValueError("Packed score transport requires unfiltered sampling: top_k=-1, top_p=1, min_p=0")
    if mode == "mc" and (args.rollout_top_logprobs_num not in (32, 64, 128) or transport != "array"):
        raise ValueError("SCMC requires 32, 64 or 128 candidates and array transport")
    # These are run-wide transport constraints, not per-request sampling overrides.
    required = {
        "return_logprob": True,
        "logprob_start_len": -1,
        "return_text_in_logprobs": False,
        "return_flat_raw_output_top_logprobs": True,
        "return_flat_raw_top_logprobs_b64": True,
        "stream": False,
    }
    for key, value in required.items():
        if key in request and request[key] != value:
            raise ValueError(f"Packed score transport requires {key}={value}")
    sample_required = {
        "array_output_logprobs": transport == "array",
        "score_centering_samples": args.rollout_top_logprobs_num if mode == "mc" else 0,
    }
    for key, value in sample_required.items():
        if key in sampling and sampling[key] != value:
            raise ValueError(f"Configured score mode/transport requires {key}={value}")
    request.update(required)
    # Flat transport remains usable on upstream servers without the array extension.
    if transport == "array":
        sampling.update(sample_required)
    if mode == "mc" and sampling.get("sampling_seed") is None:
        # The kernel requires an explicit seed. Preserve caller seeds; otherwise draw
        # a fresh seed per request instead of correlating every rollout with one seed.
        sampling["sampling_seed"] = secrets.randbits(63)


def decode_score_arrays(meta: Mapping[str, Any], *, rows: int, width: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode strict generated-token-only little-endian float32/int32 buffers.

    No tolist, sorting, deduplication or renormalization: SCMC multiplicity and
    exact behavior-policy log probabilities are part of the estimator input.
    """
    expected_mode = f"mc{width}" if mode == "mc" else "topk"
    if meta.get("output_score_mode", "topk") != expected_mode:
        raise ValueError(f"Expected output_score_mode={expected_mode}")
    shape = meta.get("output_top_logprobs_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != 2 or any(type(x) is not int for x in shape):
        raise ValueError("Packed scores require a two-dimensional integer shape")
    if list(shape) != [rows, width] or meta.get("output_top_logprobs_null_prefix") != 0:
        raise ValueError(
            f"Packed scores must cover every generated token with shape {[rows, width]} and no null prefix"
        )
    arrays = []
    for field, dtype in (("idx", "int32"), ("val", "float32")):
        key = f"output_top_logprobs_{field}_flat_b64"
        if meta.get(key + "_dtype") != dtype:
            raise ValueError(f"{key} requires dtype {dtype}")
        encoded = meta.get(key)
        size = rows * width * 4
        if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
            raise ValueError(f"{key} has an invalid encoded byte length")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"{key} is not valid base64") from exc
        if len(data) != size:
            raise ValueError(f"{key} has an invalid decoded byte length")
        arrays.append(np.frombuffer(data, dtype="<i4" if field == "idx" else "<f4").reshape(rows, width))
    ids, logps = arrays
    valid = ids >= 0
    if (ids < -1).any() or np.isnan(logps).any() or (logps > 1e-5).any():
        raise ValueError("Packed scores contain invalid token IDs or log probabilities")
    if not np.array_equal(valid, np.isfinite(logps)) or (mode == "mc" and not valid.all()):
        raise ValueError("Packed score padding must be ID -1/logp -inf; SCMC probes cannot be padded")
    return ids, logps
