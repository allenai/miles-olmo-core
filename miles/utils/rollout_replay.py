"""Numeric routing-replay transport shared by HTTP workers and sample conversion."""

import time

import numpy as np
import pybase64


def decode_int32_buffer(value: str | np.ndarray) -> np.ndarray:
    """Decode once; retain a flat array until the consumer knows the token shape."""
    if isinstance(value, np.ndarray):
        if value.dtype != np.int32 or value.ndim != 1 or not value.flags.c_contiguous:
            raise ValueError("Decoded replay must be a contiguous, flat int32 array")
        return value
    return np.frombuffer(pybase64.b64decode(value.encode("ascii")), dtype=np.int32)


def decode_response_routes(response, *, observation=None):
    """Replace only the routing blob; leave token/policy metadata unchanged.

    Shape and nonzero checks remain in the sample consumer, which knows the
    accumulated token count and model layers. This also supports multi-turn
    callers without guessing the replay length from this turn's output tokens.
    """
    if not isinstance(response, dict):
        return response
    metadata = response.get("meta_info")
    if not isinstance(metadata, dict) or metadata.get("routed_experts") is None:
        return response
    started, cpu = time.monotonic(), time.thread_time()
    routes = decode_int32_buffer(metadata["routed_experts"])
    if observation is not None:
        observation.update(
            replay_decode_seconds=time.monotonic() - started,
            replay_decode_thread_cpu_seconds=time.thread_time() - cpu,
            replay_decoded_bytes=routes.nbytes,
        )
    return {**response, "meta_info": {**metadata, "routed_experts": routes}}
