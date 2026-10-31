"""Numeric rollout transport shared by HTTP workers and sample conversion.

Routing replay travels as one flat int32 array. Per-token logprob records
travel as two flat arrays (``TokenLogprobs``) instead of one small list per
token, so Ray moves buffers rather than rebuilding thousands of Python
objects in the producer, and the garbage collector never walks them.
"""

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


class TokenLogprobs:
    """SGLang ``output_token_logprobs`` records as two flat arrays.

    The engine returns one ``[logprob, token_id, text]`` list per generated
    token. Rebuilding those lists in the producer process allocates several
    Python objects per token, and every full garbage collection then visits
    all of them. Two arrays carry the same values as buffers. Legacy readers
    that iterate records or index ``records[i][1]`` keep working; the text
    slot is always ``None``, as it is when text is not requested.
    """

    __slots__ = ("log_probs", "token_ids")

    def __init__(self, log_probs, token_ids):
        log_probs = np.asarray(log_probs, dtype=np.float64)
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if log_probs.ndim != 1 or token_ids.shape != log_probs.shape:
            raise ValueError("Token logprob records need exactly one logprob per token id")
        self.log_probs = log_probs
        self.token_ids = token_ids

    @classmethod
    def from_records(cls, records):
        """Build from the engine's nested records; equivalent to the legacy comprehensions."""
        if isinstance(records, cls):
            return records
        records = list(records)
        return cls([r[0] for r in records], [r[1] for r in records])

    def __reduce__(self):
        return (TokenLogprobs, (self.log_probs, self.token_ids))

    def __len__(self):
        return int(self.token_ids.shape[0])

    def __iter__(self):
        for log_prob, token_id in zip(self.log_probs.tolist(), self.token_ids.tolist(), strict=True):
            yield (log_prob, token_id, None)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return TokenLogprobs(self.log_probs[index], self.token_ids[index])
        return (float(self.log_probs[index]), int(self.token_ids[index]), None)

    def __eq__(self, other):
        if isinstance(other, TokenLogprobs):
            return np.array_equal(self.log_probs, other.log_probs) and np.array_equal(self.token_ids, other.token_ids)
        if isinstance(other, (list, tuple)):
            return len(other) == len(self) and all(
                tuple(record[:2]) == mine[:2] for record, mine in zip(other, self, strict=True)
            )
        return NotImplemented

    def tokens(self) -> list[int]:
        return self.token_ids.tolist()

    def logprobs(self) -> list[float]:
        return self.log_probs.tolist()


def compact_token_logprobs(response, *, observation=None):
    """Replace nested per-token records with ``TokenLogprobs``; everything else is untouched."""
    if not isinstance(response, dict):
        return response
    metadata = response.get("meta_info")
    if not isinstance(metadata, dict):
        return response
    records = metadata.get("output_token_logprobs")
    if records is None or isinstance(records, TokenLogprobs):
        return response
    started, cpu = time.monotonic(), time.thread_time()
    compact = TokenLogprobs.from_records(records)
    if observation is not None:
        observation.update(
            token_logprobs_compact_seconds=time.monotonic() - started,
            token_logprobs_compact_thread_cpu_seconds=time.thread_time() - cpu,
            token_logprobs_count=len(compact),
        )
    return {**response, "meta_info": {**metadata, "output_token_logprobs": compact}}
