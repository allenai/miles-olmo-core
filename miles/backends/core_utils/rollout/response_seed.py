"""Opening response tokens chosen by the data source for refresh generation.

A sample may request ``metadata["response_seed"] = {"text": "..."}``. The text is
tokenized after the prompt, scored by the serving policy in a prefill-only
request, and becomes the first response tokens: rewarded and trained like any
response token, carrying the scored behavior log-probabilities and the policy
version that scored them. Generation then continues from the seeded tokens.

Samples without the key are untouched. The request is never modified, so a reset
or requeued sample is seeded again from the same text; the scored outcome is
recorded separately under ``metadata["response_seed_scored"]``.
"""

import math

REQUEST_KEY = "response_seed"
RECORD_KEY = "response_seed_scored"


def requested(sample):
    """Return the requested seed text, or None when the sample has no seed."""
    request = (sample.metadata or {}).get(REQUEST_KEY)
    if request is None:
        return None
    if not isinstance(request, dict) or set(request) != {"text"}:
        raise ValueError(f'metadata["{REQUEST_KEY}"] must be {{"text": str}}, got {request!r}')
    text = request["text"]
    if not isinstance(text, str) or not text:
        raise ValueError(f'metadata["{REQUEST_KEY}"]["text"] must be a nonempty string')
    return text


def tokenize(state, sample, prompt_ids, text):
    """Encode the seed and require that it tokenizes identically after the prompt.

    Byte-level merges can join the seed with the end of the prompt (for example a
    newline after ``>``). Such a seed would be a different token sequence from what
    the policy samples in context, so it is rejected rather than silently trained.
    """
    seed_ids = state.tokenizer.encode(text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError(f"Response seed {text!r} encodes to no tokens")
    prompt = sample.prompt
    if not isinstance(prompt, str):
        prompt = state.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    joined = state.tokenizer.encode(prompt + text, add_special_tokens=False)
    if joined != list(prompt_ids) + seed_ids:
        raise ValueError(
            f"Response seed {text!r} tokenizes differently after the prompt: alone {seed_ids}, "
            f"in context {joined[len(prompt_ids) - 1:]} (from the last prompt token)"
        )
    return seed_ids


def remaining_sampling_params(args, sampling_params, seed_length):
    """Charge the seed against the response budget, as a resumed response would be."""
    params = dict(sampling_params)
    budget = params.get("max_new_tokens", args.rollout_max_response_len) - seed_length
    if budget <= 0:
        raise ValueError(f"Response seed of {seed_length} tokens leaves no response-token budget")
    params["max_new_tokens"] = budget
    return params


def scoring_payload(prompt_ids, seed_ids, sampling_params):
    """Prefill-only request returning the policy's log-probabilities for the seed."""
    return {
        "input_ids": list(prompt_ids) + list(seed_ids),
        "sampling_params": {**dict(sampling_params), "max_new_tokens": 0},
        "return_logprob": True,
        # SGLang reports input logprobs from logprob_start_len, the first as None.
        # Starting at the last prompt token makes the tail cover every seed token.
        "logprob_start_len": len(prompt_ids) - 1,
    }


def parse_scores(output, seed_ids):
    """Return the seed tokens' behavior log-probabilities and their policy version."""
    meta = output.get("meta_info", {})
    items = meta.get("input_token_logprobs") or []
    tail = items[-len(seed_ids) :]
    if len(tail) != len(seed_ids) or [item[1] for item in tail] != list(seed_ids):
        raise ValueError("Response seed scoring is not aligned with the seed tokens")
    scores = [item[0] for item in tail]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in scores):
        raise ValueError("Response seed scoring returned a missing or non-finite log-probability")
    version = meta.get("weight_version")
    if version is None:
        raise ValueError("Response seed scoring did not report the serving policy version")
    return [float(x) for x in scores], version


def apply(sample, prompt_ids, seed_ids, text, scores, version):
    """Make the seed the sample's first response tokens before generation resumes."""
    sample.tokens = list(prompt_ids) + list(seed_ids)
    sample.response = text
    sample.response_length = len(seed_ids)
    sample.rollout_log_probs = list(scores)
    sample.metadata = {
        **(sample.metadata or {}),
        RECORD_KEY: {"token_ids": list(seed_ids), "log_probs": list(scores), "weight_version": str(version)},
    }


def merged_meta_info(sample, meta):
    """Extend the continuation's token records and version spans over the seed.

    The continuation reports spans and logprobs for its own output only. The
    seed occupies the first response positions under the version that scored it,
    which cannot exceed the continuation's versions because publications only
    move forward.
    """
    record = sample.metadata[RECORD_KEY]
    seed_ids, scores = record["token_ids"], record["log_probs"]
    shift = len(seed_ids)
    spans = meta.get("weight_versions")
    merged = dict(meta)
    if spans is not None:
        merged["weight_versions"] = [dict(version=record["weight_version"], start=0, end=shift)] + [
            dict(version=span["version"], start=span["start"] + shift, end=span["end"] + shift) for span in spans
        ]
    merged["output_token_logprobs"] = [
        [score, token, None] for score, token in zip(scores, seed_ids, strict=True)
    ] + list(meta.get("output_token_logprobs") or [])
    return merged
