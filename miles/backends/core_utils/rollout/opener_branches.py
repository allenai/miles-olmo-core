"""Per-prompt opener branches for refresh generation.

``first_token`` mode ranks literal next tokens, including whitespace, excluding
special IDs and think delimiters. The default ``visible`` mode below expands
whitespace for compatibility with earlier experiments.

Before a prompt group is generated, the serving policy's most likely openers for the
prompt are read in prefill-only requests. An opener is the response's first tokens up
to and including the first token that completes visible text: whitespace-only and
incomplete-character prefixes are expanded, so ``" Okay"``, ``"Okay"`` and
``"\\nOkay"`` are distinct openers. Special tokens are never part of an opener.

The ``pool`` most likely openers are found by best-first search. A prefix's
probability bounds every opener under it, and the next tokens a probe did not return
are each bounded by the smaller of the omitted mass and the last returned token. The
search fetches more next tokens wherever such a bound could still beat the pool, so
a completed search returns the exact top ``pool``. A search cut short by its request
budget or depth limit is recorded as inexact, with the largest probability it could
have missed. All probes of one search must report the same policy version; a search
that spans a publication is repeated.

``branches`` openers are drawn from the pool uniformly without replacement and each
gets an equal share of the group's samples as a response seed (``response_seed``). The
draw does not follow the policy's probabilities and no correction is applied for it.
Seeded tokens still carry the policy's own scores, so TIS corrects only drift and
numerical mismatch.
"""

import hashlib
import heapq
import itertools
import math
import random

from miles.backends.core_utils.rollout import response_seed

RECORD_KEY = "opener_branches"
# Next tokens read per probe, widened by WIDEN up to MAX_TOP where bounds require it.
TOP = 20
WIDEN = 10
MAX_TOP = 2000
# Openers at most this many tokens; deeper unfinished prefixes are omissions.
MAX_DEPTH = 6
MAX_REQUESTS = 64
VERSION_ATTEMPTS = 3


def settings(args):
    """Return (branches, pool), or None when opener branching is off."""
    core = getattr(args, "olmo_core", None)
    branches = getattr(core, "opener_branches", 0) or 0
    if branches <= 0:
        return None
    return branches, getattr(core, "opener_pool", 20)


def probe_payload(input_ids, top=TOP):
    return {
        "input_ids": list(input_ids),
        "sampling_params": {"temperature": 1.0, "max_new_tokens": 1},
        "return_logprob": True,
        "top_logprobs_num": top,
    }


def next_tokens(output):
    """Top (log probability, token id) pairs for the token after the request's input."""
    top = output.get("meta_info", {}).get("output_top_logprobs") or []
    if not top or not top[0]:
        raise ValueError("Opener probe returned no next-token log probabilities")
    return [(float(lp), int(token)) for lp, token, *_ in top[0]]


def finished(tokenizer, ids):
    """Whether a prefix ends with visible, complete text."""
    text = tokenizer.decode(ids)
    return bool(text.strip()) and "�" not in text


def unseen_bound(prefix_logprob, top, requested):
    """Log bound on any next token a probe did not return (None when it returned all)."""
    if len(top) < requested:
        return None
    omitted = 1.0 - math.fsum(math.exp(v) for v, _ in top)
    bound = min(omitted, math.exp(min(v for v, _ in top)))
    return prefix_logprob + math.log(bound) if bound > 0 else None


async def search(probe, tokenizer, prompt_ids, pool):
    """Best-first search for the ``pool`` most likely openers.

    ``probe(input_ids, top)`` sends one probe and returns its output. Probes run one at
    a time, so a failure leaves no other probe of the search in flight. Returns a dict
    with ``found`` [(token ids, log probability)], ``versions``, ``exact``, ``missed``
    (largest probability an omitted opener could have) and ``requests``.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    order = itertools.count()
    # Entries: (-log bound, tie, kind, prefix ids, prefix log probability, seen ids, top).
    heap = [(0.0, next(order), "prefix", [], 0.0, set(), 0)]
    found, versions, requests, missed = [], set(), 0, 0.0
    while heap and len(found) < pool:
        neg, _, kind, ids, lp, seen, top = heapq.heappop(heap)
        if kind == "done":
            found.append((ids, lp))
            continue
        if kind == "deep" or requests >= MAX_REQUESTS:
            missed = max(missed, math.exp(-neg))
            if kind != "deep":
                break
            continue
        top = TOP if kind == "prefix" else min(top * WIDEN, MAX_TOP)
        output = await probe(list(prompt_ids) + ids, top)
        requests += 1
        versions.add(output.get("meta_info", {}).get("weight_version"))
        returned = next_tokens(output)
        for value, token in returned:
            if token in seen or token in special:
                continue
            seen.add(token)
            child, joint = ids + [token], lp + value
            if finished(tokenizer, child):
                heapq.heappush(heap, (-joint, next(order), "done", child, joint, set(), 0))
            else:
                deep = len(child) >= MAX_DEPTH
                heapq.heappush(heap, (-joint, next(order), "deep" if deep else "prefix", child, joint, set(), 0))
        rest = unseen_bound(lp, returned, top)
        if rest is not None and top < MAX_TOP:
            heapq.heappush(heap, (-rest, next(order), "more", ids, lp, seen, top))
        elif rest is not None:
            missed = max(missed, math.exp(rest))
    # Best-first order: whatever is left in the heap is bounded by the last admitted opener.
    if len(found) == pool:
        exact = missed < math.exp(found[-1][1])
    else:
        exact = missed == 0.0
    return {"found": found, "versions": versions, "exact": exact, "missed": missed, "requests": requests}


def protocol_token(tokenizer, token):
    """Recognize control markers that some tokenizers store as ordinary added tokens."""
    text = tokenizer.decode([token])
    return text in ("<think>", "</think>", "<functions>", "</functions>", "<function_calls>", "</function_calls>") or (
        text.startswith("<|") and text.endswith("|>")
    )


async def first_token_search(probe, tokenizer, prompt_ids, pool):
    """Rank literal next tokens; whitespace is a candidate, never an expansion.

    Atomic think delimiters can be ordinary added tokens, absent from special IDs.
    Exclude them by decoded spelling as well as excluding tokenizer special IDs.
    Widen only at the original prompt, and fail closed if the full pool is unavailable.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    top, versions, requests = max(TOP, pool), set(), 0
    if top > MAX_TOP:
        raise ValueError(f"First-token opener pool exceeds probe limit {MAX_TOP}")
    while True:
        output = await probe(list(prompt_ids), top)
        versions.add(output.get("meta_info", {}).get("weight_version"))
        requests += 1
        returned = sorted(next_tokens(output), reverse=True)
        found = [
            ([token], lp) for lp, token in returned if token not in special and not protocol_token(tokenizer, token)
        ][:pool]
        if len(found) == pool:
            return {"found": found, "versions": versions, "exact": True, "missed": 0.0, "requests": requests}
        if len(returned) < top or top == MAX_TOP:
            raise ValueError(f"First-token probe found {len(found)} eligible tokens; pool of {pool} required")
        top = min(top * WIDEN, MAX_TOP)


async def candidates(probe, tokenizer, prompt_ids, pool, mode="visible"):
    """Search under one policy version, repeating a search that spans a publication."""
    searches = {"visible": search, "first_token": first_token_search}
    if mode not in searches:
        raise ValueError(f"Unknown opener mode {mode!r}")
    for _ in range(VERSION_ATTEMPTS):
        result = await searches[mode](probe, tokenizer, prompt_ids, pool)
        if len(result["versions"]) == 1:
            result["version"] = next(iter(result["versions"]))
            result["mode"] = mode
            return result
    raise RuntimeError(f"Opener probes spanned policy versions in {VERSION_ATTEMPTS} consecutive searches")


def choose(pool, branches, key):
    """Draw ``branches`` candidates uniformly without replacement, reproducibly by ``key``."""
    if len(pool) < branches:
        raise ValueError(f"Opener probe found {len(pool)} candidates; {branches} branches are required")
    rng = random.Random(hashlib.sha256(key.encode()).digest())
    return rng.sample(pool, branches)


def assigned(prompt_group):
    """Whether every sample already carries a persisted branch assignment."""
    return all(
        RECORD_KEY in (s.metadata or {}) and response_seed.REQUEST_KEY in (s.metadata or {}) for s in prompt_group
    )


def assign(prompt_group, chosen, result):
    """Give each chosen opener an equal, contiguous share of the group as a response seed."""
    if len(prompt_group) % len(chosen):
        raise ValueError(f"Group of {len(prompt_group)} samples cannot split evenly over {len(chosen)} openers")
    share = len(prompt_group) // len(chosen)
    record = {
        "mode": result.get("mode", "visible"),
        "pool": [{"token_ids": ids, "logprob": lp} for ids, lp in result["found"]],
        "chosen": [ids for ids, _ in chosen],
        "probe_weight_version": result["version"],
        "pool_exact": result["exact"],
        "pool_missed_probability": result["missed"],
        "probe_requests": result["requests"],
    }
    for slot, sample in enumerate(prompt_group):
        ids = chosen[slot // share][0]
        sample.metadata = {
            **(sample.metadata or {}),
            response_seed.REQUEST_KEY: {"token_ids": list(ids)},
            RECORD_KEY: {**record, "branch": slot // share},
        }
