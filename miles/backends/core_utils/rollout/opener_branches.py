"""Per-prompt opener branches for refresh generation (first-token exploration).

Before a prompt group is generated, the serving policy's most likely openers for the
prompt are read in prefill-only requests. An opener is the response's first tokens up
to and including the first token that completes visible text: whitespace-only and
incomplete-character prefixes are expanded, so ``" Okay"``, ``"Okay"`` and
``"\\nOkay"`` are distinct openers. The ``pool`` most likely openers form the
candidate set; ``branches`` of them are drawn uniformly without replacement, and each
gets an equal share of the group's samples as a response seed (``response_seed``).

The draw does not follow the policy's probabilities and no correction is applied for
it: rare candidates are tried as often as the favorite. Seeded tokens still carry the
policy's own scores, so TIS corrects only drift and numerical mismatch.
"""

import asyncio
import hashlib
import math
import random

from miles.backends.core_utils.rollout import response_seed

RECORD_KEY = "opener_branches"
# Read this many next tokens at each explored prefix.
TOP = 20
# Prefixes at most this long; deeper unfinished prefixes are dropped.
MAX_DEPTH = 4
# Expand an unfinished prefix only while its probability is at least this.
EXPAND_FLOOR = 1e-6


def settings(args):
    """Return (branches, pool), or None when opener branching is off."""
    core = getattr(args, "olmo_core", None)
    branches = getattr(core, "opener_branches", 0) or 0
    if branches <= 0:
        return None
    return branches, getattr(core, "opener_pool", 20)


def probe_payload(input_ids):
    return {
        "input_ids": list(input_ids),
        "sampling_params": {"temperature": 1.0, "max_new_tokens": 1},
        "return_logprob": True,
        "top_logprobs_num": TOP,
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


async def candidates(score, tokenizer, prompt_ids, pool):
    """Return the ``pool`` most likely openers as (token ids, joint log probability), and
    the policy version that served the first probe.

    ``score(input_ids)`` sends one probe request and returns its output. Special tokens
    (end of text, closing tags) are never openers.
    """
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    frontier, found, version = [([], 0.0)], [], None
    for _ in range(MAX_DEPTH):
        if not frontier:
            break
        outputs = await asyncio.gather(*(score(list(prompt_ids) + path) for path, _ in frontier))
        if version is None:
            version = outputs[0].get("meta_info", {}).get("weight_version")
        nxt = []
        for (path, lp), output in zip(frontier, outputs, strict=True):
            for value, token in next_tokens(output):
                if token in special:
                    continue
                ids, joint = path + [token], lp + value
                if finished(tokenizer, ids):
                    found.append((ids, joint))
                elif joint >= math.log(EXPAND_FLOOR):
                    nxt.append((ids, joint))
        frontier = nxt
    found.sort(key=lambda item: -item[1])
    return found[:pool], version


def choose(pool, branches, key):
    """Draw ``branches`` candidates uniformly without replacement, reproducibly by ``key``."""
    if len(pool) < branches:
        raise ValueError(f"Opener probe found {len(pool)} candidates; {branches} branches are required")
    rng = random.Random(hashlib.sha256(key.encode()).digest())
    return rng.sample(pool, branches)


def assign(prompt_group, chosen, pool, version):
    """Give each chosen opener an equal, contiguous share of the group as a response seed."""
    if len(prompt_group) % len(chosen):
        raise ValueError(f"Group of {len(prompt_group)} samples cannot split evenly over {len(chosen)} openers")
    share = len(prompt_group) // len(chosen)
    record = {
        "pool": [{"token_ids": ids, "logprob": lp} for ids, lp in pool],
        "chosen": [ids for ids, _ in chosen],
        "probe_weight_version": version,
    }
    for slot, sample in enumerate(prompt_group):
        ids = chosen[slot // share][0]
        sample.metadata = {
            **(sample.metadata or {}),
            response_seed.REQUEST_KEY: {"token_ids": list(ids)},
            RECORD_KEY: {**record, "branch": slot // share},
        }
