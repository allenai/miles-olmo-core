"""Compare repeated sibling encoding with the Core prompt worker on local JSONL.

This measures CPU preparation and event-loop responsiveness, not GPU throughput.
Use the run's tokenizer and rendered prompt file to reproduce its input lengths.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import random
import time
from pathlib import Path

from miles.backends.core_utils.rollout import prompt_preparation
from miles.utils.processing_utils import load_tokenizer


async def measure(tokenizer, prompts, siblings, prepared):
    started = time.perf_counter()
    worker = prompt_preparation.PromptPreparer(lambda: copy.deepcopy(tokenizer)) if prepared else None
    startup_seconds = time.perf_counter() - started
    heartbeat = []
    finished = asyncio.Event()

    async def observe():
        previous = time.perf_counter()
        while not finished.is_set():
            await asyncio.sleep(0.001)
            now = time.perf_counter()
            heartbeat.append(now - previous)
            previous = now

    async def group(prompt):
        if prepared:
            ids = (await worker.prepare([prompt] * siblings))[prompt]
            return [list(ids) for _ in range(siblings)]
        results = []
        for _ in range(siblings):
            results.append(tokenizer.encode(prompt, add_special_tokens=False))
        return results

    observer = asyncio.create_task(observe())
    await asyncio.sleep(0)
    start, cpu = time.perf_counter(), time.thread_time()
    try:
        groups = await asyncio.gather(*(group(prompt) for prompt in prompts))
        elapsed, loop_cpu = time.perf_counter() - start, time.thread_time() - cpu
    finally:
        finished.set()
        await observer
        if worker is not None:
            await worker.close()
    digest = hashlib.sha256(json.dumps(groups, separators=(",", ":")).encode()).hexdigest()
    return dict(
        mode="prepared" if prepared else "repeated",
        tokenizer_startup_seconds=startup_seconds,
        wall_seconds=elapsed,
        event_loop_cpu_seconds=loop_cpu,
        max_heartbeat_gap_seconds=max(heartbeat),
        token_digest=digest,
        prompt_lengths=[len(group[0]) for group in groups],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--prompt-key", default="input")
    parser.add_argument("--groups", default=24, type=int)
    parser.add_argument("--siblings", default=8, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    prompts = [row[args.prompt_key] for row in random.Random(17).sample(rows, args.groups)]
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=True, trust_remote_code=True)
    tokenizer.encode(prompts[0], add_special_tokens=False)
    records = [
        asyncio.run(measure(tokenizer, prompts, args.siblings, prepared))
        for _ in range(args.repeats)
        for prepared in (False, True)
    ]
    assert len({record["token_digest"] for record in records}) == 1, "Tokenization changed"
    result = json.dumps(dict(groups=args.groups, siblings=args.siblings, measurements=records), indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
