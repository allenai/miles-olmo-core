"""Prepare fresh text once per prompt, without blocking the producer event loop."""

import asyncio
from concurrent.futures import ThreadPoolExecutor


class PromptPreparer:
    """One worker owns its tokenizer; at most one encoding job is submitted.

    Waiting callers are already bounded by the producer's active-group budget.
    Cancellation joins the running pure computation before releasing its slot:
    canceling an asyncio waiter cannot stop a tokenizer thread.
    """

    def __init__(self, tokenizer_factory):
        self._factory = tokenizer_factory
        self._tokenizer = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-tokenizer")
        self._lock = asyncio.Lock()
        self._closed = False

    def _encode(self, prompts):
        if self._tokenizer is None:
            self._tokenizer = self._factory()
        return {prompt: tuple(self._tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts}

    async def prepare(self, prompts):
        # Only immutable rendered text enters the worker, never live Samples.
        prompts = tuple(prompts)
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Prompt preparation requires rendered text")
        unique = tuple(dict.fromkeys(prompts))
        async with self._lock:
            if self._closed:
                raise RuntimeError("Prompt preparer is closed")
            pending = asyncio.wrap_future(self._pool.submit(self._encode, unique))
            try:
                return await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Repeated lifecycle cancellation must not abandon the worker
                # and let retries build an unbounded executor queue.
                while not pending.done():
                    try:
                        await asyncio.shield(pending)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not pending.cancelled():
                    pending.exception()
                raise

    async def close(self):
        self._closed = True
        await asyncio.to_thread(self._pool.shutdown, wait=True)
