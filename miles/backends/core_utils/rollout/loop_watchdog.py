"""TEST-ONLY diagnostics for the hardening integration (not for merge).

A daemon thread pings the producer's event loop. If a ping waits longer than
STALL_SECONDS the loop is blocked by synchronous code, so every thread's stack
is dumped (faulthandler). Once shutdown has started, unfinished asyncio tasks
are summarized every TASK_DUMP_SECONDS (distinct stacks with counts), to find
what the final generation drain waits on.
"""

import collections
import faulthandler
import io
import sys
import threading
import time

STALL_SECONDS = 20.0
TASK_DUMP_SECONDS = 120.0
_started = set()


def start(loop, producer) -> None:
    if id(loop) in _started:
        return
    _started.add(id(loop))
    threading.Thread(target=_run, args=(loop, producer), name="miles-loop-watchdog", daemon=True).start()


def _run(loop, producer) -> None:
    stop_seen = None
    last_dump = 0.0
    while not loop.is_closed():
        done = threading.Event()
        sent = time.monotonic()
        try:
            loop.call_soon_threadsafe(done.set)
        except RuntimeError:
            return
        if not done.wait(STALL_SECONDS):
            print(f"[loop-watchdog] event loop blocked > {STALL_SECONDS:g}s; thread stacks:", file=sys.stderr, flush=True)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            done.wait()
            print(f"[loop-watchdog] loop resumed after {time.monotonic() - sent:.1f}s", file=sys.stderr, flush=True)
        if getattr(producer, "_stopping", False):
            stop_seen = stop_seen or time.monotonic()
            now = time.monotonic()
            if now - stop_seen >= TASK_DUMP_SECONDS and now - last_dump >= TASK_DUMP_SECONDS:
                last_dump = now
                _dump_tasks(loop, now - stop_seen)
        time.sleep(5.0)


def _dump_tasks(loop, elapsed) -> None:
    import asyncio

    try:
        tasks = [task for task in list(asyncio.all_tasks(loop)) if not task.done()]
    except RuntimeError:
        return
    stacks = collections.Counter()
    for task in tasks:
        stream = io.StringIO()
        try:
            stream.write(_await_chain(task.get_coro()))
        except Exception as error:  # diagnostics only
            stream.write(f"<stack unavailable: {error!r}>")
        stacks[stream.getvalue()] += 1
    print(
        f"[loop-watchdog] {elapsed:.0f}s after shutdown started: {len(tasks)} unfinished task(s), "
        f"{len(stacks)} distinct stack(s)",
        file=sys.stderr,
        flush=True,
    )
    for stack, count in stacks.most_common(8):
        print(f"[loop-watchdog] x{count}:\n{stack}", file=sys.stderr, flush=True)


def _await_chain(coro) -> str:
    """Frames along the coroutine's await chain, outermost first (task.print_stack shows only the top)."""
    lines = []
    seen = 0
    while coro is not None and seen < 30:
        seen += 1
        frame = getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None) or getattr(coro, "ag_frame", None)
        if frame is not None:
            lines.append(f"  {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
        coro = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None) or getattr(coro, "ag_await", None)
    if coro is not None:
        lines.append(f"  awaiting {coro!r}"[:300])
    return "\n".join(lines)
