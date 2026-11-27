"""Adaptive just-in-time (JIT) pacing for fully asynchronous retained group ownership.

Targets near-zero trainer wait while keeping total retained inventory (in-flight +
partially completed + completed FIFO + prepared batch) as small as possible so
rollouts are generated close to the policy version that trains on them.

Control law (asymmetric reverse-AIMD):
- Fast escape: if trainer wait exceeds ``wait_escape_seconds`` (default 10s), jump
  immediately to ``max_batches`` (full steam ahead) and hold for a cooldown window.
- Gentle bump: if trainer wait is moderate (between ``wait_bump_seconds`` and
  ``wait_escape_seconds``), step up by ``bump_step_batches`` with a short cooldown.
- Slow tighten: when the last ``window_steps`` (default 3) consecutive steps all had
  near-zero trainer wait (``<= tighten_wait_threshold_seconds``) AND observed excess
  ready work (either completed groups left in the FIFO after drain, or a prepared
  batch finishing meaningfully ahead of the trainer), step down gently toward
  ``min_batches`` and reset the window so generation latency settles before the next
  reduction.
"""

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class JitPacerConfig:
    rollout_batch_size: int
    min_batches: float
    max_batches: float
    initial_batches: float | None = None
    window_steps: int = 3
    wait_escape_seconds: float = 10.0
    wait_bump_seconds: float = 2.0
    tighten_wait_threshold_seconds: float = 1.0
    slack_queue_batches: float = 0.25
    slack_ready_fraction: float = 0.15
    large_slack_queue_batches: float = 1.0
    large_slack_ready_fraction: float = 0.40
    tighten_step_batches: float = 0.25
    large_tighten_step_batches: float = 0.50
    bump_step_batches: float = 0.50
    escape_cooldown_steps: int = 5
    bump_cooldown_steps: int = 3


def default_bounds(
    *,
    rollout_batch_size: int,
    producer_group_budget: int,
    buffer_capacity_factor: float,
    max_retained_batches: float | None = None,
    prepared_batch_overlap: bool = False,
) -> tuple[float, float, float]:
    """Return ``(min_batches, max_batches, initial_batches)`` for JIT pacing."""
    raw_min = 2.0 if prepared_batch_overlap else 1.5
    if max_retained_batches is not None:
        max_batches = max_retained_batches
    else:
        max_batches = max(raw_min, producer_group_budget / rollout_batch_size + buffer_capacity_factor)
    min_batches = min(raw_min, max_batches)
    min_batches = max(1.0, min_batches)
    initial_batches = min(max_batches, min_batches + 1.0)
    return min_batches, max_batches, initial_batches


class JitRetainedPacer:
    """Dynamically adjusts the retained-group cap between ``min_batches`` and ``max_batches``."""

    def __init__(self, config: JitPacerConfig) -> None:
        if config.rollout_batch_size < 1:
            raise ValueError(f"rollout_batch_size must be >= 1, got {config.rollout_batch_size}")
        if not math.isfinite(config.min_batches) or config.min_batches < 1.0:
            raise ValueError(f"min_batches must be finite and >= 1.0, got {config.min_batches}")
        if not math.isfinite(config.max_batches) or config.max_batches < config.min_batches:
            raise ValueError(
                f"max_batches ({config.max_batches}) must be finite and >= min_batches ({config.min_batches})"
            )
        self.config = config
        initial = config.max_batches if config.initial_batches is None else config.initial_batches
        self._target_batches = min(config.max_batches, max(config.min_batches, initial))
        self._recent_waits: deque[float] = deque(maxlen=config.window_steps)
        self._recent_queue_batches: deque[float] = deque(maxlen=config.window_steps)
        self._recent_ready_fractions: deque[float] = deque(maxlen=config.window_steps)
        self._recent_trainer_steps: deque[float] = deque(maxlen=config.window_steps)
        self._cooldown_remaining = 0
        self._last_handoff_at: float | None = None
        self._last_wait_seconds = 0.0
        self._escape_count = 0
        self._bump_count = 0
        self._tighten_count = 0
        self._last_metrics: dict[str, float] = self._build_metrics()

    @property
    def target_batches(self) -> float:
        return self._target_batches

    @property
    def target_groups(self) -> int:
        return max(self.config.rollout_batch_size, math.floor(self._target_batches * self.config.rollout_batch_size))

    def on_pause(self) -> None:
        """Reset wall-clock step tracking across evals, pauses, or publication recovery."""
        self._last_handoff_at = None

    def observe_step(
        self,
        *,
        trainer_wait_seconds: float,
        queue_groups_after_drain: int,
        ready_before_consumer_seconds: float = 0.0,
        now: float | None = None,
        cold_start: bool = False,
    ) -> float:
        """Update the retained-batch target after one completed training batch handoff.

        Set ``cold_start=True`` on the initial step-0 direct drain (before any training
        step has executed) so cold-start fill latency establishes the handoff clock
        without snapping ``target_batches`` to ``max_batches`` before trainer compilation.

        Returns the updated ``target_batches``.
        """
        cfg = self.config
        wait = max(0.0, float(trainer_wait_seconds))
        ready_early = max(0.0, float(ready_before_consumer_seconds))
        queue_batches = max(0.0, queue_groups_after_drain / cfg.rollout_batch_size)
        self._last_wait_seconds = wait

        if cold_start:
            if now is not None:
                self._last_handoff_at = now
            self._last_metrics = self._build_metrics()
            return self._target_batches

        if now is not None:
            if self._last_handoff_at is not None:
                trainer_step = max(0.0, (now - self._last_handoff_at) - wait)
                if trainer_step > 0.0:
                    self._recent_trainer_steps.append(trainer_step)
            self._last_handoff_at = now

        rolling_train = (
            sum(self._recent_trainer_steps) / len(self._recent_trainer_steps) if self._recent_trainer_steps else 0.0
        )
        ready_fraction = (ready_early / rolling_train) if rolling_train > 0.0 else (1.0 if ready_early >= 5.0 else 0.0)

        # 1. Fast escape: protect trainer throughput immediately on >10s wait.
        if wait > cfg.wait_escape_seconds:
            if self._target_batches < cfg.max_batches:
                self._target_batches = cfg.max_batches
                self._escape_count += 1
            self._cooldown_remaining = cfg.escape_cooldown_steps
            self._clear_window()
            self._last_metrics = self._build_metrics()
            return self._target_batches

        # 2. Gentle bump: respond to moderate trainer wait before it becomes a stall.
        if wait > cfg.wait_bump_seconds:
            if self._target_batches < cfg.max_batches:
                self._target_batches = min(cfg.max_batches, self._target_batches + cfg.bump_step_batches)
                self._bump_count += 1
            self._cooldown_remaining = max(self._cooldown_remaining, cfg.bump_cooldown_steps)
            self._clear_window()
            self._last_metrics = self._build_metrics()
            return self._target_batches

        # 3. Cooldown after any upward adjustment so generation latency settles.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._last_metrics = self._build_metrics()
            return self._target_batches

        # 4. Accumulate calm-step window and tighten only on sustained zero-wait + real slack.
        self._recent_waits.append(wait)
        self._recent_queue_batches.append(queue_batches)
        self._recent_ready_fractions.append(ready_fraction)

        if len(self._recent_waits) < cfg.window_steps:
            self._last_metrics = self._build_metrics()
            return self._target_batches

        max_recent_wait = max(self._recent_waits)
        sustained_slack = all(
            queue >= cfg.slack_queue_batches or ready >= cfg.slack_ready_fraction
            for queue, ready in zip(self._recent_queue_batches, self._recent_ready_fractions, strict=True)
        )

        if max_recent_wait <= cfg.tighten_wait_threshold_seconds and sustained_slack:
            avg_queue_batches = sum(self._recent_queue_batches) / len(self._recent_queue_batches)
            avg_ready_fraction = sum(self._recent_ready_fractions) / len(self._recent_ready_fractions)
            large_slack = (
                avg_queue_batches >= cfg.large_slack_queue_batches
                or avg_ready_fraction >= cfg.large_slack_ready_fraction
            )
            step = cfg.large_tighten_step_batches if large_slack else cfg.tighten_step_batches
            new_target = max(cfg.min_batches, self._target_batches - step)
            if new_target < self._target_batches:
                self._target_batches = new_target
                self._tighten_count += 1
            self._clear_window()

        self._last_metrics = self._build_metrics()
        return self._target_batches

    def _clear_window(self) -> None:
        self._recent_waits.clear()
        self._recent_queue_batches.clear()
        self._recent_ready_fractions.clear()

    def _build_metrics(self) -> dict[str, float]:
        prefix = "rollout/fully_async/jit_pacer/"
        result: dict[str, float] = {
            f"{prefix}target_retained_batches": self._target_batches,
            f"{prefix}target_retained_groups": float(self.target_groups),
            f"{prefix}min_retained_batches": self.config.min_batches,
            f"{prefix}max_retained_batches": self.config.max_batches,
            f"{prefix}last_trainer_wait_seconds": self._last_wait_seconds,
            f"{prefix}escape_events": float(self._escape_count),
            f"{prefix}bump_events": float(self._bump_count),
            f"{prefix}tighten_events": float(self._tighten_count),
        }
        if self._recent_trainer_steps:
            result[f"{prefix}rolling_trainer_step_seconds"] = sum(self._recent_trainer_steps) / len(
                self._recent_trainer_steps
            )
        return result

    def metrics(self) -> dict[str, float]:
        """Return a copy of the latest producer-loop metrics snapshot (safe across loops)."""
        return dict(self._last_metrics)
