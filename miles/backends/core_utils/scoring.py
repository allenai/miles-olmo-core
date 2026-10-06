"""Core scoring policy; CPU planning counterpart checked by integration tests."""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ScoringPass:
    """Whether every update runs the standalone pre-update scoring pass."""

    standalone: bool
    reason: str
    optimizer_steps_per_collection: int | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def scoring_pass(core: Any, options: dict[str, Any]) -> ScoringPass:
    """Decide from the static recipe whether the standalone scoring pass carries information.

    The pass produces the old log-probabilities for the PPO ratio. With exactly one
    optimizer step per collection and no KL term in the advantages, those values are
    the training forward's own log-probabilities at unchanged weights, so the trainer
    reads them there instead. Rollout log-probabilities as the anchor make the pass
    diagnostic-only as well. Model-level conditions (dropout) are checked by the trainer,
    which can see the loaded configuration.
    """
    samples = options.get("global_batch_size")
    collection = options.get("rollout_batch_size", 0) * options.get("n_samples_per_prompt", 1)
    steps = collection // samples if collection and samples else None
    if core.scoring_pass_required:
        return ScoringPass(True, "core.scoring_pass_required", steps)
    if steps is None:
        return ScoringPass(True, "unknown collection size; one optimizer step per collection is unproven", steps)
    if steps != 1:
        return ScoringPass(True, f"{steps} optimizer steps per collection need the pre-update anchor", steps)
    if options.get("kl_coef", 0) != 0:
        return ScoringPass(True, "kl_coef needs actor log-probabilities before advantages", steps)
    anchor = (
        "rollout log-probabilities"
        if options.get("use_rollout_logprobs", False)
        else "the training forward at unchanged weights"
    )
    return ScoringPass(
        False, f"one optimizer step per collection with zero KL; old log-probabilities are {anchor}", steps
    )


def stochastic_fields(model_config: dict[str, Any]) -> list[str]:
    """Model settings that make a forward pass non-repeatable, so no single old log-probability exists."""
    return sorted(name for name, value in model_config.items() if "dropout" in name and value)


def scoring_check_due(core: Any, checks_done: int, completed_steps: int) -> bool:
    """Check the first update of every process (including after resume), then on the interval."""
    interval = core.scoring_check_interval
    return checks_done == 0 or (interval > 0 and completed_steps % interval == 0)
