"""How much finished work one fully-async training step collects from the data buffer.

Every step drains groups until a :class:`RolloutBudget` is met. Its unit names what the target
counts. Usually every group holds exactly ``n_samples_per_prompt`` completions, so a prompt budget
and a completion budget stop at the same group. A handler that merges several attempts at a prompt
into one group (``never_give_up``) makes them differ.

With a fixed global batch size the trainer takes exactly one optimizer step of
``rollout_batch_size * n_samples_per_prompt`` completions, so the budget trims the overshoot with
:func:`drop_oldest_completions_to_fit`. With ``--use-dynamic-global-batch-size`` the trainer
absorbs the overshoot and nothing is trimmed.
"""

from dataclasses import dataclass
from enum import StrEnum

from miles.rollout.filter_hub.common_filters import Group, group_weight_version_stats


class BudgetUnit(StrEnum):
    prompts = "prompts"
    completions = "completions"


@dataclass(frozen=True)
class RolloutBudget:
    """Drain groups until their total ``size`` reaches ``target``, then optionally trim to it."""

    unit: BudgetUnit
    target: int
    attempt_size: int
    trim_to_target: bool = False

    def __post_init__(self):
        assert self.target >= 1, f"rollout budget target must be positive, got {self.target}"
        assert self.attempt_size >= 1, f"attempt size must be positive, got {self.attempt_size}"
        assert (
            not self.trim_to_target or self.unit == BudgetUnit.completions
        ), "only a completion budget can be trimmed to its target"

    def size(self, group: Group) -> int:
        match self.unit:
            case BudgetUnit.prompts:
                return 1
            case BudgetUnit.completions:
                return len(group)

    def is_met(self, groups: list[Group]) -> bool:
        return sum(map(self.size, groups)) >= self.target

    def fit(self, groups: list[Group]) -> list[Group]:
        if not self.trim_to_target:
            return groups
        return drop_oldest_completions_to_fit(groups, target=self.target, attempt_size=self.attempt_size)


def rollout_budget(args) -> RolloutBudget:
    return RolloutBudget(
        unit=BudgetUnit.completions,
        target=args.rollout_batch_size * args.n_samples_per_prompt,
        attempt_size=args.n_samples_per_prompt,
        # A fixed global batch size trains exactly `target` completions in one optimizer step.
        trim_to_target=not args.use_dynamic_global_batch_size,
    )


def drop_oldest_completions_to_fit(groups: list[Group], *, target: int, attempt_size: int) -> list[Group]:
    """Drop the oldest completions until ``groups`` hold exactly ``target``.

    Each group is a sequence of attempts of ``attempt_size`` completions, newest last. Completions
    are dropped one whole attempt at a time, oldest weight version first, and a group's newest
    attempt is never dropped, so no group disappears. This needs at most ``target // attempt_size``
    groups and a total that exceeds ``target`` by whole attempts. Groups already at ``target`` are
    returned unchanged.
    """
    total = sum(map(len, groups))
    assert (
        len(groups) * attempt_size <= target <= total and (total - target) % attempt_size == 0
    ), f"cannot fit {len(groups)} groups of {total} completions to a target of {target}"
    if total == target:
        return groups

    attempts = [[group[i : i + attempt_size] for i in range(0, len(group), attempt_size)] for group in groups]

    def age(position: tuple[int, int]) -> tuple[float, int]:
        oldest = group_weight_version_stats(attempts[position[0]][position[1]]).oldest_version
        return (float("-inf") if oldest is None else oldest, position[1])

    droppable = [(g, a) for g, group_attempts in enumerate(attempts) for a in range(len(group_attempts) - 1)]
    dropped = set(sorted(droppable, key=age)[: (total - target) // attempt_size])
    return [
        [completion for a, attempt in enumerate(group_attempts) if (g, a) not in dropped for completion in attempt]
        for g, group_attempts in enumerate(attempts)
    ]
