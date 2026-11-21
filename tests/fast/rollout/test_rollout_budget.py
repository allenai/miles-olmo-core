from argparse import Namespace

import pytest

from miles.rollout.rollout_budget import BudgetUnit, RolloutBudget, drop_oldest_completions_to_fit, rollout_budget
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

ATTEMPT = 2


def _attempts(*versions: int | None) -> list[Sample]:
    """A group of one ATTEMPT-sized attempt per version, newest last; index = position in the group."""
    group = []
    for version in versions:
        spans = [] if version is None else [WeightVersionsPerCall(spans=[WeightVersionSpan(str(version), 0, 1)])]
        group += [Sample(index=len(group) + i, weight_versions=list(spans)) for i in range(ATTEMPT)]
    return group


def _attempt_starts(groups: list[list[Sample]]) -> list[list[int | None]]:
    return [[sample.index for sample in group[::ATTEMPT]] for group in groups]


class TestDropOldestCompletionsToFit:
    def test_the_oldest_attempts_go_first_and_each_newest_attempt_stays(self):
        groups = [_attempts(3, 5), _attempts(1, 2, 6), _attempts(6)]

        fitted = drop_oldest_completions_to_fit(groups, target=4 * ATTEMPT, attempt_size=ATTEMPT)

        assert _attempt_starts(fitted) == [[0, 2], [4], [0]]

    def test_unversioned_attempts_count_as_oldest(self):
        groups = [_attempts(4, 5), _attempts(None, 6)]

        fitted = drop_oldest_completions_to_fit(groups, target=3 * ATTEMPT, attempt_size=ATTEMPT)

        assert _attempt_starts(fitted) == [[0, 2], [2]]

    def test_groups_already_at_the_target_are_returned_unchanged(self):
        groups = [_attempts(3, 5), _attempts(6)]

        assert drop_oldest_completions_to_fit(groups, target=3 * ATTEMPT, attempt_size=ATTEMPT) is groups

    @pytest.mark.parametrize(
        "groups, target",
        [
            ([_attempts(1)], 2 * ATTEMPT),  # below the target
            ([_attempts(1), _attempts(2)], ATTEMPT),  # more groups than the target can hold
            ([_attempts(1, 2)], ATTEMPT + 1),  # not a whole number of attempts
        ],
    )
    def test_unreachable_targets_are_rejected(self, groups, target):
        with pytest.raises(AssertionError, match="cannot fit"):
            drop_oldest_completions_to_fit(groups, target=target, attempt_size=ATTEMPT)


class TestRolloutBudget:
    def test_units_measure_prompts_or_completions(self):
        groups = [_attempts(1, 2), _attempts(3)]

        prompts = RolloutBudget(unit=BudgetUnit.prompts, target=2, attempt_size=ATTEMPT)
        completions = RolloutBudget(unit=BudgetUnit.completions, target=4 * ATTEMPT, attempt_size=ATTEMPT)

        assert prompts.is_met(groups)
        assert not completions.is_met(groups)
        assert completions.is_met([*groups, _attempts(4)])

    def test_only_a_completion_budget_trims(self):
        with pytest.raises(AssertionError, match="only a completion budget"):
            RolloutBudget(unit=BudgetUnit.prompts, target=1, attempt_size=ATTEMPT, trim_to_target=True)

    def test_fit_without_trimming_keeps_the_overshoot(self):
        groups = [_attempts(1, 2), _attempts(3, 4)]
        budget = RolloutBudget(unit=BudgetUnit.completions, target=3 * ATTEMPT, attempt_size=ATTEMPT)

        assert budget.fit(groups) is groups

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_args_give_a_completion_target_trimmed_only_for_a_fixed_global_batch_size(self, dynamic):
        args = Namespace(rollout_batch_size=3, n_samples_per_prompt=4, use_dynamic_global_batch_size=dynamic)

        assert rollout_budget(args) == RolloutBudget(
            unit=BudgetUnit.completions, target=12, attempt_size=4, trim_to_target=not dynamic
        )
