"""The scoring-reuse audit stops when drift exceeds the configured tolerance."""

import pytest
import torch

from miles.backends.core_utils import contract


@pytest.mark.parametrize("mean", [0.0, 0.0009, 0.001])
def test_scoring_at_or_below_tolerance(mean):
    report = contract.validate_scoring_check(
        torch.tensor([mean, 1, 0], dtype=torch.float64), torch.tensor(mean), tolerance=0.001
    )
    assert report["tolerance"] == 0.001
    assert report["mean_abs"] == mean


@pytest.mark.parametrize("mean", [0.001001, 0.00350197, 0.01, 0.01001])
def test_scoring_above_tolerance_is_fatal(mean):
    # Include the original observed discrepancy: it must no longer just warn.
    with pytest.raises(ValueError, match="exceeds tolerance 0.001"):
        contract.validate_scoring_check(
            torch.tensor([mean, 1, 1], dtype=torch.float64), torch.tensor(mean), tolerance=0.001
        )


@pytest.mark.parametrize("tolerance", [0.0001, 0.01])
def test_custom_tolerance_applies_to_reduced_mean(tolerance):
    report = contract.validate_scoring_check(
        torch.tensor([2 * tolerance, 2, 1], dtype=torch.float64), torch.tensor(2 * tolerance), tolerance
    )
    assert report["mean_abs"] == tolerance
    with pytest.raises(ValueError, match="exceeds tolerance"):
        contract.validate_scoring_check(
            torch.tensor([2.1 * tolerance, 2, 1], dtype=torch.float64), torch.tensor(2 * tolerance), tolerance
        )


def test_nonfinite_scores_remain_fatal():
    with pytest.raises(ValueError, match="Non-finite"):
        contract.scoring_check([torch.zeros(1)], [torch.tensor([float("nan")])], [torch.ones(1)])
