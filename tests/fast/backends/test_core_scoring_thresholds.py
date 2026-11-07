"""The scoring-reuse audit warns on small drift and stops on a tenfold breach."""

import pytest
import torch

from miles.backends.core_utils import contract


@pytest.mark.parametrize(
    ("mean", "warning"), [(0.0, False), (0.001, False), (0.00350197, True), (0.01, True)]
)
def test_scoring_warning_band(mean, warning):
    report = contract.validate_scoring_check(
        torch.tensor([mean, 1, 0], dtype=torch.float64), torch.tensor(mean), tolerance=0.001
    )
    assert report["warning"] is warning
    assert report["fatal_tolerance"] == 0.01
    assert report["mean_abs"] == mean


def test_scoring_fatal_threshold():
    with pytest.raises(ValueError, match="exceeds fatal tolerance 0.01"):
        contract.validate_scoring_check(
            torch.tensor([0.01001, 1, 1], dtype=torch.float64), torch.tensor(0.01001), tolerance=0.001
        )


def test_custom_warning_threshold_scales_fatal_threshold():
    report = contract.validate_scoring_check(torch.tensor([0.04, 2, 2]), torch.tensor(0.02), tolerance=0.01)
    assert report["warning"] and report["fatal_tolerance"] == 0.1


def test_nonfinite_scores_remain_fatal():
    with pytest.raises(ValueError, match="Non-finite"):
        contract.scoring_check([torch.zeros(1)], [torch.tensor([float("nan")])], [torch.ones(1)])
