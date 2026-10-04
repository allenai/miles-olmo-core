"""Throughput denominators and missing observations must not inflate efficiency."""

import pytest
from miles.backends.core_utils import performance


def test_training_rate_counts_global_tokens_once():
    rates = performance.training_rates(1200, 800, 2, 4)
    assert rates == {
        "model_tokens_per_second": 600,
        "model_tokens_per_gpu_second": 150,
        "active_response_tokens_per_gpu_second": 100,
    }


@pytest.mark.parametrize("args", [(10, 11, 2, 1), (10, 2, 0, 1), (10, 2, 2, 0), (10, 2, float("nan"), 1)])
def test_invalid_rate_denominators_rejected(args):
    with pytest.raises(ValueError):
        performance.training_rates(*args)
