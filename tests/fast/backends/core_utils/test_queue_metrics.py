"""CPU checks of fleet sizing, explicit overrides, and bounded discard accounting."""

import pytest
from miles.backends.core_utils.rollout.queue_metrics import QueueMetrics


def test_discard_fraction_reports_token_waste_and_length_bias_separately():
    metrics = QueueMetrics()
    metrics.record([4096, 512], age=3, accepted=False)
    metrics.record([100, 200], age=1, accepted=True)
    prefix = "rollout/fully_async/completed_queue/"
    first = metrics.collect()
    assert first[prefix + "dropped_samples"] == 2
    assert first[prefix + "dropped_samples_fraction"] == 0.5
    assert first[prefix + "dropped_response_tokens_fraction"] == pytest.approx(4608 / 4908)
    assert first[prefix + "dropped_samples_by_length/4096_8191"] == 1
    assert first[prefix + "dropped_samples_fraction_by_length/4096_8191"] == 1
    assert first[prefix + "dropped_samples_by_age/3"] == 2
    second = metrics.collect()
    assert second.keys() == first.keys()
    assert all(value == 0 for value in second.values())
