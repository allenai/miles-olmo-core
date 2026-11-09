"""Preserve cache and policy observations without fabricating missing counts."""

from types import SimpleNamespace

from miles.backends.core_utils.rollout import sibling_timing


def test_cache_counts_and_policy_spans_are_preserved():
    sample = SimpleNamespace(metadata={"sibling_timing": {}})
    spans = [{"version": 0, "start": 0, "end": 8}, {"version": 1, "start": 8, "end": 16}]
    sibling_timing.response_received(
        sample,
        "r",
        {
            "cached_tokens": 0,
            "prompt_tokens": 100,
            "completion_tokens": 16,
            "weight_versions": spans,
            "weight_version": "1",
        },
    )
    record = sibling_timing.sample_record(sample)
    assert record["engine_reported_cached_tokens"] == 0
    assert record["engine_reported_prompt_tokens"] == 100
    assert record["engine_reported_completion_tokens"] == 16
    assert record["engine_weight_versions"] == spans
    spans[0]["end"] = 99
    assert record["engine_weight_versions"][0]["end"] == 8


def test_missing_or_invalid_counts_stay_unknown():
    sample = SimpleNamespace(metadata={"sibling_timing": {}})
    sibling_timing.response_received(sample, "r", {"cached_tokens": float("nan"), "prompt_tokens": -1})
    assert not any(key.startswith("engine_reported_") for key in sibling_timing.sample_record(sample))
