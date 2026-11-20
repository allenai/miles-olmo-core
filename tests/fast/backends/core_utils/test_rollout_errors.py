"""Transient serving failures retry; sustained failure and correctness errors stop."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from miles.backends.core_utils.rollout import rollout_errors
from miles.backends.core_utils.rollout.errors import GenerationRequestTimeout


def http_error(status, body, path="/generate", method="POST"):
    request = httpx.Request(method, "http://router" + path)
    response = httpx.Response(status, request=request, json=body)
    return httpx.HTTPStatusError("serving error", request=request, response=response)


@pytest.mark.parametrize(
    "error,category",
    [
        (httpx.ConnectError("refused"), "transport"),
        (httpx.ReadError("reset"), "transport"),
        (httpx.ReadTimeout("read timed out"), "timeout"),
        (GenerationRequestTimeout("generation deadline"), "timeout"),
        (http_error(503, {"detail": "Rollout worker unavailable"}), "transport"),
        (http_error(503, {"detail": "No healthy rollout workers available"}), "unavailable"),
        (http_error(503, {"message": "The request queue is full.", "object": "error"}), "overload"),
        (http_error(503, {"message": "The request is aborted by a higher priority request."}), "overload"),
        (http_error(503, {"message": "Request waiting timeout reached."}), "timeout"),
        (http_error(503, {"message": "Request running timeout reached."}), "timeout"),
        (http_error(429, {"message": "rate limit"}), "overload"),
    ],
)
def test_known_generation_failures(error, category):
    assert rollout_errors.recovery_category(error) == category


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("reward service timeout"),
        RuntimeError("invalid policy provenance"),
        asyncio.CancelledError(),
        http_error(500, {"message": "internal server error"}),
        http_error(503, {"message": "Using priority is disabled for this server."}),
        http_error(503, {"message": "unknown"}),
        http_error(503, ["Rollout worker unavailable"]),
        http_error(503, {"message": {"nested": "Rollout worker unavailable"}}),
        http_error(400, {"message": "The request queue is full."}),
        http_error(503, {"detail": "Rollout worker unavailable"}, path="/update_weights"),
        http_error(503, {"detail": "Rollout worker unavailable"}, method="GET"),
    ],
)
def test_unrecognized_and_correctness_failures_remain_fatal(error):
    assert rollout_errors.recovery_category(error) is None


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(rollout_errors, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now))
    return clock


def test_fifty_percent_for_five_minutes_stops(clock):
    errors = rollout_errors.RolloutErrors()
    for second in range(300):
        clock.now = second
        errors.record(successes=1, failures=["transport"])
        errors.check()
    clock.now = 300
    errors.record(successes=1, failures=["timeout"])
    with pytest.raises(RuntimeError, match="at or above 50% for 300 seconds"):
        errors.check()
    assert errors.metrics()["errors/failure_fraction_5m"] == 0.5
    assert len(errors._buckets) == 300


def test_less_than_half_can_keep_running(clock):
    errors = rollout_errors.RolloutErrors()
    for second in range(900):
        clock.now = second
        errors.record(successes=2, failures=["overload"])
        errors.check()
    assert errors.metrics()["errors/high_failure_seconds"] == 0
    assert errors.metrics()["errors/requeued_groups"] == 900


def test_short_burst_expires_without_killing_idle_or_slow_work(clock):
    errors = rollout_errors.RolloutErrors()
    errors.record(failures=["transport"] * 1000)
    errors.check()
    clock.now = 299
    errors.check()
    clock.now = 300
    errors.check()
    assert errors.metrics()["errors/window_group_attempts"] == 0
    assert errors.metrics()["errors/high_failure_seconds"] == 0


def test_recovery_resets_the_sustained_failure_timer(clock):
    errors = rollout_errors.RolloutErrors()
    errors.record(failures=["transport"])
    errors.check()
    clock.now = 250
    errors.record(successes=2)
    errors.check()
    assert errors.metrics()["errors/high_failure_seconds"] == 0
    clock.now = 260
    errors.record(failures=["transport"] * 5)
    errors.check()
    clock.now = 310
    errors.record(failures=["transport"])
    errors.check()
    assert errors.metrics()["errors/high_failure_seconds"] == 50


def test_idle_wall_time_does_not_count_as_completed_attempts(clock):
    errors = rollout_errors.RolloutErrors()
    clock.now = 3600
    errors.check()
    assert errors.metrics()["errors/completed_group_attempts"] == 0
    errors.record(failures=["unavailable"])
    errors.check()
    assert errors.metrics()["errors/high_failure_seconds"] == 0


def test_reports_without_completed_training_updates_and_retains_fatal_summary(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        rollout_errors.tracking, "log", lambda args, metrics, step_key: calls.append((metrics, step_key))
    )
    args = SimpleNamespace(save=str(tmp_path))
    errors = rollout_errors.RolloutErrors()
    errors.record(failures=["transport", "timeout"])
    errors.report(args)
    errors.report(args, fatal="RuntimeError: sustained serving failure")
    assert len(calls) == 2
    assert calls[0][1] == "errors/elapsed_seconds"
    assert calls[-1][0]["errors/fatal"] == 1
    assert calls[-1][0]["errors/requeued_groups"] == 2
    assert calls[-1][0]["errors/timeout_groups"] == 1
    assert all(key.startswith("errors/") for key in calls[-1][0])
    rows = [json.loads(line) for line in (tmp_path / "rollout_errors.jsonl").read_text().splitlines()]
    assert rows[-1]["fatal"] == "RuntimeError: sustained serving failure"


def test_tracking_failure_does_not_break_recovery(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise OSError("tracking unavailable")

    monkeypatch.setattr(rollout_errors.tracking, "log", failed)
    rollout_errors.RolloutErrors().report(SimpleNamespace(save=str(tmp_path)))
    assert (tmp_path / "rollout_errors.jsonl").is_file()


@pytest.mark.parametrize("limit", [0, 1, 5])
def test_isolated_group_retry_limit_is_configurable(limit):
    errors = rollout_errors.RolloutErrors(max_group_retries=limit)
    error = httpx.ConnectError("refused")
    for retry in range(1, limit + 1):
        errors.record(successes=1)
        assert errors.check_group_retry(7, error) == retry
    errors.record(successes=1)
    with pytest.raises(RuntimeError, match=f"after {limit} isolated retries"):
        errors.check_group_retry(7, error)


def test_default_isolated_group_retry_limit_is_three():
    assert rollout_errors.RolloutErrors().max_group_retries == rollout_errors.MAX_GROUP_RETRIES == 3


@pytest.mark.parametrize("limit", [-1, 2.0, True, None])
def test_invalid_isolated_group_retry_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="max_group_retries"):
        rollout_errors.RolloutErrors(max_group_retries=limit)


def test_limit_comes_from_core_config_with_default_for_older_configs():
    assert (
        rollout_errors.from_args(SimpleNamespace(olmo_core=SimpleNamespace(max_group_retries=6))).max_group_retries
        == 6
    )
    assert rollout_errors.from_args(SimpleNamespace(olmo_core=SimpleNamespace())).max_group_retries == 3
    assert rollout_errors.from_args(SimpleNamespace()).max_group_retries == 3
