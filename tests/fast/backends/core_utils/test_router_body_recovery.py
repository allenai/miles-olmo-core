"""A broken response transport retries; unknown server faults retain evidence."""

import json
import pickle

import httpx
import pytest

from miles.backends.core_utils.rollout import rollout_errors
from miles.utils.http_post_errors import HTTPPostFailure


def failure(body, status=500, method="POST", path="/generate"):
    request = httpx.Request(
        method, "http://router" + path, headers={"Authorization": "secret"}, content=b"private prompt"
    )
    response = httpx.Response(status, request=request, json=body)
    return httpx.HTTPStatusError("server error", request=request, response=response)


def test_partial_response_transport_survives_actor_boundary_and_retries():
    error = failure({"error": {"code": "read_response_body_failed", "message": "error decoding response body"}})
    envelope = HTTPPostFailure.capture(error, method="POST", url="http://router/generate")
    restored = pickle.loads(pickle.dumps(envelope)).exception()
    assert rollout_errors.recovery_category(restored) == "transport"
    assert "read_response_body_failed" in rollout_errors.failure_description(restored)


@pytest.mark.parametrize(
    "status,method,path,code",
    [
        (500, "POST", "/generate", "unknown"),
        (503, "POST", "/generate", "read_response_body_failed"),
        (500, "POST", "/update_weights", "read_response_body_failed"),
        (500, "GET", "/generate", "read_response_body_failed"),
    ],
)
def test_other_faults_still_stop(status, method, path, code):
    assert rollout_errors.recovery_category(failure({"error": {"code": code}}, status, method, path)) is None


def test_unknown_failure_retains_bounded_body_without_request_secrets():
    error = failure({"error": {"code": "unknown", "message": "x" * 5000}})
    description = rollout_errors.failure_description(error)
    assert "private prompt" not in description and "secret" not in description
    detail = json.loads(description.split(" response=", 1)[1])
    assert detail["status"] == 500 and detail["truncated"] is True
    assert len(detail["body"]) == 2048
    assert "unknown" in detail["body"]


def test_non_http_failure_description():
    assert rollout_errors.failure_description(ValueError("bad policy")) == "ValueError: bad policy"


def test_unread_response_does_not_hide_original_failure():
    request = httpx.Request("POST", "http://router/generate")
    response = httpx.Response(500, request=request, stream=httpx.ByteStream(b"unread"))
    error = httpx.HTTPStatusError("original", request=request, response=response)
    assert "original" in rollout_errors.failure_description(error)
    assert "response body not read" in rollout_errors.failure_description(error)
    assert rollout_errors.recovery_category(error) is None


def test_single_failing_group_has_finite_retries_amid_successful_traffic():
    errors = rollout_errors.RolloutErrors()
    error = failure({"error": {"code": "read_response_body_failed"}})
    for expected in range(1, 4):
        errors.record(successes=1000)
        assert errors.check_group_retry(71, error) == expected
    # An outage after the third isolated failure still gets the global grace period.
    assert errors.check_group_retry(71, error) == 3
    errors.record(successes=1)
    with pytest.raises(RuntimeError, match="group 71 failed again after 3 isolated retries") as caught:
        errors.check_group_retry(71, error)
    assert caught.value.__cause__ is error
    detail = json.loads(rollout_errors.failure_description(caught.value).split(" response=", 1)[1])
    assert detail["status"] == 500 and "read_response_body_failed" in detail["body"]
    assert errors.check_group_retry(72, error) == 1


def test_success_forgets_group_retry_history():
    errors = rollout_errors.RolloutErrors()
    error = httpx.ConnectError("refused")
    errors.check_group_retry(71, error)
    errors.group_succeeded(71)
    assert errors.check_group_retry(71, error) == 1
    errors.group_succeeded(71)
    errors.group_succeeded(72)
    assert errors._group_retries == {}


def test_same_prompt_outage_does_not_consume_isolated_retries():
    errors = rollout_errors.RolloutErrors()
    for _ in range(20):
        assert errors.check_group_retry(71, httpx.ConnectError("refused")) == 1
    errors.record(successes=1)
    assert errors.check_group_retry(71, httpx.ConnectError("refused")) == 2
