from __future__ import annotations

import httpx
import pytest

from jev_decision_router.retry import NO_RETRY, RetryPolicy, send_with_retry

REQUEST = httpx.Request("POST", "https://example.test")


class Sender:
    """順番にレスポンス（または例外）を返す send 関数。"""

    def __init__(self, *outcomes: httpx.Response | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self) -> httpx.Response:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, request=REQUEST)


def run(sender: Sender, policy: RetryPolicy) -> tuple[httpx.Response, list[float]]:
    sleeps: list[float] = []
    return send_with_retry(sender, policy, sleep=sleeps.append), sleeps


POLICY = RetryPolicy(max_retries=2, backoff_initial=0.5, backoff_max=5.0, backoff_jitter=0.0)


def test_success_is_returned_without_retry() -> None:
    sender = Sender(response(200))

    result, sleeps = run(sender, POLICY)

    assert result.status_code == 200
    assert sender.calls == 1
    assert sleeps == []


@pytest.mark.parametrize("status", [408, 429, 500, 503, 529])
def test_retryable_status_is_retried_with_exponential_backoff(status: int) -> None:
    sender = Sender(response(status), response(status), response(200))

    result, sleeps = run(sender, POLICY)

    assert result.status_code == 200
    assert sender.calls == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_client_errors_are_not_retried(status: int) -> None:
    sender = Sender(response(status))

    result, sleeps = run(sender, POLICY)

    assert result.status_code == status
    assert sleeps == []


def test_last_response_is_returned_when_retries_are_exhausted() -> None:
    sender = Sender(response(529), response(529), response(529))

    result, sleeps = run(sender, POLICY)

    assert result.status_code == 529
    assert sender.calls == 3
    assert len(sleeps) == 2


def test_retry_after_header_is_respected_and_capped() -> None:
    policy = RetryPolicy(max_retries=2, backoff_jitter=0.0, retry_after_max=10.0)
    sender = Sender(
        response(429, {"Retry-After": "3"}),
        response(429, {"Retry-After": "120"}),
        response(200),
    )

    _, sleeps = run(sender, policy)

    assert sleeps == [3.0, 10.0]


def test_invalid_retry_after_falls_back_to_backoff() -> None:
    sender = Sender(response(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), response(200))

    _, sleeps = run(sender, POLICY)

    assert sleeps == [0.5]


def test_transport_error_is_retried_then_raised() -> None:
    error = httpx.ConnectError("refused", request=REQUEST)
    sender = Sender(error, error, error)

    with pytest.raises(httpx.ConnectError):
        run(sender, POLICY)
    assert sender.calls == 3


def test_transport_error_recovers() -> None:
    sender = Sender(httpx.ReadTimeout("slow", request=REQUEST), response(200))

    result, sleeps = run(sender, POLICY)

    assert result.status_code == 200
    assert sleeps == [0.5]


def test_no_retry_policy() -> None:
    sender = Sender(response(503))

    result, sleeps = run(sender, NO_RETRY)

    assert result.status_code == 503
    assert sleeps == []


def test_backoff_is_capped_and_jittered() -> None:
    policy = RetryPolicy(backoff_initial=1.0, backoff_max=4.0, backoff_jitter=0.25)

    assert policy.backoff(0, rand=lambda: 0.0) == 1.0
    assert policy.backoff(5, rand=lambda: 0.0) == 4.0
    assert policy.backoff(5, rand=lambda: 1.0) == 5.0


def test_negative_max_retries_is_rejected() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
