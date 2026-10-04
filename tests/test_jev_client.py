from __future__ import annotations

import json

import httpx
import pytest

from jev_decision_router.interfaces import ChoiceRequest, DecisionError
from jev_decision_router.jev_client import JevClient
from jev_decision_router.retry import NO_RETRY, RetryPolicy

REQUEST = ChoiceRequest(
    state={"request": "hello"},
    instructions="choose one",
    criteria={"a": "route a", "b": "route b"},
)

OK_BODY = {
    "model": "jev-test",
    "answers": {
        "route": {
            "choice": "a",
            "confidence": 0.8,
            "probabilities": {"a": 0.8, "b": 0.2},
        }
    },
    "usage": {"input_tokens": 10},
}


def test_choose_sends_choice_question_and_parses_answer(mock_http) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=OK_BODY)

    client = JevClient(
        api_key="secret",
        base_url="https://jev.example/",
        model="jev-latest",
        retry_policy=NO_RETRY,
        http_client=mock_http(handler),
    )

    result = client.choose(REQUEST)

    sent = captured[0]
    assert str(sent.url) == "https://jev.example/v1/systemone"
    assert sent.headers["Authorization"] == "Bearer secret"
    payload = json.loads(sent.content)
    assert payload["model"] == "jev-latest"
    assert payload["state"] == {"request": "hello"}
    assert payload["questions"]["route"] == {
        "type": "choice",
        "instructions": "choose one",
        "criteria": {"a": "route a", "b": "route b"},
    }

    assert result.choice == "a"
    assert result.confidence == pytest.approx(0.8)
    assert result.probabilities == {"a": 0.8, "b": 0.2}
    assert result.model == "jev-test"
    assert result.usage == {"input_tokens": 10}


def test_http_error_raises_decision_error(mock_http) -> None:
    client = JevClient(
        api_key="k",
        retry_policy=NO_RETRY,
        http_client=mock_http(lambda _: httpx.Response(401, text="unauthorized")),
    )

    with pytest.raises(DecisionError, match="HTTP 401"):
        client.choose(REQUEST)


def test_network_error_raises_decision_error(mock_http) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = JevClient(api_key="k", retry_policy=NO_RETRY, http_client=mock_http(handler))

    with pytest.raises(DecisionError, match="Could not call Jev"):
        client.choose(REQUEST)


def test_unexpected_response_raises_decision_error(mock_http) -> None:
    client = JevClient(
        api_key="k",
        retry_policy=NO_RETRY,
        http_client=mock_http(lambda _: httpx.Response(200, json={"answers": {}})),
    )

    with pytest.raises(DecisionError, match="Unexpected Jev response"):
        client.choose(REQUEST)


def test_rate_limit_is_retried(mock_http) -> None:
    statuses = iter([429, 529, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status != 200:
            return httpx.Response(status, headers={"Retry-After": "0"})
        return httpx.Response(200, json=OK_BODY)

    client = JevClient(
        api_key="k",
        retry_policy=RetryPolicy(max_retries=2),
        http_client=mock_http(handler),
    )

    assert client.choose(REQUEST).choice == "a"
