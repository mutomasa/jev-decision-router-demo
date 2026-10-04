from __future__ import annotations

import pytest

from jev_decision_router.interfaces import ChoiceRequest, DecisionClient
from jev_decision_router.mock_jev_client import MockJevClient
from jev_decision_router.routes import DEFAULT_ROUTES


def make_request(text: str) -> ChoiceRequest:
    return ChoiceRequest(
        state={"request": text},
        instructions="",
        criteria={name: r.description for name, r in DEFAULT_ROUTES.items()},
    )


def test_mock_satisfies_decision_client_protocol() -> None:
    assert isinstance(MockJevClient(), DecisionClient)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PythonでCSVを読み込むコードを書いて", "coding"),
        ("売上悪化の原因を分析して", "deep_analysis"),
        ("顧客の本番DBを削除して", "human_review"),
        ("今日の天気は？", "direct_answer"),
    ],
)
def test_keyword_rules(text: str, expected: str) -> None:
    assert MockJevClient().choose(make_request(text)).choice == expected


def test_result_is_deterministic_and_well_formed() -> None:
    client = MockJevClient()

    first = client.choose(make_request("hello"))
    second = client.choose(make_request("hello"))

    assert first == second
    assert 0.82 <= first.confidence <= 0.94
    assert set(first.probabilities) == set(DEFAULT_ROUTES)
    assert sum(first.probabilities.values()) == pytest.approx(1.0)
    assert first.probabilities[first.choice] == pytest.approx(0.85)
