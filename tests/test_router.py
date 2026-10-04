from __future__ import annotations

import pytest

from jev_decision_router.router import DecisionRouter
from jev_decision_router.routes import DEFAULT_ROUTES, HUMAN_REVIEW, Route
from tests.fakes import FakeDecisionClient, make_choice


def test_decide_passes_route_when_confidence_is_high() -> None:
    router = DecisionRouter(FakeDecisionClient(make_choice("coding", 0.9)), 0.65)

    decision = router.decide("FastAPIでAPIを実装して")

    assert decision.raw_choice == "coding"
    assert decision.route == "coding"
    assert decision.gated is False


def test_decide_gates_to_human_review_when_confidence_is_low() -> None:
    router = DecisionRouter(FakeDecisionClient(make_choice("coding", 0.58)), 0.65)

    decision = router.decide("something")

    assert decision.raw_choice == "coding"
    assert decision.route == HUMAN_REVIEW
    assert decision.gated is True


def test_confidence_equal_to_threshold_passes() -> None:
    router = DecisionRouter(FakeDecisionClient(make_choice("coding", 0.65)), 0.65)

    assert router.decide("x").route == "coding"


def test_unknown_choice_falls_back_to_human_review() -> None:
    router = DecisionRouter(FakeDecisionClient(make_choice("unknown_route", 0.99)), 0.65)

    decision = router.decide("x")

    assert decision.raw_choice == HUMAN_REVIEW
    assert decision.route == HUMAN_REVIEW
    assert decision.gated is False


def test_decide_sends_all_routes_as_criteria() -> None:
    client = FakeDecisionClient(make_choice())
    router = DecisionRouter(client)

    router.decide("hello")

    request = client.requests[0]
    assert request.state["request"] == "hello"
    assert set(request.criteria) == set(DEFAULT_ROUTES)
    assert request.criteria["coding"] == DEFAULT_ROUTES["coding"].description


def test_custom_routes_are_used() -> None:
    routes = {
        "faq": Route("faq", "FAQ", "answer"),
        HUMAN_REVIEW: Route(HUMAN_REVIEW, "review", "review"),
    }
    client = FakeDecisionClient(make_choice("faq", 0.9))

    decision = DecisionRouter(client, routes=routes).decide("x")

    assert decision.route == "faq"
    assert set(client.requests[0].criteria) == {"faq", HUMAN_REVIEW}


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        DecisionRouter(FakeDecisionClient(make_choice()), threshold)


def test_routes_without_human_review_are_rejected() -> None:
    with pytest.raises(ValueError):
        DecisionRouter(
            FakeDecisionClient(make_choice()),
            routes={"faq": Route("faq", "FAQ", "answer")},
        )
