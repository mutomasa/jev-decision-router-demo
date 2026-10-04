from __future__ import annotations

from jev_decision_router.prompting import DecisionContextPromptBuilder, RoutePromptBuilder
from jev_decision_router.router import RouteDecision
from jev_decision_router.routes import DEFAULT_ROUTES, HUMAN_REVIEW

CODING = DEFAULT_ROUTES["coding"]


def decision(
    route: str = "coding",
    confidence: float = 0.95,
    probabilities: dict[str, float] | None = None,
    raw_choice: str | None = None,
    gated: bool = False,
) -> RouteDecision:
    return RouteDecision(
        raw_choice=raw_choice or route,
        route=route,
        confidence=confidence,
        probabilities=probabilities or {route: confidence},
        gated=gated,
        model="test-jev",
    )


def test_route_prompt_builder_ignores_decision() -> None:
    assert RoutePromptBuilder().build(CODING, decision(confidence=0.1)) == CODING.system_prompt


def test_context_is_appended_to_route_prompt() -> None:
    prompt = DecisionContextPromptBuilder().build(CODING, decision(confidence=0.93))

    assert prompt.startswith(CODING.system_prompt + "\n\n## Routing context")
    assert "- Selected route: coding" in prompt
    assert "- Decision confidence: 0.93" in prompt
    assert "Do not mention this routing context" in prompt


def test_high_confidence_asks_for_direct_answer() -> None:
    context = DecisionContextPromptBuilder().context(decision(confidence=0.95))

    assert "was confident" in context
    assert "one short sentence stating how you interpreted" not in context


def test_low_confidence_asks_to_state_interpretation() -> None:
    context = DecisionContextPromptBuilder(ambiguity_threshold=0.8).context(
        decision(confidence=0.7)
    )

    assert "was uncertain" in context
    assert "one short sentence stating how you interpreted" in context


def test_threshold_boundary_is_treated_as_confident() -> None:
    context = DecisionContextPromptBuilder(ambiguity_threshold=0.8).context(
        decision(confidence=0.8)
    )

    assert "was confident" in context


def test_alternatives_are_sorted_filtered_and_limited() -> None:
    probabilities = {
        "coding": 0.6,
        "deep_analysis": 0.25,
        "direct_answer": 0.1,
        "human_review": 0.04,
    }
    builder = DecisionContextPromptBuilder(max_alternatives=2, min_alternative_probability=0.05)

    context = builder.context(decision(confidence=0.6, probabilities=probabilities))

    assert "- Other candidate routes: deep_analysis (0.25), direct_answer (0.10)" in context
    assert "human_review (0.04)" not in context


def test_no_alternatives_line_when_all_probability_is_on_selected_route() -> None:
    context = DecisionContextPromptBuilder().context(decision(probabilities={"coding": 1.0}))

    assert "Other candidate routes" not in context


def test_gated_decision_mentions_original_choice() -> None:
    gated = decision(
        route=HUMAN_REVIEW,
        raw_choice="coding",
        confidence=0.33,
        probabilities={"coding": 0.5, "deep_analysis": 0.3, "human_review": 0.2},
        gated=True,
    )

    context = DecisionContextPromptBuilder().context(gated)

    assert "- Selected route: human_review" in context
    assert "preferred 'coding'" in context
    assert "coding (0.50)" in context
    assert "was uncertain" in context
