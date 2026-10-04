from __future__ import annotations

from jev_decision_router.flow import DecisionFlow
from jev_decision_router.prompting import RoutePromptBuilder
from jev_decision_router.router import DecisionRouter
from jev_decision_router.routes import DEFAULT_ROUTES, HUMAN_REVIEW
from tests.fakes import FakeChatClient, FakeDecisionClient, FakeStreamingChatClient, make_choice


def make_flow(
    choice: str, confidence: float, stop: bool = True
) -> tuple[DecisionFlow, FakeChatClient]:
    chat = FakeChatClient(content="generated")
    router = DecisionRouter(FakeDecisionClient(make_choice(choice, confidence)), 0.65)
    return DecisionFlow(router, chat, stop_on_human_review=stop), chat


def test_generate_uses_route_system_prompt() -> None:
    flow, chat = make_flow("coding", 0.9)

    decision = flow.decide("implement it").decision
    generated = flow.generate(decision, "implement it")

    assert flow.should_stop(decision) is False
    assert generated.response.content == "generated"
    assert generated.latency_ms >= 0
    assert chat.requests[0].system_prompt.startswith(DEFAULT_ROUTES["coding"].system_prompt)
    assert chat.requests[0].user_prompt == "implement it"


def test_human_review_stops_when_enabled() -> None:
    flow, _ = make_flow("coding", 0.3, stop=True)

    timed = flow.decide("x")

    assert timed.decision.route == HUMAN_REVIEW
    assert timed.latency_ms >= 0
    assert flow.should_stop(timed.decision) is True


def test_human_review_generates_review_when_stop_disabled() -> None:
    flow, chat = make_flow(HUMAN_REVIEW, 0.9, stop=False)

    decision = flow.decide("x").decision
    assert flow.should_stop(decision) is False

    flow.generate(decision, "x")
    assert chat.requests[0].system_prompt.startswith(DEFAULT_ROUTES[HUMAN_REVIEW].system_prompt)


def test_stream_falls_back_to_single_chunk_for_non_streaming_client() -> None:
    flow, chat = make_flow("coding", 0.9)

    stream = flow.stream(flow.decide("x").decision, "x")

    assert stream.model == "fake-model"
    assert stream.first_chunk_ms is None
    assert list(stream) == ["generated"]
    assert stream.first_chunk_ms is not None
    assert stream.total_ms is not None
    assert chat.requests[0].system_prompt.startswith(DEFAULT_ROUTES["coding"].system_prompt)


def test_stream_uses_streaming_client() -> None:
    chat = FakeStreamingChatClient(["a", "b", "c"])
    router = DecisionRouter(FakeDecisionClient(make_choice("direct_answer", 0.9)))
    flow = DecisionFlow(router, chat)

    stream = flow.stream(flow.decide("x").decision, "question")

    assert "".join(stream) == "abc"
    assert chat.requests[0].user_prompt == "question"
    assert chat.requests[0].system_prompt.startswith(DEFAULT_ROUTES["direct_answer"].system_prompt)


def test_confidence_is_passed_to_llm_by_default() -> None:
    flow, chat = make_flow("coding", 0.9)

    flow.generate(flow.decide("x").decision, "x")

    system_prompt = chat.requests[0].system_prompt
    assert "## Routing context" in system_prompt
    assert "- Decision confidence: 0.90" in system_prompt


def test_prompt_builder_can_be_replaced() -> None:
    chat = FakeChatClient()
    router = DecisionRouter(FakeDecisionClient(make_choice("coding", 0.9)))
    flow = DecisionFlow(router, chat, prompt_builder=RoutePromptBuilder())

    request = flow.build_chat_request(flow.decide("x").decision, "x")

    assert request.system_prompt == DEFAULT_ROUTES["coding"].system_prompt
    assert request.user_prompt == "x"
