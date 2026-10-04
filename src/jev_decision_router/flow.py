"""判断 → ポリシー → 生成 の処理フロー。

UI（Streamlit）から独立しているため、単体でテストできる。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from jev_decision_router.interfaces import ChatClient, ChatRequest, ChatResponse
from jev_decision_router.router import DecisionRouter, RouteDecision
from jev_decision_router.routes import HUMAN_REVIEW


@dataclass(frozen=True)
class TimedDecision:
    decision: RouteDecision
    latency_ms: float


@dataclass(frozen=True)
class TimedResponse:
    response: ChatResponse
    latency_ms: float


class DecisionFlow:
    """DecisionRouter と ChatClient を組み合わせて 1 リクエストを処理する。"""

    def __init__(
        self,
        router: DecisionRouter,
        chat_client: ChatClient,
        stop_on_human_review: bool = True,
    ) -> None:
        self._router = router
        self._chat_client = chat_client
        self._stop_on_human_review = stop_on_human_review

    def decide(self, user_request: str) -> TimedDecision:
        started = time.perf_counter()
        decision = self._router.decide(user_request)
        return TimedDecision(decision=decision, latency_ms=_elapsed_ms(started))

    def should_stop(self, decision: RouteDecision) -> bool:
        return self._stop_on_human_review and decision.route == HUMAN_REVIEW

    def generate(self, decision: RouteDecision, user_request: str) -> TimedResponse:
        route = self._router.routes[decision.route]
        started = time.perf_counter()
        response = self._chat_client.chat(
            ChatRequest(system_prompt=route.system_prompt, user_prompt=user_request)
        )
        return TimedResponse(response=response, latency_ms=_elapsed_ms(started))


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
