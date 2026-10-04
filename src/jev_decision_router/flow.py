"""判断 → ポリシー → 生成 の処理フロー。

UI（Streamlit）から独立しているため、単体でテストできる。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from jev_decision_router.interfaces import (
    ChatClient,
    ChatRequest,
    ChatResponse,
    ChatStream,
    StreamingChatClient,
)
from jev_decision_router.prompting import DecisionContextPromptBuilder, SystemPromptBuilder
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


class TimedStream:
    """ストリーミング応答を反復しながら、最初の断片までの時間と全体の時間を計測する。

    first_chunk_ms / total_ms は反復が終わるまで None。
    """

    def __init__(self, stream: ChatStream) -> None:
        self.model = stream.model
        self.first_chunk_ms: float | None = None
        self.total_ms: float | None = None
        self._chunks = stream.chunks

    def __iter__(self) -> Iterator[str]:
        started = time.perf_counter()
        for chunk in self._chunks:
            if self.first_chunk_ms is None:
                self.first_chunk_ms = _elapsed_ms(started)
            yield chunk
        self.total_ms = _elapsed_ms(started)


class DecisionFlow:
    """DecisionRouter と ChatClient を組み合わせて 1 リクエストを処理する。

    判断結果を LLM にどう渡すかは prompt_builder で決める。デフォルトでは、ルートの
    System Prompt に確信度・確率を書き添える（DecisionContextPromptBuilder）。
    """

    def __init__(
        self,
        router: DecisionRouter,
        chat_client: ChatClient,
        stop_on_human_review: bool = True,
        prompt_builder: SystemPromptBuilder | None = None,
    ) -> None:
        self._router = router
        self._chat_client = chat_client
        self._stop_on_human_review = stop_on_human_review
        self._prompt_builder = prompt_builder or DecisionContextPromptBuilder()

    def decide(self, user_request: str) -> TimedDecision:
        started = time.perf_counter()
        decision = self._router.decide(user_request)
        return TimedDecision(decision=decision, latency_ms=_elapsed_ms(started))

    def should_stop(self, decision: RouteDecision) -> bool:
        return self._stop_on_human_review and decision.route == HUMAN_REVIEW

    def generate(self, decision: RouteDecision, user_request: str) -> TimedResponse:
        started = time.perf_counter()
        response = self._chat_client.chat(self.build_chat_request(decision, user_request))
        return TimedResponse(response=response, latency_ms=_elapsed_ms(started))

    def stream(self, decision: RouteDecision, user_request: str) -> TimedStream:
        """応答をストリーミングで生成する。

        ChatClient が StreamingChatClient でない場合は、通常の応答を 1 つの断片として返す。
        """
        request = self.build_chat_request(decision, user_request)
        if isinstance(self._chat_client, StreamingChatClient):
            return TimedStream(self._chat_client.stream(request))
        response = self._chat_client.chat(request)
        return TimedStream(ChatStream(model=response.model, chunks=iter([response.content])))

    def build_chat_request(self, decision: RouteDecision, user_request: str) -> ChatRequest:
        """判断結果から、生成レイヤーに渡すリクエストを作る（Jev と LLM の接点）。"""
        route = self._router.routes[decision.route]
        return ChatRequest(
            system_prompt=self._prompt_builder.build(route, decision),
            user_prompt=user_request,
        )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
