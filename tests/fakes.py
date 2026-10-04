"""テスト用のフェイク実装とヘルパー。"""

from __future__ import annotations

from jev_decision_router.interfaces import (
    ChatRequest,
    ChatResponse,
    ChatStream,
    ChoiceRequest,
    ChoiceResult,
)


class FakeDecisionClient:
    """DecisionClient のテスト用フェイク。"""

    def __init__(self, result: ChoiceResult) -> None:
        self.result = result
        self.requests: list[ChoiceRequest] = []

    def choose(self, request: ChoiceRequest) -> ChoiceResult:
        self.requests.append(request)
        return self.result


class FakeChatClient:
    """ChatClient のテスト用フェイク。"""

    def __init__(self, content: str = "ok", model: str = "fake-model") -> None:
        self.content = content
        self.model = model
        self.requests: list[ChatRequest] = []

    def chat(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        return ChatResponse(content=self.content, model=self.model)


def make_choice(
    choice: str = "coding",
    confidence: float = 0.9,
    probabilities: dict[str, float] | None = None,
) -> ChoiceResult:
    return ChoiceResult(
        choice=choice,
        confidence=confidence,
        probabilities=probabilities or {choice: confidence},
        model="test-jev",
        usage={},
    )


class FakeStreamingChatClient(FakeChatClient):
    """StreamingChatClient のテスト用フェイク。"""

    def __init__(self, chunks: list[str], model: str = "fake-stream-model") -> None:
        super().__init__(content="".join(chunks), model=model)
        self.chunks = chunks

    def stream(self, request: ChatRequest) -> ChatStream:
        self.requests.append(request)
        return ChatStream(model=self.model, chunks=iter(self.chunks))
