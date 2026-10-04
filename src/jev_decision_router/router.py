"""Decision Router: 判断結果に決定論的なポリシー（Confidence Gate）を適用する。

DecisionRouter は DecisionClient（抽象）にのみ依存し、Jev / Mock のどちらでも動作する。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from jev_decision_router.interfaces import ChoiceRequest, ChoiceResult, DecisionClient
from jev_decision_router.routes import DEFAULT_ROUTES, HUMAN_REVIEW, Route

INSTRUCTIONS = (
    "Choose the single workflow route that should handle the human request. "
    "Choose human_review when automatic execution would be risky, irreversible, "
    "or the request lacks critical context. Otherwise choose the route that best "
    "matches the work the downstream language model should perform."
)

SYSTEM_CONTEXT = (
    "This decision is used by a software router. Jev only chooses the workflow; "
    "a separate local vLLM will generate the final response."
)


@dataclass(frozen=True)
class RouteDecision:
    """ポリシー適用後の最終的なルーティング結果。

    Attributes:
        raw_choice: 判断レイヤーの選択（未知の Choice は human_review に正規化済み）。
        route: Confidence Gate 適用後の実行ルート。
        confidence: 判断レイヤーの確信度。
        probabilities: 各 Choice の確率。
        gated: Confidence Gate によって human_review に切り替えられたか。
        model: 判断に使われたモデル名。
        usage: トークン使用量など。
    """

    raw_choice: str
    route: str
    confidence: float
    probabilities: dict[str, float]
    gated: bool
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


class DecisionRouter:
    """判断レイヤーの結果を受け取り、実行ルートを確定する。"""

    def __init__(
        self,
        decision_client: DecisionClient,
        confidence_threshold: float = 0.65,
        routes: Mapping[str, Route] = DEFAULT_ROUTES,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if HUMAN_REVIEW not in routes:
            raise ValueError(f"routes must contain the '{HUMAN_REVIEW}' fallback route")
        self._decision_client = decision_client
        self._confidence_threshold = confidence_threshold
        self._routes = routes

    @property
    def routes(self) -> Mapping[str, Route]:
        return self._routes

    def build_request(self, user_request: str) -> ChoiceRequest:
        return ChoiceRequest(
            state={"request": user_request, "system_context": SYSTEM_CONTEXT},
            instructions=INSTRUCTIONS,
            criteria={name: route.description for name, route in self._routes.items()},
        )

    def decide(self, user_request: str) -> RouteDecision:
        result = self._decision_client.choose(self.build_request(user_request))
        return self.apply_policy(result)

    def apply_policy(self, result: ChoiceResult) -> RouteDecision:
        raw_choice = result.choice if result.choice in self._routes else HUMAN_REVIEW
        gated = result.confidence < self._confidence_threshold
        route = HUMAN_REVIEW if gated else raw_choice
        return RouteDecision(
            raw_choice=raw_choice,
            route=route,
            confidence=result.confidence,
            probabilities=dict(result.probabilities),
            gated=gated,
            model=result.model,
            usage=dict(result.usage),
        )
