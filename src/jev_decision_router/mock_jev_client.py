"""UI 確認用の Mock 判断クライアント（DecisionClient の実装）。

キーワードマッチによる決定論的な判定を返す。Jev の代替ではない。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from jev_decision_router.interfaces import ChoiceRequest, ChoiceResult

DEFAULT_KEYWORD_RULES: Sequence[tuple[str, Sequence[str]]] = (
    ("coding", ("code", "python", "fastapi", "実装", "バグ", "sql", "api")),
    ("deep_analysis", ("比較", "分析", "原因", "戦略", "設計", "評価")),
    ("human_review", ("承認", "送金", "契約", "削除", "本番", "医療", "法律")),
)
FALLBACK_CHOICE = "direct_answer"


class MockJevClient:
    """キーワードルールで Choice を決める Mock。"""

    def __init__(
        self,
        keyword_rules: Sequence[tuple[str, Sequence[str]]] = DEFAULT_KEYWORD_RULES,
        fallback_choice: str = FALLBACK_CHOICE,
    ) -> None:
        self._keyword_rules = keyword_rules
        self._fallback_choice = fallback_choice

    def choose(self, request: ChoiceRequest) -> ChoiceResult:
        text = str(request.state.get("request", ""))
        names = list(request.criteria)
        choice = self._match(text)
        if choice not in names:
            choice = names[0] if names else choice
        return ChoiceResult(
            choice=choice,
            confidence=self._confidence(text),
            probabilities=self._probabilities(names, choice),
            model="mock-jev",
            usage={"input_tokens": 0, "output_tokens": 0},
        )

    def _match(self, text: str) -> str:
        lowered = text.lower()
        for choice, keywords in self._keyword_rules:
            if any(keyword in lowered for keyword in keywords):
                return choice
        return self._fallback_choice

    @staticmethod
    def _probabilities(names: list[str], choice: str) -> dict[str, float]:
        if len(names) <= 1:
            return {name: 1.0 for name in names}
        others = (1.0 - 0.85) / (len(names) - 1)
        return {name: 0.85 if name == choice else others for name in names}

    @staticmethod
    def _confidence(text: str) -> float:
        # 入力ごとに安定した小さな揺らぎを与え、UI がハードコードに見えないようにする。
        digest = hashlib.sha256(text.encode("utf-8")).digest()[0]
        return min(0.82 + (digest / 255.0) * 0.12, 0.94)
