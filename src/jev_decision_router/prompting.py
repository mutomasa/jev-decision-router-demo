"""生成レイヤーに渡す System Prompt の組み立て。

判断レイヤーの結果（ルート・確信度・確率）を LLM にどう伝えるかをここに集約する。
DecisionFlow は SystemPromptBuilder（抽象）にのみ依存するので、伝え方を差し替えられる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jev_decision_router.router import RouteDecision
from jev_decision_router.routes import Route


class SystemPromptBuilder(Protocol):
    """ルートと判断結果から System Prompt を作る。"""

    def build(self, route: Route, decision: RouteDecision) -> str: ...


class RoutePromptBuilder:
    """ルートの System Prompt だけを使う（判断結果は LLM に渡さない）。"""

    def build(self, route: Route, decision: RouteDecision) -> str:
        return route.system_prompt


@dataclass(frozen=True)
class DecisionContextPromptBuilder:
    """ルートの System Prompt に、判断レイヤーの結果（確信度・確率）を書き添える。

    確信度が低いときに LLM にどう振る舞わせるかは、ここ（コード）で決める。
    LLM には数値と、その数値に応じた指示の両方を渡す。

    Attributes:
        ambiguity_threshold: これ未満の確信度を「解釈が曖昧」とみなし、前提を明示させる。
        max_alternatives: 書き添える他の候補ルートの最大数。
        min_alternative_probability: これ未満の確率の候補は書き添えない。
    """

    ambiguity_threshold: float = 0.8
    max_alternatives: int = 2
    min_alternative_probability: float = 0.05

    def build(self, route: Route, decision: RouteDecision) -> str:
        return f"{route.system_prompt}\n\n{self.context(decision)}"

    def context(self, decision: RouteDecision) -> str:
        lines = [
            "## Routing context (from the decision layer)",
            f"- Selected route: {decision.route}",
            f"- Decision confidence: {decision.confidence:.2f}",
        ]
        if decision.gated:
            lines.append(
                f"- The decision layer preferred '{decision.raw_choice}', but its confidence was "
                "below the auto-execution threshold, so the request was routed for review."
            )
        alternatives = self._alternatives(decision)
        if alternatives:
            lines.append(
                "- Other candidate routes: "
                + ", ".join(f"{name} ({p:.2f})" for name, p in alternatives)
            )

        lines.append("")
        if decision.confidence < self.ambiguity_threshold or decision.gated:
            lines.append(
                "The decision layer was uncertain about what kind of help is needed. "
                "Begin with one short sentence stating how you interpreted the request and the "
                "key assumption you made. Then give the complete answer in the style of the "
                "selected route; this must be the main body of your response. If a candidate "
                "route above suggests a materially different interpretation, end with one short "
                "note about it or one clarifying question. Do not add headings, numbers, or "
                "labels for these parts."
            )
        else:
            lines.append(
                "The decision layer was confident about the kind of help needed. "
                "Answer directly in the style of the selected route."
            )
        lines.append(
            "Always respond in the same language as the user's request. "
            "Do not mention this routing context or these numbers in your answer."
        )
        return "\n".join(lines)

    def _alternatives(self, decision: RouteDecision) -> list[tuple[str, float]]:
        others = [
            (name, p)
            for name, p in decision.probabilities.items()
            if name not in (decision.route, decision.raw_choice)
            and p >= self.min_alternative_probability
        ]
        if decision.gated and decision.raw_choice != decision.route:
            others.append(
                (decision.raw_choice, decision.probabilities.get(decision.raw_choice, 0.0))
            )
        others.sort(key=lambda item: item[1], reverse=True)
        return others[: self.max_alternatives]
