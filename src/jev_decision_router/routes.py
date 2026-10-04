"""ルート定義。

ルートを追加する場合は DEFAULT_ROUTES に Route を追加するだけでよく、
router / app のコードを変更する必要はない（開放閉鎖の原則）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class Route:
    """判断レイヤーが選択できる処理ルート。

    Attributes:
        name: Choice 名。
        description: 判断レイヤーに渡す選択基準。
        system_prompt: 生成レイヤーに渡す System Prompt。
    """

    name: str
    description: str
    system_prompt: str


def _registry(*routes: Route) -> Mapping[str, Route]:
    return MappingProxyType({route.name: route for route in routes})


DEFAULT_ROUTES: Mapping[str, Route] = _registry(
    Route(
        name="direct_answer",
        description=(
            "Simple knowledge question, short explanation, or straightforward request "
            "that can be answered directly."
        ),
        system_prompt=(
            "You are a concise local assistant. Answer the request directly. "
            "Avoid unnecessary planning or long preambles. If you are uncertain, say so."
        ),
    ),
    Route(
        name="deep_analysis",
        description=(
            "Requires comparison, multi-step reasoning, architecture/design analysis, "
            "trade-offs, or a structured recommendation."
        ),
        system_prompt=(
            "You are a senior technical analyst. Analyze the request step by step internally, "
            "then provide a structured answer with assumptions, trade-offs, risks, and a "
            "recommendation. Do not expose hidden chain-of-thought; provide concise reasoning "
            "summaries instead."
        ),
    ),
    Route(
        name="coding",
        description=(
            "Primarily asks for software implementation, debugging, commands, configuration, "
            "or code generation."
        ),
        system_prompt=(
            "You are a senior software engineer. Produce practical, runnable guidance and code. "
            "State important assumptions, prefer minimal dependencies, and include verification "
            "steps."
        ),
    ),
    Route(
        name=HUMAN_REVIEW,
        description=(
            "High-impact/irreversible action, sensitive decision, or request with insufficient "
            "context for safe automatic execution."
        ),
        system_prompt=(
            "You are a review assistant. Do not perform irreversible actions. Summarize the "
            "request, identify missing information and risks, and propose what a human reviewer "
            "should check."
        ),
    ),
)
