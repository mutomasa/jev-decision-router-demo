"""ルーティング精度の評価。

JSONL のデータセット（1 行 1 ケース: id / request / expected_route）を DecisionRouter に流し、
判断レイヤーの生の選択（raw_choice）と Confidence Gate 適用後のルート（route）の正解率を出す。

    uv run jev-router-eval                       # .env の設定（実際の Jev）で評価
    uv run jev-router-eval --mock                # Mock で評価（動作確認用）
    uv run jev-router-eval --threshold 0.8       # しきい値を変えて評価
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from jev_decision_router.config import Settings
from jev_decision_router.factory import build_decision_client
from jev_decision_router.router import DecisionRouter, RouteDecision

# リポジトリのルートにある evals/ を指す（editable install での利用を前提とする）
DEFAULT_CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "routing_cases.jsonl"


@dataclass(frozen=True)
class EvalCase:
    id: str
    request: str
    expected_route: str


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    decision: RouteDecision

    @property
    def raw_correct(self) -> bool:
        return self.decision.raw_choice == self.case.expected_route

    @property
    def route_correct(self) -> bool:
        return self.decision.route == self.case.expected_route


@dataclass(frozen=True)
class EvalReport:
    results: list[CaseResult]
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def raw_accuracy(self) -> float:
        return _ratio(sum(r.raw_correct for r in self.results), self.total)

    @property
    def route_accuracy(self) -> float:
        return _ratio(sum(r.route_correct for r in self.results), self.total)

    @property
    def gated_count(self) -> int:
        return sum(r.decision.gated for r in self.results)

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.route_correct]

    def per_route_accuracy(self) -> dict[str, float]:
        totals: Counter[str] = Counter()
        correct: Counter[str] = Counter()
        for result in self.results:
            totals[result.case.expected_route] += 1
            correct[result.case.expected_route] += result.route_correct
        return {route: _ratio(correct[route], totals[route]) for route in sorted(totals)}


def load_cases(path: Path, known_routes: Iterable[str] | None = None) -> list[EvalCase]:
    """JSONL からケースを読み込む。known_routes を渡すと未知のルートを検出する。"""
    routes = set(known_routes) if known_routes is not None else None
    cases: list[EvalCase] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = EvalCase(
                id=str(raw["id"]),
                request=str(raw["request"]),
                expected_route=str(raw["expected_route"]),
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid case: {exc}") from exc
        if routes is not None and case.expected_route not in routes:
            raise ValueError(f"{path}:{line_no}: unknown route '{case.expected_route}'")
        cases.append(case)
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case ids: {duplicates}")
    return cases


def evaluate(router: DecisionRouter, cases: Sequence[EvalCase]) -> EvalReport:
    results = [CaseResult(case=case, decision=router.decide(case.request)) for case in cases]
    confusion: Counter[tuple[str, str]] = Counter(
        (r.case.expected_route, r.decision.route) for r in results
    )
    return EvalReport(results=results, confusion=confusion)


def format_report(report: EvalReport) -> str:
    lines = [
        f"Cases:            {report.total}",
        f"Raw accuracy:     {report.raw_accuracy:.1%}  (Jev choice before the confidence gate)",
        f"Route accuracy:   {report.route_accuracy:.1%}  (after the confidence gate)",
        f"Gated to review:  {report.gated_count}",
        "",
        "Per-route accuracy:",
    ]
    lines += [f"  {route:<15} {acc:.1%}" for route, acc in report.per_route_accuracy().items()]
    if report.failures:
        lines += ["", "Failures:"]
        lines += [
            f"  [{r.case.id}] expected={r.case.expected_route} got={r.decision.route} "
            f"(raw={r.decision.raw_choice}, confidence={r.decision.confidence:.2f}) "
            f"{r.case.request}"
            for r in report.failures
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Jev routing accuracy.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--mock", action="store_true", help="use the keyword-based mock")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    settings = Settings.from_env()
    if args.mock:
        settings = replace(settings, jev_mock_mode=True)
    threshold = settings.confidence_threshold if args.threshold is None else args.threshold

    router = DecisionRouter(build_decision_client(settings), confidence_threshold=threshold)
    cases = load_cases(args.cases, known_routes=router.routes)
    print(format_report(evaluate(router, cases)))
    return 0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
