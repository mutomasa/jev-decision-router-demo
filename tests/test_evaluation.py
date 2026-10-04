from __future__ import annotations

from pathlib import Path

import pytest

from jev_decision_router.evaluation import (
    DEFAULT_CASES_PATH,
    EvalCase,
    evaluate,
    format_report,
    load_cases,
    main,
)
from jev_decision_router.router import DecisionRouter
from jev_decision_router.routes import DEFAULT_ROUTES, HUMAN_REVIEW
from tests.fakes import FakeDecisionClient, make_choice


def test_default_dataset_is_valid_and_covers_all_routes() -> None:
    cases = load_cases(DEFAULT_CASES_PATH, known_routes=DEFAULT_ROUTES)

    assert len(cases) >= 20
    assert {case.expected_route for case in cases} == set(DEFAULT_ROUTES)


def test_load_cases_rejects_unknown_route(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "request": "x", "expected_route": "nope"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown route 'nope'"):
        load_cases(path, known_routes=DEFAULT_ROUTES)


def test_load_cases_rejects_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=":1: invalid case"):
        load_cases(path)


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    line = '{"id": "a", "request": "x", "expected_route": "coding"}\n'
    path = tmp_path / "cases.jsonl"
    path.write_text(line + "\n" + line, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case ids"):
        load_cases(path)


def test_evaluate_computes_raw_and_routed_accuracy() -> None:
    cases = [
        EvalCase("1", "a", "coding"),
        EvalCase("2", "b", "deep_analysis"),
    ]
    router = DecisionRouter(FakeDecisionClient(make_choice("coding", 0.5)), 0.65)

    report = evaluate(router, cases)

    assert report.total == 2
    assert report.raw_accuracy == 0.5
    assert report.route_accuracy == 0.0
    assert report.gated_count == 2
    assert report.per_route_accuracy() == {"coding": 0.0, "deep_analysis": 0.0}
    assert report.confusion[("coding", HUMAN_REVIEW)] == 1
    text = format_report(report)
    assert "Raw accuracy:     50.0%" in text
    assert "[1] expected=coding got=human_review" in text


def test_main_accepts_custom_cases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "1", "request": "Pythonでコードを書いて", "expected_route": "coding"}\n'
        '{"id": "2", "request": "本番DBを削除して", "expected_route": "human_review"}\n',
        encoding="utf-8",
    )

    assert main(["--mock", "--cases", str(path), "--threshold", "0.5"]) == 0

    assert "Route accuracy:   100.0%" in capsys.readouterr().out


def test_main_prints_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--mock", "--threshold", "0.5"]) == 0

    out = capsys.readouterr().out
    assert "Route accuracy:" in out
    assert "Per-route accuracy:" in out
