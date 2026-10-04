from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "jev_decision_router" / "app.py"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("JEV_MOCK_MODE", "true")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("VLLM_MODEL", "local-qwen")
    monkeypatch.setenv("VLLM_MAX_RETRIES", "0")
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_app_renders_without_error(app: AppTest) -> None:
    assert not app.exception
    assert app.title[0].value.startswith("🧭")


def test_empty_request_shows_warning(app: AppTest) -> None:
    app.button[0].click().run()

    assert [w.value for w in app.warning] == ["Please enter a request."]


def test_human_review_stops_before_vllm(app: AppTest) -> None:
    app.text_area[0].input("顧客の本番DBを削除して")
    app.button[0].click().run()

    assert not app.exception
    assert any("stopped automatic generation" in w.value for w in app.warning)
    assert app.metric[0].value == "human_review"


def test_vllm_connection_error_is_shown(app: AppTest) -> None:
    app.text_area[0].input("PythonでCSVを読み込むコードを書いて")
    app.button[0].click().run()

    assert app.metric[0].value == "coding"
    assert any("Could not call vLLM" in e.message for e in app.exception)
