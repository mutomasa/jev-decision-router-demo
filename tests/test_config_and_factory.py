from __future__ import annotations

import pytest

from jev_decision_router.config import Settings
from jev_decision_router.factory import (
    ConfigurationError,
    build_chat_client,
    build_decision_client,
)
from jev_decision_router.jev_client import JevClient
from jev_decision_router.mock_jev_client import MockJevClient
from jev_decision_router.vllm_client import VLLMClient


def test_settings_defaults_when_env_is_empty() -> None:
    assert Settings.from_env({}) == Settings()


def test_settings_from_env() -> None:
    settings = Settings.from_env(
        {
            "TYPESAFE_API_KEY": "key",
            "JEV_BASE_URL": "https://jev.example",
            "JEV_MODEL": "jev-x",
            "JEV_MOCK_MODE": "TRUE",
            "JEV_CONFIDENCE_THRESHOLD": "0.8",
            "VLLM_BASE_URL": "http://vllm/v1",
            "VLLM_MODEL": "qwen",
            "VLLM_API_KEY": "vkey",
        }
    )

    assert settings == Settings(
        typesafe_api_key="key",
        jev_base_url="https://jev.example",
        jev_model="jev-x",
        jev_mock_mode=True,
        confidence_threshold=0.8,
        vllm_base_url="http://vllm/v1",
        vllm_model="qwen",
        vllm_api_key="vkey",
    )


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("1", True), ("no", False)])
def test_mock_mode_parsing(raw: str, expected: bool) -> None:
    assert Settings.from_env({"JEV_MOCK_MODE": raw}).jev_mock_mode is expected


def test_build_decision_client_returns_mock_in_mock_mode() -> None:
    assert isinstance(build_decision_client(Settings(jev_mock_mode=True)), MockJevClient)


def test_build_decision_client_returns_jev_with_api_key() -> None:
    assert isinstance(build_decision_client(Settings(typesafe_api_key="k")), JevClient)


def test_build_decision_client_requires_api_key() -> None:
    with pytest.raises(ConfigurationError):
        build_decision_client(Settings())


def test_build_chat_client_returns_vllm() -> None:
    assert isinstance(build_chat_client(Settings()), VLLMClient)
