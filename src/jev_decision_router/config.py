"""環境変数からアプリケーション設定を読み込む。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """アプリケーション設定。"""

    typesafe_api_key: str = ""
    jev_base_url: str = "https://api.typesafe.ai"
    jev_model: str = "jev-latest"
    jev_mock_mode: bool = False
    confidence_threshold: float = 0.65
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = ""
    vllm_api_key: str = "EMPTY"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        default = cls()
        return cls(
            typesafe_api_key=env.get("TYPESAFE_API_KEY", default.typesafe_api_key),
            jev_base_url=env.get("JEV_BASE_URL", default.jev_base_url),
            jev_model=env.get("JEV_MODEL", default.jev_model),
            jev_mock_mode=_as_bool(env.get("JEV_MOCK_MODE"), default.jev_mock_mode),
            confidence_threshold=float(
                env.get("JEV_CONFIDENCE_THRESHOLD", default.confidence_threshold)
            ),
            vllm_base_url=env.get("VLLM_BASE_URL", default.vllm_base_url),
            vllm_model=env.get("VLLM_MODEL", default.vllm_model),
            vllm_api_key=env.get("VLLM_API_KEY", default.vllm_api_key),
        )
