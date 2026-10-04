"""設定から具体的なクライアントを組み立てる（Composition Root）。

具象クラス（JevClient / MockJevClient / VLLMClient）を知っているのはこのモジュールだけで、
それ以外のモジュールは interfaces の抽象を通して利用する。
"""

from __future__ import annotations

from jev_decision_router.config import Settings
from jev_decision_router.interfaces import ChatClient, DecisionClient
from jev_decision_router.jev_client import JevClient
from jev_decision_router.mock_jev_client import MockJevClient
from jev_decision_router.retry import RetryPolicy
from jev_decision_router.vllm_client import VLLMClient


class ConfigurationError(ValueError):
    """設定が不正。"""


def build_decision_client(settings: Settings) -> DecisionClient:
    if settings.jev_mock_mode:
        return MockJevClient()
    if not settings.typesafe_api_key:
        raise ConfigurationError("TYPESAFE_API_KEY is required when Jev mock mode is off.")
    return JevClient(
        api_key=settings.typesafe_api_key,
        base_url=settings.jev_base_url,
        model=settings.jev_model,
        timeout=settings.jev_timeout,
        retry_policy=RetryPolicy(max_retries=settings.jev_max_retries),
    )


def build_chat_client(settings: Settings) -> ChatClient:
    return VLLMClient(
        base_url=settings.vllm_base_url,
        model=settings.vllm_model or None,
        api_key=settings.vllm_api_key,
        timeout=settings.vllm_timeout,
        retry_policy=RetryPolicy(max_retries=settings.vllm_max_retries),
    )
