"""レイヤー間の抽象インターフェース（Ports）。

上位モジュール（router / app）はこのモジュールの抽象にのみ依存し、
Jev・vLLM などの具体的な実装には依存しない（依存性逆転の原則）。

- DecisionClient: System 1（判断）レイヤー。Jev / Mock が実装する。
- ChatClient:     System 2（生成）レイヤー。vLLM が実装する。
- StreamingChatClient: System 2 のストリーミング版。vLLM が実装する。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# System 1: Decision layer
# ---------------------------------------------------------------------------


class DecisionError(RuntimeError):
    """判断レイヤーの呼び出しに失敗した。"""


@dataclass(frozen=True)
class ChoiceRequest:
    """選択肢から 1 つを選ばせる判断リクエスト。

    Attributes:
        state: 判断材料となるアプリケーション状態（ユーザー入力など）。
        instructions: 判断方針の指示。
        criteria: 選択肢名 → 選択基準の説明。
    """

    state: dict[str, Any]
    instructions: str
    criteria: dict[str, str]


@dataclass(frozen=True)
class ChoiceResult:
    """判断結果。"""

    choice: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DecisionClient(Protocol):
    """System 1（高速な判断）を提供するクライアント。"""

    def choose(self, request: ChoiceRequest) -> ChoiceResult:
        """criteria の中から 1 つを選ぶ。失敗時は DecisionError を送出する。"""
        ...


# ---------------------------------------------------------------------------
# System 2: Generation layer (vLLM)
# ---------------------------------------------------------------------------


class ChatError(RuntimeError):
    """生成レイヤー（vLLM など）の呼び出しに失敗した。"""


@dataclass(frozen=True)
class ChatRequest:
    """生成レイヤーへのリクエスト。

    Attributes:
        system_prompt: ルートごとの System Prompt。
        user_prompt: ユーザーの入力。
        temperature: サンプリング温度。
        max_tokens: 生成する最大トークン数。
    """

    system_prompt: str
    user_prompt: str
    temperature: float = 0.2
    max_tokens: int = 1200


@dataclass(frozen=True)
class ChatResponse:
    """生成レイヤーからのレスポンス。"""

    content: str
    model: str


@runtime_checkable
class ChatClient(Protocol):
    """System 2（推論・生成）を提供するクライアント。

    vLLM の OpenAI 互換 API（POST /v1/chat/completions）を想定するが、
    このインターフェースを満たせば任意の LLM バックエンドに差し替えられる。
    """

    def chat(self, request: ChatRequest) -> ChatResponse:
        """応答を生成する。失敗時は ChatError を送出する。"""
        ...


@dataclass(frozen=True)
class ChatStream:
    """ストリーミング応答。

    Attributes:
        model: 実際に使われたモデル名。
        chunks: 生成されたテキストの断片。反復中の失敗は ChatError として送出される。
    """

    model: str
    chunks: Iterator[str]


@runtime_checkable
class StreamingChatClient(Protocol):
    """応答を逐次返せる生成クライアント。

    ChatClient とは別のインターフェースにしている（インターフェース分離の原則）。
    ストリーミングに対応しないバックエンドは ChatClient だけを実装すればよい。
    """

    def stream(self, request: ChatRequest) -> ChatStream:
        """応答をストリーミングで生成する。接続時の失敗は ChatError を送出する。"""
        ...
