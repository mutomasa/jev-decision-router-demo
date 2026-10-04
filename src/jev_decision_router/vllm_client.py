"""vLLM クライアント（ChatClient / StreamingChatClient の実装）。

vLLM の OpenAI 互換 API を利用する。

    GET  {base_url}/models            モデル未指定時に先頭のモデルを自動選択
    POST {base_url}/chat/completions  ChatRequest → ChatResponse
                                      stream=true の場合は SSE（data: {...} / data: [DONE]）

base_url は `/v1` まで含める（例: http://localhost:8000/v1）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from jev_decision_router.interfaces import ChatError, ChatRequest, ChatResponse, ChatStream
from jev_decision_router.retry import RetryPolicy, send_with_retry

SSE_DATA_PREFIX = "data:"
SSE_DONE = "[DONE]"


class VLLMClient:
    """vLLM の OpenAI 互換 Chat Completions API を呼び出す。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str | None = None,
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        retry_policy: RetryPolicy | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model or None
        self._api_key = api_key
        self._timeout = timeout
        self._retry_policy = retry_policy or RetryPolicy()
        self._http = http_client or httpx.Client()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, request: ChatRequest) -> ChatResponse:
        model = self.resolve_model()
        body = self._post_chat(self._build_payload(model, request, stream=False))
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"Unexpected vLLM response: {body}") from exc
        return ChatResponse(content=content, model=model)

    def stream(self, request: ChatRequest) -> ChatStream:
        model = self.resolve_model()
        payload = self._build_payload(model, request, stream=True)
        response = self._send(
            lambda: self._http.send(
                self._http.build_request(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                    timeout=self._timeout,
                ),
                stream=True,
            ),
            what="vLLM",
        )
        return ChatStream(model=model, chunks=self._iter_chunks(response))

    def resolve_model(self) -> str:
        """モデル名を返す。未指定の場合は /models の先頭のモデルを使う（結果はキャッシュ）。"""
        if self._model:
            return self._model
        response = self._send(
            lambda: self._http.get(f"{self._base_url}/models", headers=self._headers, timeout=10.0),
            what="vLLM /models",
        )
        try:
            data = response.json().get("data", [])
            if not data:
                raise ChatError("vLLM /models returned no served models.")
            self._model = str(data[0]["id"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ChatError(f"Could not discover vLLM model: {exc}") from exc
        return self._model

    @staticmethod
    def _build_payload(model: str, request: ChatRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if stream:
            payload["stream"] = True
        return payload

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._send(
            lambda: self._http.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=self._timeout,
            ),
            what="vLLM",
        )
        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ChatError(f"Could not call vLLM: invalid JSON response: {exc}") from exc
        return body

    def _send(self, send: Callable[[], httpx.Response], what: str) -> httpx.Response:
        """リトライ付きで送信し、エラーを ChatError に変換する。"""
        try:
            response = send_with_retry(send, self._retry_policy)
        except httpx.HTTPError as exc:
            raise ChatError(f"Could not call {what}: {exc}") from exc
        if response.is_error:
            response.read()
            response.close()
            raise ChatError(f"{what} returned HTTP {response.status_code}: {response.text}")
        return response

    @staticmethod
    def _iter_chunks(response: httpx.Response) -> Iterator[str]:
        try:
            for line in response.iter_lines():
                if not line.startswith(SSE_DATA_PREFIX):
                    continue
                data = line[len(SSE_DATA_PREFIX) :].strip()
                if data == SSE_DONE:
                    return
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                    raise ChatError(f"Unexpected vLLM stream chunk: {data}") from exc
                content = delta.get("content")
                if content:
                    yield content
        except httpx.HTTPError as exc:
            raise ChatError(f"vLLM stream was interrupted: {exc}") from exc
        finally:
            response.close()
