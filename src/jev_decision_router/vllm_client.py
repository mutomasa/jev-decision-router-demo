"""vLLM クライアント（ChatClient の実装）。

vLLM の OpenAI 互換 API を利用する。

    GET  {base_url}/models            モデル未指定時に先頭のモデルを自動選択
    POST {base_url}/chat/completions  ChatRequest → ChatResponse

base_url は `/v1` まで含める（例: http://localhost:8000/v1）。
"""

from __future__ import annotations

from typing import Any

import httpx

from jev_decision_router.interfaces import ChatError, ChatRequest, ChatResponse


class VLLMClient:
    """vLLM の OpenAI 互換 Chat Completions API を呼び出す。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str | None = None,
        api_key: str = "EMPTY",
        timeout: float = 180.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model or None
        self._api_key = api_key
        self._timeout = timeout
        self._http = http_client or httpx.Client()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, request: ChatRequest) -> ChatResponse:
        model = self.resolve_model()
        body = self._post_chat(self._build_payload(model, request))
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"Unexpected vLLM response: {body}") from exc
        return ChatResponse(content=content, model=model)

    def resolve_model(self) -> str:
        """モデル名を返す。未指定の場合は /models の先頭のモデルを使う（結果はキャッシュ）。"""
        if self._model:
            return self._model
        try:
            response = self._http.get(
                f"{self._base_url}/models", headers=self._headers, timeout=10.0
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            if not data:
                raise ChatError("vLLM /models returned no served models.")
            self._model = str(data[0]["id"])
            return self._model
        except httpx.HTTPStatusError as exc:
            raise ChatError(
                f"vLLM /models returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ChatError(f"Could not discover vLLM model: {exc}") from exc

    @staticmethod
    def _build_payload(model: str, request: ChatRequest) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ChatError(
                f"vLLM returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ChatError(f"Could not call vLLM: {exc}") from exc
