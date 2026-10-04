"""Jev API クライアント（DecisionClient の実装）。

Endpoint: POST {base_url}/v1/systemone
"""

from __future__ import annotations

from typing import Any

import httpx

from jev_decision_router.interfaces import ChoiceRequest, ChoiceResult, DecisionError

QUESTION_KEY = "route"


class JevClient:
    """TypeSafe Jev の Choice 判断を呼び出す。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._http = http_client or httpx.Client()

    def choose(self, request: ChoiceRequest) -> ChoiceResult:
        body = self._post(self._build_payload(request))
        return self._parse(body)

    def _build_payload(self, request: ChoiceRequest) -> dict[str, Any]:
        return {
            "state": request.state,
            "model": self._model,
            "questions": {
                QUESTION_KEY: {
                    "type": "choice",
                    "instructions": request.instructions,
                    "criteria": request.criteria,
                }
            },
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._http.post(
                f"{self._base_url}/v1/systemone",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise DecisionError(
                f"Jev returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise DecisionError(f"Could not call Jev: {exc}") from exc

    def _parse(self, body: dict[str, Any]) -> ChoiceResult:
        try:
            answer = body["answers"][QUESTION_KEY]
            return ChoiceResult(
                choice=answer["choice"],
                confidence=float(answer["confidence"]),
                probabilities={k: float(v) for k, v in answer["probabilities"].items()},
                model=body.get("model", self._model),
                usage=body.get("usage", {}),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise DecisionError(f"Unexpected Jev response: {body}") from exc
