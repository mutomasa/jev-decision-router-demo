from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from jev_decision_router.interfaces import (
    ChatClient,
    ChatError,
    ChatRequest,
    StreamingChatClient,
)
from jev_decision_router.retry import NO_RETRY, RetryPolicy
from jev_decision_router.vllm_client import VLLMClient

REQUEST = ChatRequest(system_prompt="sys", user_prompt="hi", temperature=0.1, max_tokens=50)


def chat_body(content: str = "hello") -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_vllm_client_satisfies_chat_client_protocol() -> None:
    assert isinstance(VLLMClient(), ChatClient)


def test_chat_posts_openai_compatible_payload(mock_http) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=chat_body("answer"))

    client = VLLMClient(
        base_url="http://vllm:8000/v1/",
        model="local-qwen",
        api_key="key",
        retry_policy=NO_RETRY,
        http_client=mock_http(handler),
    )

    response = client.chat(REQUEST)

    assert response.content == "answer"
    assert response.model == "local-qwen"
    sent = captured[0]
    assert str(sent.url) == "http://vllm:8000/v1/chat/completions"
    assert sent.headers["Authorization"] == "Bearer key"
    assert json.loads(sent.content) == {
        "model": "local-qwen",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "temperature": 0.1,
        "max_tokens": 50,
    }


def test_model_is_discovered_from_models_endpoint_and_cached(mock_http) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "served-a"}, {"id": "served-b"}]})
        assert json.loads(request.content)["model"] == "served-a"
        return httpx.Response(200, json=chat_body())

    client = VLLMClient(
        base_url="http://vllm/v1", retry_policy=NO_RETRY, http_client=mock_http(handler)
    )

    assert client.chat(REQUEST).model == "served-a"
    client.chat(REQUEST)

    assert calls.count("/v1/models") == 1


def test_no_served_models_raises_chat_error(mock_http) -> None:
    client = VLLMClient(
        retry_policy=NO_RETRY,
        http_client=mock_http(lambda _: httpx.Response(200, json={"data": []})),
    )

    with pytest.raises(ChatError, match="no served models"):
        client.chat(REQUEST)


def test_http_error_raises_chat_error(mock_http) -> None:
    client = VLLMClient(
        model="m",
        retry_policy=NO_RETRY,
        http_client=mock_http(lambda _: httpx.Response(500, text="oops")),
    )

    with pytest.raises(ChatError, match="HTTP 500"):
        client.chat(REQUEST)


def test_connection_error_raises_chat_error(mock_http) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = VLLMClient(model="m", retry_policy=NO_RETRY, http_client=mock_http(handler))

    with pytest.raises(ChatError, match="Could not call vLLM"):
        client.chat(REQUEST)


def test_unexpected_response_raises_chat_error(mock_http) -> None:
    client = VLLMClient(
        model="m",
        retry_policy=NO_RETRY,
        http_client=mock_http(lambda _: httpx.Response(200, json={"choices": []})),
    )

    with pytest.raises(ChatError, match="Unexpected vLLM response"):
        client.chat(REQUEST)


def sse(*events: str) -> bytes:
    return "".join(f"data: {event}\n\n" for event in events).encode()


def delta(content: str | None) -> str:
    body = {} if content is None else {"content": content}
    return json.dumps({"choices": [{"delta": body}]})


def test_stream_yields_content_deltas(mock_http) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = sse(delta(None), delta("Hel"), delta("lo"), delta(""), "[DONE]")
        return httpx.Response(200, content=body, headers={"Content-Type": "text/event-stream"})

    client = VLLMClient(model="m", retry_policy=NO_RETRY, http_client=mock_http(handler))

    stream = client.stream(REQUEST)

    assert stream.model == "m"
    assert list(stream.chunks) == ["Hel", "lo"]
    assert json.loads(captured[0].content)["stream"] is True


def test_stream_http_error_raises_chat_error(mock_http) -> None:
    client = VLLMClient(
        model="m",
        retry_policy=NO_RETRY,
        http_client=mock_http(lambda _: httpx.Response(400, text="bad request")),
    )

    with pytest.raises(ChatError, match="HTTP 400: bad request"):
        client.stream(REQUEST)


def test_stream_invalid_chunk_raises_chat_error(mock_http) -> None:
    client = VLLMClient(
        model="m",
        retry_policy=NO_RETRY,
        http_client=mock_http(lambda _: httpx.Response(200, content=sse("not-json"))),
    )

    with pytest.raises(ChatError, match="Unexpected vLLM stream chunk"):
        list(client.stream(REQUEST).chunks)


def test_vllm_client_satisfies_streaming_protocol() -> None:
    assert isinstance(VLLMClient(), StreamingChatClient)


def test_server_error_is_retried(mock_http) -> None:
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status != 200:
            return httpx.Response(status, headers={"Retry-After": "0"})
        return httpx.Response(200, json=chat_body("recovered"))

    client = VLLMClient(
        model="m", retry_policy=RetryPolicy(max_retries=1), http_client=mock_http(handler)
    )

    assert client.chat(REQUEST).content == "recovered"
