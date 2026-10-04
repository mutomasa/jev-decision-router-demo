from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest


@pytest.fixture
def mock_http() -> Callable[[Callable[[httpx.Request], httpx.Response]], httpx.Client]:
    """リクエストハンドラから httpx.Client を作る。"""

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    return factory
