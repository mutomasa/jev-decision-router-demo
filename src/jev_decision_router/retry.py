"""HTTP リトライポリシー（指数バックオフ）。

TypeSafe 公式 SDK の RetryPolicy のデフォルトに合わせている。
https://docs.typesafe.ai/sdk/python/api/retries
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

DEFAULT_RETRY_STATUSES: frozenset[int] = frozenset({408, 429, *range(500, 600)})


@dataclass(frozen=True)
class RetryPolicy:
    """リトライの設定。

    Attributes:
        max_retries: 初回を除く最大リトライ回数。0 でリトライしない。
        backoff_initial: 1 回目のリトライまでの待ち時間（秒）。以降は 2 倍ずつ増える。
        backoff_max: 待ち時間の上限（秒）。
        backoff_jitter: 待ち時間に加えるランダムな揺らぎの割合（0.25 なら最大 +25%）。
        retry_statuses: リトライ対象の HTTP ステータス。
        respect_retry_after: レスポンスの Retry-After ヘッダー（秒）を優先するか。
        retry_after_max: Retry-After に従う場合の待ち時間の上限（秒）。
        retry_on_transport_error: 接続エラー・タイムアウトをリトライするか。
    """

    max_retries: int = 2
    backoff_initial: float = 0.5
    backoff_max: float = 5.0
    backoff_jitter: float = 0.25
    retry_statuses: frozenset[int] = field(default=DEFAULT_RETRY_STATUSES)
    respect_retry_after: bool = True
    retry_after_max: float = 30.0
    retry_on_transport_error: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")

    def backoff(self, attempt: int, rand: Callable[[], float] = random.random) -> float:
        """attempt 回目（0 始まり）のリトライ前の待ち時間を返す。"""
        delay = min(self.backoff_initial * 2.0**attempt, self.backoff_max)
        return delay + delay * self.backoff_jitter * rand()

    def delay_for(self, response: httpx.Response, attempt: int) -> float:
        if self.respect_retry_after:
            retry_after = _parse_retry_after(response)
            if retry_after is not None:
                return min(retry_after, self.retry_after_max)
        return self.backoff(attempt)


NO_RETRY = RetryPolicy(max_retries=0)


def send_with_retry(
    send: Callable[[], httpx.Response],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """send を呼び、リトライ対象のエラーなら policy に従って再送する。

    リトライ回数を使い切った場合、最後のレスポンスを返す（または最後の例外を送出する）。
    ステータスの判定（raise_for_status）は呼び出し側で行う。
    """
    attempt = 0
    while True:
        try:
            response = send()
        except httpx.TransportError:
            if not policy.retry_on_transport_error or attempt >= policy.max_retries:
                raise
            sleep(policy.backoff(attempt))
            attempt += 1
            continue

        if response.status_code not in policy.retry_statuses or attempt >= policy.max_retries:
            return response
        delay = policy.delay_for(response, attempt)
        response.close()
        sleep(delay)
        attempt += 1


def _parse_retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None  # HTTP-date 形式は扱わず、通常のバックオフにする
