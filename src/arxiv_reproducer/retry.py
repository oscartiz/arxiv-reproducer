"""Retry with exponential backoff for transient failures.

Used around arXiv API calls and PDF downloads. Anthropic API calls rely on
the SDK's built-in retry (configured via max_retries on the client), which
already handles 429/5xx/connection errors with backoff.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

import httpx

T = TypeVar("T")

# Transient by nature: worth retrying. Everything else (400, 401, 404, ...)
# is a caller or resource problem and must surface immediately.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

DEFAULT_ATTEMPTS = 4
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0


def is_retryable_http_error(exc: Exception) -> bool:
    """Transient network/server trouble → True; caller errors → False."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool] = is_retryable_http_error,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    """Call fn(), retrying retryable failures with exponential backoff + jitter.

    Non-retryable exceptions and the final failed attempt propagate unchanged.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not is_retryable(exc):
                raise
            delay = min(base_delay * 2 ** (attempt - 1), max_delay)
            delay += random.uniform(0, delay / 4)
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            time.sleep(delay)
    raise AssertionError("unreachable")
