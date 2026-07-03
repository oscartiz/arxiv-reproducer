import httpx
import pytest

from arxiv_reproducer import retry as retry_mod
from arxiv_reproducer.retry import is_retryable_http_error, retry_with_backoff


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(retry_mod.time, "sleep", slept.append)
    return slept


def http_status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://export.arxiv.org")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)


class TestIsRetryable:
    @pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
    def test_transient_statuses_are_retryable(self, code):
        assert is_retryable_http_error(http_status_error(code))

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_caller_errors_are_fatal(self, code):
        assert not is_retryable_http_error(http_status_error(code))

    def test_network_errors_are_retryable(self):
        assert is_retryable_http_error(httpx.ConnectError("no route"))
        assert is_retryable_http_error(httpx.ReadTimeout("slow"))

    def test_unrelated_exceptions_are_fatal(self):
        assert not is_retryable_http_error(ValueError("nope"))


class TestRetryWithBackoff:
    def test_returns_first_success(self, no_sleep):
        assert retry_with_backoff(lambda: 42) == 42
        assert no_sleep == []

    def test_recovers_from_transient_failures(self, no_sleep):
        outcomes = [http_status_error(503), http_status_error(429), "ok"]

        def flaky():
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        assert retry_with_backoff(flaky) == "ok"
        assert len(no_sleep) == 2

    def test_backoff_grows_exponentially(self, no_sleep):
        attempts = iter([http_status_error(503)] * 3 + ["ok"])

        def flaky():
            outcome = next(attempts)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        retry_with_backoff(flaky, base_delay=1.0)
        assert len(no_sleep) == 3
        # jitter adds at most delay/4, so consecutive delays must still grow
        assert no_sleep[0] < no_sleep[1] < no_sleep[2]

    def test_gives_up_after_max_attempts(self, no_sleep):
        calls = []

        def always_503():
            calls.append(1)
            raise http_status_error(503)

        with pytest.raises(httpx.HTTPStatusError):
            retry_with_backoff(always_503, attempts=3)
        assert len(calls) == 3

    def test_fatal_error_is_not_retried(self, no_sleep):
        calls = []

        def not_found():
            calls.append(1)
            raise http_status_error(404)

        with pytest.raises(httpx.HTTPStatusError):
            retry_with_backoff(not_found)
        assert len(calls) == 1
        assert no_sleep == []

    def test_on_retry_callback_sees_each_attempt(self, no_sleep):
        seen = []
        attempts = iter([http_status_error(503), "ok"])

        def flaky():
            outcome = next(attempts)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        retry_with_backoff(flaky, on_retry=lambda n, d, e: seen.append((n, type(e).__name__)))
        assert seen == [(1, "HTTPStatusError")]
