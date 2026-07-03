import json
import logging

from arxiv_reproducer.logs import PACKAGE_LOGGER, JsonFormatter, get_logger, setup_logging


class TestJsonFormatter:
    def test_produces_parseable_json_with_core_fields(self):
        record = logging.LogRecord(
            name="arxiv_reproducer.sandbox",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="container %s started",
            args=("abc",),
            exc_info=None,
        )
        payload = json.loads(JsonFormatter().format(record))
        assert payload["level"] == "WARNING"
        assert payload["logger"] == "arxiv_reproducer.sandbox"
        assert payload["msg"] == "container abc started"
        assert "ts" in payload

    def test_includes_exception_text(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="x", level=logging.ERROR, pathname=__file__, lineno=1,
                msg="failed", args=(), exc_info=sys.exc_info(),
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "boom" in payload["exc"]


class TestSetupLogging:
    def test_json_option_swaps_formatter(self):
        setup_logging(json_logs=True)
        handler = logging.getLogger(PACKAGE_LOGGER).handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_default_is_human_readable(self):
        setup_logging()
        handler = logging.getLogger(PACKAGE_LOGGER).handlers[0]
        assert not isinstance(handler.formatter, JsonFormatter)

    def test_idempotent_no_duplicate_handlers(self):
        setup_logging()
        setup_logging()
        assert len(logging.getLogger(PACKAGE_LOGGER).handlers) == 1

    def test_verbose_enables_debug(self):
        setup_logging(verbose=True)
        assert logging.getLogger(PACKAGE_LOGGER).level == logging.DEBUG
        setup_logging(verbose=False)
        assert logging.getLogger(PACKAGE_LOGGER).level == logging.INFO


class TestGetLogger:
    def test_namespaced_under_package(self):
        assert get_logger("agent").name == f"{PACKAGE_LOGGER}.agent"
