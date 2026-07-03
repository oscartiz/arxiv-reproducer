"""Structured logging: leveled, with an optional JSON renderer.

The rich console remains the default human-facing progress renderer; these
logs go to stderr and are for operators and log aggregation.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

PACKAGE_LOGGER = "arxiv_reproducer"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def setup_logging(*, verbose: bool = False, json_logs: bool = False) -> None:
    """Configure the package logger. Idempotent: replaces prior handlers."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
    logger.handlers[:] = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")
