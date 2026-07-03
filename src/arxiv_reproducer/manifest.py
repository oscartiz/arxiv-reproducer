"""Machine-readable run manifest (run.json) and report parsing helpers.

Every run writes a run.json next to REPORT.md so batches of runs are
queryable in aggregate: verdicts, token spend, wall clock, failures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA_VERSION = 1

# Order matters: the qualified verdicts contain the bare word REPRODUCED.
_VERDICTS = ("PARTIALLY REPRODUCED", "NOT REPRODUCED", "REPRODUCED")


def verdict_from_report(report_text: str) -> str | None:
    """Best-effort extraction of the agent's verdict from REPORT.md."""
    for verdict in _VERDICTS:
        if re.search(rf"\b{verdict}\b", report_text):
            return verdict
    return None


def section_snippet(report_text: str, heading: str, max_chars: int = 300) -> str | None:
    """First line of a markdown section, e.g. 'Target Result'."""
    match = re.search(
        rf"^#+\s*{re.escape(heading)}\s*$(.*?)(?=^#+\s|\Z)",
        report_text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        return None
    body = match.group(1).strip()
    if not body:
        return None
    return body.splitlines()[0].strip()[:max_chars]


def write_manifest(workdir: Path, data: dict) -> Path:
    path = workdir / "run.json"
    payload = {"schema_version": SCHEMA_VERSION, **data}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
