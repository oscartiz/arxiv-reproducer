"""Batch screening: run a list of papers, produce a verdict spreadsheet.

One bad paper must never poison the batch: fetch and run failures become
rows like any other and the loop moves on. The spreadsheet (CSV + markdown,
one row per paper: verdict, confidence, cost, status) is the product —
feasibility triage over many papers at a glance.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class BatchRow:
    """One paper's outcome, whether or not the run got off the ground."""

    arxiv_id: str
    title: str | None = None
    status: str = "not_run"  # run status, or "fetch_error" when never started
    verdict: str | None = None
    confidence: int | None = None
    target_result: str | None = None
    iterations: int | None = None
    wall_clock_seconds: float | None = None
    estimated_cost_usd: float | None = None
    report: str | None = None
    error: str | None = None

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> BatchRow:
        """Build a row from a run's run.json."""
        data = json.loads(manifest_path.read_text())
        report = manifest_path.parent / "REPORT.md"
        return cls(
            arxiv_id=data.get("arxiv_id", "?"),
            title=data.get("title"),
            status=data.get("status", "unknown"),
            verdict=data.get("verdict"),
            confidence=data.get("confidence"),
            target_result=data.get("target_result"),
            iterations=data.get("iterations"),
            wall_clock_seconds=data.get("wall_clock_seconds"),
            estimated_cost_usd=data.get("estimated_cost_usd"),
            report=str(report) if report.exists() else None,
            error=data.get("error"),
        )


COLUMNS = [f.name for f in fields(BatchRow)]


def read_id_file(path: Path) -> list[str]:
    """arXiv IDs from a text file: one per line; blanks and # comments skipped."""
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ValueError(f"could not read batch file {path}: {exc}") from exc
    ids = []
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            ids.append(stripped)
    if not ids:
        raise ValueError(f"batch file {path} contains no arXiv IDs")
    return ids


def write_summary_csv(rows: Sequence[BatchRow], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in asdict(row).items()})


def _md_cell(value: object, max_chars: int = 80) -> str:
    if value is None:
        return "—"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text[: max_chars - 1] + "…" if len(text) > max_chars else text


def write_summary_md(rows: Sequence[BatchRow], path: Path) -> None:
    lines = [
        "| arXiv ID | Verdict | Confidence | Status | Target result | Cost (USD) | Report |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        confidence = f"{row.confidence}%" if row.confidence is not None else None
        cost = f"{row.estimated_cost_usd:.2f}" if row.estimated_cost_usd is not None else None
        lines.append(
            "| " + " | ".join(
                _md_cell(cell)
                for cell in (
                    row.arxiv_id, row.verdict, confidence, row.status,
                    row.target_result, cost, row.report,
                )
            ) + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def summarize(rows: Sequence[BatchRow]) -> str:
    """One-line aggregate: verdict counts and total estimated cost."""
    parts = [f"{len(rows)} paper" + ("s" if len(rows) != 1 else "")]
    for verdict in ("REPRODUCED", "PARTIALLY REPRODUCED", "NOT REPRODUCED"):
        count = sum(1 for row in rows if row.verdict == verdict)
        if count:
            parts.append(f"{count} {verdict}")
    without = sum(1 for row in rows if row.verdict is None)
    if without:
        parts.append(f"{without} without a verdict")
    total = sum(row.estimated_cost_usd for row in rows if row.estimated_cost_usd is not None)
    parts.append(f"estimated cost ${total:.2f}")
    return " · ".join(parts)
