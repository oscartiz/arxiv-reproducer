"""Run-directory layout: one timestamped workspace per run, never clobbered.

runs/<arxiv-id>/
    paper.pdf            # download cache, shared across runs
    20260703-142205/     # one workspace per run (bind-mounted into the sandbox)
        paper.pdf        # copy, so the agent and auditors see it in-workspace
        REPORT.md
        ...
    latest -> 20260703-142205
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def safe_dir_name(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def new_run_dir(base: Path, now: datetime | None = None) -> Path:
    """Create a fresh timestamped workspace under base; update `latest`."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = base / stamp
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = base / f"{stamp}-{counter}"
    candidate.mkdir(parents=True)
    _point_latest_at(base, candidate)
    return candidate


def _point_latest_at(base: Path, run_dir: Path) -> None:
    link = base / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(run_dir.name)
    except OSError:
        pass  # symlinks unavailable (e.g. some Windows setups) — best effort
