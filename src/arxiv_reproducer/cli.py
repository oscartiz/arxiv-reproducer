"""Command-line entry point: arxiv-repro <arxiv-id-or-url>."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from rich.console import Console

from .agent import run_reproduction
from .paper import fetch_paper, parse_arxiv_id


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="arxiv-repro",
        description="Attempt to reproduce a computational result from an arXiv paper.",
    )
    parser.add_argument("paper", help="arXiv ID or URL, e.g. 2301.12345")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory where per-paper workspaces are created (default: ./runs)",
    )
    args = parser.parse_args()

    console = Console()

    if shutil.which("docker") is None:
        console.print("[red]Docker is required but was not found on PATH.[/red]")
        sys.exit(1)

    arxiv_id = parse_arxiv_id(args.paper)
    workdir = args.runs_dir / arxiv_id.replace("/", "_")

    console.print(f"[bold]Fetching[/bold] arXiv:{arxiv_id} ...")
    paper = fetch_paper(arxiv_id, workdir)
    console.print(f"[bold]{paper.title}[/bold]")
    console.print(f"{', '.join(paper.authors)}\n")

    console.print("[bold]Starting reproduction agent[/bold] (this can take a while)\n")
    report = run_reproduction(paper, workdir, console)

    console.print(f"\n[green]Done.[/green] Report: [bold]{report}[/bold]")


if __name__ == "__main__":
    main()
