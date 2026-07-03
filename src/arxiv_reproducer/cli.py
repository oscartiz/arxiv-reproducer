"""Command-line entry point: arxiv-repro <arxiv-id-or-url>."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import httpx
from rich.console import Console

from .agent import run_reproduction
from .paper import fetch_paper, parse_arxiv_id
from .sandbox import SANDBOX_IMAGE, check_docker, ensure_image, image_exists


def has_anthropic_credentials() -> bool:
    """True if the Anthropic SDK will find some credential source.

    The SDK resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
    `ant auth login` OAuth profile on disk — an unset env var alone does
    not mean there are no credentials.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config_dir = Path(
        os.environ.get("ANTHROPIC_CONFIG_DIR", Path.home() / ".config" / "anthropic")
    )
    return config_dir.is_dir() and any(config_dir.glob("credentials/*.json"))


def main(argv: list[str] | None = None) -> None:
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
    args = parser.parse_args(argv)

    console = Console()

    docker_problem = check_docker()
    if docker_problem is not None:
        console.print(f"[red]{docker_problem}[/red]")
        sys.exit(1)

    if not has_anthropic_credentials():
        console.print(
            "[red]No Anthropic credentials found.[/red] "
            "Set ANTHROPIC_API_KEY (see .env.example) or run `ant auth login`."
        )
        sys.exit(1)

    try:
        arxiv_id = parse_arxiv_id(args.paper)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    if not image_exists(SANDBOX_IMAGE):
        console.print(
            f"[bold]Building sandbox image[/bold] {SANDBOX_IMAGE} "
            "(first run only — this takes a few minutes) ..."
        )
    try:
        ensure_image(SANDBOX_IMAGE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        console.print(f"[red]Failed to build sandbox image:[/red]\n{stderr.strip()[-2000:]}")
        sys.exit(1)

    workdir = args.runs_dir / arxiv_id.replace("/", "_")

    console.print(f"[bold]Fetching[/bold] arXiv:{arxiv_id} ...")
    try:
        paper = fetch_paper(arxiv_id, workdir)
    except httpx.HTTPStatusError as exc:
        console.print(
            f"[red]arXiv returned HTTP {exc.response.status_code} for {arxiv_id} — "
            "check that the ID exists.[/red]"
        )
        sys.exit(1)
    except httpx.HTTPError as exc:
        console.print(f"[red]Network error fetching {arxiv_id}: {exc}[/red]")
        sys.exit(1)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(f"[bold]{paper.title}[/bold]")
    console.print(f"{', '.join(paper.authors)}\n")

    console.print("[bold]Starting reproduction agent[/bold] (this can take a while)\n")
    report = run_reproduction(paper, workdir, console)

    console.print(f"\n[green]Done.[/green] Report: [bold]{report}[/bold]")


if __name__ == "__main__":
    main()
