"""Command-line entry point: arxiv-repro <arxiv-id-or-url>."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from rich.console import Console

from .agent import run_reproduction
from .config import ConfigError, get_config
from .logs import setup_logging
from .paper import PdfExtractionError, fetch_paper, parse_arxiv_id
from .runs import new_run_dir, safe_dir_name
from .sandbox import check_docker, ensure_image, image_exists


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
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug-level logging"
    )
    parser.add_argument(
        "--log-json", action="store_true", help="Emit logs as JSON lines on stderr"
    )
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose, json_logs=args.log_json)
    console = Console()

    try:
        cfg = get_config()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

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

    if not image_exists(cfg.sandbox_image):
        console.print(
            f"[bold]Building sandbox image[/bold] {cfg.sandbox_image} "
            "(first run only — this takes a few minutes) ..."
        )
    try:
        ensure_image(cfg.sandbox_image)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        console.print(f"[red]Failed to build sandbox image:[/red]\n{stderr.strip()[-2000:]}")
        sys.exit(1)

    base_dir = args.runs_dir / safe_dir_name(arxiv_id)

    console.print(f"[bold]Fetching[/bold] arXiv:{arxiv_id} ...")
    try:
        paper = fetch_paper(arxiv_id, base_dir)
    except httpx.HTTPStatusError as exc:
        console.print(
            f"[red]arXiv returned HTTP {exc.response.status_code} for {arxiv_id} — "
            "check that the ID exists.[/red]"
        )
        sys.exit(1)
    except httpx.HTTPError as exc:
        console.print(f"[red]Network error fetching {arxiv_id}: {exc}[/red]")
        sys.exit(1)
    except PdfExtractionError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(f"[bold]{paper.title}[/bold]")
    console.print(f"{', '.join(paper.authors)}\n")

    # Fresh timestamped workspace per run: prior reports are never clobbered.
    workdir = new_run_dir(base_dir)
    shutil.copy2(paper.pdf_path, workdir / "paper.pdf")

    console.print("[bold]Starting reproduction agent[/bold] (this can take a while)\n")
    result = run_reproduction(paper, workdir, console)

    usage = result.usage
    cost = (
        f"${result.estimated_cost_usd:.4f}" if result.estimated_cost_usd is not None else "n/a"
    )
    console.print(
        f"\n[dim]{usage.input_tokens} in / {usage.output_tokens} out / "
        f"{usage.cache_read_input_tokens} cache-read tokens · "
        f"estimated cost {cost} · {result.wall_clock_seconds:.0f}s · "
        f"{result.iterations} iterations[/dim]"
    )
    if result.status == "completed":
        console.print(f"\n[green]Done.[/green] Report: [bold]{result.report}[/bold]")
    else:
        console.print(
            f"\n[yellow]Run ended with status: {result.status}.[/yellow] "
            f"Partial report: [bold]{result.report}[/bold]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
