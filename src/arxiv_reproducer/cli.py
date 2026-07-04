"""Command-line entry point: arxiv-repro <arxiv-id-or-url> [more-ids...]."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table

from .agent import run_reproduction
from .batch import BatchRow, read_id_file, summarize, write_summary_csv, write_summary_md
from .config import ConfigError, get_config
from .figures import extract_paper_figures
from .logs import setup_logging
from .paper import Paper, PdfExtractionError, fetch_paper, parse_arxiv_id
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


class _PreparationError(Exception):
    """A paper could not be readied for a run; message is user-facing."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def _prepare(paper_arg: str, runs_dir: Path, console: Console) -> tuple[Paper, Path]:
    """Parse the ID, fetch the paper, and create a fresh run workspace."""
    try:
        arxiv_id = parse_arxiv_id(paper_arg)
    except ValueError as exc:
        raise _PreparationError(str(exc), exit_code=2) from exc

    base_dir = runs_dir / safe_dir_name(arxiv_id)
    console.print(f"[bold]Fetching[/bold] arXiv:{arxiv_id} ...")
    try:
        paper = fetch_paper(arxiv_id, base_dir)
    except httpx.HTTPStatusError as exc:
        raise _PreparationError(
            f"arXiv returned HTTP {exc.response.status_code} for {arxiv_id} — "
            "check that the ID exists."
        ) from exc
    except httpx.HTTPError as exc:
        raise _PreparationError(f"Network error fetching {arxiv_id}: {exc}") from exc
    except (PdfExtractionError, ValueError) as exc:
        raise _PreparationError(str(exc)) from exc

    console.print(f"[bold]{paper.title}[/bold]")
    console.print(f"{', '.join(paper.authors)}\n")

    # Fresh timestamped workspace per run: prior reports are never clobbered.
    workdir = new_run_dir(base_dir)
    shutil.copy2(paper.pdf_path, workdir / "paper.pdf")
    figures = extract_paper_figures(workdir / "paper.pdf", workdir / "paper-figures")
    if figures:
        console.print(
            f"[dim]Extracted {len(figures)} original figure(s) from the PDF "
            f"→ paper-figures/[/dim]"
        )
    return paper, workdir


def _run_single(paper_arg: str, runs_dir: Path, console: Console) -> None:
    try:
        paper, workdir = _prepare(paper_arg, runs_dir, console)
    except _PreparationError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(exc.exit_code)

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


def _run_batch(ids: list[str], runs_dir: Path, console: Console) -> None:
    """Run every paper, recording failures as rows; exit 1 only if none completed."""
    rows: list[BatchRow] = []
    for index, paper_arg in enumerate(ids, start=1):
        console.print(f"\n[bold]=== [{index}/{len(ids)}] {paper_arg} ===[/bold]")
        try:
            paper, workdir = _prepare(paper_arg, runs_dir, console)
        except _PreparationError as exc:
            console.print(f"[red]{exc}[/red] — continuing with the next paper")
            rows.append(BatchRow(arxiv_id=paper_arg, status="fetch_error", error=str(exc)))
            continue
        console.print("[bold]Starting reproduction agent[/bold]\n")
        result = run_reproduction(paper, workdir, console)
        manifest_path = workdir / "run.json"
        if manifest_path.exists():
            rows.append(BatchRow.from_manifest(manifest_path))
        else:
            rows.append(
                BatchRow(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    status=result.status,
                    error=result.error,
                    report=str(result.report),
                )
            )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = runs_dir / f"batch-{stamp}.csv"
    md_path = runs_dir / f"batch-{stamp}.md"
    write_summary_csv(rows, csv_path)
    write_summary_md(rows, md_path)

    table = Table(title="Batch summary")
    for column in ("arXiv ID", "Verdict", "Confidence", "Status", "Cost"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row.arxiv_id,
            row.verdict or "—",
            f"{row.confidence}%" if row.confidence is not None else "—",
            row.status,
            f"${row.estimated_cost_usd:.2f}" if row.estimated_cost_usd is not None else "—",
        )
    console.print()
    console.print(table)
    console.print(f"[dim]{summarize(rows)}[/dim]")
    console.print(f"Summary: [bold]{csv_path}[/bold] · [bold]{md_path}[/bold]")

    if not any(row.status == "completed" for row in rows):
        console.print("[red]No paper in the batch completed a run.[/red]")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="arxiv-repro",
        description="Attempt to reproduce computational results from arXiv papers.",
    )
    parser.add_argument(
        "papers",
        nargs="*",
        metavar="paper",
        help="arXiv ID or URL, e.g. 2301.12345 — several IDs switch on batch mode",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        metavar="FILE",
        help="Read additional arXiv IDs from FILE (one per line, # comments allowed)",
    )
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

    ids: list[str] = list(args.papers)
    if args.batch is not None:
        try:
            ids += read_id_file(args.batch)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(2)
    ids = list(dict.fromkeys(ids))  # de-duplicate, preserving order
    if not ids:
        parser.error("provide at least one arXiv ID, or --batch FILE")

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

    if len(ids) == 1:
        _run_single(ids[0], args.runs_dir, console)
    else:
        _run_batch(ids, args.runs_dir, console)


if __name__ == "__main__":
    main()
