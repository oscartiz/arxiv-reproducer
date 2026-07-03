"""The reproduction agent: a Claude tool-use loop wired to the Docker sandbox."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import anthropic
import httpx
from anthropic import beta_tool
from rich.console import Console

from .paper import Paper
from .prompts import SYSTEM_PROMPT, initial_user_message
from .sandbox import DockerSandbox

MODEL = "claude-opus-4-8"
MAX_TOKENS = 16_000

# Caps on the whole run, so a confused or adversarially-prompted agent cannot
# loop forever burning API spend. Per-command timeouts live in sandbox.py.
MAX_ITERATIONS = 60
MAX_WALL_CLOCK_SECONDS = 3600

# Anthropic SDK built-in retry handles 429/5xx/connection errors with backoff.
API_MAX_RETRIES = 5


@dataclass
class RunResult:
    report: Path
    status: str  # "completed" | "error" | "iteration_cap" | "wall_clock_cap"
    error: str | None = None
    iterations: int = 0


MAX_READ_CHARS = 50_000


def _resolve_in_workspace(workdir: Path, path: str) -> Path | None:
    """Resolve an agent-supplied path, returning None unless it stays inside the
    workspace after following symlinks. Paths are UNTRUSTED model output."""
    try:
        target = (workdir / path).resolve()
    except (ValueError, OSError):
        # e.g. embedded null bytes, paths too long for the OS
        return None
    if not target.is_relative_to(workdir):
        return None
    return target


def build_tools(sandbox: DockerSandbox):
    """Create the agent's tools, bound to one sandbox instance."""

    @beta_tool
    def write_file(path: str, content: str) -> str:
        """Write a file into the workspace (overwrites if it exists).

        Args:
            path: Path relative to the workspace root, e.g. "simulate.py".
            content: Full file content.
        """
        target = _resolve_in_workspace(sandbox.workdir, path)
        if target is None:
            return "Error: path escapes the workspace"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except OSError as exc:
            return f"Error: could not write {path}: {exc}"
        return f"Wrote {len(content)} chars to {path}"

    @beta_tool
    def read_file(path: str) -> str:
        """Read a text file from the workspace.

        Args:
            path: Path relative to the workspace root.
        """
        target = _resolve_in_workspace(sandbox.workdir, path)
        if target is None:
            return "Error: path escapes the workspace"
        if not target.exists():
            return f"Error: {path} does not exist"
        try:
            text = target.read_text()
        except UnicodeDecodeError:
            return f"Error: {path} is not a text file"
        except OSError as exc:
            return f"Error: could not read {path}: {exc}"
        if len(text) > MAX_READ_CHARS:
            omitted = len(text) - MAX_READ_CHARS
            return text[:MAX_READ_CHARS] + f"\n... [truncated, {omitted} chars omitted]"
        return text

    @beta_tool
    def run_python(path: str) -> str:
        """Execute a Python script in the sandbox and return its output.

        Args:
            path: Path of the script relative to the workspace root.
        """
        return sandbox.run_python_file(path).render()

    @beta_tool
    def install_packages(packages: str) -> str:
        """Install Python packages into the sandbox with pip.

        Only plain PyPI package names with optional version pins are accepted
        (pre-built wheels only — no URLs, flags, or source builds). The
        scientific stack (numpy, scipy, matplotlib, pandas, sympy,
        scikit-learn, networkx, pillow) is already installed.

        Args:
            packages: Space-separated package names, e.g. "statsmodels emcee".
        """
        return sandbox.pip_install(packages.split()).render()

    return [write_file, read_file, run_python, install_packages]


def run_reproduction(paper: Paper, workdir: Path, console: Console) -> RunResult:
    """Run the full agent loop against a paper.

    Always leaves the workspace in a coherent state: REPORT.md exists when
    this returns, whatever happened mid-run, and the container is torn down.
    """
    client = anthropic.Anthropic(max_retries=API_MAX_RETRIES)
    status = "completed"
    error: str | None = None
    iterations = 0

    with DockerSandbox(workdir) as sandbox:
        runner = client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=build_tools(sandbox),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": initial_user_message(
                                paper.title, paper.arxiv_id, paper.full_text
                            ),
                            # The paper text is resent on every loop iteration;
                            # caching it cuts cost and latency substantially.
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        )

        started = time.monotonic()
        try:
            for message in runner:
                iterations += 1
                for block in message.content:
                    if block.type == "text":
                        console.print(block.text)
                    elif block.type == "tool_use":
                        console.print(f"[dim]→ {block.name}[/dim]")
                if iterations >= MAX_ITERATIONS:
                    status = "iteration_cap"
                    console.print(
                        f"[yellow]Stopping: iteration cap reached ({MAX_ITERATIONS}).[/yellow]"
                    )
                    break
                if time.monotonic() - started > MAX_WALL_CLOCK_SECONDS:
                    status = "wall_clock_cap"
                    console.print(
                        f"[yellow]Stopping: wall-clock cap reached "
                        f"({MAX_WALL_CLOCK_SECONDS}s).[/yellow]"
                    )
                    break
        except (anthropic.APIError, httpx.HTTPError) as exc:
            # SDK retries are exhausted by the time we get here.
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            console.print(f"[red]Agent loop aborted: {error}[/red]")

    report = workdir / "REPORT.md"
    if not report.exists():
        lines = [
            "# Reproduction report missing",
            "",
            f"The run ended with status `{status}` before the agent wrote REPORT.md.",
        ]
        if error:
            lines += ["", f"Error: {error}"]
        report.write_text("\n".join(lines) + "\n")
    return RunResult(report=report, status=status, error=error, iterations=iterations)
