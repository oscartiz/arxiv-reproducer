"""The reproduction agent: a Claude tool-use loop wired to the Docker sandbox."""

from __future__ import annotations

from pathlib import Path

import anthropic
from anthropic import beta_tool
from rich.console import Console

from .paper import Paper
from .prompts import SYSTEM_PROMPT, initial_user_message
from .sandbox import DockerSandbox

MODEL = "claude-opus-4-8"
MAX_TOKENS = 16_000


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

        Args:
            packages: Space-separated package names, e.g. "numpy scipy matplotlib".
        """
        return sandbox.pip_install(packages.split()).render()

    return [write_file, read_file, run_python, install_packages]


def run_reproduction(paper: Paper, workdir: Path, console: Console) -> Path:
    """Run the full agent loop against a paper. Returns the path to REPORT.md."""
    client = anthropic.Anthropic()

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

        for message in runner:
            for block in message.content:
                if block.type == "text":
                    console.print(block.text)
                elif block.type == "tool_use":
                    console.print(f"[dim]→ {block.name}[/dim]")

    report = workdir / "REPORT.md"
    if not report.exists():
        report.write_text(
            "# Reproduction report missing\n\nThe agent finished without writing REPORT.md.\n"
        )
    return report
