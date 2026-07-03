"""Docker-based sandbox for executing agent-generated code.

Each run gets a long-lived container with the run's workspace mounted at
/workspace, so pip installs and intermediate files persist across tool calls
within a run but never touch the host Python environment.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_IMAGE = "python:3.12-slim"
EXEC_TIMEOUT = 600  # seconds per command


def check_docker() -> str | None:
    """Return None if Docker is usable, else a human-readable problem."""
    if shutil.which("docker") is None:
        return "Docker CLI not found on PATH — install Docker (https://docs.docker.com/get-docker/)."
    probe = subprocess.run(["docker", "info"], capture_output=True)
    if probe.returncode != 0:
        return "Docker daemon is not running — start Docker and try again."
    return None


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str

    def render(self, max_chars: int = 20_000) -> str:
        """Format for return to the model, truncating runaway output."""
        out = f"exit code: {self.exit_code}\n--- stdout ---\n{self.stdout}\n--- stderr ---\n{self.stderr}"
        if len(out) > max_chars:
            out = out[:max_chars] + f"\n... [truncated, {len(out) - max_chars} chars omitted]"
        return out


class DockerSandbox:
    def __init__(self, workdir: Path, image: str = DEFAULT_IMAGE):
        self.workdir = workdir.resolve()
        self.image = image
        self.name = f"arxiv-repro-{uuid.uuid4().hex[:12]}"
        self._started = False

    def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", self.name,
                "--memory", "4g",
                "--cpus", "2",
                "-v", f"{self.workdir}:/workspace",
                "-w", "/workspace",
                self.image,
                "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
        )
        self._started = True

    def exec(self, command: list[str], timeout: int = EXEC_TIMEOUT) -> ExecResult:
        if not self._started:
            raise RuntimeError("Sandbox not started — call start() first")
        try:
            proc = subprocess.run(
                ["docker", "exec", self.name, *command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecResult(proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return ExecResult(-1, "", f"Command timed out after {timeout}s")

    def run_python_file(self, relpath: str, timeout: int = EXEC_TIMEOUT) -> ExecResult:
        return self.exec(["python", relpath], timeout=timeout)

    def pip_install(self, packages: list[str]) -> ExecResult:
        return self.exec(["pip", "install", "--quiet", *packages])

    def stop(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self._started = False

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
