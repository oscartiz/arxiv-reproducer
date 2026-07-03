"""Docker-based sandbox for executing agent-generated code.

The sandbox is the trust boundary of this tool: papers are untrusted input,
and the code the agent writes under their influence is untrusted output. All
execution happens in a long-lived container with:

- no network (``--network none``) — the scientific stack is pre-baked into
  the image at build time, so run-time code cannot exfiltrate or phone home;
- a read-only root filesystem, writable only at the bind-mounted /workspace
  (kept on the host for auditability) and a size-capped /tmp tmpfs;
- all capabilities dropped, no-new-privileges, a non-root user, and memory /
  CPU / pid limits.

Package installs run in a separate ephemeral container that does have
network but only ever executes a validated ``pip install --only-binary``
into /workspace/.deps — agent code never runs with network access.
"""

from __future__ import annotations

import atexit
import re
import shutil
import signal
import subprocess
import uuid
from types import FrameType
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .config import get_config
from .logs import get_logger

logger = get_logger("sandbox")

DEPS_DIR = ".deps"  # inside /workspace; importable via PYTHONPATH

# PEP 508-ish subset: name, optional extras, optional version pin. No flags,
# URLs, paths, or spaces — anything else is refused before docker is invoked.
_PACKAGE_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:\[[A-Za-z0-9,._-]+\])?"
    r"(?:(?:==|>=|<=|~=|!=)[A-Za-z0-9.*+!_-]+)?$"
)

# Containers that exist right now; emptied by stop()/atexit/SIGTERM so a
# crash or kill never leaks a running container.
_ACTIVE_CONTAINERS: set[str] = set()
_HOOKS_INSTALLED = False


def check_docker() -> str | None:
    """Return None if Docker is usable, else a human-readable problem."""
    if shutil.which("docker") is None:
        return "Docker CLI not found on PATH — install Docker (https://docs.docker.com/get-docker/)."
    probe = subprocess.run(["docker", "info"], capture_output=True)
    if probe.returncode != 0:
        return "Docker daemon is not running — start Docker and try again."
    return None


def sandbox_dockerfile() -> str:
    """The Dockerfile for the pre-baked sandbox image (shipped as package data)."""
    return resources.files("arxiv_reproducer").joinpath("docker/sandbox.Dockerfile").read_text()


def image_exists(image: str | None = None) -> bool:
    image = image or get_config().sandbox_image
    probe = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    return probe.returncode == 0


def ensure_image(image: str | None = None) -> None:
    """Build the sandbox image from the packaged Dockerfile if it is missing."""
    image = image or get_config().sandbox_image
    if image_exists(image):
        return
    logger.info("building sandbox image %s", image)
    subprocess.run(
        ["docker", "build", "-t", image, "-"],
        input=sandbox_dockerfile().encode(),
        check=True,
        capture_output=True,
    )


def validate_packages(packages: list[str]) -> str | None:
    """Return None if every entry is a plain package spec, else the problem."""
    if not packages:
        return "no packages given"
    for pkg in packages:
        if not _PACKAGE_RE.match(pkg):
            return f"invalid package spec: {pkg!r} (plain PyPI names with optional == pins only)"
    return None


def _remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _cleanup_all_containers() -> None:
    for name in list(_ACTIVE_CONTAINERS):
        logger.warning("emergency cleanup of container %s", name)
        _remove_container(name)
        _ACTIVE_CONTAINERS.discard(name)


def _install_cleanup_hooks() -> None:
    """Ensure containers die with the process: atexit + SIGTERM.

    SIGINT needs no handler — KeyboardInterrupt unwinds the context manager.
    """
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True
    atexit.register(_cleanup_all_containers)

    previous = signal.getsignal(signal.SIGTERM)

    def on_sigterm(signum: int, frame: FrameType | None) -> None:
        _cleanup_all_containers()
        if callable(previous):
            previous(signum, frame)
        else:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.raise_signal(signal.SIGTERM)

    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except ValueError:
        pass  # not in the main thread; atexit still covers us


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


@dataclass
class DockerSandbox:
    workdir: Path
    image: str = ""  # empty → config's sandbox_image
    name: str = field(default_factory=lambda: f"arxiv-repro-{uuid.uuid4().hex[:12]}")
    installed_packages: list[str] = field(default_factory=list)
    _started: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir).resolve()
        if not self.image:
            self.image = get_config().sandbox_image

    def _hardening_flags(self) -> list[str]:
        """Flags shared by the exec container and the ephemeral installer."""
        cfg = get_config()
        return [
            "--memory", cfg.memory_limit,
            "--memory-swap", cfg.memory_limit,  # equal to --memory: no swap
            "--cpus", cfg.cpu_limit,
            "--pids-limit", str(cfg.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "1000:1000",
            "--read-only",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={cfg.tmp_size}",
            "-e", "HOME=/tmp",
            "-e", "MPLBACKEND=Agg",
            "-e", "MPLCONFIGDIR=/tmp/mpl",
            "-e", f"PYTHONPATH=/workspace/{DEPS_DIR}",
            "-v", f"{self.workdir}:/workspace",
            "-w", "/workspace",
        ]

    def start(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        ensure_image(self.image)
        _install_cleanup_hooks()
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", self.name,
                "--network", "none",
                *self._hardening_flags(),
                self.image,
                "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
        )
        self._started = True
        _ACTIVE_CONTAINERS.add(self.name)
        logger.info("container %s started (image=%s, workdir=%s)", self.name, self.image, self.workdir)

    def exec(self, command: list[str], timeout: int | None = None) -> ExecResult:
        if not self._started:
            raise RuntimeError("Sandbox not started — call start() first")
        timeout = timeout if timeout is not None else get_config().exec_timeout
        logger.debug("exec in %s: %s", self.name, " ".join(command))
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

    def run_python_file(self, relpath: str, timeout: int | None = None) -> ExecResult:
        return self.exec(["python", relpath], timeout=timeout)

    def pip_install(self, packages: list[str]) -> ExecResult:
        """Install wheels into /workspace/.deps via an ephemeral networked container.

        The exec container never gets network; this one gets network but only
        ever runs a validated pip command. --only-binary means no sdist
        setup.py ever executes with network access.
        """
        problem = validate_packages(packages)
        if problem is not None:
            logger.warning("refused pip install: %s", problem)
            return ExecResult(2, "", f"Refused: {problem}")
        logger.info("pip install (networked installer): %s", " ".join(packages))
        install_timeout = get_config().install_timeout
        try:
            proc = subprocess.run(
                [
                    "docker", "run", "--rm",
                    *self._hardening_flags(),
                    self.image,
                    "pip", "install",
                    "--quiet",
                    "--no-cache-dir",
                    "--only-binary=:all:",
                    "--target", f"/workspace/{DEPS_DIR}",
                    *packages,
                ],
                capture_output=True,
                text=True,
                timeout=install_timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(-1, "", f"pip install timed out after {install_timeout}s")
        if proc.returncode == 0:
            self.installed_packages.extend(packages)
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)

    def stop(self) -> None:
        if self._started:
            logger.info("container %s stopped", self.name)
            _remove_container(self.name)
            _ACTIVE_CONTAINERS.discard(self.name)
            self._started = False

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
