"""Runtime configuration: defaults < TOML config file < environment variables.

Every tunable lives here — model name, token/iteration/wall-clock caps,
container resource limits, timeouts, image tag. Nothing security-relevant is
configurable (the hardening flags in sandbox.py are not tunables).

Sources, later wins:
1. Built-in defaults (the dataclass field defaults below).
2. A TOML file: ``./arxiv-repro.toml`` if present, or the path in
   ``ARXIV_REPRO_CONFIG``. Keys are flat and match the field names.
3. ``ARXIV_REPRO_<FIELD>`` environment variables, e.g.
   ``ARXIV_REPRO_MODEL=claude-sonnet-5``.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

ENV_PREFIX = "ARXIV_REPRO_"
DEFAULT_CONFIG_FILENAME = "arxiv-repro.toml"


class ConfigError(ValueError):
    """Bad configuration — the message says which key and why."""


@dataclass(frozen=True)
class Config:
    # Agent
    model: str = "claude-opus-4-8"
    max_tokens: int = 16_000
    max_iterations: int = 60
    max_wall_clock_seconds: int = 3600
    api_max_retries: int = 5
    # Sandbox
    sandbox_image: str = "arxiv-repro-sandbox:latest"
    exec_timeout: int = 600
    install_timeout: int = 300
    memory_limit: str = "4g"
    cpu_limit: str = "2"
    pids_limit: int = 256
    tmp_size: str = "512m"


_INT_FIELDS = {
    "max_tokens",
    "max_iterations",
    "max_wall_clock_seconds",
    "api_max_retries",
    "exec_timeout",
    "install_timeout",
    "pids_limit",
}
_SIZE_RE = re.compile(r"^\d+[bkmg]$", re.IGNORECASE)
_CPU_RE = re.compile(r"^\d+(\.\d+)?$")


def _coerce(name: str, value: object, source: str) -> object:
    if name in _INT_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ConfigError(f"{source}: {name} must be an integer, got {value!r}")
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"{source}: {name} must be an integer, got {value!r}") from None
    if not isinstance(value, (str, int, float)):
        raise ConfigError(f"{source}: {name} must be a string, got {value!r}")
    return str(value)


def _validate(cfg: Config) -> None:
    problems = []
    if not cfg.model.strip():
        problems.append("model must be non-empty")
    for name in sorted(_INT_FIELDS):
        if getattr(cfg, name) <= 0:
            problems.append(f"{name} must be positive")
    if not _SIZE_RE.match(cfg.memory_limit):
        problems.append(f"memory_limit must look like '4g' or '512m', got {cfg.memory_limit!r}")
    if not _SIZE_RE.match(cfg.tmp_size):
        problems.append(f"tmp_size must look like '512m', got {cfg.tmp_size!r}")
    if not _CPU_RE.match(cfg.cpu_limit):
        problems.append(f"cpu_limit must be a number like '2' or '1.5', got {cfg.cpu_limit!r}")
    if problems:
        raise ConfigError("invalid configuration: " + "; ".join(problems))


def load_config(
    env: Mapping[str, str] | None = None, config_file: str | Path | None = None
) -> Config:
    environ: Mapping[str, str] = os.environ if env is None else env
    field_names = {f.name for f in fields(Config)}
    values: dict[str, object] = {}

    path: str | Path | None = config_file or environ.get(ENV_PREFIX + "CONFIG")
    if path is None and Path(DEFAULT_CONFIG_FILENAME).is_file():
        path = DEFAULT_CONFIG_FILENAME
    if path is not None:
        toml_path = Path(path)
        if not toml_path.is_file():
            raise ConfigError(f"config file not found: {toml_path}")
        try:
            data = tomllib.loads(toml_path.read_text())
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {toml_path}: {exc}") from exc
        unknown = sorted(set(data) - field_names)
        if unknown:
            raise ConfigError(
                f"unknown config keys in {toml_path}: {', '.join(unknown)} "
                f"(valid keys: {', '.join(sorted(field_names))})"
            )
        for name, value in data.items():
            values[name] = _coerce(name, value, str(toml_path))

    for name in sorted(field_names):
        env_key = ENV_PREFIX + name.upper()
        if env_key in environ:
            values[name] = _coerce(name, environ[env_key], env_key)

    cfg = Config(**values)  # type: ignore[arg-type]
    _validate(cfg)
    return cfg


_current: Config | None = None


def get_config() -> Config:
    """The process-wide config, loaded lazily on first use."""
    global _current
    if _current is None:
        _current = load_config()
    return _current


def set_config(cfg: Config | None) -> None:
    """Override (or reset with None) the process-wide config. For tests/tools."""
    global _current
    _current = cfg
