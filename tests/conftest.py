import os

import pytest

from arxiv_reproducer import config as config_module
from arxiv_reproducer.config import set_config


@pytest.fixture(autouse=True)
def fresh_config(monkeypatch, tmp_path):
    """Each test sees a freshly-loaded config (env changes take effect),
    isolated from the developer's ./arxiv-repro.toml and ARXIV_REPRO_* env vars."""
    for key in list(os.environ):
        if key.startswith(config_module.ENV_PREFIX):
            monkeypatch.delenv(key)
    monkeypatch.setattr(
        config_module, "DEFAULT_CONFIG_FILENAME", str(tmp_path / "absent-config.toml")
    )
    set_config(None)
    yield
    set_config(None)


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker",
        action="store_true",
        default=False,
        help="run integration tests that need a real Docker daemon",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-docker"):
        return
    skip_docker = pytest.mark.skip(reason="needs a real Docker daemon; pass --run-docker")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)
