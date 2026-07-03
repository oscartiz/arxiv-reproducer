import pytest

from arxiv_reproducer.config import set_config


@pytest.fixture(autouse=True)
def fresh_config():
    """Each test sees a freshly-loaded config (env changes take effect)."""
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
