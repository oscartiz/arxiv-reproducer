from pathlib import Path

import httpx
import pytest

from arxiv_reproducer import cli as cli_mod
from arxiv_reproducer.cli import has_anthropic_credentials, main
from arxiv_reproducer.paper import Paper


@pytest.fixture
def ready_environment(monkeypatch, tmp_path):
    """Docker up, credentials present — the default happy baseline."""
    monkeypatch.setattr(cli_mod, "check_docker", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return tmp_path


def fake_paper(tmp_path):
    return Paper(
        arxiv_id="2301.12345",
        title="A Test Paper",
        abstract="An abstract.",
        authors=["A. Author", "B. Author"],
        full_text="Full text.",
        pdf_path=tmp_path / "2301.12345.pdf",
    )


class TestArgumentAndPreflightErrors:
    def test_no_arguments_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2
        assert "usage" in capsys.readouterr().err.lower()

    def test_unparseable_id_exits_with_clear_message(self, ready_environment, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["not a paper at all"])
        assert excinfo.value.code == 2
        assert "Could not parse an arXiv ID" in capsys.readouterr().out

    def test_docker_unavailable_exits_before_any_network_call(self, monkeypatch, capsys):
        monkeypatch.setattr(cli_mod, "check_docker", lambda: "Docker daemon is not running")
        monkeypatch.setattr(
            cli_mod, "fetch_paper", lambda *a, **kw: pytest.fail("should not fetch")
        )
        with pytest.raises(SystemExit) as excinfo:
            main(["2301.12345"])
        assert excinfo.value.code == 1
        assert "Docker daemon" in capsys.readouterr().out

    def test_missing_credentials_exits_with_guidance(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli_mod, "check_docker", lambda: None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "empty"))
        with pytest.raises(SystemExit) as excinfo:
            main(["2301.12345"])
        assert excinfo.value.code == 1
        assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


class TestFetchFailures:
    def test_http_status_error_is_reported(self, ready_environment, monkeypatch, capsys):
        def raise_status(*a, **kw):
            request = httpx.Request("GET", "https://export.arxiv.org")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

        monkeypatch.setattr(cli_mod, "fetch_paper", raise_status)
        with pytest.raises(SystemExit) as excinfo:
            main(["2301.12345"])
        assert excinfo.value.code == 1
        assert "HTTP 503" in capsys.readouterr().out

    def test_network_error_is_reported(self, ready_environment, monkeypatch, capsys):
        def raise_connect(*a, **kw):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(cli_mod, "fetch_paper", raise_connect)
        with pytest.raises(SystemExit) as excinfo:
            main(["2301.12345"])
        assert excinfo.value.code == 1
        assert "Network error" in capsys.readouterr().out

    def test_unknown_id_value_error_is_reported(self, ready_environment, monkeypatch, capsys):
        def raise_value(*a, **kw):
            raise ValueError("arXiv API returned no entry for 2301.12345")

        monkeypatch.setattr(cli_mod, "fetch_paper", raise_value)
        with pytest.raises(SystemExit) as excinfo:
            main(["2301.12345"])
        assert excinfo.value.code == 1
        assert "no entry" in capsys.readouterr().out


class TestHappyPath:
    def test_end_to_end_wiring(self, ready_environment, monkeypatch, tmp_path, capsys):
        runs_dir = tmp_path / "runs"
        seen = {}

        def fake_fetch(arxiv_id, workdir):
            seen["fetch"] = (arxiv_id, workdir)
            return fake_paper(tmp_path)

        def fake_run(paper, workdir, console):
            seen["run"] = (paper.arxiv_id, workdir)
            report = workdir / "REPORT.md"
            workdir.mkdir(parents=True, exist_ok=True)
            report.write_text("# ok")
            return report

        monkeypatch.setattr(cli_mod, "fetch_paper", fake_fetch)
        monkeypatch.setattr(cli_mod, "run_reproduction", fake_run)

        main(["https://arxiv.org/abs/2301.12345", "--runs-dir", str(runs_dir)])

        assert seen["fetch"][0] == "2301.12345"
        assert seen["fetch"][1] == runs_dir / "2301.12345"
        assert seen["run"] == ("2301.12345", runs_dir / "2301.12345")
        out = capsys.readouterr().out
        assert "A Test Paper" in out
        assert "Done." in out

    def test_old_style_id_maps_to_safe_directory_name(
        self, ready_environment, monkeypatch, tmp_path
    ):
        seen = {}

        def fake_fetch(arxiv_id, workdir):
            seen["w"] = workdir
            return fake_paper(tmp_path)

        monkeypatch.setattr(cli_mod, "fetch_paper", fake_fetch)
        monkeypatch.setattr(
            cli_mod, "run_reproduction", lambda p, w, c: w / "REPORT.md"
        )
        main(["hep-th/9901001", "--runs-dir", str(tmp_path / "runs")])
        assert seen["w"].name == "hep-th_9901001"  # no path separator in dir name


class TestCredentialDetection:
    def test_env_api_key_counts(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
        assert has_anthropic_credentials()

    def test_auth_token_counts(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oat-x")
        assert has_anthropic_credentials()

    def test_oauth_profile_on_disk_counts(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        config_dir = tmp_path / "anthropic"
        (config_dir / "credentials").mkdir(parents=True)
        (config_dir / "credentials" / "default.json").write_text("{}")
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(config_dir))
        assert has_anthropic_credentials()

    def test_nothing_available(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "missing"))
        assert not has_anthropic_credentials()

    def test_config_dir_default_is_under_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_CONFIG_DIR", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert not has_anthropic_credentials()
