from pathlib import Path

import httpx
import pytest

from arxiv_reproducer import cli as cli_mod
from arxiv_reproducer.agent import RunResult
from arxiv_reproducer.cli import has_anthropic_credentials, main
from arxiv_reproducer.paper import Paper, PdfExtractionError


@pytest.fixture
def ready_environment(monkeypatch, tmp_path):
    """Docker up, image built, credentials present — the happy baseline."""
    monkeypatch.setattr(cli_mod, "check_docker", lambda: None)
    monkeypatch.setattr(cli_mod, "image_exists", lambda image=None: True)
    monkeypatch.setattr(cli_mod, "ensure_image", lambda image=None: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return tmp_path


def fake_paper(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(b"%PDF-fake")
    return Paper(
        arxiv_id="2301.12345",
        title="A Test Paper",
        abstract="An abstract.",
        authors=["A. Author", "B. Author"],
        full_text="Full text.",
        pdf_path=pdf_path,
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


class TestImagePreflight:
    def test_build_failure_is_reported_cleanly(self, monkeypatch, capsys):
        import subprocess

        monkeypatch.setattr(cli_mod, "check_docker", lambda: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setattr(cli_mod, "image_exists", lambda image=None: False)

        def boom(image=None):
            raise subprocess.CalledProcessError(1, ["docker", "build"], stderr=b"build exploded")

        monkeypatch.setattr(cli_mod, "ensure_image", boom)
        with pytest.raises(SystemExit) as excinfo:
            main(["2301.12345"])
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "Building sandbox image" in out  # first-run notice
        assert "build exploded" in out


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

    def test_scanned_pdf_is_reported_clearly(self, ready_environment, monkeypatch, capsys):
        def raise_extraction(*a, **kw):
            raise PdfExtractionError("this looks like a scanned/image PDF")

        monkeypatch.setattr(cli_mod, "fetch_paper", raise_extraction)
        with pytest.raises(SystemExit) as excinfo:
            main(["2301.12345"])
        assert excinfo.value.code == 1
        assert "scanned" in capsys.readouterr().out


class TestHappyPath:
    def test_end_to_end_wiring(self, ready_environment, monkeypatch, tmp_path, capsys):
        runs_dir = tmp_path / "runs"
        seen = {}

        def fake_fetch(arxiv_id, base_dir):
            seen["fetch"] = (arxiv_id, base_dir)
            return fake_paper(tmp_path)

        def fake_run(paper, workdir, console):
            seen["run_workdir"] = workdir
            report = workdir / "REPORT.md"
            report.write_text("# ok")
            return RunResult(report=report, status="completed")

        monkeypatch.setattr(cli_mod, "fetch_paper", fake_fetch)
        monkeypatch.setattr(cli_mod, "run_reproduction", fake_run)

        main(["https://arxiv.org/abs/2301.12345", "--runs-dir", str(runs_dir)])

        base_dir = runs_dir / "2301.12345"
        assert seen["fetch"] == ("2301.12345", base_dir)
        # The agent works in a fresh timestamped dir under the paper's base dir,
        # with the cached PDF copied in for auditability.
        workdir = seen["run_workdir"]
        assert workdir.parent == base_dir
        assert workdir.name != ""
        assert (workdir / "paper.pdf").exists()
        out = capsys.readouterr().out
        assert "A Test Paper" in out
        assert "Done." in out

    def test_rerun_gets_fresh_workspace_and_keeps_old_report(
        self, ready_environment, monkeypatch, tmp_path
    ):
        runs_dir = tmp_path / "runs"
        workdirs = []

        def fake_run(paper, workdir, console):
            workdirs.append(workdir)
            report = workdir / "REPORT.md"
            report.write_text(f"# run {len(workdirs)}")
            return RunResult(report=report, status="completed")

        monkeypatch.setattr(cli_mod, "fetch_paper", lambda i, b: fake_paper(tmp_path))
        monkeypatch.setattr(cli_mod, "run_reproduction", fake_run)

        main(["2301.12345", "--runs-dir", str(runs_dir)])
        main(["2301.12345", "--runs-dir", str(runs_dir)])

        assert len(workdirs) == 2
        assert workdirs[0] != workdirs[1]
        assert (workdirs[0] / "REPORT.md").read_text() == "# run 1"  # not clobbered

    def test_failed_run_exits_nonzero_but_names_partial_report(
        self, ready_environment, monkeypatch, tmp_path, capsys
    ):
        def fake_run(paper, workdir, console):
            report = workdir / "REPORT.md"
            report.write_text("# aborted")
            return RunResult(report=report, status="error", error="api died")

        monkeypatch.setattr(cli_mod, "fetch_paper", lambda i, b: fake_paper(tmp_path))
        monkeypatch.setattr(cli_mod, "run_reproduction", fake_run)

        with pytest.raises(SystemExit) as excinfo:
            main(["2301.12345", "--runs-dir", str(tmp_path / "runs")])
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "error" in out
        assert "REPORT.md" in out

    def test_old_style_id_maps_to_safe_directory_name(
        self, ready_environment, monkeypatch, tmp_path
    ):
        seen = {}

        def fake_fetch(arxiv_id, base_dir):
            seen["base"] = base_dir
            return fake_paper(tmp_path)

        def fake_run(paper, workdir, console):
            (workdir / "REPORT.md").write_text("# ok")
            return RunResult(report=workdir / "REPORT.md", status="completed")

        monkeypatch.setattr(cli_mod, "fetch_paper", fake_fetch)
        monkeypatch.setattr(cli_mod, "run_reproduction", fake_run)
        main(["hep-th/9901001", "--runs-dir", str(tmp_path / "runs")])
        assert seen["base"].name == "hep-th_9901001"  # no path separator in dir name


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
