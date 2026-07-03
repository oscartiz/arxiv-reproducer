import subprocess

import pytest

from arxiv_reproducer import sandbox as sandbox_mod
from arxiv_reproducer.sandbox import DockerSandbox, ExecResult, check_docker


@pytest.fixture
def fake_docker(monkeypatch):
    """Replace subprocess.run inside sandbox.py with a recorder.

    Returns the list of recorded (argv, kwargs) calls. Individual tests can
    swap the behavior via the `.responder` attribute.
    """

    class Recorder:
        def __init__(self):
            self.calls = []
            self.responder = lambda cmd, kw: subprocess.CompletedProcess(cmd, 0, "", "")

        def __call__(self, cmd, **kwargs):
            self.calls.append((cmd, kwargs))
            return self.responder(cmd, kwargs)

    recorder = Recorder()
    monkeypatch.setattr(sandbox_mod.subprocess, "run", recorder)
    return recorder


def docker_run_argv(recorder):
    return next(cmd for cmd, _ in recorder.calls if cmd[:2] == ["docker", "run"])


class TestLifecycle:
    def test_start_creates_workdir_and_container(self, tmp_path, fake_docker):
        workdir = tmp_path / "ws"
        sb = DockerSandbox(workdir)
        sb.start()

        assert workdir.is_dir()
        argv = docker_run_argv(fake_docker)
        # Workspace is bind-mounted and used as the working directory.
        assert f"{workdir.resolve()}:/workspace" in argv
        assert argv[argv.index("-w") + 1] == "/workspace"
        assert sb.name in argv
        assert sb.image in argv
        # Resource limits are applied.
        assert "--memory" in argv
        assert "--cpus" in argv

    def test_container_names_are_unique(self, tmp_path):
        names = {DockerSandbox(tmp_path).name for _ in range(20)}
        assert len(names) == 20

    def test_exec_before_start_raises(self, tmp_path):
        sb = DockerSandbox(tmp_path)
        with pytest.raises(RuntimeError, match="not started"):
            sb.exec(["echo", "hi"])

    def test_stop_removes_container(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()
        sb.stop()
        rm_calls = [cmd for cmd, _ in fake_docker.calls if cmd[:2] == ["docker", "rm"]]
        assert rm_calls == [["docker", "rm", "-f", sb.name]]

    def test_stop_is_idempotent(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()
        sb.stop()
        sb.stop()
        rm_calls = [cmd for cmd, _ in fake_docker.calls if cmd[:2] == ["docker", "rm"]]
        assert len(rm_calls) == 1

    def test_context_manager_stops_on_exception(self, tmp_path, fake_docker):
        with pytest.raises(RuntimeError, match="boom"):
            with DockerSandbox(tmp_path) as sb:
                raise RuntimeError("boom")
        rm_calls = [cmd for cmd, _ in fake_docker.calls if cmd[:2] == ["docker", "rm"]]
        assert rm_calls == [["docker", "rm", "-f", sb.name]]

    def test_failed_start_does_not_mark_started(self, tmp_path, fake_docker):
        def fail(cmd, kw):
            raise subprocess.CalledProcessError(125, cmd, stderr=b"daemon down")

        fake_docker.responder = fail
        sb = DockerSandbox(tmp_path)
        with pytest.raises(subprocess.CalledProcessError):
            sb.start()
        # No container was created, so stop() must not try to remove one.
        fake_docker.responder = lambda cmd, kw: subprocess.CompletedProcess(cmd, 0, "", "")
        sb.stop()
        assert not any(cmd[:2] == ["docker", "rm"] for cmd, _ in fake_docker.calls)


class TestExec:
    def test_exec_returns_output(self, tmp_path, fake_docker):
        fake_docker.responder = lambda cmd, kw: subprocess.CompletedProcess(cmd, 3, "out", "err")
        sb = DockerSandbox(tmp_path)
        sb.start()
        result = sb.exec(["python", "-V"])
        assert (result.exit_code, result.stdout, result.stderr) == (3, "out", "err")
        exec_argv = fake_docker.calls[-1][0]
        assert exec_argv == ["docker", "exec", sb.name, "python", "-V"]

    def test_exec_timeout_is_reported_not_raised(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()

        def time_out(cmd, kw):
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])

        fake_docker.responder = time_out
        result = sb.exec(["python", "spin.py"], timeout=7)
        assert result.exit_code == -1
        assert "timed out after 7s" in result.stderr

    def test_run_python_file_and_pip_install_dispatch(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()
        sb.run_python_file("sim.py")
        assert fake_docker.calls[-1][0][-2:] == ["python", "sim.py"]
        sb.pip_install(["numpy", "scipy"])
        assert fake_docker.calls[-1][0][-5:] == ["pip", "install", "--quiet", "numpy", "scipy"]


class TestExecResult:
    def test_render_includes_streams_and_exit_code(self):
        rendered = ExecResult(2, "hello", "oops").render()
        assert "exit code: 2" in rendered
        assert "hello" in rendered
        assert "oops" in rendered

    def test_render_truncates_runaway_output(self):
        rendered = ExecResult(0, "x" * 100_000, "").render(max_chars=500)
        assert len(rendered) < 700
        assert "truncated" in rendered

    def test_render_short_output_not_truncated(self):
        rendered = ExecResult(0, "short", "").render()
        assert "truncated" not in rendered


class TestCheckDocker:
    def test_no_cli_on_path(self, monkeypatch):
        monkeypatch.setattr(sandbox_mod.shutil, "which", lambda _: None)
        problem = check_docker()
        assert problem is not None and "not found" in problem

    def test_daemon_not_running(self, monkeypatch):
        monkeypatch.setattr(sandbox_mod.shutil, "which", lambda _: "/usr/local/bin/docker")
        monkeypatch.setattr(
            sandbox_mod.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, b"", b""),
        )
        problem = check_docker()
        assert problem is not None and "daemon" in problem

    def test_docker_usable(self, monkeypatch):
        monkeypatch.setattr(sandbox_mod.shutil, "which", lambda _: "/usr/local/bin/docker")
        monkeypatch.setattr(
            sandbox_mod.subprocess,
            "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""),
        )
        assert check_docker() is None


@pytest.mark.docker
class TestRealDocker:
    """Opt-in integration tests: pytest --run-docker."""

    def test_container_roundtrip_and_teardown(self, tmp_path):
        sb = DockerSandbox(tmp_path)
        with sb:
            (tmp_path / "hello.py").write_text("print('hi from sandbox')")
            result = sb.run_python_file("hello.py")
            assert result.exit_code == 0
            assert "hi from sandbox" in result.stdout

        leftovers = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={sb.name}", "-q"],
            capture_output=True,
            text=True,
        )
        assert leftovers.stdout.strip() == ""
