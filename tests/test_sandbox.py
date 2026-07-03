import os
import subprocess

import pytest

from arxiv_reproducer import sandbox as sandbox_mod
from arxiv_reproducer.sandbox import (
    DEPS_DIR,
    DockerSandbox,
    ExecResult,
    check_docker,
    ensure_image,
    validate_packages,
)


@pytest.fixture(autouse=True)
def clean_registry():
    sandbox_mod._ACTIVE_CONTAINERS.clear()
    yield
    sandbox_mod._ACTIVE_CONTAINERS.clear()


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

    def test_start_applies_isolation_hardening(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()
        argv = docker_run_argv(fake_docker)

        # The exec container must never have network access.
        assert argv[argv.index("--network") + 1] == "none"
        # Immutable root fs with a size-capped writable /tmp.
        assert "--read-only" in argv
        tmpfs = argv[argv.index("--tmpfs") + 1]
        assert tmpfs.startswith("/tmp:") and "noexec" in tmpfs and "size=" in tmpfs
        # Privilege reduction.
        assert argv[argv.index("--cap-drop") + 1] == "ALL"
        assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
        user = argv[argv.index("--user") + 1]
        assert user == f"{os.getuid()}:{os.getgid()}"
        assert not user.startswith("0:")
        # Fork bombs and swap thrash are capped.
        assert "--pids-limit" in argv
        assert "--memory-swap" in argv

    def test_start_builds_image_when_missing(self, tmp_path, fake_docker):
        def responder(cmd, kw):
            if cmd[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(cmd, 1, "", "no such image")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        fake_docker.responder = responder
        DockerSandbox(tmp_path).start()
        build = next(cmd for cmd, kw in fake_docker.calls if cmd[:2] == ["docker", "build"])
        assert "-t" in build

    def test_start_registers_container_for_crash_cleanup(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()
        assert sb.name in sandbox_mod._ACTIVE_CONTAINERS
        sb.stop()
        assert sb.name not in sandbox_mod._ACTIVE_CONTAINERS

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

    def test_run_python_file_dispatches(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()
        sb.run_python_file("sim.py")
        assert fake_docker.calls[-1][0][-2:] == ["python", "sim.py"]


class TestPipInstall:
    """Installs run in an ephemeral container that HAS network, so the
    command surface must be locked down: validated names, wheels only."""

    def install_argv(self, recorder):
        return next(
            cmd for cmd, _ in recorder.calls if cmd[:2] == ["docker", "run"] and "pip" in cmd
        )

    def test_installer_is_ephemeral_and_wheels_only(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()
        result = sb.pip_install(["statsmodels", "emcee==3.1.4"])
        assert result.exit_code == 0

        argv = self.install_argv(fake_docker)
        assert "--rm" in argv
        # The installer may reach PyPI, so it must never execute package code:
        assert "--only-binary=:all:" in argv
        # Installs land inside the workspace, importable via PYTHONPATH.
        assert f"/workspace/{DEPS_DIR}" in argv[argv.index("--target") + 1]
        # Same privilege reduction as the exec container.
        assert argv[argv.index("--cap-drop") + 1] == "ALL"
        assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
        # But NOT --network none — this is the sanctioned install phase.
        assert "none" not in argv
        assert argv[-2:] == ["statsmodels", "emcee==3.1.4"]
        assert sb.installed_packages == ["statsmodels", "emcee==3.1.4"]

    def test_failed_install_is_not_recorded(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()
        fake_docker.responder = lambda cmd, kw: subprocess.CompletedProcess(
            cmd, 1, "", "no matching distribution"
        )
        result = sb.pip_install(["nosuchpackage12345"])
        assert result.exit_code == 1
        assert sb.installed_packages == []

    MALICIOUS = [
        "--index-url=http://evil.example",
        "-r/workspace/reqs.txt",
        "git+https://github.com/evil/evil",
        "https://evil.example/pkg.whl",
        "numpy; import os",
        "pkg && curl evil",
        "../../../etc",
        "pkg name",
        ".",
        "",
    ]

    @pytest.mark.parametrize("spec", MALICIOUS)
    def test_malicious_specs_refused_before_docker_runs(self, tmp_path, fake_docker, spec):
        sb = DockerSandbox(tmp_path)
        sb.start()
        calls_before = len(fake_docker.calls)
        result = sb.pip_install([spec])
        assert result.exit_code != 0
        assert "Refused" in result.stderr
        assert len(fake_docker.calls) == calls_before  # docker never invoked

    def test_install_timeout_reported(self, tmp_path, fake_docker):
        sb = DockerSandbox(tmp_path)
        sb.start()

        def time_out(cmd, kw):
            if "pip" in cmd:
                raise subprocess.TimeoutExpired(cmd, kw["timeout"])
            return subprocess.CompletedProcess(cmd, 0, "", "")

        fake_docker.responder = time_out
        result = sb.pip_install(["numpy"])
        assert result.exit_code == -1
        assert "timed out" in result.stderr


class TestValidatePackages:
    GOOD = ["numpy", "scipy==1.11.4", "pandas>=2.0", "scikit-learn", "typing_extensions",
            "uncertainties[arrays]", "emcee~=3.1"]

    @pytest.mark.parametrize("spec", GOOD)
    def test_accepts_plain_specs(self, spec):
        assert validate_packages([spec]) is None

    def test_rejects_empty_list(self):
        assert validate_packages([]) is not None


class TestEnsureImage:
    def test_present_image_is_not_rebuilt(self, fake_docker):
        ensure_image("some-image:latest")
        assert not any(cmd[:2] == ["docker", "build"] for cmd, _ in fake_docker.calls)

    def test_missing_image_is_built_from_packaged_dockerfile(self, fake_docker):
        def responder(cmd, kw):
            if cmd[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(cmd, 1, "", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        fake_docker.responder = responder
        ensure_image("some-image:latest")
        build_call = next(
            (cmd, kw) for cmd, kw in fake_docker.calls if cmd[:2] == ["docker", "build"]
        )
        assert b"FROM python" in build_call[1]["input"]


class TestCrashCleanup:
    def test_cleanup_all_removes_every_registered_container(self, fake_docker):
        sandbox_mod._ACTIVE_CONTAINERS.update({"c1", "c2"})
        sandbox_mod._cleanup_all_containers()
        removed = {cmd[3] for cmd, _ in fake_docker.calls if cmd[:3] == ["docker", "rm", "-f"]}
        assert removed == {"c1", "c2"}
        assert sandbox_mod._ACTIVE_CONTAINERS == set()

    def test_hooks_install_only_once(self, monkeypatch):
        monkeypatch.setattr(sandbox_mod, "_HOOKS_INSTALLED", False)
        registered = []
        monkeypatch.setattr(sandbox_mod.atexit, "register", registered.append)
        monkeypatch.setattr(sandbox_mod.signal, "signal", lambda *a: None)
        sandbox_mod._install_cleanup_hooks()
        sandbox_mod._install_cleanup_hooks()
        assert registered == [sandbox_mod._cleanup_all_containers]


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
    """Opt-in integration tests: pytest --run-docker (builds the image on first use)."""

    def test_container_roundtrip_and_teardown(self, tmp_path):
        sb = DockerSandbox(tmp_path)
        with sb:
            (tmp_path / "hello.py").write_text("import numpy\nprint('numpy', numpy.__version__)")
            result = sb.run_python_file("hello.py")
            assert result.exit_code == 0
            assert "numpy" in result.stdout  # scientific stack is pre-baked

        leftovers = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={sb.name}", "-q"],
            capture_output=True,
            text=True,
        )
        assert leftovers.stdout.strip() == ""

    def test_no_network_inside_sandbox(self, tmp_path):
        with DockerSandbox(tmp_path) as sb:
            (tmp_path / "phone_home.py").write_text(
                "import urllib.request\n"
                "urllib.request.urlopen('https://example.com', timeout=5)\n"
            )
            result = sb.run_python_file("phone_home.py")
            assert result.exit_code != 0

    def test_runs_as_non_root_with_read_only_rootfs(self, tmp_path):
        with DockerSandbox(tmp_path) as sb:
            uid = sb.exec(["python", "-c", "import os; print(os.getuid())"])
            assert uid.stdout.strip() == str(os.getuid())
            assert uid.stdout.strip() != "0"
            poke = sb.exec(["python", "-c", "open('/etc/pwned', 'w')"])
            assert poke.exit_code != 0

    def test_two_phase_install_then_import_offline(self, tmp_path):
        with DockerSandbox(tmp_path) as sb:
            install = sb.pip_install(["six"])
            assert install.exit_code == 0, install.stderr
            result = sb.exec(["python", "-c", "import six; print(six.__version__)"])
            assert result.exit_code == 0, result.stderr
            assert result.stdout.strip()
