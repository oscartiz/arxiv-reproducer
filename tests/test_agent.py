from pathlib import Path

import pytest
from rich.console import Console

from arxiv_reproducer import agent as agent_mod
from arxiv_reproducer.agent import MAX_READ_CHARS, build_tools, run_reproduction
from arxiv_reproducer.paper import Paper
from arxiv_reproducer.sandbox import ExecResult


class FakeSandbox:
    """Stands in for DockerSandbox: same surface the tools use."""

    def __init__(self, workdir):
        self.workdir = Path(workdir).resolve()
        self.calls = []

    def run_python_file(self, relpath, timeout=600):
        self.calls.append(("run_python", relpath))
        return ExecResult(0, f"ran {relpath}", "")

    def pip_install(self, packages):
        self.calls.append(("pip_install", packages))
        return ExecResult(0, f"installed {' '.join(packages)}", "")

    def start(self):
        self.calls.append(("start",))

    def stop(self):
        self.calls.append(("stop",))

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


@pytest.fixture
def sandbox(tmp_path):
    return FakeSandbox(tmp_path)


@pytest.fixture
def tools(sandbox):
    return {tool.name: tool for tool in build_tools(sandbox)}


class TestToolSurface:
    def test_exposes_exactly_the_four_tools(self, tools):
        assert set(tools) == {"write_file", "read_file", "run_python", "install_packages"}


class TestWriteAndReadFile:
    def test_roundtrip(self, tools):
        result = tools["write_file"].call({"path": "sim.py", "content": "print(1)"})
        assert "Wrote" in result
        assert tools["read_file"].call({"path": "sim.py"}) == "print(1)"

    def test_write_creates_nested_directories(self, tools, sandbox):
        tools["write_file"].call({"path": "figs/deep/plot.py", "content": "x"})
        assert (sandbox.workdir / "figs" / "deep" / "plot.py").read_text() == "x"

    def test_read_missing_file(self, tools):
        assert "does not exist" in tools["read_file"].call({"path": "nope.py"})

    def test_read_binary_file_is_an_error_not_a_crash(self, tools, sandbox):
        (sandbox.workdir / "blob.bin").write_bytes(b"\x89PNG\x00\xff\xfe")
        assert "not a text file" in tools["read_file"].call({"path": "blob.bin"})

    def test_read_truncates_huge_files(self, tools, sandbox):
        (sandbox.workdir / "big.txt").write_text("x" * (MAX_READ_CHARS + 1000))
        result = tools["read_file"].call({"path": "big.txt"})
        assert len(result) < MAX_READ_CHARS + 200
        assert "truncated" in result


class TestPathEscapes:
    """Paths come from untrusted model output driven by untrusted paper text."""

    ESCAPES = [
        "../outside.txt",
        "../../etc/passwd",
        "a/../../outside.txt",
        "/etc/passwd",
        "/tmp/evil.sh",
        "a\x00b",
        "..",
    ]

    @pytest.mark.parametrize("path", ESCAPES)
    def test_write_rejects_escape(self, tools, path):
        result = tools["write_file"].call({"path": path, "content": "pwned"})
        assert result.startswith("Error"), f"{path!r} was not rejected: {result}"

    @pytest.mark.parametrize("path", ESCAPES)
    def test_read_rejects_escape(self, tools, path):
        result = tools["read_file"].call({"path": path})
        assert result.startswith("Error"), f"{path!r} was not rejected: {result}"

    def test_write_escape_leaves_no_file_behind(self, tools, sandbox, tmp_path):
        tools["write_file"].call({"path": "../pwned.txt", "content": "pwned"})
        assert not (sandbox.workdir.parent / "pwned.txt").exists()

    def test_symlinked_file_pointing_outside_is_rejected(self, tools, sandbox):
        # Container code can create symlinks inside the bind-mounted workspace;
        # on the host they resolve to host files and must be refused.
        (sandbox.workdir / "innocent.txt").symlink_to("/etc/passwd")
        assert tools["read_file"].call({"path": "innocent.txt"}).startswith("Error")
        assert tools["write_file"].call(
            {"path": "innocent.txt", "content": "pwned"}
        ).startswith("Error")
        assert Path("/etc/passwd").read_text()  # untouched

    def test_symlinked_directory_pointing_outside_is_rejected(self, tools, sandbox):
        (sandbox.workdir / "sneaky").symlink_to("/etc")
        assert tools["read_file"].call({"path": "sneaky/passwd"}).startswith("Error")
        assert tools["write_file"].call(
            {"path": "sneaky/new.conf", "content": "x"}
        ).startswith("Error")


class TestExecutionTools:
    def test_run_python_dispatches_to_sandbox(self, tools, sandbox):
        result = tools["run_python"].call({"path": "sim.py"})
        assert ("run_python", "sim.py") in sandbox.calls
        assert "ran sim.py" in result

    def test_install_packages_splits_names(self, tools, sandbox):
        tools["install_packages"].call({"packages": "numpy scipy matplotlib"})
        assert ("pip_install", ["numpy", "scipy", "matplotlib"]) in sandbox.calls


class _Block:
    def __init__(self, type, **attrs):
        self.type = type
        for key, value in attrs.items():
            setattr(self, key, value)


class _Message:
    def __init__(self, blocks):
        self.content = blocks


class FakeAnthropic:
    """Mimics anthropic.Anthropic().beta.messages.tool_runner(...)."""

    last_kwargs = None
    init_kwargs = None
    messages_to_yield = []
    raise_after_yield: Exception | None = None

    def __init__(self, *args, **kwargs):
        FakeAnthropic.init_kwargs = kwargs

        def runner():
            yield from FakeAnthropic.messages_to_yield
            if FakeAnthropic.raise_after_yield is not None:
                raise FakeAnthropic.raise_after_yield

        class _Messages:
            def tool_runner(self, **kwargs):
                FakeAnthropic.last_kwargs = kwargs
                return runner()

        class _Beta:
            messages = _Messages()

        self.beta = _Beta()


@pytest.fixture
def fake_agent_env(monkeypatch, tmp_path):
    """Patch out the real Anthropic client and DockerSandbox."""
    FakeAnthropic.last_kwargs = None
    FakeAnthropic.init_kwargs = None
    FakeAnthropic.messages_to_yield = []
    FakeAnthropic.raise_after_yield = None
    monkeypatch.setattr(agent_mod.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(agent_mod, "DockerSandbox", FakeSandbox)
    paper = Paper(
        arxiv_id="2301.12345",
        title="A Test Paper",
        abstract="An abstract.",
        authors=["A. Author"],
        full_text="The full text of the paper.",
        pdf_path=tmp_path / "2301.12345.pdf",
    )
    return paper, tmp_path


class TestRunReproduction:
    def test_passes_tools_model_and_cached_paper_text(self, fake_agent_env):
        paper, workdir = fake_agent_env
        run_reproduction(paper, workdir, Console(record=True, width=200))

        kwargs = FakeAnthropic.last_kwargs
        assert kwargs["model"] == agent_mod.MODEL
        assert len(kwargs["tools"]) == 4
        first_block = kwargs["messages"][0]["content"][0]
        assert first_block["cache_control"] == {"type": "ephemeral"}
        assert paper.full_text in first_block["text"]

    def test_streams_text_and_tool_use_to_console(self, fake_agent_env):
        paper, workdir = fake_agent_env
        FakeAnthropic.messages_to_yield = [
            _Message([_Block("text", text="Planning the reproduction.")]),
            _Message([_Block("tool_use", name="write_file")]),
        ]
        console = Console(record=True, width=200)
        run_reproduction(paper, workdir, console)
        output = console.export_text()
        assert "Planning the reproduction." in output
        assert "write_file" in output

    def test_client_uses_sdk_retries(self, fake_agent_env):
        paper, workdir = fake_agent_env
        run_reproduction(paper, workdir, Console(record=True, width=200))
        assert FakeAnthropic.init_kwargs == {"max_retries": agent_mod.API_MAX_RETRIES}

    def test_missing_report_gets_placeholder(self, fake_agent_env):
        paper, workdir = fake_agent_env
        result = run_reproduction(paper, workdir, Console(record=True, width=200))
        assert result.report == workdir / "REPORT.md"
        assert result.status == "completed"
        assert "missing" in result.report.read_text().lower()

    def test_agent_written_report_is_preserved(self, fake_agent_env):
        paper, workdir = fake_agent_env
        (workdir / "REPORT.md").write_text("# Verdict: REPRODUCED")
        result = run_reproduction(paper, workdir, Console(record=True, width=200))
        assert result.report.read_text() == "# Verdict: REPRODUCED"


class TestMidRunFailure:
    """A dying API mid-run must still leave a coherent workspace."""

    def test_api_failure_yields_error_status_and_report(self, fake_agent_env):
        import httpx

        paper, workdir = fake_agent_env
        FakeAnthropic.messages_to_yield = [_Message([_Block("text", text="working...")])]
        FakeAnthropic.raise_after_yield = httpx.ConnectError("api unreachable")

        console = Console(record=True, width=200)
        result = run_reproduction(paper, workdir, console)

        assert result.status == "error"
        assert "api unreachable" in (result.error or "")
        assert result.iterations == 1
        report_text = result.report.read_text()
        assert "error" in report_text.lower()
        assert "api unreachable" in report_text
        assert "aborted" in console.export_text().lower()

    def test_sandbox_is_stopped_after_api_failure(self, fake_agent_env, monkeypatch):
        import httpx

        paper, workdir = fake_agent_env
        stopped = []

        class TrackingSandbox(FakeSandbox):
            def stop(self):
                stopped.append(self.name if hasattr(self, "name") else "sandbox")
                super().stop()

        monkeypatch.setattr(agent_mod, "DockerSandbox", TrackingSandbox)
        FakeAnthropic.raise_after_yield = httpx.ConnectError("api unreachable")
        run_reproduction(paper, workdir, Console(record=True, width=200))
        assert stopped  # teardown happened despite the failure


class TestRunCaps:
    """A run must not loop forever burning API spend."""

    def test_iteration_cap_stops_the_loop(self, fake_agent_env, monkeypatch):
        paper, workdir = fake_agent_env
        monkeypatch.setattr(agent_mod, "MAX_ITERATIONS", 3)
        FakeAnthropic.messages_to_yield = [
            _Message([_Block("text", text=f"message-{i}")]) for i in range(10)
        ]
        console = Console(record=True, width=200)
        result = run_reproduction(paper, workdir, console)
        output = console.export_text()
        assert "message-2" in output
        assert "message-3" not in output
        assert "iteration cap" in output.lower()
        assert result.status == "iteration_cap"
        assert result.iterations == 3

    def test_wall_clock_cap_stops_the_loop(self, fake_agent_env, monkeypatch):
        paper, workdir = fake_agent_env
        monkeypatch.setattr(agent_mod, "MAX_WALL_CLOCK_SECONDS", 0)
        FakeAnthropic.messages_to_yield = [
            _Message([_Block("text", text=f"message-{i}")]) for i in range(5)
        ]
        console = Console(record=True, width=200)
        result = run_reproduction(paper, workdir, console)
        output = console.export_text()
        assert "message-0" in output
        assert "message-1" not in output
        assert "wall-clock cap" in output.lower()
        assert result.status == "wall_clock_cap"
