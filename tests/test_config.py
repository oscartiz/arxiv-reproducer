import pytest

from arxiv_reproducer.config import (
    Config,
    ConfigError,
    get_config,
    load_config,
    set_config,
)


class TestDefaults:
    def test_sane_out_of_the_box(self):
        cfg = load_config(env={})
        assert cfg.model == "claude-opus-4-8"
        assert cfg.max_iterations == 60
        assert cfg.sandbox_image == "arxiv-repro-sandbox:latest"
        assert cfg.memory_limit == "4g"


class TestEnvOverrides:
    def test_string_and_int_overrides(self):
        cfg = load_config(
            env={
                "ARXIV_REPRO_MODEL": "claude-sonnet-5",
                "ARXIV_REPRO_MAX_ITERATIONS": "10",
                "ARXIV_REPRO_MEMORY_LIMIT": "8g",
            }
        )
        assert cfg.model == "claude-sonnet-5"
        assert cfg.max_iterations == 10
        assert cfg.memory_limit == "8g"

    def test_non_integer_value_is_a_clear_error(self):
        with pytest.raises(ConfigError, match="ARXIV_REPRO_MAX_ITERATIONS.*integer"):
            load_config(env={"ARXIV_REPRO_MAX_ITERATIONS": "lots"})

    def test_unrelated_env_vars_ignored(self):
        cfg = load_config(env={"ARXIV_REPRO_UNRELATED_THING": "x", "PATH": "/bin"})
        assert cfg == Config()


class TestTomlFile:
    def test_file_values_applied(self, tmp_path):
        toml = tmp_path / "arxiv-repro.toml"
        toml.write_text('model = "claude-haiku-4-5"\nmax_iterations = 5\ncpu_limit = "1.5"\n')
        cfg = load_config(env={}, config_file=toml)
        assert cfg.model == "claude-haiku-4-5"
        assert cfg.max_iterations == 5
        assert cfg.cpu_limit == "1.5"

    def test_env_beats_file(self, tmp_path):
        toml = tmp_path / "cfg.toml"
        toml.write_text('model = "claude-haiku-4-5"\n')
        cfg = load_config(env={"ARXIV_REPRO_MODEL": "claude-opus-4-8"}, config_file=toml)
        assert cfg.model == "claude-opus-4-8"

    def test_file_path_from_env(self, tmp_path):
        toml = tmp_path / "custom.toml"
        toml.write_text("max_tokens = 8000\n")
        cfg = load_config(env={"ARXIV_REPRO_CONFIG": str(toml)})
        assert cfg.max_tokens == 8000

    def test_unknown_keys_are_rejected_with_the_valid_list(self, tmp_path):
        toml = tmp_path / "cfg.toml"
        toml.write_text("max_iteratoins = 5\n")  # typo
        with pytest.raises(ConfigError, match="unknown config keys.*max_iteratoins"):
            load_config(env={}, config_file=toml)

    def test_invalid_toml_is_a_clear_error(self, tmp_path):
        toml = tmp_path / "cfg.toml"
        toml.write_text("model = [unclosed\n")
        with pytest.raises(ConfigError, match="invalid TOML"):
            load_config(env={}, config_file=toml)

    def test_missing_explicit_file_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(env={}, config_file=tmp_path / "nope.toml")


class TestValidation:
    def test_empty_model_rejected(self):
        with pytest.raises(ConfigError, match="model must be non-empty"):
            load_config(env={"ARXIV_REPRO_MODEL": "  "})

    def test_nonpositive_ints_rejected(self):
        with pytest.raises(ConfigError, match="max_iterations must be positive"):
            load_config(env={"ARXIV_REPRO_MAX_ITERATIONS": "0"})

    def test_bad_memory_format_rejected(self):
        with pytest.raises(ConfigError, match="memory_limit"):
            load_config(env={"ARXIV_REPRO_MEMORY_LIMIT": "four gigs"})

    def test_bad_cpu_format_rejected(self):
        with pytest.raises(ConfigError, match="cpu_limit"):
            load_config(env={"ARXIV_REPRO_CPU_LIMIT": "many"})


class TestProcessWideConfig:
    def test_get_config_caches(self, monkeypatch):
        set_config(None)
        monkeypatch.delenv("ARXIV_REPRO_MODEL", raising=False)
        first = get_config()
        assert get_config() is first

    def test_set_config_overrides(self):
        override = Config(max_iterations=3)
        set_config(override)
        assert get_config() is override

    def test_env_override_reaches_sandbox_flags(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ARXIV_REPRO_MEMORY_LIMIT", "8g")
        monkeypatch.setenv("ARXIV_REPRO_PIDS_LIMIT", "64")
        set_config(None)  # force re-load with the new env
        from arxiv_reproducer.sandbox import DockerSandbox

        flags = DockerSandbox(tmp_path)._hardening_flags()
        assert flags[flags.index("--memory") + 1] == "8g"
        assert flags[flags.index("--pids-limit") + 1] == "64"
