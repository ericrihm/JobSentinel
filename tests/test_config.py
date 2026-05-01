"""Tests for sentinel/config.py."""

import os
import pytest

import sentinel.config as config_module
from sentinel.config import SentinelConfig, load_config, get_config


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after each test."""
    config_module._config = None
    yield
    config_module._config = None


# ---------------------------------------------------------------------------
# test_defaults
# ---------------------------------------------------------------------------

def test_defaults(tmp_path):
    """load_config with a nonexistent path returns a SentinelConfig with defaults."""
    nonexistent = str(tmp_path / "does_not_exist.toml")
    cfg = load_config(nonexistent)
    assert isinstance(cfg, SentinelConfig)
    assert cfg.ai_enabled is True
    assert cfg.ai_model == "claude-haiku-4-5"
    assert cfg.ai_model_deep == "claude-sonnet-4-6"
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.max_ai_calls_per_hour == 100
    assert cfg.rate_limit_rpm == 60
    assert cfg.cors_origins == ["http://localhost:3000"]
    assert cfg.log_level == "INFO"
    assert cfg.db_path.endswith("sentinel.db")


# ---------------------------------------------------------------------------
# test_load_from_file
# ---------------------------------------------------------------------------

def test_load_from_file(tmp_path):
    """Writing a TOML file and loading it overrides the specified fields."""
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        'ai_model = "claude-opus-4-5"\n'
        'log_level = "DEBUG"\n'
        'rate_limit_rpm = 30\n'
        'cors_origins = ["https://myapp.example.com"]\n',
        encoding="utf-8",
    )
    cfg = load_config(str(toml_file))
    assert cfg.ai_model == "claude-opus-4-5"
    assert cfg.log_level == "DEBUG"
    assert cfg.rate_limit_rpm == 30
    assert cfg.cors_origins == ["https://myapp.example.com"]
    # un-overridden fields keep defaults
    assert cfg.ai_enabled is True
    assert cfg.max_ai_calls_per_hour == 100


# ---------------------------------------------------------------------------
# test_partial_toml
# ---------------------------------------------------------------------------

def test_partial_toml(tmp_path):
    """Only fields present in the TOML are overridden; others keep defaults."""
    toml_file = tmp_path / "partial.toml"
    toml_file.write_text('ai_enabled = false\n', encoding="utf-8")
    cfg = load_config(str(toml_file))
    assert cfg.ai_enabled is False
    # Everything else unchanged
    assert cfg.ai_model == "claude-haiku-4-5"
    assert cfg.log_level == "INFO"
    assert cfg.cors_origins == ["http://localhost:3000"]


# ---------------------------------------------------------------------------
# test_malformed_toml
# ---------------------------------------------------------------------------

def test_malformed_toml(tmp_path):
    """A TOML file with syntax errors causes load_config to silently return defaults."""
    toml_file = tmp_path / "bad.toml"
    toml_file.write_text("this is [[[not valid toml\n", encoding="utf-8")
    cfg = load_config(str(toml_file))
    assert isinstance(cfg, SentinelConfig)
    assert cfg.ai_model == "claude-haiku-4-5"
    assert cfg.log_level == "INFO"


# ---------------------------------------------------------------------------
# test_get_config_singleton
# ---------------------------------------------------------------------------

def test_get_config_singleton():
    """Calling get_config() twice returns the exact same object."""
    first = get_config()
    second = get_config()
    assert first is second


# ---------------------------------------------------------------------------
# test_expanduser
# ---------------------------------------------------------------------------

def test_expanduser(tmp_path):
    """A db_path value starting with '~' is expanded to an absolute path."""
    toml_file = tmp_path / "expand.toml"
    toml_file.write_text('db_path = "~/test.db"\n', encoding="utf-8")
    cfg = load_config(str(toml_file))
    assert not cfg.db_path.startswith("~")
    assert cfg.db_path == os.path.expanduser("~/test.db")
    assert os.path.isabs(cfg.db_path)
