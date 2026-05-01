"""TOML configuration for Sentinel."""

import os
import tomllib
from dataclasses import dataclass, field

_DEFAULT_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".config", "sentinel", "config.toml")

@dataclass
class SentinelConfig:
    db_path: str = os.path.join(os.path.expanduser("~"), ".sentinel", "sentinel.db")
    ai_enabled: bool = True
    ai_model: str = "claude-haiku-4-5"
    ai_model_deep: str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_ai_calls_per_hour: int = 100
    rate_limit_rpm: int = 60
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    log_level: str = "INFO"

_config: SentinelConfig | None = None

def load_config(path: str | None = None) -> SentinelConfig:
    config_path = path or _DEFAULT_CONFIG_PATH
    defaults = SentinelConfig()
    if not os.path.exists(config_path):
        return defaults
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        for key in ("db_path", "ai_enabled", "ai_model", "ai_model_deep", "api_key_env",
                     "max_ai_calls_per_hour", "rate_limit_rpm", "cors_origins", "log_level"):
            if key in data:
                setattr(defaults, key, data[key])
        if defaults.db_path.startswith("~"):
            defaults.db_path = os.path.expanduser(defaults.db_path)
        return defaults
    except Exception:
        return SentinelConfig()

def get_config() -> SentinelConfig:
    global _config
    if _config is None:
        _config = load_config()
    # Always honour the env-var so tests (and containers) can override the
    # cached config without needing to reset the module-level singleton.
    env_db = os.environ.get("SENTINEL_DB_PATH")
    if env_db:
        _config.db_path = env_db
    return _config


def setup_logging(level: str = "") -> None:
    import logging
    lvl = level or get_config().log_level
    logging.basicConfig(
        level=getattr(logging, lvl.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
