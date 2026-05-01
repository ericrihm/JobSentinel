# Task: Build sentinel/config.py -- TOML Configuration Module

## Task

Build a centralized TOML configuration module at `sentinel/config.py` that loads settings from `~/.config/sentinel/config.toml`, provides typed dataclass defaults, and wires into the existing modules.

## Context

- The project is JobSentinel, an AI-powered LinkedIn job scam detection platform at the repo root.
- `sentinel/config.py` is listed in `CLAUDE.md` but currently does not exist.
- The codebase uses Python 3.12+ stdlib-first design. `tomllib` is available in stdlib since 3.11.
- The following modules need config wired in:
  - `sentinel/analyzer.py` -- uses hardcoded AI model names (`_HAIKU_MODEL`, `_SONNET_MODEL`) and has no API key configuration.
  - `sentinel/api.py` -- hardcodes `allow_origins=["*"]` in CORS middleware.
  - `sentinel/db.py` -- hardcodes `DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".sentinel", "sentinel.db")`.

## What To Do

### 1. Create `sentinel/config.py`

```python
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
```

Define a `SentinelConfig` dataclass with these fields and defaults:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `db_path` | `str` | `~/.sentinel/sentinel.db` | SQLite database path |
| `ai_enabled` | `bool` | `True` | Enable AI analysis tier |
| `ai_model` | `str` | `claude-haiku-4-5` | Default AI model for analysis |
| `ai_model_deep` | `str` | `claude-sonnet-4-6` | Deep analysis model |
| `api_key_env` | `str` | `ANTHROPIC_API_KEY` | Env var name for API key |
| `max_ai_calls_per_hour` | `int` | `100` | Rate limit for AI API calls |
| `rate_limit_rpm` | `int` | `60` | API rate limit (requests per minute) |
| `cors_origins` | `list[str]` | `["http://localhost:3000"]` | Allowed CORS origins |
| `log_level` | `str` | `INFO` | Logging level |

Implement:

- `load_config(path: str | None = None) -> SentinelConfig`: Load from TOML file at `path` (default `~/.config/sentinel/config.toml`). If file doesn't exist, return defaults. Merge TOML values over defaults -- missing keys keep defaults.
- `get_config() -> SentinelConfig`: Module-level singleton getter. Calls `load_config()` on first access, caches the result. Thread-safe is not required.
- Handle malformed TOML gracefully -- log a warning and return defaults.

### 2. Wire config into `sentinel/analyzer.py`

- Import `get_config` from `sentinel.config`.
- Replace the hardcoded `_HAIKU_MODEL` and `_SONNET_MODEL` with values from `config.ai_model` and `config.ai_model_deep`.
- In `_escalate_to_ai()`, check `config.ai_enabled` before proceeding with AI calls.
- Read the API key from `os.environ.get(config.api_key_env)` when constructing the Anthropic client.

### 3. Wire config into `sentinel/api.py`

- In `create_app()`, import `get_config` and use `config.cors_origins` for the `allow_origins` parameter instead of `["*"]`.
- Use `config.rate_limit_rpm` to set a rate limit value (store it as app state for now; the actual rate-limiting middleware is task 02).

### 4. Wire config into `sentinel/db.py`

- Import `get_config` from `sentinel.config`.
- Change `DEFAULT_DB_PATH` to use `get_config().db_path` as the fallback in `SentinelDB.__init__()`.

### 5. Add tests in `tests/test_config.py`

Write tests covering:
- `load_config()` returns `SentinelConfig` with correct defaults when no file exists.
- `load_config()` correctly reads a TOML file and overrides individual fields.
- `load_config()` merges partial TOML (only some fields set) with defaults.
- `load_config()` handles malformed TOML without raising.
- `get_config()` returns the same instance on repeated calls.
- Config values propagate: create a temp TOML, load it, confirm fields.

Use `tmp_path` fixtures and write actual TOML files for testing.

## Acceptance Criteria

- [ ] `sentinel/config.py` exists with `SentinelConfig` dataclass and `load_config()`/`get_config()` functions.
- [ ] Config loads from `~/.config/sentinel/config.toml` with fallback defaults.
- [ ] `analyzer.py` reads AI model names and `ai_enabled` flag from config.
- [ ] `api.py` reads CORS origins from config instead of hardcoding `["*"]`.
- [ ] `db.py` reads `db_path` from config as its default.
- [ ] All new and existing tests pass: `python -m pytest tests/ -v`.

## Constraints

- Use only Python stdlib (`tomllib`, `dataclasses`, `pathlib`, `os`). No third-party config libraries.
- Do not break any existing tests.
- Do not change function signatures of public APIs -- only internal implementation.
- Expand `~` in `db_path` using `os.path.expanduser()`.

## Test Command

```bash
python -m pytest tests/test_config.py tests/test_core.py -v
```
