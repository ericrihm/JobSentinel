# Task: Fix pyproject.toml Optional Dependencies

## Task

Update `pyproject.toml` to add proper optional dependency groups and fix the project description.

## Context

- `pyproject.toml` is at the repo root: `pyproject.toml`.
- Currently it only lists `click>=8.1` as a runtime dependency.
- The project has several optional dependencies documented in `CLAUDE.md`:
  - `anthropic` -- for AI analysis tier (used in `sentinel/analyzer.py`)
  - `fastapi` + `uvicorn` -- for REST API server (used in `sentinel/api.py`, `sentinel/cli.py` serve command)
  - `httpx` -- for URL fetching and company validation (used in `sentinel/scanner.py`, `sentinel/validator.py`)
  - `beautifulsoup4` -- for HTML parsing (mentioned in CLAUDE.md as optional)
- The current `description` field says `"sentinel -- security tool"` which is generic and unhelpful.
- The `dev` optional dependency group already exists with pytest, ruff, mutmut, and pytest-cov.
- The current `dependencies` list has `"click>=8.1"` with inconsistent formatting (no indentation alignment).

## What To Do

### 1. Update the project description

Change the `description` field from:
```
description = "sentinel -- security tool"
```
to:
```
description = "AI-powered LinkedIn job scam detection and validation platform"
```

### 2. Add optional dependency groups

Add these groups under `[project.optional-dependencies]`, replacing the existing section:

```toml
[project.optional-dependencies]
ai = ["anthropic>=0.40"]
api = ["fastapi>=0.115", "uvicorn>=0.32"]
web = ["httpx>=0.27", "beautifulsoup4>=4.12"]
full = [
    "anthropic>=0.40",
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
]
dev = [
    "mutmut>=3.0",
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "ruff>=0.8",
]
```

### 3. Clean up formatting

Ensure the `dependencies` list uses consistent formatting:

```toml
dependencies = [
    "click>=8.1",
]
```

### 4. Verify the complete file

After editing, the complete `pyproject.toml` should look like this:

```toml
[build-system]
requires = ["setuptools>=75.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sentinel"
version = "0.1.0"
description = "AI-powered LinkedIn job scam detection and validation platform"
requires-python = ">=3.12"
license = "MIT"
authors = [
    {name = "Eric Rihm", email = "eric@cobaltsystems.io"},
]
dependencies = [
    "click>=8.1",
]

[project.optional-dependencies]
ai = ["anthropic>=0.40"]
api = ["fastapi>=0.115", "uvicorn>=0.32"]
web = ["httpx>=0.27", "beautifulsoup4>=4.12"]
full = [
    "anthropic>=0.40",
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
]
dev = [
    "mutmut>=3.0",
    "pytest>=8.0",
    "pytest-cov>=6.0",
    "ruff>=0.8",
]

[project.scripts]
sentinel = "sentinel.cli:main"

[tool.setuptools.packages.find]
include = ["sentinel*"]

[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.coverage.run]
source = ["sentinel"]

[tool.coverage.report]
fail_under = 80
show_missing = true
skip_covered = true

[tool.mutmut]
source_paths = ["sentinel/"]
pytest_add_cli_args_test_selection = ["tests/"]
```

## Acceptance Criteria

- [ ] `pyproject.toml` has updated description: "AI-powered LinkedIn job scam detection and validation platform".
- [ ] Four new optional dependency groups exist: `ai`, `api`, `web`, `full`.
- [ ] `pip install sentinel[ai]` would install anthropic.
- [ ] `pip install sentinel[api]` would install fastapi and uvicorn.
- [ ] `pip install sentinel[web]` would install httpx and beautifulsoup4.
- [ ] `pip install sentinel[full]` would install all optional deps.
- [ ] `dev` group is preserved with existing dev dependencies.
- [ ] The TOML file is syntactically valid (verify with `python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"`).
- [ ] All existing tests still pass.

## Constraints

- Only modify `pyproject.toml`. Do not modify any Python files.
- Keep the `dependencies` list (click) unchanged.
- Keep all existing `[tool.*]` sections unchanged.
- Use minimum version specifiers (`>=`) not exact pins (`==`).

## Test Command

```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" && python -m pytest tests/ -v
```
