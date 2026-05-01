# Task: Fill Test Coverage Gaps for Untested Modules

## Task

Write comprehensive tests for the three modules with zero test coverage: flywheel.py, api.py, and cli.py.

## Context

- Current test suite: 154 tests in `tests/test_core.py` and `tests/test_advanced.py`
- `sentinel/flywheel.py` — self-improving detection loop, CUSUMDetector, learn_from_report, evolve_patterns — zero tests
- `sentinel/api.py` — FastAPI REST API with 5 endpoints — zero tests
- `sentinel/cli.py` — Click CLI with 8+ commands — zero tests
- Shared fixtures in `tests/conftest.py`: `sample_job`, `scam_job`, `ghost_job`, `temp_db`, `seeded_db`

## What To Do

### 1. Create `tests/test_flywheel.py`

Test the `DetectionFlywheel` class and `CUSUMDetector`:

```python
class TestCUSUMDetector:
    def test_no_alarm_under_threshold(self): ...
    def test_alarm_at_threshold(self): ...
    def test_reset_clears_alarm(self): ...
    def test_cumulative_drift_detection(self): ...

class TestDetectionFlywheel:
    def test_learn_from_report_scam(self): ...
    def test_learn_from_report_legitimate(self): ...
    def test_evolve_patterns_promotes_high_precision(self): ...
    def test_evolve_patterns_deprecates_low_precision(self): ...
    def test_run_cycle_returns_metrics(self): ...
    def test_get_health_structure(self): ...
    def test_detect_regression_no_alarm_baseline(self): ...
```

Use `seeded_db` fixture. Mock DB path to temp directory.

### 2. Create `tests/test_api.py`

Test all 5 endpoints using FastAPI TestClient:

```python
from fastapi.testclient import TestClient
from sentinel.api import create_app

class TestAnalyzeEndpoint:
    def test_analyze_text(self): ...
    def test_analyze_missing_input(self): ...
    def test_analyze_with_job_data(self): ...

class TestReportEndpoint:
    def test_report_scam(self): ...
    def test_report_legitimate(self): ...

class TestPatternsEndpoint:
    def test_list_patterns(self): ...
    def test_filter_by_category(self): ...

class TestStatsEndpoint:
    def test_stats_returns_structure(self): ...

class TestHealthEndpoint:
    def test_health_check(self): ...
```

Mock DB to use temp directory.

### 3. Create `tests/test_cli.py`

Test CLI commands using Click CliRunner:

```python
from click.testing import CliRunner
from sentinel.cli import main

class TestAnalyzeCommand:
    def test_analyze_text_scam(self): ...
    def test_analyze_text_legit(self): ...

class TestValidateCommand:
    def test_validate_known_company(self): ...
    def test_validate_unknown_company(self): ...

class TestReportCommand:
    def test_report_scam(self): ...

class TestPatternsCommand:
    def test_list_patterns(self): ...

class TestStatsCommand:
    def test_stats_output(self): ...

class TestInitCommand:
    def test_init_seed(self): ...
```

Mock DB paths to temp directory.

## Acceptance Criteria

- [ ] `tests/test_flywheel.py` — at least 10 tests, all passing
- [ ] `tests/test_api.py` — at least 8 tests covering all 5 endpoints, all passing
- [ ] `tests/test_cli.py` — at least 8 tests covering key commands, all passing
- [ ] Total test count reaches 180+
- [ ] All tests pass: `python -m pytest tests/ -v`

## Constraints

- Use existing conftest.py fixtures where applicable
- Mock file system paths to avoid writing to ~/.sentinel/ during tests
- Do not modify existing test files
- Each test file should be independent (no cross-file dependencies)
- Use `unittest.mock.patch` for DB path mocking

## Test Command

```bash
python -m pytest tests/ -v
```
