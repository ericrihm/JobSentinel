# Task: Fill Test Coverage for Untested Modules

## Task

Write comprehensive tests for modules with zero or minimal direct test coverage: `flywheel.py`, `api.py`, `cli.py`, `ecosystem.py`, and `innovation.py`. Target 90%+ line coverage on these modules.

## Context

- The test suite is in `tests/` using pytest. Existing tests are in `tests/test_core.py` and `tests/test_advanced.py`.
- `tests/conftest.py` currently has only a placeholder fixture (`sample_sentinel` returning `{}`).
- Key modules needing test coverage:
  - `sentinel/flywheel.py` -- `CUSUMDetector`, `SignalWeightTracker`, `DetectionFlywheel` with methods: `learn_from_report`, `evolve_patterns`, `compute_accuracy`, `detect_regression`, `run_cycle`, `get_health`. Also standalone functions `_gamma_sample` and `_normal_sample`.
  - `sentinel/api.py` -- 5 endpoints via FastAPI `TestClient`. App is created by `create_app()` factory.
  - `sentinel/cli.py` -- Key commands via Click `CliRunner`: `analyze`, `validate`, `report`, `patterns`, `stats`, `evolve`, `innovate`, `ecosystem`, `serve`.
  - `sentinel/ecosystem.py` -- `publish_observation`, `publish_event`, `publish_flywheel_state`, `publish_detection_result`, `publish_to_engram`, `read_ecosystem_context` (partially tested in test_advanced.py).
  - `sentinel/innovation.py` -- `InnovationEngine` with `run_cycle`, strategy methods (partially tested in test_advanced.py).

- FastAPI is an optional dependency. Tests for `api.py` should skip if FastAPI is not installed.
- The `analyzer.py` `analyze_job()` function has a known bug (passes undefined variable `score` to `_escalate_to_ai` on line 44). Tests should use the `build_result` pipeline from `scorer.py` directly when testing the full pipeline (as `test_core.py` already does).

## What To Do

### 1. Create `tests/test_flywheel.py`

**Test `CUSUMDetector`:**
- `__init__` sets default target=0, slack=0.5, threshold=5.0.
- `update()` returns False when value is above target (no regression).
- `update()` returns True (alarm) after sustained values well below the target minus slack (e.g., feeding 20 zeros with target=0.8 should eventually trigger alarm).
- `reset()` resets the statistic to 0.
- `statistic` property tracks the cumulative sum correctly.
- After `reset()`, `statistic` is 0.0.

**Test `SignalWeightTracker` (the one in flywheel.py):**
- `update()` increments alpha for true positives.
- `update()` increments beta for false positives.
- `expected_weight()` returns `alpha / (alpha + beta)`.
- `sample()` returns a value in [0, 1] (test 100 samples).
- `all_weights()` returns dict of all tracked signals.
- `get_posterior()` returns correct (alpha, beta) tuple.
- After 10 true-positive updates, `expected_weight()` > 0.8.
- After 10 false-positive updates, `expected_weight()` < 0.2.

**Test `DetectionFlywheel`:**
- `__init__` creates with a temp DB (use `tmp_path`).
- `learn_from_report()` with a scam report updates signal weights and returns summary dict with expected keys: `report_url`, `is_scam`, `signals_updated`, `signal_count`, `updated_at`.
- `learn_from_report()` with a legitimate report updates beta values.
- `evolve_patterns()` promotes high-precision candidate patterns (seed a candidate pattern with obs=15, tp=14, fp=1, then call evolve and verify it becomes active).
- `evolve_patterns()` deprecates low-precision active patterns (seed an active pattern with obs=25, tp=5, fp=20, then call evolve and verify it becomes deprecated).
- `evolve_patterns()` retains patterns that don't meet thresholds.
- `compute_accuracy()` returns correct metrics with seeded report data.
- `compute_accuracy()` returns zeros with empty DB.
- `detect_regression()` returns no alarm with insufficient data (< 5 reports).
- `detect_regression()` returns a result dict with expected keys.
- `run_cycle()` executes full cycle and returns metrics dict with keys: `cycle_ts`, `cycle_number`, `total_analyzed`, `precision`, `recall`, `f1`, `patterns_promoted`, `patterns_deprecated`, `regression_alarm`, `cusum_statistic`.
- `get_health()` returns health dict with keys: `healthy`, `grade`, `precision`, `recall`, `f1`, `active_patterns`, `total_jobs_analyzed`, `regression_alarm`, `checked_at`.
- Health grade is "A" when precision >= 0.85 (seed reports to achieve this).
- Health grade is "F" when precision < 0.40.

### 2. Create `tests/test_api_endpoints.py`

Skip all tests if FastAPI not installed:
```python
pytest.importorskip("fastapi")
from starlette.testclient import TestClient
from sentinel.api import create_app
```

Monkeypatch the default DB path for each test to use a temp directory.

**POST /api/analyze:**
- With `text` field containing scam language: returns 200 with response containing `scam_score`, `risk_level`, `signal_count`.
- With `text` field containing legitimate job description: returns 200 with low scam_score.
- With `job_data` dict: returns 200 with valid response structure.
- With no input fields (empty body): returns 422.
- Response contains `risk_label` string.

**POST /api/report:**
- Submit scam report with `url` and `is_scam=true`: returns 200 with `recorded: true` and `verdict: "scam"`.
- Submit legitimate report with `is_scam=false`: returns 200 with `verdict: "legitimate"`.
- Missing required `url` field: returns 422.

**GET /api/patterns:**
- Returns JSON with `patterns` list and `count` integer.
- With `status=all` query param: returns patterns from all statuses.
- With `category=red_flag` query param: returns only red_flag patterns.
- Empty DB returns `count: 0`.

**GET /api/stats:**
- Returns JSON with all expected keys: `total_jobs_analyzed`, `scam_jobs_detected`, `avg_scam_score`, `total_user_reports`, `prediction_accuracy`, `active_patterns`.
- Empty DB returns zeros for all numeric fields.

**GET /api/health:**
- Returns JSON with `status`, `healthy`, `grade`, `precision`, `recall`.
- `status` is either "ok" or "degraded".

### 3. Create `tests/test_cli_commands.py`

Use Click's `CliRunner` for all tests. Monkeypatch DB paths to temp directories.

**analyze command:**
- `sentinel analyze "Some job description text"` exits with code 0 and produces output.
- `sentinel --json-output analyze "text"` produces valid JSON output.
- `sentinel analyze --no-ai "text"` works (AI disabled).

**validate command:**
- `sentinel validate Google` shows "[VERIFIED]" in output.
- `sentinel validate "Unknown Corp XYZ123"` shows "[UNVERIFIED]" in output.
- `sentinel --json-output validate Google` produces valid JSON.

**report command:**
- `sentinel report "https://scam.com/1" --reason "asked for money"` exits 0 and shows "Report recorded".
- `sentinel report "https://legit.com/1" --legitimate` marks as legitimate.

**patterns command:**
- `sentinel patterns` runs without error.
- `sentinel patterns --type red-flag` filters correctly (or shows no patterns on empty DB).

**stats command:**
- `sentinel stats` runs without error and shows "Sentinel Detection Statistics" in output.
- `sentinel --json-output stats` produces valid JSON.

**evolve command:**
- `sentinel evolve` runs without error and shows "Flywheel Cycle Complete".

### 4. Create `tests/test_ecosystem_extended.py`

- `publish_flywheel_state()` writes to both events and observations files (monkeypatch file paths).
- `publish_to_engram()` falls back to `publish_observation()` when engram is not installed.
- `publish_observation()` handles OSError gracefully when directory doesn't exist and can't be created.
- `read_ecosystem_context()` returns a dict even when briefing file doesn't exist.
- `read_ecosystem_context()` reads briefing file when it exists (create a temp briefing file).

## Acceptance Criteria

- [ ] `tests/test_flywheel.py` has 15+ tests covering CUSUMDetector, SignalWeightTracker, and DetectionFlywheel.
- [ ] `tests/test_api_endpoints.py` has 10+ tests covering all 5 API endpoints with TestClient.
- [ ] `tests/test_cli_commands.py` has 10+ tests covering analyze, validate, report, patterns, stats, evolve commands.
- [ ] `tests/test_ecosystem_extended.py` has 5+ tests for ecosystem functions.
- [ ] All tests pass, including existing tests in test_core.py and test_advanced.py.
- [ ] Tests use tmp_path for DB isolation -- no side effects on the real DB.

## Constraints

- Do not modify any source files. This task is test-only.
- Skip API tests when FastAPI is not installed (use `pytest.importorskip`).
- Use `monkeypatch` to redirect DB paths and ecosystem file paths to tmp directories.
- Do not call real external APIs (LinkedIn, WHOIS, Anthropic). Mock all network calls.
- Use `CliRunner(mix_stderr=False)` for CLI tests to capture output properly.
- For flywheel tests, seed test data using `db.save_report()` and `db.save_pattern()` directly.

## Test Command

```bash
python -m pytest tests/test_flywheel.py tests/test_api_endpoints.py tests/test_cli_commands.py tests/test_ecosystem_extended.py tests/test_core.py tests/test_advanced.py -v --tb=short
```
