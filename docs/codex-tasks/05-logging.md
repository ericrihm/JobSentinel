# Task: Add Structured Logging Throughout the Codebase

## Task

Add Python `logging` module throughout the Sentinel codebase using a `sentinel.*` logger hierarchy. Replace all bare `except: pass` and `except Exception: pass` blocks with proper exception logging.

## Context

- The Sentinel codebase currently has no logging. Multiple modules use bare `except: pass` blocks that silently swallow errors.
- The project structure has these Python modules: `analyzer.py`, `api.py`, `cli.py`, `db.py`, `ecosystem.py`, `flywheel.py`, `innovation.py`, `knowledge.py`, `scanner.py`, `scorer.py`, `signals.py`, `validator.py`.
- `sentinel/config.py` may or may not exist. If it exists, read `log_level` from it. If not, default to `INFO`.
- The CLI entry point is `sentinel/cli.py` using Click.

## What To Do

### 1. Set up logging configuration

In `sentinel/__init__.py`, add basic logging setup:

```python
import logging

logging.getLogger("sentinel").addHandler(logging.NullHandler())
```

This ensures the `sentinel` logger exists but doesn't output anything unless configured by the application (CLI or API server).

### 2. Configure logging in CLI entry point

In `sentinel/cli.py`, in the `main()` Click group function, configure logging:

```python
import logging

def main(ctx, use_json):
    ...
    # Configure logging
    log_level = "INFO"
    try:
        from sentinel.config import get_config
        log_level = get_config().log_level
    except (ImportError, Exception):
        pass
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
```

### 3. Add loggers to each module

At the top of each module, add:

```python
import logging
logger = logging.getLogger(__name__)
```

This creates loggers like `sentinel.analyzer`, `sentinel.db`, etc.

### 4. Add log statements at appropriate levels

**INFO level (normal operations):**
- `analyzer.py`: Log when analysis starts (`"Analyzing job: {title} from {company}"`) and completes with result (`"Analysis complete: score={score:.2f} risk={risk_level}"`).
- `api.py`: Log incoming requests to each endpoint (`"POST /api/analyze request received"`).
- `cli.py`: Log command invocations at the `main` group level.
- `flywheel.py`: Log `run_cycle()` start and completion with metrics.
- `knowledge.py`: Log `report_scam()` calls and `seed_default_patterns()` count.
- `innovation.py`: Log strategy selection and execution results.

**WARNING level (recoverable issues):**
- `analyzer.py`: Log when AI escalation fails (`"AI escalation failed: {exc}"`).
- `db.py`: Log when database operations fail in non-critical paths.
- `ecosystem.py`: Log when ecosystem publish operations fail (`"Failed to publish observation: {exc}"`).
- `validator.py`: Log when WHOIS lookup or LinkedIn check fails.

**DEBUG level (detailed internals):**
- `signals.py`: Log each signal extraction result (`"Signal {name}: detected={bool}"`).
- `scorer.py`: Log scoring math (`"Scoring {n} signals: log_odds={lo:.3f} -> score={score:.3f}"`).
- `scanner.py`: Log salary/location/experience extraction results.

### 5. Replace all bare except blocks

Find every instance of:
```python
except Exception:
    pass
```
or:
```python
except:
    pass
```

Replace with:
```python
except Exception:
    logger.exception("Description of what failed")
```

Specific locations to fix (search the codebase for `except.*pass`):
- `analyzer.py`: DB save in `analyze_job` (line ~101 area), and inside `_escalate_to_ai`.
- `api.py`: DB save in `analyze_endpoint` (line ~179), DB read in `report_endpoint` (line ~206).
- `cli.py`: DB save in `analyze` command (line ~121), DB read in `report` command (line ~276).
- `ecosystem.py`: File write in `publish_observation` and `publish_event`.
- `innovation.py`: State load/save in `_load_state` and `_save_state`.
- `scanner.py`: URL fetch in `parse_job_url` (line ~542).
- `validator.py`: Domain check and LinkedIn check in `validate_company`.

### 6. Add tests in `tests/test_logging.py`

Write tests:
- Verify that `logging.getLogger("sentinel")` exists and has `NullHandler`.
- Verify that `logging.getLogger("sentinel.analyzer")` is a child of `sentinel`.
- Use `caplog` pytest fixture to verify that analyzing a job produces INFO log messages.
- Verify that a simulated failure in ecosystem publish produces a WARNING log message.
- Verify that signal extraction produces DEBUG log messages when log level is DEBUG.

## Acceptance Criteria

- [ ] Every module has `logger = logging.getLogger(__name__)` at module level.
- [ ] INFO logs cover: analysis requests, scam detections, report submissions, flywheel cycles.
- [ ] WARNING logs cover: AI failures, DB errors, ecosystem publish failures.
- [ ] DEBUG logs cover: signal extraction details, scoring math.
- [ ] All bare `except: pass` and `except Exception: pass` blocks are replaced with `logger.exception(...)` or `logger.warning(...)`.
- [ ] `sentinel/__init__.py` registers a NullHandler on the root `sentinel` logger.
- [ ] CLI configures logging with basicConfig.
- [ ] All existing tests pass.

## Constraints

- Use only `logging` from stdlib. No structlog, loguru, or other libraries.
- Do not change any function signatures or return values.
- Log messages should be concise and useful, not verbose dumps.
- Use `logger.exception()` (not `logger.error()`) when inside an except block -- it automatically includes the traceback.
- In tests, use `caplog` fixture for log capture.

## Test Command

```bash
python -m pytest tests/test_logging.py tests/test_core.py tests/test_advanced.py -v
```
