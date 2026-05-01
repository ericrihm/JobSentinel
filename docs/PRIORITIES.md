# JobSentinel — Build Priorities

## Priority Lanes (highest → lowest)

### 1. Security Hardening
- Input validation on all API request fields (URL format, string length, no injection)
- Parameterized SQL queries (verify db.py, no f-string SQL)
- Sanitize HTML before processing in scanner.py
- Validate domain names before subprocess calls in validator.py (prevent command injection)
- API rate limiting + optional API key auth

### 2. Correctness / Bug Fixes
- Wire flywheel learned weights back into scorer.py (currently disconnected)
- Fix flywheel_metrics table schema mismatch (missing columns)
- Fix date parsing for real LinkedIn data (ISO 8601 with timezone, relative dates)
- Fix salary heuristic edge case (bare numbers < 1000 wrongly scaled)

### 3. Test Coverage
- flywheel.py: CUSUMDetector, learn_from_report, evolve_patterns, run_cycle, get_health — zero tests
- api.py: All 5 endpoints via TestClient — zero tests
- cli.py: Key commands via CliRunner — zero tests
- ecosystem.py: publish functions — zero tests
- Target: 90%+ module coverage, 200+ total tests

### 4. Core Features
- config.py module (TOML config loading, settings dataclass)
- Batch analysis (--file flag in CLI)
- scan command (LinkedIn search scraping)
- Company validation caching (persist to DB, 7-day TTL)
- Browser extension scaffold (Chrome MV3)

### 5. Innovation Engine
- Implement stub strategies: _correlate_signals, _expand_keywords, _mine_patterns
- Add structured logging throughout codebase

### 6. Polish
- pyproject.toml optional dependency groups
- Shared test fixtures in conftest.py
- Proper error messages for missing optional deps
