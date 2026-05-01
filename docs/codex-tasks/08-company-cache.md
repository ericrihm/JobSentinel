# Task: Cache Validated Companies in Database

## Task

Make `validator.py` persist company validation results to SQLite and check the cache before re-validating.

## Context

- `sentinel/validator.py` has `validate_company(name)` which does WHOIS lookups and LinkedIn checks on every call.
- `sentinel/db.py` has a `companies` table but `validate_company` never reads or writes to it.
- The DB has `save_company()` and `get_company()` methods (verify they exist; create if not).
- Lookups are slow (network I/O) and LinkedIn may rate-limit repeated checks.

## What To Do

### 1. Verify/add DB methods

In `sentinel/db.py`, ensure these methods exist:
- `save_company(data: dict) -> None` — upsert into companies table
- `get_company(name: str) -> dict | None` — lookup by name

If they don't exist, add them with parameterized SQL.

### 2. Add caching to `validate_company`

In `sentinel/validator.py`:

```python
def validate_company(name: str, force_refresh: bool = False) -> CompanyProfile:
    # 1. Check cache (unless force_refresh)
    if not force_refresh:
        cached = _get_cached_company(name)
        if cached and _is_fresh(cached, max_age_days=7):
            return cached

    # 2. Do actual validation (existing logic)
    profile = _do_validation(name)

    # 3. Cache the result
    _cache_company(profile)

    return profile
```

### 3. Add `--refresh` flag to CLI

In `sentinel/cli.py`, add `--refresh` flag to the `validate` command:
```python
@click.option("--refresh", is_flag=True, help="Force re-validation, bypass cache")
```

### 4. Add tests

- Test cache hit: validate, then validate again — second call should not make HTTP requests
- Test cache miss: first call goes through full validation
- Test TTL expiry: mock a stale cached entry, verify re-validation happens
- Test `--refresh` flag: verify it bypasses cache

## Acceptance Criteria

- [ ] `validate_company("Google")` caches result to SQLite
- [ ] Subsequent calls within 7 days return cached result
- [ ] `force_refresh=True` bypasses cache
- [ ] CLI `--refresh` flag works
- [ ] All tests pass: `python -m pytest tests/ -v`

## Constraints

- Import SentinelDB inside validate_company (keep it optional — validator should still work without DB)
- Use `last_checked` column for TTL comparison
- Do not change the CompanyProfile dataclass

## Test Command

```bash
python -m pytest tests/ -v
```
