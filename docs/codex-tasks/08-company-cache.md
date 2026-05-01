# Task: Cache Validated Companies in DB

## Task

Add caching to `validator.py`'s `validate_company()` so it checks the DB first and only performs external lookups (WHOIS, LinkedIn) when the cached result is stale or missing.

## Context

- `sentinel/validator.py` defines `validate_company(company_name, domain) -> CompanyProfile`.
- Every call currently performs fresh WHOIS + LinkedIn checks, which are slow (network I/O) and unnecessary for recently-validated companies.
- `sentinel/db.py` already has `save_company(company_data)` and `get_company(name) -> dict | None`.
- The `companies` table schema includes: `name TEXT PRIMARY KEY`, `domain TEXT`, `employee_count INTEGER`, `is_verified INTEGER DEFAULT 0`, `linkedin_url TEXT`, `glassdoor_rating REAL`, `whois_age_days INTEGER DEFAULT 0`, `last_checked TEXT`.
- `sentinel/cli.py` has a `validate` command at line ~191 that calls `validate_company(company_name, domain=domain)`.
- `sentinel/models.py` defines `CompanyProfile` with fields: `name`, `linkedin_url`, `website`, `domain`, `employee_count`, `founded_year`, `industry`, `glassdoor_rating`, `is_verified`, `whois_age_days`, `has_linkedin_page`, `linkedin_followers`, `verification_source`.

## What To Do

### 1. Add cache check at the start of `validate_company()`

Modify `validate_company()` in `sentinel/validator.py` to accept a `refresh` parameter:

```python
def validate_company(company_name: str, domain: str = "", refresh: bool = False) -> CompanyProfile:
```

At the start of the function, before any external checks:

1. Import `SentinelDB` from `sentinel.db` (inside the function, to keep it optional).
2. Try to load cached data: `db.get_company(company_name)`.
3. If cached data exists and `refresh` is False:
   - Parse `last_checked` timestamp from the cached dict.
   - If `last_checked` is within the last 7 days, convert the cached dict to a `CompanyProfile` and return it immediately.
   - If `last_checked` is older than 7 days, proceed with fresh validation.
4. If no cached data exists or `refresh` is True, proceed with the current validation logic.

### 2. Save validation results to DB after fresh validation

After the existing validation logic completes (at the end of the function), save the result:

```python
try:
    from sentinel.db import SentinelDB
    db = SentinelDB()
    db.save_company({
        "name": profile.name,
        "domain": profile.domain,
        "employee_count": profile.employee_count,
        "is_verified": profile.is_verified,
        "linkedin_url": getattr(profile, "linkedin_url", ""),
        "glassdoor_rating": profile.glassdoor_rating,
        "whois_age_days": profile.whois_age_days,
        "last_checked": _now_iso(),
    })
    db.close()
except Exception:
    pass  # best-effort caching
```

### 3. Add helper functions

Add at module level in `validator.py`:

```python
from datetime import datetime, timezone, timedelta

CACHE_TTL_DAYS = 7

def _is_cache_fresh(last_checked: str) -> bool:
    """Return True if last_checked is within CACHE_TTL_DAYS."""
    if not last_checked:
        return False
    try:
        checked_dt = datetime.fromisoformat(last_checked)
        if checked_dt.tzinfo is None:
            checked_dt = checked_dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - checked_dt) < timedelta(days=CACHE_TTL_DAYS)
    except (ValueError, TypeError):
        return False

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _cached_to_profile(cached: dict) -> CompanyProfile:
    """Convert a DB company dict to a CompanyProfile."""
    return CompanyProfile(
        name=cached.get("name", ""),
        linkedin_url=cached.get("linkedin_url", ""),
        domain=cached.get("domain", ""),
        employee_count=cached.get("employee_count", 0),
        is_verified=bool(cached.get("is_verified", False)),
        whois_age_days=cached.get("whois_age_days", 0),
        has_linkedin_page=bool(cached.get("linkedin_url", "")),
        linkedin_followers=0,
        glassdoor_rating=cached.get("glassdoor_rating", 0.0),
        verification_source="cached",
    )
```

### 4. Add `--refresh` flag to CLI validate command

In `sentinel/cli.py`, modify the `validate` command at line ~191:

Add the option:
```python
@click.option("--refresh", is_flag=True, default=False, help="Force re-validation (ignore cache).")
```

Pass `refresh=refresh` to `validate_company()`.

When cache is used (verification_source is "cached"), add a note to the output:
```
  [VERIFIED]  Google
  Source:     cached
  (cached -- last checked 2 days ago)
```

### 5. Add tests in `tests/test_company_cache.py`

Write tests:

- **Cache miss (first call):**
  - Call `validate_company("Google")` with a fresh temp DB (monkeypatch DB path).
  - Verify the result is returned (from known_companies_list).
  - Verify the company is now saved in the DB.

- **Cache hit (second call):**
  - Call `validate_company("Google")` twice.
  - Monkeypatch `check_company_linkedin` to track call count.
  - Verify LinkedIn check is NOT called on the second invocation (cache hit).

- **Cache TTL expiry:**
  - Save a company to DB with `last_checked` set to 10 days ago.
  - Call `validate_company()` and verify it performs fresh validation.

- **Refresh flag:**
  - Save a company to DB with fresh `last_checked`.
  - Call `validate_company(..., refresh=True)`.
  - Verify fresh validation is performed despite cache being fresh.

- **`_is_cache_fresh()` unit tests:**
  - Fresh timestamp (1 day ago) returns True.
  - Stale timestamp (10 days ago) returns False.
  - Empty string returns False.
  - Invalid string returns False.

- **`_cached_to_profile()` unit test:**
  - Convert a dict to CompanyProfile, verify all fields.

- **CLI with `--refresh`:**
  - Use Click `CliRunner` to run `sentinel validate Google --refresh`.
  - Verify it completes without error.

## Acceptance Criteria

- [ ] `validate_company()` checks DB cache before performing external lookups.
- [ ] Cached results younger than 7 days are returned without network calls.
- [ ] Cached results older than 7 days trigger fresh validation.
- [ ] `--refresh` flag on CLI forces re-validation.
- [ ] Validation results are persisted to DB after every fresh validation.
- [ ] `_is_cache_fresh()` correctly handles fresh, stale, empty, and invalid timestamps.
- [ ] All existing tests pass.

## Constraints

- DB access is best-effort -- failures in cache read/write must not break validation.
- Do not change the return type of `validate_company()`.
- The `refresh` parameter should default to `False` for backward compatibility.
- `CACHE_TTL_DAYS = 7` should be a module-level constant.
- Use `datetime.fromisoformat()` for parsing ISO timestamps (Python 3.11+).
- Do not change the `CompanyProfile` dataclass in `models.py`.

## Test Command

```bash
python -m pytest tests/test_company_cache.py tests/test_core.py -v
```
