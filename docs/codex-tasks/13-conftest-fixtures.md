# Task: Build Shared Test Fixtures

## Task

Build shared pytest fixtures in `tests/conftest.py` and update existing tests to use them where they duplicate setup.

## Context

- `tests/conftest.py` currently has only a placeholder fixture:
  ```python
  @pytest.fixture
  def sample_sentinel():
      """Override with project-specific fixture."""
      return {}
  ```
- Existing tests in `test_core.py` and `test_advanced.py` repeatedly create `JobPosting` objects and `SentinelDB` instances with `tmp_path`.
- The project uses `sentinel.models.JobPosting`, `sentinel.models.ScamSignal`, `sentinel.models.ValidationResult`, `sentinel.models.ScamPattern`, `sentinel.models.SignalCategory`, `sentinel.models.RiskLevel`.
- `sentinel.db.SentinelDB` takes a `path` parameter for the SQLite file.
- `sentinel.knowledge.KnowledgeBase` takes an optional `db` parameter and has `seed_default_patterns()`.

## What To Do

### 1. Build fixtures in `tests/conftest.py`

Replace the placeholder content with these fixtures:

**`sample_job_posting`** -- returns a realistic, legitimate JobPosting:
```python
@pytest.fixture
def sample_job_posting():
    """A realistic, legitimate job posting."""
    from sentinel.models import JobPosting
    return JobPosting(
        url="https://linkedin.com/jobs/view/12345",
        title="Senior Backend Engineer",
        company="Acme Technologies",
        company_linkedin_url="https://linkedin.com/company/acme-technologies",
        company_size="5000",
        location="San Francisco, CA",
        description=(
            "We are looking for a Senior Backend Engineer with 5+ years of experience "
            "in Python, Go, or Java. You will design and build scalable microservices "
            "using PostgreSQL, Redis, and Kubernetes. Must have a bachelor's degree in "
            "Computer Science or equivalent. We offer health insurance, dental, vision, "
            "401k matching, equity, and generous PTO."
        ),
        salary_min=160000.0,
        salary_max=220000.0,
        experience_level="senior",
        employment_type="full-time",
        is_remote=False,
        posted_date="2025-04-15",
    )
```

**`scam_job_posting`** -- returns an obvious scam JobPosting:
```python
@pytest.fixture
def scam_job_posting():
    """An obvious scam job posting with multiple red flags."""
    from sentinel.models import JobPosting
    return JobPosting(
        url="https://linkedin.com/jobs/view/99999",
        title="Work From Home Data Entry",
        company="",
        company_linkedin_url="",
        location="Remote",
        description=(
            "Earn guaranteed income of $500 per day! No experience needed. "
            "You must pay a registration fee of $49.99 upfront to get started. "
            "Send your SSN and bank account number to apply. "
            "Apply now! Only 5 spots remaining! "
            "Contact us at recruiter@gmail.com for more info. "
            "Payment via Bitcoin only."
        ),
        salary_min=0.0,
        salary_max=0.0,
        is_remote=True,
        is_repost=True,
    )
```

**`ghost_job_posting`** -- returns a stale/reposted ghost job:
```python
@pytest.fixture
def ghost_job_posting():
    """A stale, reposted ghost job posting."""
    from sentinel.models import JobPosting
    return JobPosting(
        url="https://linkedin.com/jobs/view/55555",
        title="Software Engineer",
        company="Zombie Corp",
        company_linkedin_url="https://linkedin.com/company/zombie-corp",
        location="New York, NY",
        description=(
            "We are always looking for talented engineers to join our team. "
            "General duties include software development and other tasks as needed."
        ),
        posted_date="2024-01-01",
        is_repost=True,
    )
```

**`temp_db`** -- creates a temporary SentinelDB, yields it, cleans up:
```python
@pytest.fixture
def temp_db(tmp_path):
    """A temporary SentinelDB instance that is cleaned up after the test."""
    from sentinel.db import SentinelDB
    db_path = str(tmp_path / "test_sentinel.db")
    db = SentinelDB(db_path)
    yield db
    db.close()
```

**`seeded_db`** -- temp_db with seed_default_patterns() called:
```python
@pytest.fixture
def seeded_db(tmp_path):
    """A temporary SentinelDB seeded with default scam patterns."""
    from sentinel.db import SentinelDB
    from sentinel.knowledge import KnowledgeBase
    db_path = str(tmp_path / "seeded_sentinel.db")
    db = SentinelDB(db_path)
    kb = KnowledgeBase(db)
    kb.seed_default_patterns()
    yield db
    db.close()
```

**`mock_httpx`** -- patches httpx.Client to return configurable responses:
```python
@pytest.fixture
def mock_httpx(monkeypatch):
    """Patches httpx.Client to return configurable responses.
    
    Usage:
        def test_something(mock_httpx):
            mock_httpx(status_code=200, text="<html>...</html>")
    """
    class MockResponse:
        def __init__(self, status_code=200, text="", json_data=None):
            self.status_code = status_code
            self.text = text
            self._json_data = json_data
        def json(self):
            return self._json_data or {}
        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")
    
    class MockClient:
        def __init__(self, response):
            self._response = response
        def get(self, url, **kwargs):
            return self._response
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    def _configure(status_code=200, text="", json_data=None):
        response = MockResponse(status_code, text, json_data)
        try:
            import httpx
            monkeypatch.setattr(httpx, "Client", lambda **kwargs: MockClient(response))
        except ImportError:
            pass
        return response
    
    return _configure
```

### 2. Update existing tests to use fixtures

Scan `tests/test_core.py` for DB tests that manually create `SentinelDB(str(tmp_path / "test.db"))` and replace with the `temp_db` fixture. The DB tests in `TestDB` class all follow this pattern:

```python
def test_some_db_thing(self, tmp_path):
    from sentinel.db import SentinelDB
    db_path = str(tmp_path / "test.db")
    with SentinelDB(db_path) as db:
        ...
```

These can be refactored to use `temp_db`:

```python
def test_some_db_thing(self, temp_db):
    temp_db.save_job({...})
    result = temp_db.get_job(...)
    ...
```

**Important:** The existing tests use `with SentinelDB(...) as db:` context manager. The `temp_db` fixture handles cleanup in the fixture teardown, so tests using it should NOT use `with` blocks. Be careful to preserve all assertions and test logic.

Specific tests to update in `TestDB`:
- `test_save_and_get_job_round_trip`
- `test_get_job_not_found`
- `test_search_jobs_fts`
- `test_save_report`
- `test_get_stats_empty`
- `test_get_stats_with_data`

Specific tests to update in `TestKnowledge`:
- `test_seed_default_patterns_populates_db`
- `test_seed_default_patterns_idempotent`
- `test_report_scam_saves_correctly`
- `test_search_returns_matching_jobs`
- `test_get_accuracy_stats_empty`
- `test_get_accuracy_stats_perfect_predictions`

### 3. Add fixture validation tests

Add `tests/test_fixtures.py` with tests that validate the fixtures themselves:

- `sample_job_posting` has populated title, company, description, salary_min, salary_max.
- `scam_job_posting` has empty company, scam language in description.
- `ghost_job_posting` has `is_repost=True` and old `posted_date`.
- `temp_db` is writable (save a job, read it back).
- `seeded_db` has default patterns loaded (count matches `_DEFAULT_PATTERNS`).
- `mock_httpx` returns the configured status code and text.

## Acceptance Criteria

- [ ] `conftest.py` defines all 6 fixtures: `sample_job_posting`, `scam_job_posting`, `ghost_job_posting`, `temp_db`, `seeded_db`, `mock_httpx`.
- [ ] Fixtures provide realistic, reusable test data.
- [ ] Existing DB tests in `test_core.py` are updated to use `temp_db` fixture where applicable.
- [ ] All existing test assertions pass without modification.
- [ ] `test_fixtures.py` validates each fixture works correctly.
- [ ] The old `sample_sentinel` placeholder fixture is removed.

## Constraints

- Do not change any test assertions or expected values. Only change setup code.
- Keep `tmp_path` as the underlying isolation mechanism (via `temp_db` fixture).
- The `mock_httpx` fixture must not require httpx to be installed -- handle ImportError gracefully.
- Remove the existing `sample_sentinel` placeholder fixture.
- Do not add any new dependencies.

## Test Command

```bash
python -m pytest tests/ -v
```
