# Task: Security Hardening Audit and Fixes

## Task

Audit and fix security vulnerabilities across the codebase: input validation, SQL injection prevention, command injection prevention, and HTML sanitization.

## Context

- `sentinel/api.py` — CORS allows all origins, no input validation on request fields, no rate limiting
- `sentinel/db.py` — Uses parameterized queries in most places but needs verification
- `sentinel/scanner.py` — Processes raw HTML without sanitization
- `sentinel/validator.py` — Passes domain names directly to `subprocess.run(["whois", domain])` which could allow command injection

## What To Do

### 1. Input validation in `api.py`

Add Pydantic validators to request models:

```python
class AnalyzeRequest(BaseModel):
    text: Optional[str] = Field(None, max_length=50000)
    url: Optional[str] = Field(None, max_length=2048)
    title: str = Field("", max_length=500)
    company: str = Field("", max_length=500)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

class ReportRequest(BaseModel):
    url: str = Field(..., max_length=2048)
    reason: str = Field("", max_length=5000)
```

### 2. SQL verification in `db.py`

Audit every SQL query. Ensure:
- All values use `?` parameter placeholders
- No f-strings or string concatenation in SQL
- No user input in table/column names

### 3. Command injection prevention in `validator.py`

In `check_domain_age()`:

```python
import re
_DOMAIN_RE = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$')

def check_domain_age(domain: str) -> int:
    if not _DOMAIN_RE.match(domain):
        return 0  # reject suspicious domain strings
    # ... existing subprocess.run code
```

### 4. HTML sanitization in `scanner.py`

Before processing HTML in `parse_job_html()`, strip potentially dangerous elements:

```python
import re
def _sanitize_html(html: str) -> str:
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'on\w+\s*=\s*"[^"]*"', '', html, flags=re.IGNORECASE)
    return html
```

### 5. Add security-focused tests

Create `tests/test_security.py`:

```python
class TestInputValidation:
    def test_analyze_rejects_oversized_text(self): ...
    def test_analyze_rejects_invalid_url(self): ...
    def test_report_rejects_oversized_reason(self): ...

class TestCommandInjection:
    def test_domain_validation_rejects_shell_metacharacters(self): ...
    def test_domain_validation_accepts_valid_domains(self): ...

class TestSQLInjection:
    def test_save_job_with_sql_in_title(self): ...
    def test_search_with_sql_in_query(self): ...

class TestHTMLSanitization:
    def test_script_tags_removed(self): ...
    def test_event_handlers_removed(self): ...
```

## Acceptance Criteria

- [ ] API rejects text > 50,000 chars with 422
- [ ] API rejects invalid URLs with 422
- [ ] Domain validation rejects strings with shell metacharacters (;, |, &, $, `)
- [ ] No SQL injection possible via any user-facing input
- [ ] HTML processing strips `<script>` and event handlers
- [ ] All tests pass: `python -m pytest tests/ -v`

## Constraints

- Use Pydantic's built-in validation (Field max_length, validators)
- Do not add external security libraries
- Keep changes backward-compatible
- Security tests should prove the vulnerability is actually prevented, not just test the happy path

## Test Command

```bash
python -m pytest tests/ -v
```
