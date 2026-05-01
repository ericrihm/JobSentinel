# Task: Security Audit and Fixes

## Task

Audit and fix security vulnerabilities across the Sentinel codebase: input validation in the API, SQL injection prevention, HTML sanitization, command injection prevention, and CLI input sanitization.

## Context

- `sentinel/api.py` -- FastAPI endpoints accept user input (URLs, text, company names) with no validation on length or format. CORS is `allow_origins=["*"]`.
- `sentinel/db.py` -- SQLite queries mostly use parameterized statements but need comprehensive verification. The FTS5 `search_jobs()` method constructs queries from user input.
- `sentinel/scanner.py` -- Processes raw HTML input. The `_strip_html()` function removes tags but doesn't sanitize for XSS-relevant patterns like `javascript:` URIs.
- `sentinel/validator.py` -- `check_domain_age()` passes user-supplied domain names to `subprocess.run(["whois", domain])` -- this is a command injection vector if the domain contains shell metacharacters.
- `sentinel/cli.py` -- Takes user input from command line arguments and passes to analyzers/validators.

## What To Do

### 1. Input validation in `sentinel/api.py`

Add Pydantic validators to request models inside `create_app()`:

**URL validation on `AnalyzeRequest`:**
```python
from pydantic import field_validator

class AnalyzeRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw job description text.", max_length=50000)
    url: Optional[str] = Field(None, description="LinkedIn job posting URL.", max_length=2048)
    job_data: Optional[dict] = Field(None, description="Structured job dict (JSON).")
    title: str = Field("", description="Job title.", max_length=500)
    company: str = Field("", description="Company name.", max_length=200)
    use_ai: bool = Field(True, description="Enable AI escalation.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if v is None:
            return v
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        lower = v.lower()
        if lower.startswith(("javascript:", "data:", "vbscript:")):
            raise ValueError("Invalid URL scheme")
        return v[:2048]
```

**ReportRequest validation:**
```python
class ReportRequest(BaseModel):
    url: str = Field(..., description="Job posting URL.", max_length=2048)
    is_scam: bool = Field(..., description="True if scam.")
    reason: str = Field("", description="Optional explanation.", max_length=2000)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v[:2048]
```

**Add text sanitization function** (inside `create_app()` or at module level):
```python
import re as _re

def _sanitize_text(text: str) -> str:
    """Remove potentially dangerous HTML/script content from user input."""
    if not text:
        return text
    text = _re.sub(r'<script[^>]*>.*?</script>', '', text, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=_re.IGNORECASE)
    return text
```

Apply `_sanitize_text()` to `req.text`, `req.title`, `req.company` in the `analyze_endpoint` handler, and to `req.reason` in the `report_endpoint` handler.

### 2. Verify and harden SQL in `sentinel/db.py`

Audit every SQL query in the file. Confirm all use parameterized statements. Specifically:

**Harden `search_jobs()`:**
The FTS5 MATCH query currently only escapes double quotes. Add sanitization of FTS5 operators:

```python
import re

def search_jobs(self, query: str, limit: int = 20) -> list[dict]:
    safe_query = query.replace('"', '""')
    # Strip FTS5 special operators to prevent query injection
    safe_query = re.sub(r'\b(AND|OR|NOT|NEAR)\b', '', safe_query, flags=re.IGNORECASE)
    safe_query = safe_query.replace('*', '').replace('^', '')
    safe_query = safe_query.strip()
    if not safe_query:
        return []
    ...
```

Add `import re` to the top of `db.py` if not already present.

**Verify no f-string SQL:** Search the entire file for any SQL constructed with f-strings or string concatenation. Confirm there are none (the current code looks clean but verify).

### 3. HTML sanitization in `sentinel/scanner.py`

Add a sanitization function after `_strip_html()`:

```python
def _sanitize_extracted(text: str) -> str:
    """Sanitize text extracted from HTML to prevent XSS if displayed."""
    if not text:
        return text
    text = text.replace('\x00', '')
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'data:text/html', '', text, flags=re.IGNORECASE)
    return text
```

Call `_sanitize_extracted()` on outputs of `_strip_html()` in `parse_job_html()` -- specifically on the extracted title, company, and description values at the end of Pass 2 and Pass 3.

### 4. Command injection prevention in `sentinel/validator.py`

Add domain validation before the subprocess call:

```python
import re

_VALID_DOMAIN_RE = re.compile(
    r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?'
    r'(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
)

_SHELL_METACHARACTERS = set(';|&`$(){}[]<>\n\r\t\\\'\"')

def _validate_domain(domain: str) -> str | None:
    """Validate and sanitize a domain name. Returns None if invalid."""
    domain = domain.strip().lower().removeprefix("www.")
    if not domain:
        return None
    if len(domain) > 253:
        return None
    if any(c in _SHELL_METACHARACTERS for c in domain):
        return None
    if not _VALID_DOMAIN_RE.match(domain):
        return None
    return domain
```

Update `check_domain_age()` to use validation:
```python
def check_domain_age(domain: str) -> int:
    validated = _validate_domain(domain)
    if validated is None:
        return 0
    # ... rest uses validated instead of domain
    # Also make shell=False explicit:
    result = subprocess.run(
        ["whois", validated],
        capture_output=True,
        text=True,
        timeout=10,
        shell=False,
    )
```

### 5. CLI input sanitization in `sentinel/cli.py`

Add basic validation in CLI command handlers:

**In `analyze` command:**
```python
if is_url and not input_text.startswith(("http://", "https://")):
    click.echo(click.style("Error: URL must start with http:// or https://", fg="red"), err=True)
    sys.exit(1)
```

**In `validate` command:**
```python
if len(company_name) > 200:
    click.echo(click.style("Error: Company name too long (max 200 chars)", fg="red"), err=True)
    sys.exit(1)
```

**In `report` command:**
```python
if not url.startswith(("http://", "https://")):
    click.echo(click.style("Error: URL must start with http:// or https://", fg="red"), err=True)
    sys.exit(1)
```

### 6. Add security-focused tests in `tests/test_security.py`

**API input validation tests** (skip if FastAPI not installed):
- POST `/api/analyze` with `url="javascript:alert(1)"` returns 422.
- POST `/api/analyze` with `url="data:text/html,<script>alert(1)</script>"` returns 422.
- POST `/api/analyze` with text containing `<script>alert('xss')</script>` -- the response should not contain the script tag.
- POST `/api/analyze` with extremely long text (100,000+ chars) returns 422.
- POST `/api/report` with invalid URL (not http/https) returns 422.
- POST `/api/report` with very long reason (5,000+ chars) returns 422.

**SQL injection prevention tests:**
- `db.search_jobs('"; DROP TABLE jobs; --')` does not crash or drop tables. Verify the jobs table still exists afterward.
- `db.search_jobs("test AND 1=1")` sanitizes FTS5 operators (returns results or empty, no error).
- `db.get_company("'; DROP TABLE companies; --")` does not crash.
- `db.save_job({"url": "test", "title": "'; DROP TABLE jobs;--", ...})` succeeds and data is stored literally.

**HTML sanitization tests:**
- `_strip_html("<script>alert(1)</script>hello")` removes the script tag.
- `_sanitize_extracted("click javascript:alert(1)")` removes `javascript:`.
- `parse_job_text()` with input containing `javascript:` URIs sanitizes them.

**Command injection prevention tests:**
- `_validate_domain("google.com")` returns `"google.com"`.
- `_validate_domain("example.co.uk")` returns `"example.co.uk"`.
- `_validate_domain("google.com; rm -rf /")` returns `None`.
- `_validate_domain("$(whoami).evil.com")` returns `None`.
- `_validate_domain("google.com | cat /etc/passwd")` returns `None`.
- `_validate_domain("evil.com\nrm -rf /")` returns `None`.
- `_validate_domain("")` returns `None`.
- `_validate_domain("a" * 300)` returns `None` (too long).
- `check_domain_age("evil.com; rm -rf /")` returns 0 (validation catches it before subprocess).

**CLI input validation tests** (using CliRunner):
- `sentinel analyze "javascript:alert(1)"` shows URL error.
- `sentinel validate "A" * 300` shows length error.
- `sentinel report "not-a-url"` shows URL error.

## Acceptance Criteria

- [ ] API endpoints validate URL format (must start with http/https), reject dangerous schemes.
- [ ] API endpoints enforce text length limits via Pydantic `max_length`.
- [ ] Text inputs to API are sanitized (script tags, event handlers stripped).
- [ ] All SQL queries in db.py use parameterized statements (verified, no f-strings).
- [ ] FTS5 search input is sanitized against operator injection.
- [ ] HTML extracted from job postings is sanitized (javascript: URIs removed).
- [ ] Domain names are validated before passing to subprocess (shell metacharacters rejected).
- [ ] `subprocess.run` explicitly uses `shell=False`.
- [ ] CLI validates URL format and input length.
- [ ] Security tests cover all 5 attack vectors with both positive and negative test cases.
- [ ] All existing tests still pass.

## Constraints

- Do not add new dependencies. Use stdlib `re` for sanitization.
- Do not break existing functionality -- legitimate inputs must still work.
- Pydantic validators should return helpful error messages.
- Sanitization should be silent (strip bad content) rather than rejecting entire requests, except for clearly malformed inputs (invalid URL schemes).
- The `subprocess.run` call must use `shell=False` (the default, but make it explicit for clarity).

## Test Command

```bash
python -m pytest tests/test_security.py tests/test_core.py tests/test_advanced.py -v
```
