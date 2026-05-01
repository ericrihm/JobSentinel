# Task: Fix Date Parsing for Real LinkedIn Data

## Task

Fix `_days_since_posted()` in `sentinel/signals.py` to handle timezone-aware ISO 8601 dates and relative date strings that LinkedIn uses in search results.

## Context

- `sentinel/signals.py` defines `_days_since_posted(posted_date: str) -> int | None` at line ~101.
- Currently it only handles 3 formats via `strptime`: `"%Y-%m-%d"`, `"%Y-%m-%dT%H:%M:%S"`, `"%Y-%m-%dT%H:%M:%SZ"`.
- Real LinkedIn data uses:
  - ISO 8601 with timezone offset: `"2024-03-15T10:30:00+00:00"`, `"2024-03-15T10:30:00+05:30"`, `"2024-03-15T10:30:00-08:00"`.
  - ISO 8601 with milliseconds: `"2024-03-15T10:30:00.000Z"`, `"2024-03-15T10:30:00.000+00:00"`.
  - Relative date strings from search results: `"2 weeks ago"`, `"3 days ago"`, `"1 month ago"`, `"just now"`, `"today"`, `"yesterday"`, `"1 hour ago"`, `"30+ days ago"`.
- This function is used by `check_stale_posting()` (line ~338) which detects ghost jobs (posted > 30 or > 60 days with no activity). Incorrect parsing means ghost jobs are missed.
- Python 3.11+ `datetime.fromisoformat()` handles most ISO 8601 formats natively, including timezone offsets.

## What To Do

### 1. Add `_parse_relative_date()` helper to `sentinel/signals.py`

Add near the top of the file (after the existing regex definitions):

```python
_RELATIVE_DATE_RE = re.compile(
    r"(\d+)\+?\s*(second|minute|hour|day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)

def _parse_relative_date(text: str) -> int | None:
    """Parse relative date strings like '3 days ago', '2 weeks ago'.
    
    Returns number of days, or None if not a relative date.
    """
    text = text.strip().lower()
    
    if text in ("just now", "today", "just posted", "moments ago"):
        return 0
    if text == "yesterday":
        return 1
    
    m = _RELATIVE_DATE_RE.match(text)
    if not m:
        return None
    
    count = int(m.group(1))
    unit = m.group(2).lower()
    
    multipliers = {
        "second": 0,
        "minute": 0,
        "hour": 0,
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365,
    }
    
    return count * multipliers.get(unit, 0)
```

### 2. Rewrite `_days_since_posted()` in `sentinel/signals.py`

Replace the current implementation (lines ~101-110) with:

```python
def _days_since_posted(posted_date: str) -> int | None:
    """Calculate days since a job was posted.
    
    Handles:
    - ISO 8601: "2024-03-15", "2024-03-15T10:30:00Z", "2024-03-15T10:30:00+00:00"
    - ISO 8601 with milliseconds: "2024-03-15T10:30:00.000Z"
    - Relative dates: "3 days ago", "2 weeks ago", "1 month ago"
    - Special strings: "just now", "today", "yesterday", "30+ days ago"
    """
    if not posted_date:
        return None
    
    text = posted_date.strip()
    
    # Try relative date parsing first
    relative = _parse_relative_date(text)
    if relative is not None:
        return relative
    
    # Try ISO 8601 via fromisoformat (Python 3.11+ handles timezone offsets)
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - dt).days)
    except (ValueError, TypeError):
        pass
    
    # Handle Z suffix (fromisoformat in 3.11 handles this, but be safe)
    if text.endswith("Z"):
        try:
            dt = datetime.fromisoformat(text[:-1] + "+00:00")
            return max(0, (datetime.now(timezone.utc) - dt).days)
        except (ValueError, TypeError):
            pass
    
    # Fallback: try common date formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return max(0, (datetime.now(timezone.utc) - dt).days)
        except ValueError:
            continue
    
    return None
```

### 3. Add tests in `tests/test_date_parsing.py`

Write comprehensive tests:

**ISO 8601 formats (use dates far enough in the past to get predictable day counts):**
- `"2024-03-15"` -- plain date, returns a positive int.
- `"2024-03-15T10:30:00"` -- datetime without timezone.
- `"2024-03-15T10:30:00Z"` -- UTC with Z suffix.
- `"2024-03-15T10:30:00+00:00"` -- UTC with offset.
- `"2024-03-15T10:30:00+05:30"` -- non-UTC positive offset.
- `"2024-03-15T10:30:00-08:00"` -- negative offset.
- `"2024-03-15T10:30:00.000Z"` -- with milliseconds and Z.
- `"2024-03-15T10:30:00.000+00:00"` -- milliseconds with offset.
- For all ISO tests, verify the result is a reasonable positive integer (> 300 days for dates in early 2024).

**Relative date formats:**
- `"just now"` -> 0
- `"today"` -> 0
- `"yesterday"` -> 1
- `"3 days ago"` -> 3
- `"1 day ago"` -> 1
- `"2 weeks ago"` -> 14
- `"1 week ago"` -> 7
- `"1 month ago"` -> 30
- `"3 months ago"` -> 90
- `"1 year ago"` -> 365
- `"30+ days ago"` -> 30 (the `+` is handled by regex)
- `"1 hour ago"` -> 0
- `"just posted"` -> 0
- `"moments ago"` -> 0

**Edge cases:**
- Empty string `""` -> None.
- `None`-like input (just test empty string, since the function takes str).
- Random garbage string `"not a date at all"` -> None.
- Very old date `"1990-01-01"` -> large positive number.

**Integration with `check_stale_posting`:**
- Job with `posted_date="45 days ago"` triggers `stale_posting` signal (>30 days).
- Job with `posted_date="2024-01-01T00:00:00+00:00"` (old date) triggers `stale_posting` signal.
- Job with `posted_date="2 days ago"` does NOT trigger `stale_posting` signal.
- Job with `posted_date="just now"` does NOT trigger `stale_posting` signal.

## Acceptance Criteria

- [ ] `_days_since_posted()` handles all ISO 8601 variants with timezone offsets.
- [ ] `_days_since_posted()` handles relative date strings ("X days/weeks/months ago").
- [ ] `_days_since_posted()` handles "just now", "today", "yesterday", "30+ days ago".
- [ ] `check_stale_posting()` correctly detects ghost jobs with any supported date format.
- [ ] All existing tests pass (the old formats still work).
- [ ] New tests cover all format variations.

## Constraints

- Use only Python stdlib (`datetime`, `re`). No `python-dateutil` or `arrow`.
- `datetime.fromisoformat()` in Python 3.11+ handles most ISO 8601 -- lean on it.
- Return `None` for unparseable dates (never raise).
- Relative date conversion to days is approximate (1 month = 30 days, 1 year = 365 days).
- Do not modify any other signal functions or the `check_stale_posting` function.
- Return non-negative values (use `max(0, ...)` for computed days).

## Test Command

```bash
python -m pytest tests/test_date_parsing.py tests/test_core.py -v
```
