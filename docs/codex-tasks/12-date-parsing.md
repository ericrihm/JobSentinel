# Task: Fix Date Parsing for Real LinkedIn Data

## Task

Expand `_days_since_posted()` in `scanner.py` to handle real-world date formats from LinkedIn.

## Context

- `sentinel/scanner.py` has `_days_since_posted(date_str)` that only parses 3 formats: `%Y-%m-%d`, `%Y-%m-%dT%H:%M:%S`, `%Y-%m-%dT%H:%M:%SZ`
- LinkedIn API returns ISO 8601 with timezone offsets: `2024-03-15T10:30:00+00:00`
- LinkedIn search results show relative dates: "2 weeks ago", "3 days ago", "1 month ago", "yesterday", "just now"
- Python 3.11+ `datetime.fromisoformat()` handles timezone offsets natively

## What To Do

### 1. Update `_days_since_posted()` in `scanner.py`

Replace the try-each-format approach with:

```python
def _days_since_posted(date_str: str) -> int | None:
    if not date_str:
        return None

    # Try relative dates first
    relative = _parse_relative_date(date_str)
    if relative is not None:
        return relative

    # Try ISO 8601 (handles timezone offsets in Python 3.11+)
    try:
        dt = datetime.fromisoformat(date_str)
        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (now - dt).days
    except ValueError:
        pass

    return None
```

### 2. Add `_parse_relative_date()` helper

```python
_RELATIVE_PATTERNS = {
    r"just now": 0,
    r"today": 0,
    r"yesterday": 1,
    r"(\d+)\s*hours?\s*ago": lambda m: 0,
    r"(\d+)\s*days?\s*ago": lambda m: int(m.group(1)),
    r"(\d+)\s*weeks?\s*ago": lambda m: int(m.group(1)) * 7,
    r"(\d+)\s*months?\s*ago": lambda m: int(m.group(1)) * 30,
}
```

### 3. Add tests

Test each format:
- `"2026-04-15"` → correct days
- `"2026-04-15T10:30:00"` → correct days
- `"2026-04-15T10:30:00Z"` → correct days
- `"2026-04-15T10:30:00+00:00"` → correct days
- `"2026-04-15T10:30:00-05:00"` → correct days
- `"3 days ago"` → 3
- `"2 weeks ago"` → 14
- `"1 month ago"` → 30
- `"yesterday"` → 1
- `"just now"` → 0
- `""` → None
- `"invalid"` → None

## Acceptance Criteria

- [ ] ISO 8601 with timezone offsets parsed correctly
- [ ] Relative date strings ("3 days ago") parsed correctly
- [ ] Empty string and invalid input return None
- [ ] `check_stale_posting` signal now works with real LinkedIn dates
- [ ] All tests pass: `python -m pytest tests/ -v`

## Constraints

- Use only stdlib (`datetime`, `re`). No dateutil or arrow.
- Do not change the return type (int | None)
- Relative date parsing should be case-insensitive

## Test Command

```bash
python -m pytest tests/ -v
```
