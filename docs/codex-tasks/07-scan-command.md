# Task: Build `sentinel scan` CLI Command

## Task

Implement the `sentinel scan` command documented in CLAUDE.md that searches LinkedIn for job postings and analyzes them.

## Context

- CLAUDE.md documents `sentinel scan --query "software engineer" --location "remote"` but it doesn't exist.
- The CLI uses Click (see `sentinel/cli.py`).
- `sentinel/scanner.py` has `parse_job_html()` for parsing individual job pages.
- `sentinel/analyzer.py` has `batch_analyze()` for analyzing multiple jobs.
- httpx is an optional dependency for HTTP requests.

## What To Do

### 1. Add scan command to `sentinel/cli.py`

```python
@main.command()
@click.option("--query", "-q", required=True, help="Job search query")
@click.option("--location", "-l", default="", help="Location filter")
@click.option("--limit", default=10, help="Max results to analyze")
@click.option("--no-ai", is_flag=True, help="Skip AI analysis")
def scan(query, location, limit, no_ai):
    """Scan LinkedIn job search results for scams."""
```

### 2. Build search URL construction in `scanner.py`

Add `build_search_url(query: str, location: str) -> str` that constructs a LinkedIn job search URL from parameters.

Add `parse_search_results(html: str) -> list[dict]` that extracts job listing cards from a search results page. Each card should have: title, company, url, location. Use regex patterns targeting LinkedIn's search result HTML structure.

### 3. Wire into the scan command

The command should:
1. Build the search URL
2. Fetch the HTML (httpx with graceful ImportError)
3. Parse search results
4. For each result (up to `limit`), analyze the job
5. Output a ranked table sorted by scam score (highest first)
6. Show: risk icon, score, title, company

### 4. Add tests

- Test `build_search_url` produces valid URL
- Test `parse_search_results` with sample HTML
- Test scan CLI command with CliRunner (mock httpx)

## Acceptance Criteria

- [ ] `sentinel scan --query "software engineer"` produces output
- [ ] Results are sorted by scam score (most suspicious first)
- [ ] Graceful handling when httpx is not installed
- [ ] Graceful handling when LinkedIn returns non-200 or changes HTML structure
- [ ] All tests pass: `python -m pytest tests/ -v`

## Constraints

- httpx is optional — fail with a clear message if not installed
- Do not store LinkedIn credentials
- Be respectful of rate limits — add a 1-second delay between requests
- Do not break existing tests or commands

## Test Command

```bash
python -m pytest tests/ -v
```
