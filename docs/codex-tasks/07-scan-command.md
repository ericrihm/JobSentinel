# Task: Build the LinkedIn Scan Command

## Task

Build the `sentinel scan` CLI command that searches LinkedIn for job postings and analyzes each result for scam signals.

## Context

- `CLAUDE.md` documents `sentinel scan --query "software engineer" --location "remote"` but this command does not exist in `sentinel/cli.py`.
- The CLI uses Click and is defined in `sentinel/cli.py`. Commands are registered via `@main.command()`.
- `sentinel/scanner.py` has `parse_job_html(html, url)` for parsing LinkedIn HTML and `parse_job_text(text, title, company, url)` for raw text parsing.
- `sentinel/analyzer.py` has `analyze_job(job, use_ai)` for analyzing a single job posting.
- LinkedIn job search results at `https://www.linkedin.com/jobs/search/?keywords=...&location=...` are publicly accessible (no auth required for search results pages), but the HTML structure may change and scraping may be blocked.
- `httpx` is an optional dependency (same as elsewhere in the project).

## What To Do

### 1. Add `scan` command to `sentinel/cli.py`

```python
@main.command()
@click.option("--query", "-q", required=True, help="Job search query (e.g., 'software engineer').")
@click.option("--location", "-l", default="", help="Location filter (e.g., 'remote', 'New York, NY').")
@click.option("--limit", default=10, show_default=True, help="Maximum number of results to analyze.")
@click.option("--no-ai", is_flag=True, default=False, help="Disable AI escalation.")
@click.pass_context
def scan(ctx, query, location, limit, no_ai):
    """Scan LinkedIn job search results and analyze each for scam signals."""
```

### 2. Implement search URL construction

Build the LinkedIn job search URL using `urllib.parse`:

```python
import urllib.parse

params = {"keywords": query}
if location:
    params["location"] = location
search_url = "https://www.linkedin.com/jobs/search/?" + urllib.parse.urlencode(params)
```

### 3. Implement HTML fetching and parsing in `sentinel/scanner.py`

Add a new function `scrape_search_results`:

```python
def scrape_search_results(query: str, location: str = "", limit: int = 10) -> list[JobPosting]:
    """Scrape LinkedIn job search results page.
    
    Returns a list of JobPosting objects with basic info (title, company, URL).
    This is best-effort -- LinkedIn may block or change HTML structure.
    Requires httpx (optional dependency).
    """
```

Implementation details:
1. Import httpx (raise ImportError with clear message if not installed).
2. Construct the search URL from `query` and `location`.
3. Fetch the page with a realistic User-Agent header and `follow_redirects=True`.
4. Parse job listing cards from the HTML using regex. LinkedIn search results use patterns like:
   - Job cards containing links to `/jobs/view/` URLs.
   - Titles in `<h3>` elements with class `base-search-card__title`.
   - Companies in `<h4>` elements with class `base-search-card__subtitle`.
   - Location in `<span>` elements with class `job-search-card__location`.
5. Use regex-based HTML parsing (consistent with the rest of `scanner.py` which uses regex, not BeautifulSoup).
6. Return up to `limit` job postings.
7. If parsing finds no results (HTML structure changed, blocked, etc.), return an empty list -- do not raise.

Add these regex patterns near the other regexes in `scanner.py`:

```python
_SEARCH_CARD_RE = re.compile(
    r'<a[^>]*href="(https://www\.linkedin\.com/jobs/view/[^"?]*)[^"]*"',
    re.IGNORECASE,
)

_SEARCH_TITLE_RE = re.compile(
    r'class="[^"]*base-search-card__title[^"]*"[^>]*>(.*?)</h3>',
    re.DOTALL | re.IGNORECASE,
)

_SEARCH_COMPANY_RE = re.compile(
    r'class="[^"]*base-search-card__subtitle[^"]*"[^>]*>(.*?)</h4>',
    re.DOTALL | re.IGNORECASE,
)

_SEARCH_LOCATION_RE = re.compile(
    r'class="[^"]*job-search-card__location[^"]*"[^>]*>(.*?)</span>',
    re.DOTALL | re.IGNORECASE,
)
```

### 4. Implement the scan flow in the CLI command

1. Call `scrape_search_results(query, location, limit)`.
2. If no results found, display a message explaining LinkedIn may have blocked the request, and suggest using `--file` instead.
3. For each found job posting, run `analyze_job(job, use_ai=not no_ai)` using the `build_result` + `extract_signals` pipeline from `scorer.py` and `signals.py` (to avoid the known bug in `analyze_job`'s AI escalation path).
4. Sort results by scam score descending.
5. Display a ranked table:

```
  LinkedIn Scan: "software engineer" in "remote"
  Found 10 job listings
  ──────────────────────────────────────────────────────────────
  Score  Risk         Title                        Company
  ──────────────────────────────────────────────────────────────
  0.87   [SCAM]       Work From Home Data Entry     ScamCo
  0.65   [HIGH]       Remote Assistant              Unknown Corp
  0.42   [SUSPICIOUS] Marketing Coordinator         Sunrise Media
  0.15   [SAFE]       Backend Engineer              Google
  0.08   [SAFE]       Senior SWE                    Meta
```

6. If `--json-output` is set, output the full results as a JSON array.
7. Persist results to DB (best-effort).

### 5. Add tests in `tests/test_scan.py`

Write tests:

- **scrape_search_results with mocked httpx:**
  - Mock `httpx.Client` to return sample HTML containing 3 job cards with titles, companies, and LinkedIn URLs.
  - Verify 3 `JobPosting` objects are returned with title, company, and URL populated.

- **scrape_search_results with empty results:**
  - Mock httpx to return HTML with no job cards.
  - Verify empty list is returned (no crash).

- **scrape_search_results without httpx:**
  - Mock httpx import to fail.
  - Verify `ImportError` is raised with helpful message.

- **URL construction:**
  - Verify `build_search_url("software engineer", "remote")` produces a valid URL with proper encoding.

- **CLI scan command with mocked scraper:**
  - Use Click's `CliRunner`.
  - Monkeypatch `scrape_search_results` to return sample job postings.
  - Verify output contains the job titles and scores.

- **CLI scan command JSON output:**
  - Monkeypatch scraper, run with `--json-output`.
  - Verify output is valid JSON.

## Acceptance Criteria

- [ ] `sentinel scan --query "..." --location "..."` command exists and runs.
- [ ] `scrape_search_results()` function in scanner.py fetches and parses LinkedIn search HTML.
- [ ] Results are analyzed and displayed in a ranked table sorted by scam score.
- [ ] Graceful handling when LinkedIn blocks the request or returns unexpected HTML.
- [ ] `--json-output` flag produces JSON array output.
- [ ] All existing tests pass.

## Constraints

- This is best-effort scraping. LinkedIn actively blocks scrapers, so the code must handle failures gracefully.
- Do NOT use Selenium, Playwright, or any browser automation.
- Do NOT store LinkedIn credentials or session cookies.
- Use regex-based HTML parsing (consistent with `scanner.py`'s existing approach).
- `httpx` is an optional dependency -- guard the import.
- Limit is capped at 25 results maximum to be respectful.
- Do not break existing tests or commands.

## Test Command

```bash
python -m pytest tests/test_scan.py tests/test_core.py -v
```
