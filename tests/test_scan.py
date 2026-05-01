"""Tests for the scan command and LinkedIn search scraper."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sentinel.scanner import build_search_url, scrape_search_results
from sentinel.cli import main


# ---------------------------------------------------------------------------
# Helper: patch httpx.Client in sentinel.scanner
# ---------------------------------------------------------------------------

@contextmanager
def _mock_httpx(status: int, html: str = "", exc=None):
    """Context manager that patches sentinel.scanner.httpx.Client.

    Usage::

        with _mock_httpx(200, "<html>...") as mock_client:
            jobs = scrape_search_results("engineer")
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.text = html

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    if exc is not None:
        mock_client.get.side_effect = exc
    else:
        mock_client.get.return_value = mock_resp

    import sentinel.scanner as _scanner

    mock_httpx = MagicMock()
    mock_httpx.Client.return_value = mock_client

    with patch.object(_scanner, "httpx", mock_httpx):
        yield mock_client


# ===========================================================================
# build_search_url
# ===========================================================================


class TestBuildSearchUrl:
    def test_build_search_url_query_only(self):
        url = build_search_url("software engineer")
        assert url.startswith("https://www.linkedin.com/jobs/search/?")
        assert "keywords=software+engineer" in url or "keywords=software%20engineer" in url

    def test_build_search_url_with_location(self):
        url = build_search_url("data scientist", location="San Francisco")
        assert "keywords=data+scientist" in url or "keywords=data%20scientist" in url
        assert "location=" in url
        assert "San" in url

    def test_build_search_url_no_location_param(self):
        url = build_search_url("DevOps engineer")
        assert "location" not in url

    def test_build_search_url_empty_query(self):
        url = build_search_url("")
        assert "keywords=" in url


# ===========================================================================
# scrape_search_results — unit tests with mocked httpx
# ===========================================================================

_FAKE_HTML = """
<html>
<body>
  <div class="job-card-container">
    <a href="https://www.linkedin.com/jobs/view/1111111111?refId=abc"
       class="base-card__full-link">Job 1</a>
    <h3 class="base-search-card__title">Software Engineer</h3>
    <h4 class="base-search-card__subtitle">Acme Corp</h4>
  </div>
  <div class="job-card-container">
    <a href="https://www.linkedin.com/jobs/view/2222222222?refId=def"
       class="base-card__full-link">Job 2</a>
    <h3 class="base-search-card__title">Data Analyst</h3>
    <h4 class="base-search-card__subtitle">DataCo LLC</h4>
  </div>
  <div class="job-card-container">
    <a href="https://www.linkedin.com/jobs/view/3333333333?refId=ghi"
       class="base-card__full-link">Job 3</a>
    <h3 class="base-search-card__title">Product Manager</h3>
    <h4 class="base-search-card__subtitle">TechStart Inc</h4>
  </div>
</body>
</html>
"""


class TestScrapeSearchResults:
    def test_scrape_with_mock_html(self):
        """Returns up to `limit` JobPosting objects when HTML contains job cards."""
        with _mock_httpx(200, _FAKE_HTML):
            jobs = scrape_search_results("engineer", limit=10)

        # Should have found the 3 job URLs in the fake HTML
        assert len(jobs) == 3
        urls = [j.url for j in jobs]
        assert any("1111111111" in u for u in urls)
        assert any("2222222222" in u for u in urls)
        assert any("3333333333" in u for u in urls)

    def test_scrape_respects_limit(self):
        """limit parameter caps the returned list."""
        with _mock_httpx(200, _FAKE_HTML):
            jobs = scrape_search_results("engineer", limit=2)

        assert len(jobs) == 2

    def test_scrape_empty_results_on_non_200(self):
        """Non-200 HTTP response yields an empty list."""
        with _mock_httpx(403, "Forbidden"):
            jobs = scrape_search_results("engineer")

        assert jobs == []

    def test_scrape_empty_results_on_network_error(self):
        """Network exception yields an empty list (never raises)."""
        with _mock_httpx(0, exc=OSError("connection refused")):
            jobs = scrape_search_results("engineer")

        assert jobs == []

    def test_scrape_empty_results_on_empty_html(self):
        """HTML with no job cards returns an empty list."""
        with _mock_httpx(200, "<html><body>No jobs here.</body></html>"):
            jobs = scrape_search_results("obscure-role-xyz")

        assert jobs == []


# ===========================================================================
# scan CLI command
# ===========================================================================


class TestScanCli:
    def test_scan_cli_with_mock(self):
        """scan command runs end-to-end with a mocked scraper and prints a table."""
        from sentinel.models import JobPosting

        fake_jobs = [
            JobPosting(
                url="https://www.linkedin.com/jobs/view/9000000001",
                title="Remote Data Entry — Earn $3000/week GUARANTEED",
                company="Quick Cash LLC",
                description="Easy money, no experience needed, pay registration fee upfront.",
                source="linkedin",
            ),
            JobPosting(
                url="https://www.linkedin.com/jobs/view/9000000002",
                title="Senior Python Engineer",
                company="Reliable Tech",
                description="5+ years Python required. Design distributed systems.",
                source="linkedin",
            ),
        ]

        runner = CliRunner()
        with patch("sentinel.scanner.scrape_search_results", return_value=fake_jobs):
            result = runner.invoke(main, ["scan", "--query", "engineer", "--no-ai"])

        assert result.exit_code == 0, result.output
        # Table header should be present
        assert "Score" in result.output
        assert "Risk" in result.output

    def test_scan_cli_no_results(self):
        """scan command handles empty scraper results gracefully."""
        runner = CliRunner()
        with patch("sentinel.scanner.scrape_search_results", return_value=[]):
            result = runner.invoke(main, ["scan", "--query", "impossible-role-xyz"])

        assert result.exit_code == 0
        assert result.output.strip() != ""

    def test_scan_cli_json_output(self):
        """--json-output flag produces valid JSON."""
        import json as _json

        from sentinel.models import JobPosting

        fake_jobs = [
            JobPosting(
                url="https://www.linkedin.com/jobs/view/9000000003",
                title="Accountant",
                company="FinanceCo",
                description="Manage accounts payable and receivable. CPA preferred.",
                source="linkedin",
            ),
        ]

        runner = CliRunner()
        with patch("sentinel.scanner.scrape_search_results", return_value=fake_jobs):
            result = runner.invoke(
                main, ["--json-output", "scan", "--query", "accountant", "--no-ai"]
            )

        assert result.exit_code == 0, result.output
        data = _json.loads(result.output)
        assert "results" in data
        assert data["query"] == "accountant"
