"""Tests for sentinel.sources — job board adapter system."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sentinel.models import JobPosting
from sentinel.sources import (
    AdzunaSource,
    AshbySource,
    GreenhouseSource,
    JobSource,
    LeverSource,
    RemoteOKSource,
    RemotiveSource,
    SmartRecruitersSource,
    TheMuseSource,
    USAJobsSource,
    fetch_from_all,
    get_all_sources,
)


# ---------------------------------------------------------------------------
# Sample API response fixtures
# ---------------------------------------------------------------------------

REMOTEOK_RESPONSE = [
    {"legal": "This is a legal notice"},  # metadata row, should be skipped
    {
        "id": 1,
        "position": "Senior Python Developer",
        "company": "AcmeCorp",
        "description": "<p>Build cool <b>stuff</b></p>",
        "tags": ["python", "django"],
        "salary_min": 100000,
        "salary_max": 150000,
        "location": "Worldwide",
        "date": "2026-04-01T00:00:00",
        "url": "https://remoteok.com/jobs/1",
    },
    {
        "id": 2,
        "position": "Java Engineer",
        "company": "OtherCo",
        "description": "Java work",
        "tags": ["java"],
        "salary_min": 80000,
        "salary_max": 120000,
        "location": "US",
        "date": "2026-04-02",
        "url": "https://remoteok.com/jobs/2",
    },
]

ADZUNA_RESPONSE = {
    "results": [
        {
            "title": "Data Analyst",
            "company": {"display_name": "DataCo"},
            "description": "<b>Analyze</b> data sets",
            "salary_min": 70000,
            "salary_max": 90000,
            "location": {"display_name": "New York"},
            "created": "2026-04-10T12:00:00Z",
            "redirect_url": "https://adzuna.com/jobs/123",
        }
    ]
}

THEMUSE_RESPONSE = {
    "results": [
        {
            "name": "Product Manager",
            "company": {"name": "MuseCo"},
            "contents": "<div>Lead product <em>strategy</em></div>",
            "locations": [{"name": "San Francisco, CA"}],
            "levels": [{"name": "Senior"}],
            "publication_date": "2026-04-15",
            "refs": {"landing_page": "https://themuse.com/jobs/456"},
        }
    ]
}

USAJOBS_RESPONSE = {
    "SearchResult": {
        "SearchResultItems": [
            {
                "MatchedObjectDescriptor": {
                    "PositionTitle": "IT Specialist",
                    "OrganizationName": "Department of Defense",
                    "QualificationSummary": "<p>Must have clearance</p>",
                    "PositionRemuneration": [
                        {"MinimumRange": "80000", "MaximumRange": "120000"}
                    ],
                    "PositionLocation": [{"LocationName": "Washington, DC"}],
                    "PositionURI": "https://usajobs.gov/jobs/789",
                    "PublicationStartDate": "2026-04-20",
                }
            }
        ]
    }
}

REMOTIVE_RESPONSE = {
    "jobs": [
        {
            "title": "Frontend Engineer",
            "company_name": "RemotiveCo",
            "description": "<h2>Build UIs</h2>",
            "candidate_required_location": "Anywhere",
            "salary": "$100k - $140k",
            "url": "https://remotive.com/jobs/101",
            "publication_date": "2026-04-18",
            "category": "Software Development",
        }
    ]
}


# ---------------------------------------------------------------------------
# Helper: build a mock httpx response
# ---------------------------------------------------------------------------

def _mock_response(data, status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


def _mock_client(response):
    """Return a context-manager mock that yields a client whose .get returns response."""
    client = MagicMock()
    client.get.return_value = response
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# RemoteOK tests
# ---------------------------------------------------------------------------

class TestRemoteOKSource:
    def test_parse_response(self):
        client = _mock_client(_mock_response(REMOTEOK_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = RemoteOKSource()
            jobs = source.fetch(query="python", limit=10)

        assert len(jobs) == 1  # only "python" match
        job = jobs[0]
        assert job.title == "Senior Python Developer"
        assert job.company == "AcmeCorp"
        assert job.salary_min == 100000
        assert job.salary_max == 150000
        assert "cool" in job.description
        assert "<b>" not in job.description  # HTML stripped
        assert job.source == "remoteok"
        assert job.is_remote is True

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ConnectError("Connection refused")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            jobs = RemoteOKSource().fetch(query="test")
        assert jobs == []

    def test_malformed_json_returns_empty(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 500
        resp.raise_for_status.return_value = None
        resp.json.side_effect = json.JSONDecodeError("fail", "", 0)
        client = _mock_client(resp)
        with patch("sentinel.sources.httpx.Client", return_value=client):
            jobs = RemoteOKSource().fetch()
        assert jobs == []

    def test_no_query_returns_all(self):
        client = _mock_client(_mock_response(REMOTEOK_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            jobs = RemoteOKSource().fetch(query="", limit=50)
        assert len(jobs) == 2  # Both items (metadata row skipped)


# ---------------------------------------------------------------------------
# Adzuna tests
# ---------------------------------------------------------------------------

class TestAdzunaSource:
    def test_parse_response(self):
        client = _mock_client(_mock_response(ADZUNA_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = AdzunaSource(app_id="test_id", app_key="test_key")
            jobs = source.fetch(query="data", location="New York")

        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "Data Analyst"
        assert job.company == "DataCo"
        assert job.location == "New York"
        assert job.salary_min == 70000
        assert job.salary_max == 90000
        assert "Analyze" in job.description
        assert "<b>" not in job.description
        assert job.source == "adzuna"
        assert job.url == "https://adzuna.com/jobs/123"

    def test_missing_credentials_returns_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            source = AdzunaSource(app_id="", app_key="")
            assert source.available is False
            jobs = source.fetch(query="test")
        assert jobs == []

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ConnectError("timeout")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = AdzunaSource(app_id="x", app_key="y")
            jobs = source.fetch()
        assert jobs == []


# ---------------------------------------------------------------------------
# The Muse tests
# ---------------------------------------------------------------------------

class TestTheMuseSource:
    def test_parse_response(self):
        client = _mock_client(_mock_response(THEMUSE_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = TheMuseSource()
            jobs = source.fetch(query="Product")

        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "Product Manager"
        assert job.company == "MuseCo"
        assert "San Francisco" in job.location
        assert job.experience_level == "Senior"
        assert "strategy" in job.description
        assert "<em>" not in job.description
        assert job.source == "themuse"
        assert job.url == "https://themuse.com/jobs/456"

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ReadTimeout("timeout")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            jobs = TheMuseSource().fetch()
        assert jobs == []

    def test_malformed_json_returns_empty(self):
        resp = MagicMock(spec=httpx.Response)
        resp.raise_for_status.return_value = None
        resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        client = _mock_client(resp)
        with patch("sentinel.sources.httpx.Client", return_value=client):
            jobs = TheMuseSource().fetch()
        assert jobs == []


# ---------------------------------------------------------------------------
# USAJobs tests
# ---------------------------------------------------------------------------

class TestUSAJobsSource:
    def test_parse_response(self):
        client = _mock_client(_mock_response(USAJOBS_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = USAJobsSource(api_key="test_key", email="test@test.com")
            jobs = source.fetch(query="IT", location="Washington")

        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "IT Specialist"
        assert job.company == "Department of Defense"
        assert "Washington" in job.location
        assert job.salary_min == 80000
        assert job.salary_max == 120000
        assert "clearance" in job.description
        assert "<p>" not in job.description
        assert job.source == "usajobs"

    def test_missing_credentials_returns_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            source = USAJobsSource(api_key="", email="")
            assert source.available is False
            jobs = source.fetch(query="test")
        assert jobs == []

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ConnectError("refused")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = USAJobsSource(api_key="k", email="e@e.com")
            jobs = source.fetch()
        assert jobs == []


# ---------------------------------------------------------------------------
# Remotive tests
# ---------------------------------------------------------------------------

class TestRemotiveSource:
    def test_parse_response(self):
        client = _mock_client(_mock_response(REMOTIVE_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = RemotiveSource()
            jobs = source.fetch(query="frontend")

        assert len(jobs) == 1
        job = jobs[0]
        assert job.title == "Frontend Engineer"
        assert job.company == "RemotiveCo"
        assert job.location == "Anywhere"
        assert "Build UIs" in job.description
        assert "<h2>" not in job.description
        assert job.source == "remotive"
        assert job.is_remote is True
        assert job.industry == "Software Development"
        assert job.url == "https://remotive.com/jobs/101"
        # Salary should be parsed from the "$100k - $140k" string
        assert job.salary_min > 0
        assert job.salary_max > 0

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ConnectError("fail")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            jobs = RemotiveSource().fetch()
        assert jobs == []

    def test_malformed_json_returns_empty(self):
        resp = MagicMock(spec=httpx.Response)
        resp.raise_for_status.return_value = None
        resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        client = _mock_client(resp)
        with patch("sentinel.sources.httpx.Client", return_value=client):
            jobs = RemotiveSource().fetch()
        assert jobs == []


# ---------------------------------------------------------------------------
# get_all_sources tests
# ---------------------------------------------------------------------------

class TestGetAllSources:
    def test_skips_adzuna_without_env(self):
        with patch.dict("os.environ", {}, clear=True):
            sources = get_all_sources()
        names = [s.name for s in sources]
        assert "remoteok" in names
        assert "themuse" in names
        assert "remotive" in names
        assert "adzuna" not in names
        assert "usajobs" not in names

    def test_includes_adzuna_with_env(self):
        env = {"ADZUNA_APP_ID": "id", "ADZUNA_APP_KEY": "key"}
        with patch.dict("os.environ", env, clear=True):
            sources = get_all_sources()
        names = [s.name for s in sources]
        assert "adzuna" in names

    def test_includes_usajobs_with_env(self):
        env = {"USAJOBS_API_KEY": "key", "USAJOBS_EMAIL": "a@b.com"}
        with patch.dict("os.environ", env, clear=True):
            sources = get_all_sources()
        names = [s.name for s in sources]
        assert "usajobs" in names


# ---------------------------------------------------------------------------
# fetch_from_all tests
# ---------------------------------------------------------------------------

class TestFetchFromAll:
    def test_deduplicates_by_url(self):
        """Jobs with the same URL from different sources should be deduplicated."""
        job1 = JobPosting(url="https://example.com/job/1", title="Job 1", source="a")
        job2 = JobPosting(url="https://example.com/job/1", title="Job 1 dup", source="b")
        job3 = JobPosting(url="https://example.com/job/2", title="Job 2", source="c")

        mock_source_a = MagicMock(spec=JobSource)
        mock_source_a.name = "a"
        mock_source_a.fetch.return_value = [job1]

        mock_source_b = MagicMock(spec=JobSource)
        mock_source_b.name = "b"
        mock_source_b.fetch.return_value = [job2, job3]

        with patch("sentinel.sources.get_all_sources", return_value=[mock_source_a, mock_source_b]):
            results = fetch_from_all(query="test")

        assert len(results) == 2
        urls = [j.url for j in results]
        assert "https://example.com/job/1" in urls
        assert "https://example.com/job/2" in urls

    def test_handles_source_failure(self):
        """If one source raises, the others should still return results."""
        job = JobPosting(url="https://ok.com/1", title="OK", source="ok")

        failing = MagicMock(spec=JobSource)
        failing.name = "fail"
        failing.fetch.side_effect = RuntimeError("boom")

        working = MagicMock(spec=JobSource)
        working.name = "ok"
        working.fetch.return_value = [job]

        with patch("sentinel.sources.get_all_sources", return_value=[failing, working]):
            results = fetch_from_all(query="test")

        assert len(results) == 1
        assert results[0].title == "OK"


# ---------------------------------------------------------------------------
# Greenhouse tests
# ---------------------------------------------------------------------------

GREENHOUSE_RESPONSE = {
    "jobs": [
        {
            "id": 101,
            "title": "Senior Python Engineer",
            "location": {"name": "Remote"},
            "content": "<p>Build <b>awesome</b> systems with Python and Django.</p>",
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/101",
            "updated_at": "2026-04-10T09:00:00Z",
        },
        {
            "id": 102,
            "title": "Java Developer",
            "location": {"name": "New York, NY"},
            "content": "<p>Enterprise Java work</p>",
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/102",
            "updated_at": "2026-04-11T09:00:00Z",
        },
    ]
}


class TestGreenhouseSource:
    def test_fetch_maps_fields(self):
        client = _mock_client(_mock_response(GREENHOUSE_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = GreenhouseSource(companies=["acme"])
            jobs = source.fetch(query="", limit=10)

        assert len(jobs) == 2
        job = jobs[0]
        assert job.title == "Senior Python Engineer"
        assert job.company == "acme"
        assert job.location == "Remote"
        assert "awesome" in job.description
        assert "<b>" not in job.description  # HTML stripped
        assert job.url == "https://boards.greenhouse.io/acme/jobs/101"
        assert job.posted_date == "2026-04-10T09:00:00Z"
        assert job.source == "greenhouse"

    def test_empty_companies_returns_empty(self):
        source = GreenhouseSource(companies=[])
        jobs = source.fetch(query="python")
        assert jobs == []

    def test_no_companies_default_returns_empty(self):
        source = GreenhouseSource()
        jobs = source.fetch()
        assert jobs == []

    def test_query_filtering(self):
        client = _mock_client(_mock_response(GREENHOUSE_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = GreenhouseSource(companies=["acme"])
            jobs = source.fetch(query="python", limit=10)

        # Only the Python job should match
        assert len(jobs) == 1
        assert jobs[0].title == "Senior Python Engineer"

    def test_http_error_returns_empty(self):
        client = _mock_client(_mock_response({}, status_code=500))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = GreenhouseSource(companies=["acme"])
            jobs = source.fetch()
        assert jobs == []

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ConnectError("refused")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = GreenhouseSource(companies=["acme"])
            jobs = source.fetch()
        assert jobs == []


# ---------------------------------------------------------------------------
# Lever tests
# ---------------------------------------------------------------------------

# createdAt is milliseconds since epoch; 1712750400000 ms = 2024-04-10T12:00:00+00:00
LEVER_RESPONSE = [
    {
        "id": "abc-123",
        "text": "Staff Software Engineer",
        "categories": {
            "location": "San Francisco, CA",
            "team": "Infrastructure",
        },
        "description": "<h2>About the role</h2><p>Build <em>reliable</em> systems.</p>",
        "hostedUrl": "https://jobs.lever.co/cloudflare/abc-123",
        "createdAt": 1712750400000,
    },
    {
        "id": "def-456",
        "text": "Marketing Manager",
        "categories": {
            "location": "Austin, TX",
            "team": "Marketing",
        },
        "description": "<p>Drive campaigns</p>",
        "hostedUrl": "https://jobs.lever.co/cloudflare/def-456",
        "createdAt": 1712836800000,
    },
]


class TestLeverSource:
    def test_fetch_maps_fields(self):
        client = _mock_client(_mock_response(LEVER_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = LeverSource(companies=["cloudflare"])
            jobs = source.fetch(query="", limit=10)

        assert len(jobs) == 2
        job = jobs[0]
        assert job.title == "Staff Software Engineer"
        assert job.company == "cloudflare"
        assert job.location == "San Francisco, CA"
        assert "reliable" in job.description
        assert "<em>" not in job.description  # HTML stripped
        assert job.url == "https://jobs.lever.co/cloudflare/abc-123"
        assert job.source == "lever"

    def test_timestamp_ms_to_iso(self):
        """createdAt in milliseconds should be converted to an ISO date string."""
        client = _mock_client(_mock_response(LEVER_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = LeverSource(companies=["cloudflare"])
            jobs = source.fetch(query="", limit=10)

        assert len(jobs) == 2
        # posted_date should be a non-empty ISO string containing the date
        posted = jobs[0].posted_date
        assert posted != ""
        assert "2024-04-10" in posted  # 1712750400000 ms → 2024-04-10

    def test_empty_companies_returns_empty(self):
        source = LeverSource(companies=[])
        jobs = source.fetch(query="engineer")
        assert jobs == []

    def test_query_filtering(self):
        client = _mock_client(_mock_response(LEVER_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = LeverSource(companies=["cloudflare"])
            jobs = source.fetch(query="engineer", limit=10)

        assert len(jobs) == 1
        assert "Engineer" in jobs[0].title

    def test_404_returns_empty(self):
        client = _mock_client(_mock_response({}, status_code=404))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = LeverSource(companies=["nonexistent"])
            jobs = source.fetch()
        assert jobs == []

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ConnectError("refused")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = LeverSource(companies=["cloudflare"])
            jobs = source.fetch()
        assert jobs == []


# ---------------------------------------------------------------------------
# Ashby tests
# ---------------------------------------------------------------------------

ASHBY_RESPONSE = {
    "jobs": [
        {
            "id": "job-001",
            "title": "Backend Engineer",
            "location": "Remote (US)",
            "descriptionHtml": "<p>Build <strong>scalable</strong> APIs.</p>",
            "jobUrl": "https://jobs.ashbyhq.com/linear/job-001",
            "publishedAt": "2026-04-12T00:00:00.000Z",
            "employmentType": "FullTime",
        },
        {
            "id": "job-002",
            "title": "Design Lead",
            "location": "San Francisco",
            "descriptionHtml": "<p>Lead design systems.</p>",
            "jobUrl": "https://jobs.ashbyhq.com/linear/job-002",
            "publishedAt": "2026-04-13T00:00:00.000Z",
            "employmentType": "FullTime",
        },
    ]
}


class TestAshbySource:
    def test_fetch_maps_fields(self):
        client = _mock_client(_mock_response(ASHBY_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = AshbySource(companies=["linear"])
            jobs = source.fetch(query="", limit=10)

        assert len(jobs) == 2
        job = jobs[0]
        assert job.title == "Backend Engineer"
        assert job.company == "linear"
        assert job.location == "Remote (US)"
        assert "scalable" in job.description
        assert "<strong>" not in job.description  # HTML stripped
        assert job.url == "https://jobs.ashbyhq.com/linear/job-001"
        assert job.posted_date == "2026-04-12T00:00:00.000Z"
        assert job.source == "ashby"

    def test_employment_type_mapped(self):
        client = _mock_client(_mock_response(ASHBY_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = AshbySource(companies=["linear"])
            jobs = source.fetch(query="", limit=10)

        assert jobs[0].employment_type == "FullTime"

    def test_empty_companies_returns_empty(self):
        source = AshbySource(companies=[])
        jobs = source.fetch()
        assert jobs == []

    def test_query_filtering(self):
        client = _mock_client(_mock_response(ASHBY_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = AshbySource(companies=["linear"])
            jobs = source.fetch(query="backend", limit=10)

        assert len(jobs) == 1
        assert jobs[0].title == "Backend Engineer"

    def test_http_error_returns_empty(self):
        client = _mock_client(_mock_response({}, status_code=500))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = AshbySource(companies=["linear"])
            jobs = source.fetch()
        assert jobs == []

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ConnectError("refused")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = AshbySource(companies=["linear"])
            jobs = source.fetch()
        assert jobs == []


# ---------------------------------------------------------------------------
# SmartRecruiters tests
# ---------------------------------------------------------------------------

SMARTRECRUITERS_RESPONSE = {
    "content": [
        {
            "name": "Product Manager",
            "location": {"city": "Austin", "country": "US"},
            "company": {"name": "Acme Inc."},
            "ref": "https://careers.smartrecruiters.com/AcmeInc/product-manager",
            "releasedDate": "2026-04-14",
        },
        {
            "name": "Data Scientist",
            "location": {"city": "Remote", "country": ""},
            "company": {"name": "Acme Inc."},
            "ref": "https://careers.smartrecruiters.com/AcmeInc/data-scientist",
            "releasedDate": "2026-04-15",
        },
    ]
}


class TestSmartRecruitersSource:
    def test_fetch_maps_fields(self):
        client = _mock_client(_mock_response(SMARTRECRUITERS_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = SmartRecruitersSource(companies=["AcmeInc"])
            jobs = source.fetch(query="", limit=10)

        assert len(jobs) == 2
        job = jobs[0]
        assert job.title == "Product Manager"
        assert job.company == "Acme Inc."
        assert job.url == "https://careers.smartrecruiters.com/AcmeInc/product-manager"
        assert job.posted_date == "2026-04-14"
        assert job.source == "smartrecruiters"

    def test_location_formatting_city_and_country(self):
        """City and country should be joined with ', '."""
        client = _mock_client(_mock_response(SMARTRECRUITERS_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = SmartRecruitersSource(companies=["AcmeInc"])
            jobs = source.fetch(query="", limit=10)

        assert jobs[0].location == "Austin, US"

    def test_location_formatting_city_only(self):
        """When country is empty, location should be just the city (no trailing comma)."""
        client = _mock_client(_mock_response(SMARTRECRUITERS_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = SmartRecruitersSource(companies=["AcmeInc"])
            jobs = source.fetch(query="", limit=10)

        # Second posting has empty country
        assert jobs[1].location == "Remote"
        assert not jobs[1].location.endswith(",")

    def test_company_falls_back_to_company_id(self):
        """If the nested company.name is missing, fall back to company_id."""
        response = {
            "content": [
                {
                    "name": "SRE Lead",
                    "location": {"city": "NYC", "country": "US"},
                    "company": {},
                    "ref": "https://careers.smartrecruiters.com/Foo/sre-lead",
                    "releasedDate": "2026-04-16",
                }
            ]
        }
        client = _mock_client(_mock_response(response))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = SmartRecruitersSource(companies=["Foo"])
            jobs = source.fetch(query="", limit=10)

        assert len(jobs) == 1
        assert jobs[0].company == "Foo"

    def test_empty_companies_returns_empty(self):
        source = SmartRecruitersSource(companies=[])
        jobs = source.fetch()
        assert jobs == []

    def test_query_filtering(self):
        client = _mock_client(_mock_response(SMARTRECRUITERS_RESPONSE))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = SmartRecruitersSource(companies=["AcmeInc"])
            jobs = source.fetch(query="data scientist", limit=10)

        assert len(jobs) == 1
        assert jobs[0].title == "Data Scientist"

    def test_http_error_returns_empty(self):
        client = _mock_client(_mock_response({}, status_code=500))
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = SmartRecruitersSource(companies=["AcmeInc"])
            jobs = source.fetch()
        assert jobs == []

    def test_network_error_returns_empty(self):
        client = _mock_client(None)
        client.get.side_effect = httpx.ConnectError("refused")
        with patch("sentinel.sources.httpx.Client", return_value=client):
            source = SmartRecruitersSource(companies=["AcmeInc"])
            jobs = source.fetch()
        assert jobs == []
