"""Pluggable job board adapter system for fetching real job postings.

Each adapter wraps a free public API and returns sentinel.models.JobPosting
objects. Adapters handle their own rate limiting, parsing, and error handling.
All network errors are swallowed and logged -- callers always get a list.
"""

import contextlib
import logging
import os
from abc import ABC, abstractmethod

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from sentinel.models import JobPosting
from sentinel.scanner import _strip_html

logger = logging.getLogger(__name__)

_USER_AGENT = "Sentinel/0.1 (job-scam-detection)"
_TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class JobSource(ABC):
    """Abstract base for job board adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable name, e.g. 'remoteok'."""
        ...

    @abstractmethod
    def fetch(self, query: str = "", location: str = "", limit: int = 25) -> list[JobPosting]:
        """Fetch job postings. Returns empty list on any failure."""
        ...


# ---------------------------------------------------------------------------
# RemoteOK
# ---------------------------------------------------------------------------

class RemoteOKSource(JobSource):
    """Adapter for https://remoteok.com/api (no auth required)."""

    @property
    def name(self) -> str:
        return "remoteok"

    def fetch(self, query: str = "", location: str = "", limit: int = 25) -> list[JobPosting]:
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(
                    "https://remoteok.com/api",
                    headers={"User-Agent": _USER_AGENT},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.exception("RemoteOK fetch failed")
            return []

        # The first element is usually a metadata/legal notice dict -- skip it
        if isinstance(data, list) and data and isinstance(data[0], dict) and "id" not in data[0]:
            data = data[1:]

        query_lower = query.lower()
        jobs: list[JobPosting] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            # Local filtering since API has no search param
            if query_lower:
                searchable = " ".join([
                    str(item.get("position", "")),
                    str(item.get("company", "")),
                    str(item.get("description", "")),
                    " ".join(item.get("tags", []) if isinstance(item.get("tags"), list) else []),
                ]).lower()
                if query_lower not in searchable:
                    continue

            try:
                description_html = str(item.get("description", ""))
                description = _strip_html(description_html)

                salary_min = 0.0
                salary_max = 0.0
                with contextlib.suppress(ValueError, TypeError):
                    salary_min = float(item.get("salary_min", 0) or 0)
                with contextlib.suppress(ValueError, TypeError):
                    salary_max = float(item.get("salary_max", 0) or 0)

                job = JobPosting(
                    url=str(item.get("url", "")),
                    title=str(item.get("position", "")),
                    company=str(item.get("company", "")),
                    location=str(item.get("location", "Remote")),
                    description=description,
                    salary_min=salary_min,
                    salary_max=salary_max,
                    posted_date=str(item.get("date", "")),
                    is_remote=True,
                    source=self.name,
                )
                jobs.append(job)
            except Exception:
                logger.exception("RemoteOK: failed to parse item")
                continue

            if len(jobs) >= limit:
                break

        return jobs


# ---------------------------------------------------------------------------
# Adzuna
# ---------------------------------------------------------------------------

class AdzunaSource(JobSource):
    """Adapter for Adzuna Jobs API (requires ADZUNA_APP_ID and ADZUNA_APP_KEY)."""

    def __init__(self, app_id: str = "", app_key: str = "", country: str = "us"):
        self._app_id = app_id or os.environ.get("ADZUNA_APP_ID", "")
        self._app_key = app_key or os.environ.get("ADZUNA_APP_KEY", "")
        self._country = country

    @property
    def name(self) -> str:
        return "adzuna"

    @property
    def available(self) -> bool:
        return bool(self._app_id and self._app_key)

    def fetch(self, query: str = "", location: str = "", limit: int = 25) -> list[JobPosting]:
        if not self.available:
            logger.warning("Adzuna: missing ADZUNA_APP_ID or ADZUNA_APP_KEY")
            return []

        params: dict[str, str | int] = {
            "app_id": self._app_id,
            "app_key": self._app_key,
            "results_per_page": min(limit, 50),
        }
        if query:
            params["what"] = query
        if location:
            params["where"] = location

        try:
            url = f"https://api.adzuna.com/v1/api/jobs/{self._country}/search/1"
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(url, params=params, headers={"User-Agent": _USER_AGENT})
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.exception("Adzuna fetch failed")
            return []

        results = data.get("results", [])
        jobs: list[JobPosting] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                description_html = str(item.get("description", ""))
                description = _strip_html(description_html)

                company_obj = item.get("company", {})
                company_name = ""
                if isinstance(company_obj, dict):
                    company_name = str(company_obj.get("display_name", ""))

                location_obj = item.get("location", {})
                loc = ""
                if isinstance(location_obj, dict):
                    loc = str(location_obj.get("display_name", ""))

                salary_min = 0.0
                salary_max = 0.0
                with contextlib.suppress(ValueError, TypeError):
                    salary_min = float(item.get("salary_min", 0) or 0)
                with contextlib.suppress(ValueError, TypeError):
                    salary_max = float(item.get("salary_max", 0) or 0)

                job = JobPosting(
                    url=str(item.get("redirect_url", "")),
                    title=str(item.get("title", "")),
                    company=company_name,
                    location=loc,
                    description=description,
                    salary_min=salary_min,
                    salary_max=salary_max,
                    posted_date=str(item.get("created", "")),
                    source=self.name,
                )
                jobs.append(job)
            except Exception:
                logger.exception("Adzuna: failed to parse item")
                continue

        return jobs[:limit]


# ---------------------------------------------------------------------------
# The Muse
# ---------------------------------------------------------------------------

class TheMuseSource(JobSource):
    """Adapter for The Muse public API (no auth required)."""

    @property
    def name(self) -> str:
        return "themuse"

    def fetch(self, query: str = "", location: str = "", limit: int = 25) -> list[JobPosting]:
        params: dict[str, str | int] = {"page": 0}
        if query:
            params["category"] = query

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(
                    "https://www.themuse.com/api/public/jobs",
                    params=params,
                    headers={"User-Agent": _USER_AGENT},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.exception("TheMuse fetch failed")
            return []

        results = data.get("results", [])
        jobs: list[JobPosting] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                contents_html = str(item.get("contents", ""))
                description = _strip_html(contents_html)

                company_obj = item.get("company", {})
                company_name = ""
                if isinstance(company_obj, dict):
                    company_name = str(company_obj.get("name", ""))

                locations = item.get("locations", [])
                loc_parts = []
                if isinstance(locations, list):
                    for loc_item in locations:
                        if isinstance(loc_item, dict):
                            loc_parts.append(str(loc_item.get("name", "")))
                loc = ", ".join(loc_parts) if loc_parts else ""

                levels = item.get("levels", [])
                experience = ""
                if isinstance(levels, list) and levels:
                    first_level = levels[0]
                    if isinstance(first_level, dict):
                        experience = str(first_level.get("name", ""))

                refs = item.get("refs", {})
                url = ""
                if isinstance(refs, dict):
                    url = str(refs.get("landing_page", ""))

                job = JobPosting(
                    url=url,
                    title=str(item.get("name", "")),
                    company=company_name,
                    location=loc,
                    description=description,
                    experience_level=experience,
                    posted_date=str(item.get("publication_date", "")),
                    source=self.name,
                )
                jobs.append(job)
            except Exception:
                logger.exception("TheMuse: failed to parse item")
                continue

            if len(jobs) >= limit:
                break

        return jobs


# ---------------------------------------------------------------------------
# USAJobs
# ---------------------------------------------------------------------------

class USAJobsSource(JobSource):
    """Adapter for data.usajobs.gov (requires USAJOBS_API_KEY and USAJOBS_EMAIL)."""

    def __init__(self, api_key: str = "", email: str = ""):
        self._api_key = api_key or os.environ.get("USAJOBS_API_KEY", "")
        self._email = email or os.environ.get("USAJOBS_EMAIL", "")

    @property
    def name(self) -> str:
        return "usajobs"

    @property
    def available(self) -> bool:
        return bool(self._api_key and self._email)

    def fetch(self, query: str = "", location: str = "", limit: int = 25) -> list[JobPosting]:
        if not self.available:
            logger.warning("USAJobs: missing USAJOBS_API_KEY or USAJOBS_EMAIL")
            return []

        params: dict[str, str | int] = {
            "ResultsPerPage": min(limit, 25),
        }
        if query:
            params["Keyword"] = query
        if location:
            params["LocationName"] = location

        headers = {
            "Authorization-Key": self._api_key,
            "User-Agent": self._email,
        }

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(
                    "https://data.usajobs.gov/api/search",
                    params=params,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.exception("USAJobs fetch failed")
            return []

        search_result = data.get("SearchResult", {})
        items = search_result.get("SearchResultItems", [])
        jobs: list[JobPosting] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                desc = item.get("MatchedObjectDescriptor", {})
                if not isinstance(desc, dict):
                    continue

                title = str(desc.get("PositionTitle", ""))
                org = str(desc.get("OrganizationName", ""))
                summary = _strip_html(str(desc.get("QualificationSummary", "")))
                uri = str(desc.get("PositionURI", ""))
                pub_date = str(desc.get("PublicationStartDate", ""))

                # Salary
                salary_min = 0.0
                salary_max = 0.0
                remuneration = desc.get("PositionRemuneration", [])
                if isinstance(remuneration, list) and remuneration:
                    rem = remuneration[0]
                    if isinstance(rem, dict):
                        with contextlib.suppress(ValueError, TypeError):
                            salary_min = float(rem.get("MinimumRange", 0) or 0)
                        with contextlib.suppress(ValueError, TypeError):
                            salary_max = float(rem.get("MaximumRange", 0) or 0)

                # Location
                loc_parts = []
                pos_location = desc.get("PositionLocation", [])
                if isinstance(pos_location, list):
                    for pl in pos_location:
                        if isinstance(pl, dict):
                            loc_name = pl.get("LocationName", "")
                            if loc_name:
                                loc_parts.append(str(loc_name))
                loc = ", ".join(loc_parts) if loc_parts else ""

                job = JobPosting(
                    url=uri,
                    title=title,
                    company=org,
                    location=loc,
                    description=summary,
                    salary_min=salary_min,
                    salary_max=salary_max,
                    posted_date=pub_date,
                    source=self.name,
                )
                jobs.append(job)
            except Exception:
                logger.exception("USAJobs: failed to parse item")
                continue

        return jobs[:limit]


# ---------------------------------------------------------------------------
# Remotive
# ---------------------------------------------------------------------------

class RemotiveSource(JobSource):
    """Adapter for https://remotive.com/api/remote-jobs (no auth required)."""

    @property
    def name(self) -> str:
        return "remotive"

    def fetch(self, query: str = "", location: str = "", limit: int = 25) -> list[JobPosting]:
        params: dict[str, str | int] = {"limit": limit}
        if query:
            params["search"] = query

        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.get(
                    "https://remotive.com/api/remote-jobs",
                    params=params,
                    headers={"User-Agent": _USER_AGENT},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.exception("Remotive fetch failed")
            return []

        results = data.get("jobs", [])
        jobs: list[JobPosting] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            try:
                description_html = str(item.get("description", ""))
                description = _strip_html(description_html)

                job = JobPosting(
                    url=str(item.get("url", "")),
                    title=str(item.get("title", "")),
                    company=str(item.get("company_name", "")),
                    location=str(item.get("candidate_required_location", "Remote")),
                    description=description,
                    posted_date=str(item.get("publication_date", "")),
                    is_remote=True,
                    source=self.name,
                    industry=str(item.get("category", "")),
                )

                # Parse salary field if present
                salary_str = str(item.get("salary", "") or "")
                if salary_str:
                    from sentinel.scanner import extract_salary
                    sal_min, sal_max, cur = extract_salary(salary_str)
                    job.salary_min = sal_min
                    job.salary_max = sal_max
                    if sal_min > 0:
                        job.salary_currency = cur

                jobs.append(job)
            except Exception:
                logger.exception("Remotive: failed to parse item")
                continue

            if len(jobs) >= limit:
                break

        return jobs


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def get_all_sources() -> list[JobSource]:
    """Return instances of all available sources, skipping those missing required env vars."""
    sources: list[JobSource] = []

    # No-auth sources are always available
    sources.append(RemoteOKSource())
    sources.append(TheMuseSource())
    sources.append(RemotiveSource())

    # Auth-required sources -- only include if credentials are present
    adzuna = AdzunaSource()
    if adzuna.available:
        sources.append(adzuna)

    usajobs = USAJobsSource()
    if usajobs.available:
        sources.append(usajobs)

    return sources


def fetch_from_all(
    query: str = "",
    location: str = "",
    limit_per_source: int = 25,
) -> list[JobPosting]:
    """Fetch from all available sources, deduplicate by URL, return combined list."""
    sources = get_all_sources()
    all_jobs: list[JobPosting] = []
    seen_urls: set[str] = set()

    for source in sources:
        try:
            jobs = source.fetch(query=query, location=location, limit=limit_per_source)
        except Exception:
            logger.exception("Unexpected error from source %s", source.name)
            jobs = []

        for job in jobs:
            if job.url and job.url in seen_urls:
                continue
            if job.url:
                seen_urls.add(job.url)
            all_jobs.append(job)

    return all_jobs
