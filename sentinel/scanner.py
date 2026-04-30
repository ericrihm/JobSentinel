"""LinkedIn job parsing and ingestion.

Parses job postings from raw text, HTML, JSON dicts, or batch files.
No runtime dependencies beyond Python stdlib.
"""

import json
import re
from typing import Optional

from sentinel.models import JobPosting

# ---------------------------------------------------------------------------
# Salary extraction
# ---------------------------------------------------------------------------

_SALARY_RANGE_RE = re.compile(
    r"([\$£€])?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k)?"
    r"\s*[-–—to]+\s*"
    r"([\$£€])?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k)?"
    r"(?:\s*/\s*(yr|year|hour|hr|mo|month))?"
    r"(?:\s*(USD|CAD|GBP|EUR|AUD))?",
    re.IGNORECASE,
)

_SALARY_HOURLY_RE = re.compile(
    r"([\$£€])\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:/\s*(?:hr|hour)|per\s+hour)",
    re.IGNORECASE,
)

_SALARY_ANNUAL_RE = re.compile(
    r"([\$£€])\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(k)?"
    r"\s*(?:/\s*(?:yr|year)|per\s+year|annually)",
    re.IGNORECASE,
)

_SALARY_BARE_RE = re.compile(
    r"([\$£€])\s*(\d{2,3}(?:,\d{3})+|\d{3,7})\s*(k)?\b",
    re.IGNORECASE,
)

_CURRENCY_SYMBOLS = {"$": "USD", "£": "GBP", "€": "EUR"}
_CURRENCY_CODES = {"USD", "CAD", "GBP", "EUR", "AUD"}


def _clean_num(digits: str, k_suffix: Optional[str]) -> float:
    val = float(digits.replace(",", ""))
    if k_suffix and k_suffix.lower() == "k":
        val *= 1000
    return val


def _detect_currency(text: str) -> str:
    """Return the first explicit currency code found, else infer from symbol."""
    for code in _CURRENCY_CODES:
        if re.search(r"\b" + code + r"\b", text, re.IGNORECASE):
            return code.upper()
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in text:
            return code
    return "USD"


def extract_salary(text: str) -> tuple[float, float, str]:
    """Extract salary range from text.

    Returns (min_salary, max_salary, currency).
    Handles:
    - $120k - $180k
    - $120,000 - $180,000/year
    - $50/hr  (converted to annual @ 2080 hrs)
    - 120k-180k USD
    Returns (0.0, 0.0, "USD") when nothing is found.
    """
    currency = _detect_currency(text)

    # Explicit range: 120k - 180k / $120,000 - $180,000
    m = _SALARY_RANGE_RE.search(text)
    if m:
        lo = _clean_num(m.group(2), m.group(3))
        hi = _clean_num(m.group(5), m.group(6))
        # Bare numbers like "120 - 180" are thousands
        if lo < 1000 and hi < 1000:
            lo *= 1000
            hi *= 1000
        period = (m.group(7) or "").lower()
        if period in ("hr", "hour"):
            lo *= 2080
            hi *= 2080
        elif period in ("mo", "month"):
            lo *= 12
            hi *= 12
        if m.group(8):
            currency = m.group(8).upper()
        return lo, hi, currency

    # Hourly single: $50/hr
    m = _SALARY_HOURLY_RE.search(text)
    if m:
        rate = _clean_num(m.group(2), None)
        annual = rate * 2080
        return annual, annual, currency

    # Annual single: $120k/year
    m = _SALARY_ANNUAL_RE.search(text)
    if m:
        val = _clean_num(m.group(2), m.group(3))
        if val < 1000:
            val *= 1000
        return val, val, currency

    # Bare dollar figure
    m = _SALARY_BARE_RE.search(text)
    if m:
        val = _clean_num(m.group(2), m.group(3))
        if val < 1000:
            val *= 1000
        return val, val, currency

    return 0.0, 0.0, currency


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------

_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}

_REMOTE_RE = re.compile(
    r"\b(remote|fully[- ]remote|work[- ]from[- ]home|wfh|distributed|"
    r"anywhere|location[- ]independent)\b",
    re.IGNORECASE,
)

_CITY_STATE_RE = re.compile(r"\b([A-Z][a-zA-Z .]{2,24}),\s*([A-Z]{2})\b")

_CITY_COUNTRY_RE = re.compile(
    r"\b([A-Z][a-zA-Z .]{2,24}),\s*"
    r"(United States|United Kingdom|Canada|Australia|Germany|France|India|"
    r"Netherlands|Singapore|Ireland|Israel|Poland|Brazil|Mexico|Spain|Italy)\b",
    re.IGNORECASE,
)

_LOCATION_LABEL_RE = re.compile(
    r"[Ll]ocation\s*[:\-]\s*([^\n\r,;]{3,60})"
)


def extract_location(text: str) -> str:
    """Extract location from job text. Returns 'Remote', 'City, ST', etc."""
    if _REMOTE_RE.search(text):
        return "Remote"

    m = _CITY_STATE_RE.search(text)
    if m and m.group(2) in _US_STATES:
        return f"{m.group(1).strip()}, {m.group(2)}"

    m = _CITY_COUNTRY_RE.search(text)
    if m:
        return f"{m.group(1).strip()}, {m.group(2)}"

    # Any city, XX pattern
    m = _CITY_STATE_RE.search(text)
    if m:
        return f"{m.group(1).strip()}, {m.group(2)}"

    m = _LOCATION_LABEL_RE.search(text)
    if m:
        return m.group(1).strip()

    return ""


# ---------------------------------------------------------------------------
# Experience level detection
# ---------------------------------------------------------------------------

_EXPERIENCE_LEVELS: list[tuple[str, re.Pattern]] = [
    ("executive", re.compile(
        r"\b(chief\s+\w+\s+officer|cto|ceo|cfo|coo|c[- ]suite|"
        r"vice\s+president|vp\s+of|director\s+of|executive\s+director)\b",
        re.IGNORECASE,
    )),
    ("lead", re.compile(
        r"\b(lead\s+\w*engineer|staff\s+\w*engineer|principal\s+\w*engineer|"
        r"tech\s+lead|engineering\s+manager|head\s+of|group\s+lead|"
        r"\bstaff\b|\bprincipal\b)\b",
        re.IGNORECASE,
    )),
    ("senior", re.compile(
        r"\b(senior|sr\.?\s|(?:[5-9]|10)\+?\s*years?\s+(?:of\s+)?experience)\b",
        re.IGNORECASE,
    )),
    ("mid", re.compile(
        r"\b(mid[- ]?level|mid[- ]?senior|intermediate|"
        r"[34]\+?\s*years?\s+(?:of\s+)?experience)\b",
        re.IGNORECASE,
    )),
    ("entry", re.compile(
        r"\b(entry[- ]?level|junior|jr\.?\s|new\s+grad|recent\s+grad|"
        r"graduate|internship|intern|0[- ]?2\s*years|1\+?\s*year)\b",
        re.IGNORECASE,
    )),
]


def detect_experience_level(text: str) -> str:
    """Detect experience level: entry, mid, senior, lead, executive."""
    for level, pattern in _EXPERIENCE_LEVELS:
        if pattern.search(text):
            return level
    return ""


# ---------------------------------------------------------------------------
# Employment type detection
# ---------------------------------------------------------------------------

_EMPLOYMENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("contract", re.compile(
        r"\b(contract|contractor|c2c|corp[- ]to[- ]corp|freelance|1099|"
        r"independent\s+contractor)\b",
        re.IGNORECASE,
    )),
    ("internship", re.compile(r"\b(internship|intern)\b", re.IGNORECASE)),
    ("part-time", re.compile(r"\bpart[- ]?time\b", re.IGNORECASE)),
    ("temporary", re.compile(r"\b(temporary|temp)\b", re.IGNORECASE)),
    ("full-time", re.compile(r"\bfull[- ]?time\b", re.IGNORECASE)),
]


def _detect_employment_type(text: str) -> str:
    for etype, pattern in _EMPLOYMENT_PATTERNS:
        if pattern.search(text):
            return etype
    return ""


# ---------------------------------------------------------------------------
# Helper: strip HTML tags and decode common entities
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]{2,}")
_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
    "&ndash;": "–", "&mdash;": "—", "&hellip;": "…",
}


def _strip_html(html: str) -> str:
    text = _HTML_TAG_RE.sub(" ", html)
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Core text parser
# ---------------------------------------------------------------------------

_POSTED_DATE_RE = re.compile(
    r"(?:posted|date)\s*[:\-]?\s*"
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

_APPLICANTS_RE = re.compile(
    r"(\d[\d,]*)\s+(?:applicants?|applications?|people\s+(?:have\s+)?applied)",
    re.IGNORECASE,
)

_RECRUITER_RE = re.compile(
    r"(?:posted\s+by|recruiter|contact\s+person)\s*[:\-]?\s*"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
    re.IGNORECASE,
)

_TITLE_LABEL_RE = re.compile(
    r"(?:job\s+title|position|role)\s*[:\-]\s*([^\n\r]{3,80})",
    re.IGNORECASE,
)

_COMPANY_LABEL_RE = re.compile(
    r"(?:company|employer|organization|hiring\s+company)\s*[:\-]\s*([^\n\r]{2,60})",
    re.IGNORECASE,
)


def parse_job_text(
    text: str,
    title: str = "",
    company: str = "",
    url: str = "",
) -> JobPosting:
    """Parse raw job posting text into a structured JobPosting.

    Extracts: salary, location, experience level, employment type, remote status,
    posted date, applicant count, and recruiter name via regex.
    """
    sal_min, sal_max, currency = extract_salary(text)
    location = extract_location(text)
    experience = detect_experience_level(text)
    employment_type = _detect_employment_type(text)
    is_remote = bool(_REMOTE_RE.search(text))

    if not title:
        m = _TITLE_LABEL_RE.search(text)
        if m:
            title = m.group(1).strip()

    if not company:
        m = _COMPANY_LABEL_RE.search(text)
        if m:
            company = m.group(1).strip()

    posted_date = ""
    m = _POSTED_DATE_RE.search(text)
    if m:
        posted_date = m.group(1).strip()

    applicant_count = 0
    m = _APPLICANTS_RE.search(text)
    if m:
        try:
            applicant_count = int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    recruiter_name = ""
    m = _RECRUITER_RE.search(text)
    if m:
        recruiter_name = m.group(1).strip()

    return JobPosting(
        url=url,
        title=title,
        company=company,
        location=location,
        description=text,
        salary_min=sal_min,
        salary_max=sal_max,
        salary_currency=currency,
        posted_date=posted_date,
        applicant_count=applicant_count,
        experience_level=experience,
        employment_type=employment_type,
        is_remote=is_remote,
    )


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

_JSON_LD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)

# LinkedIn-specific HTML selectors (class names are stable across versions)
_LI_TITLE_RE = re.compile(
    r'<h1[^>]*class="[^"]*(?:job-title|topcard__title)[^"]*"[^>]*>(.*?)</h1>',
    re.DOTALL | re.IGNORECASE,
)
_LI_COMPANY_RE = re.compile(
    r'<(?:a|span)[^>]*class="[^"]*(?:topcard__org-name-link|company-name)[^"]*"[^>]*>(.*?)</(?:a|span)>',
    re.DOTALL | re.IGNORECASE,
)
_LI_LOCATION_RE = re.compile(
    r'<span[^>]*class="[^"]*topcard__flavor--bullet[^"]*"[^>]*>(.*?)</span>',
    re.DOTALL | re.IGNORECASE,
)
_LI_DESCRIPTION_RE = re.compile(
    r'<div[^>]*class="[^"]*(?:description__text|show-more-less-html)[^"]*"[^>]*>(.*?)</div\s*>',
    re.DOTALL | re.IGNORECASE,
)


def _extract_json_ld(html: str) -> Optional[dict]:
    """Return the first JobPosting JSON-LD block found in HTML, or None."""
    for m in _JSON_LD_RE.finditer(html):
        try:
            data = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("@type") == "JobPosting":
                    return item
        elif isinstance(data, dict) and data.get("@type") == "JobPosting":
            return data
    return None


def parse_job_html(html: str, url: str = "") -> JobPosting:
    """Parse LinkedIn job page HTML into a JobPosting.

    Strategy:
    1. Extract JSON-LD structured data (most reliable).
    2. Regex-match LinkedIn-specific HTML class patterns.
    3. Strip all HTML and call parse_job_text as final fallback.
    """
    # --- Pass 1: JSON-LD ---
    ld = _extract_json_ld(html)
    if ld:
        ld_description = _strip_html(ld.get("description", ""))
        ld_title = ld.get("title", "")
        hiring_org = ld.get("hiringOrganization") or {}
        ld_company = hiring_org.get("name", "") if isinstance(hiring_org, dict) else ""
        ld_url = url or ld.get("url", "")

        job = parse_job_text(ld_description, ld_title, ld_company, ld_url)

        # Overlay reliable structured fields
        location_data = ld.get("jobLocation") or {}
        if isinstance(location_data, dict):
            address = location_data.get("address") or {}
            if isinstance(address, dict):
                locality = address.get("addressLocality", "")
                region = address.get("addressRegion", "")
                country = address.get("addressCountry", "")
                parts = [p for p in (locality, region or country) if p]
                if parts:
                    job.location = ", ".join(parts)

        emp_type = ld.get("employmentType", "")
        if emp_type:
            job.employment_type = str(emp_type).lower()

        date_posted = ld.get("datePosted", "")
        if date_posted:
            job.posted_date = str(date_posted)

        base_salary = ld.get("baseSalary") or {}
        if isinstance(base_salary, dict):
            value = base_salary.get("value") or {}
            if isinstance(value, dict):
                lo = value.get("minValue") or value.get("value")
                hi = value.get("maxValue") or value.get("value")
                if lo is not None:
                    try:
                        job.salary_min = float(lo)
                    except (ValueError, TypeError):
                        pass
                if hi is not None:
                    try:
                        job.salary_max = float(hi)
                    except (ValueError, TypeError):
                        pass
            cur = base_salary.get("currency", "")
            if cur:
                job.salary_currency = str(cur).upper()

        company_url = hiring_org.get("sameAs", "") if isinstance(hiring_org, dict) else ""
        if company_url:
            job.company_linkedin_url = str(company_url)

        return job

    # --- Pass 2: LinkedIn HTML class patterns ---
    title = ""
    m = _LI_TITLE_RE.search(html)
    if m:
        title = _strip_html(m.group(1))

    company = ""
    m = _LI_COMPANY_RE.search(html)
    if m:
        company = _strip_html(m.group(1))

    location = ""
    m = _LI_LOCATION_RE.search(html)
    if m:
        location = _strip_html(m.group(1))

    description = ""
    m = _LI_DESCRIPTION_RE.search(html)
    if m:
        description = _strip_html(m.group(1))

    if description:
        job = parse_job_text(description, title, company, url)
        if location:
            job.location = location
        job.raw_html = html
        return job

    # --- Pass 3: strip everything ---
    job = parse_job_text(_strip_html(html), title, company, url)
    if location:
        job.location = location
    job.raw_html = html
    return job


# ---------------------------------------------------------------------------
# JSON dict parser
# ---------------------------------------------------------------------------

# Maps JobPosting field names to lists of known aliases in external JSON
_FIELD_ALIASES: dict[str, list[str]] = {
    "url": ["url", "link", "job_url", "jobUrl", "job_link"],
    "title": ["title", "job_title", "jobTitle", "position", "role", "name"],
    "company": ["company", "company_name", "companyName", "employer", "organization"],
    "location": ["location", "city", "place", "jobLocation", "job_location"],
    "description": ["description", "body", "content", "text", "details",
                    "job_description", "jobDescription", "full_description"],
    "salary_min": ["salary_min", "salaryMin", "min_salary", "salary_from",
                   "compensation_min", "pay_min"],
    "salary_max": ["salary_max", "salaryMax", "max_salary", "salary_to",
                   "compensation_max", "pay_max"],
    "salary_currency": ["salary_currency", "currency", "salaryCurrency", "pay_currency"],
    "salary": ["salary", "pay", "compensation"],  # free-form string fallback
    "posted_date": ["posted_date", "postedDate", "date_posted", "datePosted",
                    "date", "listed_at", "created_at"],
    "applicant_count": ["applicant_count", "applicantCount", "applicants",
                        "applications", "num_applicants"],
    "experience_level": ["experience_level", "experienceLevel", "seniority",
                         "level", "experience", "seniority_level"],
    "employment_type": ["employment_type", "employmentType", "job_type",
                        "jobType", "type", "work_type"],
    "industry": ["industry", "sector", "job_function"],
    "company_size": ["company_size", "companySize", "employees",
                     "employee_count", "company_employees"],
    "company_linkedin_url": ["company_linkedin_url", "companyLinkedinUrl",
                             "company_url", "companyUrl", "org_url"],
    "recruiter_name": ["recruiter_name", "recruiterName", "recruiter",
                       "posted_by", "hiring_manager", "contact"],
    "is_remote": ["is_remote", "isRemote", "remote", "work_from_home",
                  "wfh", "remote_ok"],
    "is_repost": ["is_repost", "isRepost", "repost", "reposted"],
    "source": ["source", "platform", "origin"],
}


def _pick(data: dict, aliases: list[str], default=None):
    """Return the first value found for any of the given key aliases."""
    for key in aliases:
        if key in data:
            return data[key]
    return default


def _coerce_bool(val) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return bool(val)
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1", "remote", "y")
    return None


def parse_job_json(data: dict) -> JobPosting:
    """Parse a JSON dict (from LinkedIn API, scraper, or file) into a JobPosting."""
    description = str(_pick(data, _FIELD_ALIASES["description"], "") or "")
    title = str(_pick(data, _FIELD_ALIASES["title"], "") or "")
    company = str(_pick(data, _FIELD_ALIASES["company"], "") or "")
    url = str(_pick(data, _FIELD_ALIASES["url"], "") or "")

    # Base: use text parser to fill any gaps regex can fill
    job = parse_job_text(description, title, company, url)

    # Overlay authoritative JSON fields
    loc = _pick(data, _FIELD_ALIASES["location"])
    if loc:
        job.location = str(loc)

    sal_min = _pick(data, _FIELD_ALIASES["salary_min"])
    if sal_min is not None:
        try:
            job.salary_min = float(sal_min)
        except (ValueError, TypeError):
            pass

    sal_max = _pick(data, _FIELD_ALIASES["salary_max"])
    if sal_max is not None:
        try:
            job.salary_max = float(sal_max)
        except (ValueError, TypeError):
            pass

    currency = _pick(data, _FIELD_ALIASES["salary_currency"])
    if currency:
        job.salary_currency = str(currency).upper()

    # Free-form salary string fallback (only if no numeric fields present)
    if job.salary_min == 0.0:
        sal_str = _pick(data, _FIELD_ALIASES["salary"])
        if sal_str and isinstance(sal_str, str):
            lo, hi, cur = extract_salary(sal_str)
            job.salary_min, job.salary_max = lo, hi
            if lo > 0:
                job.salary_currency = cur

    posted = _pick(data, _FIELD_ALIASES["posted_date"])
    if posted:
        job.posted_date = str(posted)

    applicants = _pick(data, _FIELD_ALIASES["applicant_count"])
    if applicants is not None:
        try:
            job.applicant_count = int(applicants)
        except (ValueError, TypeError):
            pass

    exp = _pick(data, _FIELD_ALIASES["experience_level"])
    if exp:
        job.experience_level = str(exp).lower()

    emp_type = _pick(data, _FIELD_ALIASES["employment_type"])
    if emp_type:
        job.employment_type = str(emp_type).lower()

    industry = _pick(data, _FIELD_ALIASES["industry"])
    if industry:
        job.industry = str(industry)

    company_size = _pick(data, _FIELD_ALIASES["company_size"])
    if company_size is not None:
        job.company_size = str(company_size)

    company_url = _pick(data, _FIELD_ALIASES["company_linkedin_url"])
    if company_url:
        job.company_linkedin_url = str(company_url)

    recruiter = _pick(data, _FIELD_ALIASES["recruiter_name"])
    if recruiter:
        job.recruiter_name = str(recruiter)

    is_remote = _coerce_bool(_pick(data, _FIELD_ALIASES["is_remote"]))
    if is_remote is not None:
        job.is_remote = is_remote

    is_repost = _coerce_bool(_pick(data, _FIELD_ALIASES["is_repost"]))
    if is_repost is not None:
        job.is_repost = is_repost

    source = _pick(data, _FIELD_ALIASES["source"], "linkedin")
    job.source = str(source)

    return job


# ---------------------------------------------------------------------------
# Batch file loader
# ---------------------------------------------------------------------------

def load_jobs_from_file(path: str) -> list[JobPosting]:
    """Load jobs from a JSON file (array of job objects).

    Supports:
    - Array of job dicts:  [{"title": ..., "description": ...}, ...]
    - Single job dict:     {"title": ..., "description": ...}
    - NDJSON:              one JSON object per line
    """
    with open(path, encoding="utf-8") as fh:
        content = fh.read().strip()

    if not content:
        return []

    # Try parsing as standard JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [parse_job_json(item) for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parse_job_json(parsed)]
        return []
    except json.JSONDecodeError:
        pass

    # Fall back to NDJSON (newline-delimited JSON)
    jobs: list[JobPosting] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                jobs.append(parse_job_json(item))
        except json.JSONDecodeError:
            continue

    return jobs
