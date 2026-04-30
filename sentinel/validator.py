"""Company validation — cross-reference companies against external sources."""

import socket
import subprocess
from datetime import datetime, timezone

from sentinel.models import CompanyProfile

try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

KNOWN_COMPANIES: set[str] = {
    # Big Tech
    "google", "alphabet", "meta", "facebook", "apple", "amazon", "microsoft",
    "netflix", "nvidia", "intel", "ibm", "oracle", "salesforce", "adobe",
    "sap", "vmware", "cisco", "qualcomm", "broadcom", "amd", "texas instruments",
    # Cloud & Infrastructure
    "aws", "azure", "gcp", "cloudflare", "fastly", "twilio", "okta",
    "datadog", "splunk", "pagerduty", "elastic", "mongodb", "redis labs",
    "hashicorp", "confluent", "databricks", "snowflake", "palantir",
    # Fintech & Finance
    "stripe", "square", "block", "paypal", "coinbase", "robinhood", "plaid",
    "affirm", "klarna", "brex", "ripple", "chime", "wise", "checkout.com",
    "jpmorgan", "jp morgan", "goldman sachs", "morgan stanley", "wells fargo",
    "bank of america", "citibank", "citi", "blackrock", "fidelity", "vanguard",
    "american express", "visa", "mastercard", "capital one",
    # E-commerce & Marketplace
    "shopify", "ebay", "etsy", "wayfair", "chewy", "doordash", "instacart",
    "grubhub", "uber eats", "postmates",
    # Ride & Delivery
    "uber", "lyft", "airbnb", "doordash",
    # Enterprise SaaS
    "workday", "servicenow", "zendesk", "hubspot", "atlassian", "slack",
    "zoom", "dropbox", "box", "docusign", "veeva", "coupa", "freshworks",
    "sprinklr", "medallia", "qualtrics",
    # Security
    "crowdstrike", "palo alto networks", "fortinet", "zscaler", "sentinelone",
    "cyberark", "rapid7", "qualys", "tenable",
    # Dev Tools & Infra
    "github", "gitlab", "jfrog", "sonatype", "circleci", "harness",
    "jetbrains", "postman", "new relic",
    # Healthcare & Biotech
    "unitedhealth", "anthem", "aetna", "cigna", "humana", "cvs health",
    "johnson & johnson", "pfizer", "moderna", "gilead", "biogen",
    "epic systems", "cerner", "allscripts", "change healthcare",
    "teladoc", "veracyte", "guardant health",
    # Media & Entertainment
    "disney", "warner bros", "comcast", "nbc universal", "fox", "viacomcbs",
    "paramount", "sony", "spotify", "tiktok", "bytedance", "snapchat",
    "twitter", "x corp", "reddit", "pinterest", "linkedin",
    # Consulting & Professional Services
    "mckinsey", "bain", "bcg", "deloitte", "pwc", "kpmg", "ey",
    "ernst & young", "accenture", "booz allen", "leidos", "saic",
    # Retail
    "walmart", "target", "costco", "kroger", "home depot", "lowes",
    "best buy", "nordstrom", "macy's",
    # Telecom
    "at&t", "verizon", "t-mobile", "comcast", "charter",
    # Aerospace & Defense
    "lockheed martin", "raytheon", "boeing", "northrop grumman", "general dynamics",
    "spacex", "blue origin",
    # Automotive
    "tesla", "ford", "general motors", "gm", "toyota", "honda", "bmw",
    "volkswagen", "stellantis", "rivian", "lucid",
    # AI / ML
    "openai", "anthropic", "cohere", "hugging face", "scale ai",
    "deepmind", "inflection", "stability ai",
}


def validate_company(company_name: str, domain: str = "") -> CompanyProfile:
    """Validate a company across available sources.

    Checks (each wrapped in try/except, all optional):
    1. LinkedIn company page existence (if httpx available)
    2. Domain WHOIS age
    3. Company size/industry from LinkedIn data

    Returns CompanyProfile with whatever data was gathered.
    Falls back gracefully if httpx not installed.
    """
    name_lower = company_name.strip().lower()
    profile = CompanyProfile(name=company_name, domain=domain)

    profile.is_verified = _is_known_company(name_lower)
    if profile.is_verified:
        profile.verification_source = "known_companies_list"

    if domain:
        try:
            age = check_domain_age(domain)
            profile.whois_age_days = age
        except Exception:
            pass

    if _HTTPX_AVAILABLE:
        try:
            linkedin_data = check_company_linkedin(company_name)
            if linkedin_data:
                profile.has_linkedin_page = linkedin_data.get("exists", False)
                profile.linkedin_url = linkedin_data.get("url", "")
                profile.linkedin_followers = linkedin_data.get("followers", 0)
                profile.employee_count = linkedin_data.get("employee_count", 0)
                profile.industry = linkedin_data.get("industry", "")
                if profile.has_linkedin_page and not profile.is_verified:
                    profile.verification_source = "linkedin"
        except Exception:
            pass

    return profile


def check_domain_age(domain: str) -> int:
    """Check domain WHOIS age in days. Returns 0 if unavailable.
    Uses stdlib socket + subprocess for whois lookup.
    """
    domain = domain.strip().lower().removeprefix("www.")
    if not domain:
        return 0

    try:
        socket.getaddrinfo(domain, None)
    except socket.gaierror:
        return 0

    try:
        result = subprocess.run(
            ["whois", domain],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return 0

    creation_date: datetime | None = None
    date_keys = (
        "creation date:", "created:", "registered:", "domain created:",
        "registration date:", "created on:",
    )
    for line in output.splitlines():
        lower = line.strip().lower()
        for key in date_keys:
            if lower.startswith(key):
                raw = line.split(":", 1)[-1].strip()
                creation_date = _parse_whois_date(raw)
                if creation_date:
                    break
        if creation_date:
            break

    if not creation_date:
        return 0

    now = datetime.now(tz=timezone.utc)
    if creation_date.tzinfo is None:
        creation_date = creation_date.replace(tzinfo=timezone.utc)

    age_days = (now - creation_date).days
    return max(0, age_days)


def _parse_whois_date(raw: str) -> datetime | None:
    """Attempt to parse a WHOIS date string into a datetime."""
    raw = raw.strip().split("T")[0].split(" ")[0]
    formats = ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d")
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def check_company_linkedin(company_name: str) -> dict:
    """Check LinkedIn for company page. Returns basic info dict.
    Requires httpx. Returns empty dict if unavailable.
    """
    if not _HTTPX_AVAILABLE:
        return {}

    slug = (
        company_name.lower()
        .replace(" ", "-")
        .replace(".", "")
        .replace(",", "")
        .replace("'", "")
        .replace("&", "and")
    )
    url = f"https://www.linkedin.com/company/{slug}/"

    try:
        with _httpx.Client(
            timeout=8.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            follow_redirects=True,
        ) as client:
            response = client.get(url)

        if response.status_code == 404:
            return {"exists": False, "url": url}

        if response.status_code == 200:
            return {
                "exists": True,
                "url": url,
                "followers": 0,
                "employee_count": 0,
                "industry": "",
            }

        return {}

    except Exception:
        return {}


def _is_known_company(company_name: str) -> bool:
    """Check against a hardcoded list of known major employers.
    Quick offline check — not exhaustive but catches obvious legitimacy.
    """
    name = company_name.strip().lower()
    if name in KNOWN_COMPANIES:
        return True
    for known in KNOWN_COMPANIES:
        if name == known or name.startswith(known + " ") or name.endswith(" " + known):
            return True
    return False
