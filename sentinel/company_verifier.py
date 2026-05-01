"""Company legitimacy verification for job postings.

Checks whether the employer listed in a job posting actually exists and is
legitimate.  All network calls are optional and degrade gracefully so the
module works fully offline using heuristic checks only.
"""

from __future__ import annotations

import logging
import re
import socket
import ssl
import unicodedata
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from sentinel.models import CompanyProfile, JobPosting, ScamSignal, SignalCategory

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known-legitimate companies (Fortune 100 + top tech/consulting/finance firms)
# ---------------------------------------------------------------------------

_KNOWN_COMPANIES: set[str] = {
    # Big Tech
    "google", "alphabet", "meta", "facebook", "apple", "amazon", "microsoft",
    "netflix", "nvidia", "intel", "ibm", "oracle", "salesforce", "adobe",
    "sap", "vmware", "cisco", "qualcomm", "broadcom", "amd",
    "texas instruments", "western digital", "seagate", "micron",
    # Cloud & Infrastructure
    "aws", "cloudflare", "fastly", "twilio", "okta", "datadog", "splunk",
    "pagerduty", "elastic", "mongodb", "redis labs", "hashicorp", "confluent",
    "databricks", "snowflake", "palantir", "digitalocean", "linode", "rackspace",
    # Fintech & Finance
    "stripe", "square", "block", "paypal", "coinbase", "robinhood", "plaid",
    "affirm", "klarna", "brex", "chime", "wise", "checkout.com",
    "jpmorgan", "jp morgan", "goldman sachs", "morgan stanley", "wells fargo",
    "bank of america", "citibank", "citi", "blackrock", "fidelity", "vanguard",
    "american express", "visa", "mastercard", "capital one", "discover",
    "charles schwab", "td ameritrade", "merrill lynch", "raymond james",
    "state street", "pnc financial", "us bancorp", "truist",
    # E-commerce & Marketplace
    "shopify", "ebay", "etsy", "wayfair", "chewy", "doordash", "instacart",
    "grubhub", "uber eats", "postmates", "alibaba", "jd.com",
    # Ride & Delivery
    "uber", "lyft", "airbnb", "grab", "didi",
    # Enterprise SaaS
    "workday", "servicenow", "zendesk", "hubspot", "atlassian", "slack",
    "zoom", "dropbox", "box", "docusign", "veeva", "coupa", "freshworks",
    "qualtrics", "sprinklr", "medallia", "samsara", "toast", "bill.com",
    "asana", "monday.com", "notion", "figma", "miro", "lucidchart",
    # Security
    "crowdstrike", "palo alto networks", "fortinet", "zscaler", "sentinelone",
    "cyberark", "rapid7", "qualys", "tenable", "proofpoint", "mimecast",
    "carbon black", "sophos", "trend micro", "checkpoint",
    # Dev Tools & Infra
    "github", "gitlab", "jfrog", "sonatype", "circleci", "harness",
    "jetbrains", "postman", "new relic", "dynatrace", "appdynamics",
    # Healthcare & Biotech
    "unitedhealth", "anthem", "aetna", "cigna", "humana", "cvs health",
    "johnson & johnson", "pfizer", "moderna", "gilead", "biogen",
    "epic systems", "cerner", "allscripts", "change healthcare",
    "teladoc", "abbvie", "merck", "bristol myers squibb", "eli lilly",
    "amgen", "genentech", "roche", "novartis", "sanofi",
    # Media & Entertainment
    "disney", "warner bros", "comcast", "nbc universal", "fox", "viacomcbs",
    "paramount", "sony", "spotify", "tiktok", "bytedance", "snapchat",
    "twitter", "x corp", "reddit", "pinterest", "linkedin",
    # Consulting & Professional Services
    "mckinsey", "bain", "bcg", "deloitte", "pwc", "kpmg", "ey",
    "ernst & young", "accenture", "booz allen", "leidos", "saic",
    "infosys", "tata consultancy", "wipro", "hcl", "cognizant", "capgemini",
    # Retail
    "walmart", "target", "costco", "kroger", "home depot", "lowes",
    "best buy", "nordstrom", "macy's", "dollar general", "dollar tree",
    "cvs", "walgreens", "rite aid", "publix", "whole foods",
    # Telecom
    "at&t", "verizon", "t-mobile", "charter", "comcast", "dish network",
    # Aerospace & Defense
    "lockheed martin", "raytheon", "boeing", "northrop grumman",
    "general dynamics", "spacex", "blue origin", "l3harris",
    # Automotive
    "tesla", "ford", "general motors", "gm", "toyota", "honda", "bmw",
    "volkswagen", "stellantis", "rivian", "lucid", "waymo",
    # AI / ML
    "openai", "anthropic", "cohere", "hugging face", "scale ai",
    "deepmind", "stability ai", "mistral",
    # Energy & Utilities
    "exxonmobil", "chevron", "conocophillips", "nextera energy",
    "duke energy", "southern company", "dominion energy",
    # Logistics
    "fedex", "ups", "dhl", "usps", "xpo logistics", "ch robinson",
    "j.b. hunt", "werner enterprises",
    # Food & Beverage
    "nestle", "pepsi", "coca-cola", "ab inbev", "mondelez",
    "kraft heinz", "general mills", "kelloggs", "tyson foods",
    # Real Estate / RE Tech
    "zillow", "redfin", "cbre", "jll", "colliers", "cushman & wakefield",
    # Education
    "coursera", "udemy", "chegg", "duolingo", "2u", "pearson",
    # HR / Recruiting
    "adp", "paychex", "workday", "bamboohr", "greenhouse", "lever",
    # Travel
    "booking.com", "expedia", "marriott", "hilton", "hyatt", "united airlines",
    "delta", "american airlines", "southwest",
}

# ---------------------------------------------------------------------------
# Virtual-office address patterns (commonly used by scammers)
# ---------------------------------------------------------------------------

_VIRTUAL_OFFICE_PROVIDERS = re.compile(
    r"\b(regus|wework|we work|iws|intelligent office|davinci|opus virtual|"
    r"servcorp|alliance virtual|virtually there|earth class mail|"
    r"virtual office|shared office|coworking|co-working|executive suites|"
    r"postal connections|ups store|mailbox|pmb \d|suite \d{3,4}[^a-z])\b",
    re.IGNORECASE,
)

_PO_BOX = re.compile(
    r"\b(p\.?\s*o\.?\s*box|post\s+office\s+box|po\s+box)\b",
    re.IGNORECASE,
)

_GENERIC_REMOTE = re.compile(
    r"^\s*(remote|anywhere|work from home|wfh|virtual|online)\s*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Suspicious company name patterns
# ---------------------------------------------------------------------------

_RANDOM_CHARS = re.compile(r"[A-Z]{4,}\d|[A-Z]\d[A-Z]\d|[0-9]{3,}[A-Za-z]")

_GENERIC_SUFFIX = re.compile(
    r"^[A-Za-z]{3,20}\s+(Solutions|Global|International|Enterprises?|"
    r"Worldwide|Unlimited|Ventures?|Associates?|Consulting|Services?|"
    r"Group|Partners?|Network|Systems|Resources|Staffing|Placement|"
    r"Opportunities|Connections)$",
    re.IGNORECASE,
)

_EXCESSIVE_ENTITY = re.compile(
    r"(LLC|Inc|Corp|Ltd|Co\.?)\s*,?\s*(LLC|Inc|Corp|Ltd|Co\.?)",
    re.IGNORECASE,
)

# Known brand names that scammers commonly misspell or impersonate
_BRAND_NAMES = [
    "google", "amazon", "microsoft", "apple", "facebook", "meta",
    "netflix", "tesla", "paypal", "walmart", "target", "costco",
    "deloitte", "accenture", "ibm", "oracle", "salesforce", "nvidia",
]

# ---------------------------------------------------------------------------
# LinkedIn slug heuristics
# ---------------------------------------------------------------------------

_AUTO_GENERATED_SLUG = re.compile(
    r"company-\d{6,}|[a-z]+-\d{5,}|\d{6,}",
    re.IGNORECASE,
)

_LINKEDIN_COMPANY_URL = re.compile(
    r"^https?://(www\.)?linkedin\.com/company/([a-zA-Z0-9\-]+)/?$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Domain → company name fuzzy matching helpers
# ---------------------------------------------------------------------------

def _normalize_for_compare(text: str) -> str:
    """Lowercase, remove punctuation/spaces, strip unicode accents."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9]", "", text.lower())
    return text


def _domain_matches_company(company_name: str, domain: str) -> bool:
    """Return True if domain reasonably matches company name (fuzzy)."""
    cn = _normalize_for_compare(company_name)
    domain_lower = domain.lower().lstrip("www.")

    if not cn or not domain_lower:
        return False

    # Check if company name appears anywhere in the full domain
    domain_flat = _normalize_for_compare(domain_lower)
    if cn in domain_flat or domain_flat in cn:
        return True

    # Check each subdomain/domain part individually
    parts = domain_lower.replace(".", " ").split()
    for part in parts:
        dp = _normalize_for_compare(part)
        if not dp or len(dp) < 3:
            continue
        if cn in dp or dp in cn:
            return True
        ratio = SequenceMatcher(None, cn, dp).ratio()
        if ratio >= 0.70:
            return True

    return False


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def _is_misspelled_brand(name: str) -> str | None:
    """Return the closest brand name if company name looks like a misspelling."""
    normalized = _normalize_for_compare(name)
    for brand in _BRAND_NAMES:
        dist = _levenshtein(normalized, brand)
        # Allow 1 typo per 4 chars, minimum distance 1 (not exact)
        threshold = max(1, len(brand) // 4)
        if 0 < dist <= threshold:
            return brand
    return None


# ---------------------------------------------------------------------------
# CompanyVerifier
# ---------------------------------------------------------------------------

class CompanyVerifier:
    """Checks whether the employer in a job posting is legitimate.

    All network operations (DNS, HTTPS, WHOIS) are optional and wrapped in
    try/except so the module degrades gracefully when offline.
    """

    # ------------------------------------------------------------------
    # 1. Domain verification
    # ------------------------------------------------------------------

    def verify_domain(self, company_name: str, claimed_url: str = "") -> dict:
        """Check if the company's domain resolves and appears legitimate.

        Returns a dict with keys:
            domain (str), resolves (bool), has_https (bool),
            name_matches_domain (bool), is_recent_domain (bool),
            whois_age_days (int), flags (list[str])
        """
        result: dict = {
            "domain": "",
            "resolves": False,
            "has_https": False,
            "name_matches_domain": False,
            "is_recent_domain": False,
            "whois_age_days": 0,
            "flags": [],
        }

        domain = self._extract_domain_from_url(claimed_url)
        if not domain:
            result["flags"].append("no_domain_provided")
            return result

        result["domain"] = domain

        # DNS resolution check
        try:
            socket.getaddrinfo(domain, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            result["resolves"] = True
        except (socket.gaierror, OSError):
            result["resolves"] = False
            result["flags"].append("domain_does_not_resolve")
            return result

        # HTTPS validation
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with ctx.wrap_socket(sock, server_hostname=domain):
                    result["has_https"] = True
        except Exception:
            result["has_https"] = False
            result["flags"].append("no_valid_https")

        # Fuzzy name-to-domain match
        result["name_matches_domain"] = _domain_matches_company(company_name, domain)
        if not result["name_matches_domain"]:
            result["flags"].append("domain_name_mismatch")

        # WHOIS age check (optional — requires subprocess whois)
        age_days = self._get_whois_age(domain)
        result["whois_age_days"] = age_days
        if 0 < age_days < 180:
            result["is_recent_domain"] = True
            result["flags"].append("recently_registered_domain")

        return result

    def _extract_domain_from_url(self, url: str) -> str:
        """Extract bare hostname from a URL string."""
        if not url:
            return ""
        url = url.strip()
        # Strip scheme
        url = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        # Strip path and query
        domain = url.split("/")[0].split("?")[0].split("#")[0]
        # Strip port
        domain = domain.split(":")[0].lower().strip()
        # Validate it looks like a hostname
        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$", domain):
            return ""
        return domain

    def _get_whois_age(self, domain: str) -> int:
        """Return domain age in days from WHOIS, or 0 if unavailable."""
        import subprocess
        from datetime import UTC, datetime

        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$", domain):
            return 0

        try:
            result = subprocess.run(
                ["whois", domain],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout
        except Exception:
            return 0

        date_keys = (
            "creation date:", "created:", "registered:",
            "domain created:", "registration date:", "created on:",
        )
        for line in output.splitlines():
            lower = line.strip().lower()
            for key in date_keys:
                if lower.startswith(key):
                    raw = line.split(":", 1)[-1].strip()
                    creation_date = self._parse_whois_date(raw)
                    if creation_date:
                        now = datetime.now(tz=UTC)
                        if creation_date.tzinfo is None:
                            creation_date = creation_date.replace(tzinfo=UTC)
                        return max(0, (now - creation_date).days)
        return 0

    def _parse_whois_date(self, raw: str):
        """Parse a WHOIS date string. Returns datetime or None."""
        from datetime import datetime
        raw = raw.strip().split("T")[0].split(" ")[0]
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
        return None

    # ------------------------------------------------------------------
    # 2. Local heuristic company existence check
    # ------------------------------------------------------------------

    def check_company_exists(self, company_name: str) -> dict:
        """Heuristic checks to determine if a company name looks legitimate.

        Returns a dict with keys:
            is_known (bool), confidence (float), flags (list[str]),
            misspelled_brand (str | None), matched_known_name (str | None)
        """
        result: dict = {
            "is_known": False,
            "confidence": 0.50,
            "flags": [],
            "misspelled_brand": None,
            "matched_known_name": None,
        }

        if not company_name or not company_name.strip():
            result["flags"].append("empty_company_name")
            result["confidence"] = 0.0
            return result

        name = company_name.strip()
        name_lower = name.lower()

        # 1. Check built-in known companies list
        if name_lower in _KNOWN_COMPANIES:
            result["is_known"] = True
            result["confidence"] = 0.95
            result["matched_known_name"] = name_lower
            return result

        # Partial / prefix matching for variants like "Google LLC"
        for known in _KNOWN_COMPANIES:
            if name_lower == known:
                result["is_known"] = True
                result["confidence"] = 0.95
                result["matched_known_name"] = known
                return result
            if name_lower.startswith(known + " ") or name_lower.endswith(" " + known):
                result["is_known"] = True
                result["confidence"] = 0.90
                result["matched_known_name"] = known
                return result

        # 2. Check for misspelled brand names
        misspelled = _is_misspelled_brand(name)
        if misspelled:
            result["misspelled_brand"] = misspelled
            result["flags"].append("misspelled_brand_name")
            result["confidence"] = max(0.10, result["confidence"] - 0.25)

        # 3. Check for random-character patterns
        if _RANDOM_CHARS.search(name):
            result["flags"].append("random_char_pattern")
            result["confidence"] = max(0.10, result["confidence"] - 0.20)

        # 4. Check for excessive entity suffixing (e.g. "MyBiz LLC LLC")
        if _EXCESSIVE_ENTITY.search(name):
            result["flags"].append("excessive_entity_suffix")
            result["confidence"] = max(0.10, result["confidence"] - 0.15)

        # 5. Generic single-word + buzzword suffix pattern
        if _GENERIC_SUFFIX.match(name):
            parts = name.split()
            if len(parts) <= 3:
                result["flags"].append("generic_buzzword_name")
                result["confidence"] = max(0.15, result["confidence"] - 0.15)

        # 6. All-caps name (not an acronym of ≤4 chars)
        if name == name.upper() and len(name) > 4 and re.search(r"[A-Z]{5,}", name):
            result["flags"].append("all_caps_name")
            result["confidence"] = max(0.15, result["confidence"] - 0.10)

        # 7. Check against scam_data if available
        scam_names = self._load_scam_company_names()
        if scam_names:
            name_norm = _normalize_for_compare(name)
            for scam_name in scam_names:
                if _normalize_for_compare(scam_name) == name_norm:
                    result["flags"].append("known_scam_company_name")
                    result["confidence"] = 0.05
                    return result

        return result

    def _load_scam_company_names(self) -> list[str]:
        """Load known scam company names from DB or scam_data module if available."""
        names: list[str] = []
        # Try DB
        try:
            from sentinel.db import SentinelDB
            with SentinelDB() as db:
                entities = db.get_scam_entities()
            names.extend(e["name"] for e in entities if e.get("name"))
        except Exception:
            pass

        # Try importing sentinel.scam_data if it exists
        try:
            import importlib
            scam_data = importlib.import_module("sentinel.scam_data")
            if hasattr(scam_data, "SCAM_COMPANY_NAMES"):
                names.extend(scam_data.SCAM_COMPANY_NAMES)
        except (ImportError, ModuleNotFoundError):
            pass

        return names

    # ------------------------------------------------------------------
    # 3. LinkedIn presence verification
    # ------------------------------------------------------------------

    def verify_linkedin_presence(self, company_linkedin_url: str) -> dict:
        """Analyze a LinkedIn company URL for legitimacy.

        Returns a dict with keys:
            is_valid_format (bool), slug (str), slug_looks_legitimate (bool),
            flags (list[str])
        """
        result: dict = {
            "is_valid_format": False,
            "slug": "",
            "slug_looks_legitimate": False,
            "flags": [],
        }

        if not company_linkedin_url or not company_linkedin_url.strip():
            result["flags"].append("no_linkedin_url")
            return result

        url = company_linkedin_url.strip()
        match = _LINKEDIN_COMPANY_URL.match(url)
        if not match:
            result["flags"].append("invalid_linkedin_url_format")
            return result

        result["is_valid_format"] = True
        slug = match.group(2)
        result["slug"] = slug

        # Check if slug looks auto-generated (random digits, long numeric suffix)
        if _AUTO_GENERATED_SLUG.search(slug):
            result["flags"].append("auto_generated_linkedin_slug")
            result["slug_looks_legitimate"] = False
            return result

        # Slug should be all lowercase letters, hyphens, occasionally digits
        # Legitimate slugs are usually a company name in kebab-case
        if not re.match(r"^[a-z0-9][a-z0-9\-]{1,60}[a-z0-9]$", slug):
            result["flags"].append("unusual_linkedin_slug")
        else:
            result["slug_looks_legitimate"] = True

        # Overly short slugs (< 3 chars) are suspicious
        if len(slug) < 3:
            result["flags"].append("linkedin_slug_too_short")
            result["slug_looks_legitimate"] = False

        return result

    # ------------------------------------------------------------------
    # 4. Address legitimacy check
    # ------------------------------------------------------------------

    def check_address_legitimacy(self, location: str) -> dict:
        """Check for suspicious address patterns.

        Returns a dict with keys:
            is_virtual_office (bool), is_po_box (bool),
            is_generic_remote (bool), flags (list[str])
        """
        result: dict = {
            "is_virtual_office": False,
            "is_po_box": False,
            "is_generic_remote": False,
            "flags": [],
        }

        if not location or not location.strip():
            result["flags"].append("no_location_provided")
            return result

        loc = location.strip()

        if _VIRTUAL_OFFICE_PROVIDERS.search(loc):
            result["is_virtual_office"] = True
            result["flags"].append("virtual_office_address")

        if _PO_BOX.search(loc):
            result["is_po_box"] = True
            result["flags"].append("po_box_only")

        if _GENERIC_REMOTE.match(loc):
            result["is_generic_remote"] = True
            result["flags"].append("generic_remote_no_company_location")

        return result

    # ------------------------------------------------------------------
    # 5. Full verification
    # ------------------------------------------------------------------

    def full_verification(self, job: JobPosting) -> CompanyProfile:
        """Run all verification checks and return a populated CompanyProfile.

        Network calls are attempted but degrade gracefully when offline.
        """
        company_name = (job.company or "").strip()
        profile = CompanyProfile(name=company_name)

        if not company_name:
            return profile

        # --- Heuristic company existence check ---
        existence = self.check_company_exists(company_name)
        profile.is_verified = existence["is_known"]
        if existence["is_known"]:
            profile.verification_source = "known_companies_list"

        # --- LinkedIn verification ---
        if job.company_linkedin_url:
            li_result = self.verify_linkedin_presence(job.company_linkedin_url)
            profile.linkedin_url = job.company_linkedin_url
            profile.has_linkedin_page = li_result["is_valid_format"] and li_result["slug_looks_legitimate"]
            if profile.has_linkedin_page and not profile.is_verified:
                profile.verification_source = "linkedin_url"

        # --- Domain / website verification ---
        # Try to infer a URL from company name if not provided
        claimed_url = self._infer_company_url(company_name, job)
        if claimed_url:
            domain_result = self.verify_domain(company_name, claimed_url)
            profile.domain = domain_result.get("domain", "")
            profile.whois_age_days = domain_result.get("whois_age_days", 0)
            if domain_result.get("resolves") and domain_result.get("has_https"):
                if domain_result.get("name_matches_domain") and not domain_result.get("is_recent_domain"):
                    if not profile.is_verified:
                        profile.is_verified = True
                        profile.verification_source = "domain_verification"

        # --- Address check ---
        if job.location:
            addr_result = self.check_address_legitimacy(job.location)
            # Degrade is_verified if suspicious address detected
            if addr_result["is_virtual_office"] or addr_result["is_po_box"]:
                if profile.is_verified and profile.verification_source not in (
                    "known_companies_list",
                ):
                    profile.is_verified = False

        return profile

    _JOB_BOARD_DOMAINS = {
        "linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com",
        "monster.com", "dice.com", "lever.co", "greenhouse.io",
        "workday.com", "adp.com", "micronapps.com", "smartrecruiters.com",
        "icims.com", "taleo.net", "brassring.com", "ultipro.com",
        "myworkdayjobs.com", "jobvite.com", "applytojob.com",
    }

    def _infer_company_url(self, company_name: str, job: JobPosting) -> str:
        """Try to find a URL from job posting fields or infer from company name."""
        if job.url:
            url_lower = job.url.lower()
            is_job_board = any(jb in url_lower for jb in self._JOB_BOARD_DOMAINS)
            if not is_job_board:
                return job.url

        slug = re.sub(r"[^a-z0-9]", "", company_name.lower())
        if slug:
            return f"https://{slug}.com"
        return ""

    # ------------------------------------------------------------------
    # 6. Extract ScamSignal objects from verification results
    # ------------------------------------------------------------------

    def extract_verification_signals(self, job: JobPosting) -> list[ScamSignal]:
        """Generate ScamSignal objects from all verification checks.

        Signal names and weights:
            company_not_found          RED_FLAG    0.75
            company_domain_mismatch    WARNING     0.65
            virtual_office_address     WARNING     0.45
            suspicious_company_name    WARNING     0.60
            company_verified           POSITIVE    0.30
        """
        signals: list[ScamSignal] = []
        company_name = (job.company or "").strip()

        if not company_name:
            return signals

        # --- Existence / name checks ---
        existence = self.check_company_exists(company_name)

        if "misspelled_brand_name" in existence["flags"]:
            signals.append(ScamSignal(
                name="suspicious_company_name",
                category=SignalCategory.WARNING,
                weight=0.60,
                confidence=0.70,
                detail=(
                    f"Company name '{company_name}' looks like a misspelling of "
                    f"'{existence['misspelled_brand']}'"
                ),
                evidence=f"{company_name} ≈ {existence['misspelled_brand']}",
            ))

        if "known_scam_company_name" in existence["flags"]:
            signals.append(ScamSignal(
                name="suspicious_company_name",
                category=SignalCategory.WARNING,
                weight=0.60,
                confidence=0.85,
                detail=f"Company name '{company_name}' matches a known scam company",
                evidence=company_name,
            ))

        suspicious_name_flags = {
            "random_char_pattern", "excessive_entity_suffix",
            "generic_buzzword_name", "all_caps_name",
        }
        hit_name_flags = suspicious_name_flags & set(existence["flags"])
        if hit_name_flags and not existence["is_known"]:
            signals.append(ScamSignal(
                name="suspicious_company_name",
                category=SignalCategory.WARNING,
                weight=0.60,
                confidence=0.55,
                detail=f"Company name '{company_name}' matches suspicious naming patterns",
                evidence=", ".join(sorted(hit_name_flags)),
            ))

        # --- Domain verification ---
        claimed_url = self._infer_company_url(company_name, job)
        domain_result: dict = {}
        if claimed_url:
            try:
                domain_result = self.verify_domain(company_name, claimed_url)
            except Exception:
                logger.debug("Domain verification failed for %r", company_name, exc_info=True)

        has_linkedin = bool((getattr(job, "company_linkedin_url", "") or "").strip())
        if domain_result:
            if not domain_result.get("resolves", True) and not has_linkedin:
                signals.append(ScamSignal(
                    name="company_not_found",
                    category=SignalCategory.RED_FLAG,
                    weight=0.75,
                    confidence=0.70,
                    detail=f"Company domain '{domain_result.get('domain', '')}' does not resolve",
                    evidence=f"DNS lookup failed for {domain_result.get('domain', '')}",
                ))

            if domain_result.get("resolves") and not domain_result.get("name_matches_domain"):
                existence = self.check_company_exists(company_name)
                if not existence.get("is_known"):
                    signals.append(ScamSignal(
                        name="company_domain_mismatch",
                        category=SignalCategory.WARNING,
                        weight=0.50,
                        confidence=0.50,
                        detail=(
                            f"Company name '{company_name}' does not match "
                            f"domain '{domain_result.get('domain', '')}'"
                        ),
                        evidence=f"name={company_name}, domain={domain_result.get('domain', '')}",
                    ))

        # --- Address check ---
        if job.location:
            try:
                addr_result = self.check_address_legitimacy(job.location)
                if addr_result["is_virtual_office"]:
                    signals.append(ScamSignal(
                        name="virtual_office_address",
                        category=SignalCategory.WARNING,
                        weight=0.45,
                        confidence=0.65,
                        detail=(
                            f"Location '{job.location}' appears to be a virtual office "
                            "address commonly used by scammers"
                        ),
                        evidence=job.location,
                    ))
                if addr_result["is_po_box"]:
                    signals.append(ScamSignal(
                        name="virtual_office_address",
                        category=SignalCategory.WARNING,
                        weight=0.45,
                        confidence=0.55,
                        detail=f"Location '{job.location}' is a PO Box — no physical office",
                        evidence=job.location,
                    ))
            except Exception:
                logger.debug("Address check failed for %r", job.location, exc_info=True)

        # --- Positive: company passes all checks ---
        profile = self.full_verification(job)
        if profile.is_verified:
            signals.append(ScamSignal(
                name="company_verified",
                category=SignalCategory.POSITIVE,
                weight=0.30,
                confidence=0.75,
                detail=f"Company '{company_name}' passed verification checks",
                evidence=f"verification_source={profile.verification_source}",
            ))

        return signals
