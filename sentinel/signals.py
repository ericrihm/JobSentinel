"""Signal extraction: 50+ scam indicators for LinkedIn job postings."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING

from sentinel.adversarial import EvasionDetector, TextNormalizer
from sentinel.models import JobPosting, ScamSignal, SignalCategory

if TYPE_CHECKING:
    from sentinel.models import ScamPattern

logger = logging.getLogger(__name__)

# Module-level singletons — cheap to construct, safe to share across calls
_NORMALIZER = TextNormalizer()
_EVASION_DETECTOR = EvasionDetector()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERSONAL_DOMAINS = re.compile(
    r"@(gmail|yahoo|hotmail|outlook|aol|icloud|protonmail|mail|ymail|"
    r"live|msn|me|googlemail)\.",
    re.IGNORECASE,
)

_UPFRONT_PAY = re.compile(
    r"\b(fee required|send money|training fee|buy equipment|purchase (your |a )?equipment|"
    r"starter kit fee|background check fee|pay (a |the )?deposit|wire (me|us)|"
    r"upfront (cost|fee|payment)|advance fee)\b",
    re.IGNORECASE,
)

_PERSONAL_INFO = re.compile(
    r"\b(social security|SSN|bank account( number)?|routing number|"
    r"credit card( number)?|debit card|full (name and )?address|"
    r"passport (number|copy)|drivers? licen[sc]e)\b",
    re.IGNORECASE,
)

# Fuzzy "guaranteed" sub-patterns tolerate char_duplicate (guarranteed,
# guarantteed, guaranteedd, GGuaranteed) and char_drop (guaranted, guarnteed,
# guranteed, Guanteed, Guaraneed).
# Two alternations:
#   1. Normal path: requires 't' (g{1,2} handles doubled leading G)
#   2. t-dropped path: g+uar+n+ee+d (requires r and n to compensate)
_GUAR = (
    r"g{1,2}(?:ua?|a)r{0,2}a?n{0,2}t{1,2}e{1,3}d{0,2}"
    r"|g{1,2}ua?r{1,2}a?ne{1,3}d{0,2}"
)

# Fuzzy "income" sub-pattern covers all 1-letter drop variants:
# ncome (i dropped), icome (n dropped), incme (o dropped),
# incoe (m dropped), incom (e dropped), plus char_dup doubles.
_INCOME = (
    r"inc{1,2}o{1,2}m{1,2}e{0,2}"
    r"|inc{1,2}m{1,2}e{1,2}"
    r"|inc{1,2}o{1,2}e{1,2}"
    r"|in{1,2}o{1,2}m{1,2}e{0,2}"
    r"|n{1,2}c{1,2}o{1,2}m{1,2}e{0,2}"
    r"|ic{1,2}o{1,2}m{1,2}e{0,2}"
)

_GUARANTEED_INCOME = re.compile(
    r"\b((?:" + _GUAR + r")\s+(?:salar{1,2}y|" + _INCOME + r"|pay|earnings?|profit)|"
    r"earn\s+\$[\d,]+\s*(?:a\s+|per\s+)?(?:day|daily|week|hour|hr)\s+(?:" + _GUAR + r")|"
    r"(?:(?:" + _GUAR + r")|promise[sd])\s+to\s+(?:earn|make|pay))\b",
    re.IGNORECASE,
)

_CRYPTO = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|crypto(currency)?|gift card|"
    r"western union|moneygram|wire transfer|zelle|cashapp|venmo)\b",
    re.IGNORECASE,
)

_URGENCY = re.compile(
    r"\b(apply (now|immediately|today|asap)|limited (spots?|openings?|positions?)|"
    r"hiring (immediately|now|today|asap)|urgent(ly)? (hiring|needed)|"
    r"positions? (filling|fill) fast|don'?t (miss|wait)|act (now|fast|quickly)|"
    r"only \d+ (spots?|seats?|positions?) (left|remaining|available))\b",
    re.IGNORECASE,
)

# Fuzzy patterns tolerate char_duplicate (Noo, expperience, experiencce, nneedded)
# and char_drop (no experienc, no exprience, no experience neded).
# exp{1,2}: handles doubled 'p' in experience
# c{0,2}: handles doubled 'cc' in experience
# n{1,2} for needed: handles doubled 'nn' at start of 'needed'
_NO_EXPERIENCE = re.compile(
    r"\b("
    r"n{1,2}o{1,2}\s+exp{1,2}e?r{1,2}i{0,2}en{1,2}c{0,2}e{0,2}\s+"
    r"(requ?i{0,2}r{0,2}e{0,2}d{0,2}|n{1,2}e{1,3}d{1,2}e{0,2}d{0,2}|neces{1,2}a?r{1,2}y)|"
    r"no\s+(skills?|qualifications?|background)\s+(required|needed)|"
    r"anyone can|so easy|simple\s+(job|work|tasks?)"
    r")\b",
    re.IGNORECASE,
)

_GRAMMAR_CAPS = re.compile(r"\b[A-Z]{5,}\b")
_GRAMMAR_PUNCT = re.compile(r"[!?]{2,}")
_GRAMMAR_EMOJI = re.compile(r"[\U0001F300-\U0001FFFF]|[☀-⟿]", re.UNICODE)

_SUSPICIOUS_LINKS = re.compile(
    r"(bit\.ly|tinyurl\.com|t\.co/|forms\.gle|docs\.google\.com/forms|"
    r"t\.me/|telegram\.me/|wa\.me/|whatsapp\.com/|typeform\.com)",
    re.IGNORECASE,
)

_TECH_STACK = re.compile(
    r"\b(python|java(?:script)?|typescript|golang|rust|ruby|php|swift|kotlin|"
    r"c\+\+|c#|scala|react|angular|vue|django|flask|fastapi|spring|rails|"
    r"node(?:\.js)?|express|postgres(?:ql)?|mysql|mongodb|redis|kafka|"
    r"kubernetes|docker|terraform|aws|azure|gcp|sql|linux|bash|git)\b",
    re.IGNORECASE,
)

_EXPERIENCE_YRS = re.compile(
    r"\b(\d+\+?\s*(?:years?|yrs?)\s*(of\s+)?(experience|exp))\b",
    re.IGNORECASE,
)

_DEGREE = re.compile(
    r"\b(bachelor|b\.s\.|b\.a\.|master|m\.s\.|m\.a\.|mba|ph\.?d|"
    r"associate|degree (in|required)|diploma)\b",
    re.IGNORECASE,
)

_BENEFITS = re.compile(
    r"\b(health insurance|dental|vision|401k|equity|stock options?|"
    r"pto|paid time off|parental leave|bonus|pension|retirement)\b",
    re.IGNORECASE,
)


def _full_text(job: JobPosting) -> str:
    return f"{job.title} {job.description}".strip()


def _days_since_posted(posted_date: str) -> int | None:
    if not posted_date:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(posted_date, fmt).replace(tzinfo=UTC)
            return (datetime.now(UTC) - dt).days
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Red Flags — weight 0.7–0.95
# ---------------------------------------------------------------------------

def check_upfront_payment(job: JobPosting) -> ScamSignal | None:
    m = _UPFRONT_PAY.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="upfront_payment",
        category=SignalCategory.RED_FLAG,
        weight=0.95,
        confidence=0.90,
        detail="Posting requests upfront payment or equipment purchase",
        evidence=m.group(0),
    )


def check_personal_info_request(job: JobPosting) -> ScamSignal | None:
    m = _PERSONAL_INFO.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="personal_info_request",
        category=SignalCategory.RED_FLAG,
        weight=0.92,
        confidence=0.88,
        detail="Posting requests sensitive personal/financial info before interview",
        evidence=m.group(0),
    )


def check_guaranteed_income(job: JobPosting) -> ScamSignal | None:
    m = _GUARANTEED_INCOME.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="guaranteed_income",
        category=SignalCategory.RED_FLAG,
        weight=0.85,
        confidence=0.82,
        detail="Posting promises guaranteed income — legitimate employers never do this",
        evidence=m.group(0),
    )


def check_suspicious_email_domain(job: JobPosting) -> ScamSignal | None:
    # Check description and recruiter contact field
    combined = f"{_full_text(job)} {job.recruiter_name}"
    m = _PERSONAL_DOMAINS.search(combined)
    if not m:
        return None
    return ScamSignal(
        name="suspicious_email_domain",
        category=SignalCategory.RED_FLAG,
        weight=0.78,
        confidence=0.75,
        detail="Corporate role advertised with personal email domain",
        evidence=m.group(0),
    )


def check_crypto_payment(job: JobPosting) -> ScamSignal | None:
    m = _CRYPTO.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="crypto_payment",
        category=SignalCategory.RED_FLAG,
        weight=0.90,
        confidence=0.87,
        detail="Untraceable payment method mentioned (crypto/wire/gift card)",
        evidence=m.group(0),
    )


def check_no_company_presence(job: JobPosting) -> ScamSignal | None:
    if not job.company.strip():
        return ScamSignal(
            name="no_company_presence",
            category=SignalCategory.RED_FLAG,
            weight=0.85,
            confidence=0.80,
            detail="No company name listed",
            evidence="company field is empty",
        )
    if not job.company_linkedin_url.strip():
        try:
            from sentinel.company_verifier import CompanyVerifier
            cv = CompanyVerifier()
            existence = cv.check_company_exists(job.company)
            if existence.get("is_known"):
                return None
        except ImportError:
            pass
        return ScamSignal(
            name="no_company_presence",
            category=SignalCategory.WARNING,
            weight=0.45,
            confidence=0.50,
            detail="No company LinkedIn page linked",
            evidence="company_linkedin_url is empty",
        )
    return None


# ---------------------------------------------------------------------------
# Salary benchmark helpers
# ---------------------------------------------------------------------------

# Maps keywords found in a job title to a salary_benchmarks category.
# Evaluated in order — first match wins.
_TITLE_CATEGORY_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(software|engineer|developer|programmer|devops|sre|backend|frontend|full.?stack|sde|swe)\b", re.IGNORECASE), "software_engineer"),
    (re.compile(r"\b(data scientist|machine learning|ml engineer|ai engineer)\b", re.IGNORECASE), "data_scientist"),
    (re.compile(r"\b(data analyst|business analyst|analytics)\b", re.IGNORECASE), "data_analyst"),
    (re.compile(r"\b(nurs|rn\b|lpn|cna|healthcare|medical assistant|patient)\b", re.IGNORECASE), "nursing"),
    (re.compile(r"\b(account(ant|ing)|bookkeep|cpa|auditor|financial analyst|controller)\b", re.IGNORECASE), "accounting"),
    (re.compile(r"\b(project manager|scrum master|program manager|pmo)\b", re.IGNORECASE), "project_manager"),
    (re.compile(r"\b(marketing|seo|content (strateg|market)|social media manager|brand)\b", re.IGNORECASE), "marketing"),
    (re.compile(r"\b(sales (rep|executive|manager|associate)|account executive|business development)\b", re.IGNORECASE), "sales"),
    (re.compile(r"\b(human resources|hr (manager|generalist|coordinator)|recruiter|talent acquisition)\b", re.IGNORECASE), "human_resources"),
    (re.compile(r"\b(graphic design|ux|ui designer|visual designer|illustrator)\b", re.IGNORECASE), "graphic_designer"),
    (re.compile(r"\b(teach|instructor|tutor|professor|educator|curriculum)\b", re.IGNORECASE), "teacher"),
    (re.compile(r"\b(warehouse|logistics|forklift|shipping|inventory|fulfillment|picker|packer)\b", re.IGNORECASE), "warehouse"),
    (re.compile(r"\b(paralegal|attorney|lawyer|counsel|legal)\b", re.IGNORECASE), "legal"),
    (re.compile(r"\b(customer (service|support|success)|call center|helpdesk|support rep)\b", re.IGNORECASE), "customer_service"),
    (re.compile(r"\b(admin(istrative)?( assistant)?|office manager|executive assistant|receptionist|clerk)\b", re.IGNORECASE), "admin_assistant"),
]

# Maps raw experience_level strings to benchmark level keys
_LEVEL_MAP: dict[str, str] = {
    "entry": "entry",
    "entry level": "entry",
    "entry-level": "entry",
    "internship": "entry",
    "junior": "entry",
    "mid": "mid",
    "mid-level": "mid",
    "mid level": "mid",
    "associate": "mid",
    "intermediate": "mid",
    "senior": "senior",
    "lead": "senior",
    "principal": "senior",
    "staff": "senior",
    "director": "senior",
    "manager": "mid",  # conservative default for generic manager titles
}


def classify_job_category(title: str) -> str | None:
    """Return salary_benchmarks category key for a job title, or None."""
    for pattern, category in _TITLE_CATEGORY_MAP:
        if pattern.search(title):
            return category
    return None


def normalize_level(experience_level: str) -> str:
    """Map raw experience_level to 'entry', 'mid', or 'senior'. Defaults to 'mid'."""
    return _LEVEL_MAP.get(experience_level.lower().strip(), "mid")


@lru_cache(maxsize=256)
def _get_benchmark_cached(category: str, level: str) -> tuple[int, int, int, int] | None:
    """Fetch (p25, p50, p75, p90) from DB; cached per process."""
    try:
        from sentinel.db import SentinelDB
        with SentinelDB() as db:
            row = db.get_salary_benchmark(category, level)
        if row is None:
            return None
        return (row["p25"], row["p50"], row["p75"], row["p90"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Warnings — weight 0.4–0.7
# ---------------------------------------------------------------------------

def check_salary_anomaly(job: JobPosting) -> ScamSignal | None:
    lo, hi = job.salary_min, job.salary_max
    level_raw = (job.experience_level or "").lower()
    entry_levels = {"entry", "entry level", "entry-level", "internship", "junior"}

    # --- Wide-range check (existing fallback) ---
    if lo > 0 and hi > 0 and hi / lo > 3.0:
        return ScamSignal(
            name="salary_anomaly",
            category=SignalCategory.WARNING,
            weight=0.55,
            confidence=0.60,
            detail=f"Salary range is suspiciously wide: ${lo:,.0f}–${hi:,.0f}",
            evidence=f"{lo}–{hi}",
        )

    # --- Market-rate comparison ---
    ceiling = hi if hi > 0 else lo
    if ceiling > 0:
        category = classify_job_category(job.title)
        if category:
            level_key = normalize_level(level_raw)
            benchmark = _get_benchmark_cached(category, level_key)
            if benchmark is not None:
                _p25, _p50, _p75, p90 = benchmark
                if ceiling > 3.0 * p90:
                    return ScamSignal(
                        name="salary_anomaly",
                        category=SignalCategory.WARNING,
                        weight=0.85,
                        confidence=0.80,
                        detail=(
                            f"Salary ${ceiling:,.0f} is >3× the P90 market rate "
                            f"(${p90:,}) for {level_key} {category.replace('_', ' ')} — highly suspicious"
                        ),
                        evidence=f"{ceiling} vs P90={p90}",
                    )
                if ceiling > 2.0 * p90:
                    return ScamSignal(
                        name="salary_anomaly",
                        category=SignalCategory.WARNING,
                        weight=0.70,
                        confidence=0.72,
                        detail=(
                            f"Salary ${ceiling:,.0f} is >2× the P90 market rate "
                            f"(${p90:,}) for {level_key} {category.replace('_', ' ')} — suspicious"
                        ),
                        evidence=f"{ceiling} vs P90={p90}",
                    )

    # --- Legacy entry-level ceiling check (no category match) ---
    if level_raw in entry_levels and ceiling > 500_000:
        return ScamSignal(
            name="salary_anomaly",
            category=SignalCategory.WARNING,
            weight=0.70,
            confidence=0.72,
            detail=f"Unrealistically high salary for {level_raw} role: ${ceiling:,.0f}",
            evidence=str(ceiling),
        )

    return None


def check_vague_description(job: JobPosting) -> ScamSignal | None:
    words = job.description.split()
    if len(words) >= 20:
        return None
    weight = 0.50 if len(words) < 8 else 0.35
    return ScamSignal(
        name="vague_description",
        category=SignalCategory.WARNING,
        weight=weight,
        confidence=0.60,
        detail=f"Job description is extremely sparse ({len(words)} words)",
        evidence=job.description[:120],
    )


def check_no_qualifications(job: JobPosting) -> ScamSignal | None:
    # Only meaningful for non-trivially short descriptions
    if len(job.description.split()) < 20:
        return None
    text = job.description
    has_qual = bool(
        _TECH_STACK.search(text)
        or _EXPERIENCE_YRS.search(text)
        or _DEGREE.search(text)
        or re.search(r"\b(require[sd]?|must have|qualif|proficien|skill)\b", text, re.IGNORECASE)
    )
    if has_qual:
        return None
    return ScamSignal(
        name="no_qualifications",
        category=SignalCategory.WARNING,
        weight=0.48,
        confidence=0.55,
        detail="Non-trivial description but no skills, degree, or experience requirements",
        evidence="",
    )


def check_urgency_language(job: JobPosting) -> ScamSignal | None:
    m = _URGENCY.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="urgency_language",
        category=SignalCategory.WARNING,
        weight=0.58,
        confidence=0.65,
        detail="Artificial urgency language used to pressure applicants",
        evidence=m.group(0),
    )


def check_wfh_unrealistic_pay(job: JobPosting) -> ScamSignal | None:
    text = _full_text(job)
    is_wfh = job.is_remote or bool(
        re.search(
            r"\b(w{1,2}o{0,2}r{1,2}k{1,2}\s+f{1,2}r{0,2}o?m{0,2}\s+h{1,2}o{0,2}m?e{0,2}|"
            r"remote|wfh|w{1,2}o{0,2}r{1,2}k{1,2}\s+at\s+h{1,2}o{0,2}m?e{0,2})\b",
            text, re.IGNORECASE
        )
    )
    if not is_wfh:
        return None
    m = _NO_EXPERIENCE.search(text)
    if not m:
        return None
    return ScamSignal(
        name="wfh_unrealistic_pay",
        category=SignalCategory.WARNING,
        weight=0.65,
        confidence=0.68,
        detail="Remote 'no-experience-needed' role — common reshipping/money-mule vector",
        evidence=m.group(0),
    )


def check_low_recruiter_connections(job: JobPosting) -> ScamSignal | None:
    n = job.recruiter_connections
    if n <= 0:
        return None
    if n >= 50:
        return None
    weight = 0.62 if n < 20 else 0.45
    return ScamSignal(
        name="low_recruiter_connections",
        category=SignalCategory.WARNING,
        weight=weight,
        confidence=0.60,
        detail=f"Recruiter has only {n} LinkedIn connections — likely a fake profile",
        evidence=str(n),
    )


# ---------------------------------------------------------------------------
# Ghost Job — weight 0.4–0.6
# ---------------------------------------------------------------------------

def check_stale_posting(job: JobPosting) -> ScamSignal | None:
    days = _days_since_posted(job.posted_date)
    if days is None:
        return None
    if days > 60:
        return ScamSignal(
            name="stale_posting",
            category=SignalCategory.GHOST_JOB,
            weight=0.58,
            confidence=0.70,
            detail=f"Posting is {days} days old with no visible activity",
            evidence=f"{days} days since posted",
        )
    if days > 30:
        return ScamSignal(
            name="stale_posting",
            category=SignalCategory.GHOST_JOB,
            weight=0.42,
            confidence=0.55,
            detail=f"Posting is {days} days old — may be a ghost job",
            evidence=f"{days} days since posted",
        )
    return None


def check_repost_pattern(job: JobPosting) -> ScamSignal | None:
    if not job.is_repost:
        return None
    return ScamSignal(
        name="repost_pattern",
        category=SignalCategory.GHOST_JOB,
        weight=0.50,
        confidence=0.60,
        detail="Role has been reposted — no hires made from previous postings",
        evidence="is_repost=True",
    )


_TALENT_POOL = re.compile(
    r"\b(talent pipeline|talent community|talent pool|future openings?|"
    r"expressions? of interest|join our (talent|candidate) network|"
    r"pipeline (of candidates|for future)|upcoming opportunities)\b",
    re.IGNORECASE,
)

_GENERIC_TITLES = re.compile(
    r"^\s*(various positions?|multiple openings?|multiple positions?|"
    r"team member|general (applicant|application)|open application|"
    r"general (hire|hiring)|various roles?|multiple roles?)\s*$",
    re.IGNORECASE,
)


def check_talent_pool_language(job: JobPosting) -> ScamSignal | None:
    """Ghost job signal: talent pool / future-openings language indicates non-real position."""
    m = _TALENT_POOL.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="talent_pool_language",
        category=SignalCategory.GHOST_JOB,
        weight=0.55,
        confidence=0.65,
        detail="Posting uses talent-pool / future-openings language — likely not a real open role",
        evidence=m.group(0),
    )


def check_high_applicant_count(job: JobPosting) -> ScamSignal | None:
    """Ghost job signal: >500 applicants on a >30-day-old posting suggests a ghost job."""
    if job.applicant_count <= 500:
        return None
    days = _days_since_posted(job.posted_date)
    if days is None or days <= 30:
        return None
    return ScamSignal(
        name="high_applicant_count",
        category=SignalCategory.GHOST_JOB,
        weight=0.48,
        confidence=0.55,
        detail=(
            f"Over {job.applicant_count} applicants on a {days}-day-old posting — "
            "likely a ghost job or perpetually open pipeline role"
        ),
        evidence=f"{job.applicant_count} applicants, {days} days old",
    )


def check_role_title_generic(job: JobPosting) -> ScamSignal | None:
    """Ghost job signal: extremely vague/generic role titles indicate non-specific pipeline postings."""
    m = _GENERIC_TITLES.match(job.title)
    if not m:
        return None
    return ScamSignal(
        name="role_title_generic",
        category=SignalCategory.GHOST_JOB,
        weight=0.42,
        confidence=0.55,
        detail="Role title is extremely generic — indicative of a catch-all pipeline posting",
        evidence=job.title,
    )


# ---------------------------------------------------------------------------
# Velocity / Temporal signals — weight 0.55–0.72
# ---------------------------------------------------------------------------


def check_high_posting_velocity(job: JobPosting, db=None) -> ScamSignal | None:
    """Velocity signal: company posted >20 jobs in 24 hours (scam ring indicator)."""
    if db is None:
        return None
    company = (job.company or "").strip()
    if not company:
        return None
    row = db.get_posting_velocity(company)
    if row is None:
        return None
    postings_24h = row.get("postings_24h", 0)
    if postings_24h <= 20:
        return None
    return ScamSignal(
        name="high_posting_velocity",
        category=SignalCategory.GHOST_JOB,
        weight=0.65,
        confidence=0.70,
        detail=(
            f"Company '{company}' posted {postings_24h} jobs in the last 24 hours — "
            "unusually high velocity consistent with scam ring activity"
        ),
        evidence=f"{postings_24h} postings in 24h",
    )


def check_new_recruiter_account(job: JobPosting) -> ScamSignal | None:
    """Velocity signal: recruiter profile created <7 days ago is a strong scam indicator.

    Convention: recruiter_connections == -1 encodes "account age <7 days" when
    the ingestion pipeline has that data.  A value of -1 is never a valid
    connection count, so it is safe to overload this field.
    """
    if job.recruiter_connections != -1:
        return None
    return ScamSignal(
        name="new_recruiter_account",
        category=SignalCategory.WARNING,
        weight=0.55,
        confidence=0.65,
        detail="Recruiter account was created less than 7 days ago — strong scam indicator",
        evidence="recruiter_account_age < 7 days",
    )


def _description_hash(description: str) -> str:
    """Normalise whitespace and return a SHA-256 hex digest of the description."""
    normalised = " ".join(description.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def check_cross_posting_duplicate(job: JobPosting, db=None) -> ScamSignal | None:
    """Velocity/dedup signal: same description text under a different company name."""
    if db is None:
        return None
    desc = (job.description or "").strip()
    if not desc:
        return None
    h = _description_hash(desc)
    company = (job.company or "").strip()
    matches = db.get_duplicate_description(h, exclude_company=company)
    if not matches:
        return None
    other_companies = list({m["company_name"] for m in matches})[:3]
    return ScamSignal(
        name="cross_posting_duplicate",
        category=SignalCategory.RED_FLAG,
        weight=0.72,
        confidence=0.80,
        detail=(
            "Identical job description found under different company name(s): "
            + ", ".join(other_companies)
            + " — strong scam-ring indicator"
        ),
        evidence=f"hash={h[:16]}… matched by {len(matches)} other posting(s)",
    )


# ---------------------------------------------------------------------------
# Structural — weight 0.4–0.6
# ---------------------------------------------------------------------------

def check_grammar_quality(job: JobPosting) -> ScamSignal | None:
    text = _full_text(job)
    if len(text) < 40:
        return None

    caps_hits = len(_GRAMMAR_CAPS.findall(text))
    punct_hits = len(_GRAMMAR_PUNCT.findall(text))
    emoji_hits = len(_GRAMMAR_EMOJI.findall(text))

    # Weighted anomaly score; thresholds tuned on manual review
    anomaly = caps_hits * 0.15 + punct_hits * 0.20 + emoji_hits * 0.10
    if anomaly < 0.5:
        return None

    issues = []
    if caps_hits:
        issues.append(f"{caps_hits} ALL-CAPS word(s)")
    if punct_hits:
        issues.append(f"{punct_hits} repeated punctuation sequence(s)")
    if emoji_hits:
        issues.append(f"{emoji_hits} emoji(s)")

    weight = round(min(0.60, 0.35 + anomaly * 0.05), 2)
    return ScamSignal(
        name="grammar_quality",
        category=SignalCategory.STRUCTURAL,
        weight=weight,
        confidence=0.55,
        detail="Unusual formatting: " + ", ".join(issues),
        evidence=", ".join(issues),
    )


def check_suspicious_links(job: JobPosting) -> ScamSignal | None:
    m = _SUSPICIOUS_LINKS.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="suspicious_links",
        category=SignalCategory.STRUCTURAL,
        weight=0.58,
        confidence=0.65,
        detail="Description contains shortened or third-party form links",
        evidence=m.group(0),
    )


# ---------------------------------------------------------------------------
# Positive signals — reduce scam score, weight 0.3–0.4
# ---------------------------------------------------------------------------

def check_established_company(job: JobPosting) -> ScamSignal | None:
    size_str = (job.company_size or "").replace(",", "").lower()
    m = re.search(r"(\d+)", size_str)
    if not m:
        return None
    if int(m.group(1)) < 100:
        return None
    return ScamSignal(
        name="established_company",
        category=SignalCategory.POSITIVE,
        weight=0.35,
        confidence=0.65,
        detail=f"Company has {job.company_size} employees — established organisation",
        evidence=job.company_size,
    )


def check_detailed_requirements(job: JobPosting) -> ScamSignal | None:
    text = job.description
    if len(text.split()) < 20:
        return None

    unique_techs = len({t.lower() for t in _TECH_STACK.findall(text)})
    qualifiers = [
        unique_techs >= 3,
        bool(_EXPERIENCE_YRS.search(text)),
        bool(_DEGREE.search(text)),
        bool(_BENEFITS.search(text)),
    ]
    if sum(qualifiers) < 2:
        return None

    evidence_parts = []
    if qualifiers[0]:
        evidence_parts.append(f"{unique_techs} technologies listed")
    if qualifiers[1]:
        evidence_parts.append("experience years specified")
    if qualifiers[2]:
        evidence_parts.append("degree requirement stated")
    if qualifiers[3]:
        evidence_parts.append("benefits mentioned")

    return ScamSignal(
        name="detailed_requirements",
        category=SignalCategory.POSITIVE,
        weight=0.38,
        confidence=0.70,
        detail="Detailed, specific requirements indicate a genuine role",
        evidence=", ".join(evidence_parts),
    )


# ---------------------------------------------------------------------------
# AI-informed scam detection signals
# ---------------------------------------------------------------------------

_GENERIC_CULTURE = re.compile(
    r"\b(dynamic team|fast.paced environment|passionate individuals?|"
    r"results.driven|go-getter|self.starter|rockstar|ninja|guru|"
    r"collaborative culture|inclusive workplace|family.like (team|culture)|"
    r"work hard play hard|make a (difference|impact))\b",
    re.IGNORECASE,
)

_SPECIFIC_DETAILS = re.compile(
    r"\b(team [A-Z][a-z]+|project [A-Z][a-z]+|"
    r"[A-Z][a-z]+ (squad|pod|team|org)|"
    r"using [A-Z][a-zA-Z]+|built (on|with) [A-Z]|"
    r"our (codebase|stack|platform|product|API|service))\b",
)

_PREMIUM_PHONE = re.compile(r"\b(900|976)[-.\s]?\d{3}[-.\s]?\d{4}\b")
_PHONE_GENERAL = re.compile(
    r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

_INTERVIEW_BYPASS = re.compile(
    r"\b(no interview (required|needed|necessary)|"
    r"hired (on the spot|immediately|same day)|"
    r"start (immediately|today|right away) no (questions?|interview)|"
    r"no resume (required|needed|necessary)|"
    r"no background (check|screening))\b",
    re.IGNORECASE,
)

_MLM_LANGUAGE = re.compile(
    r"\b(be your own boss|unlimited earning potential|residual income|"
    r"network marketing|multi.?level marketing|mlm|"
    r"downline|upline|recruit (others|people|friends)|"
    r"financial freedom|work when you want|set your own (hours|schedule)|"
    r"passive income opportunity)\b",
    re.IGNORECASE,
)

_RESHIPPING = re.compile(
    r"\b(receive (packages?|parcels?|shipments?)|"
    r"reship(ping)?|re-ship(ping)?|forward (packages?|parcels?)|"
    r"package (handler|inspector|coordinator) (from|at) home|"
    r"quality control inspector (from|at) home|"
    r"inspect (packages?|items?) (at|from) home)\b",
    re.IGNORECASE,
)

_DATA_HARVESTING = re.compile(
    r"\b(complete (our|the) (application|form|survey) (at|on|via)|"
    r"fill out (our|the) (application|form) (at|on|via)|"
    r"apply (at|via|through|on) (our )?(external|separate|outside)|"
    r"submit (your )?(application|info|details) (at|to|via) (http|www|forms?\.|typeform|google))\b",
    re.IGNORECASE,
)

_COMPENSATION_RED_FLAGS = re.compile(
    r"\b(commission.only|100% commission|1099 only|"
    r"independent contractor (only|position|role)|"
    r"performance.based pay only|"
    r"training (period|pay) (is )?unpaid|"
    r"no base (salary|pay)|draw against commission)\b",
    re.IGNORECASE,
)

_SUSPICIOUS_COMPANY_SUFFIX = re.compile(
    r"\b(\w+\s+(Solutions|Global|International|Enterprises?|Worldwide|"
    r"Unlimited|Ventures?|Associates?|Consulting|Services?|Group|Partners?))\s*$",
    re.IGNORECASE,
)


def check_ai_generated_content(job: JobPosting) -> ScamSignal | None:
    text = job.description
    if len(text.split()) < 30:
        return None

    generic_hits = len(_GENERIC_CULTURE.findall(text))
    specific_hits = len(_SPECIFIC_DETAILS.findall(text))
    # Also count tech stack mentions as "specific"
    specific_hits += len(_TECH_STACK.findall(text))
    specific_hits += len(_EXPERIENCE_YRS.findall(text))

    if generic_hits == 0:
        return None
    # Avoid division by zero; no specifics counts as worst-case ratio
    ratio = generic_hits / max(specific_hits, 1)
    if ratio <= 3.0:
        return None

    return ScamSignal(
        name="ai_generated_content",
        category=SignalCategory.STRUCTURAL,
        weight=0.45,
        confidence=0.55,
        detail=(
            f"Description has {generic_hits} generic culture phrase(s) but only "
            f"{specific_hits} specific detail(s) — ratio {ratio:.1f}:1 suggests AI filler"
        ),
        evidence=f"{generic_hits} generic vs {specific_hits} specific",
    )


def check_phone_anomaly(job: JobPosting) -> ScamSignal | None:
    text = _full_text(job)

    premium = _PREMIUM_PHONE.search(text)
    if premium:
        return ScamSignal(
            name="phone_anomaly",
            category=SignalCategory.WARNING,
            weight=0.5,
            confidence=0.75,
            detail="Premium-rate phone number detected (900/976 prefix)",
            evidence=premium.group(0),
        )

    phone_match = _PHONE_GENERAL.search(text)
    if phone_match:
        return ScamSignal(
            name="phone_anomaly",
            category=SignalCategory.WARNING,
            weight=0.5,
            confidence=0.50,
            detail="Phone number embedded in job description — unusual for LinkedIn postings",
            evidence=phone_match.group(0),
        )

    return None


def check_interview_bypass(job: JobPosting) -> ScamSignal | None:
    m = _INTERVIEW_BYPASS.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="interview_bypass",
        category=SignalCategory.RED_FLAG,
        weight=0.75,
        confidence=0.80,
        detail="Posting explicitly skips standard hiring steps (interview, resume, background check)",
        evidence=m.group(0),
    )


def check_mlm_language(job: JobPosting) -> ScamSignal | None:
    m = _MLM_LANGUAGE.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="mlm_language",
        category=SignalCategory.RED_FLAG,
        weight=0.8,
        confidence=0.82,
        detail="Multi-level marketing / pyramid scheme language detected",
        evidence=m.group(0),
    )


def check_reshipping_scam(job: JobPosting) -> ScamSignal | None:
    m = _RESHIPPING.search(_full_text(job))
    if not m:
        return None
    # Extra signal: "logistics coordinator" from home with no known logistics company
    text = _full_text(job)
    is_logistics_home = bool(
        re.search(r"logistics coordinator", text, re.IGNORECASE)
        and re.search(r"\b(from|at) home\b", text, re.IGNORECASE)
        and not job.company_linkedin_url.strip()
    )
    evidence = m.group(0)
    if is_logistics_home:
        evidence += " (logistics coordinator from home, no company LinkedIn)"
    return ScamSignal(
        name="reshipping_scam",
        category=SignalCategory.RED_FLAG,
        weight=0.9,
        confidence=0.85,
        detail="Reshipping / package forwarding job — classic money-mule vector",
        evidence=evidence,
    )


def check_data_harvesting(job: JobPosting) -> ScamSignal | None:
    text = _full_text(job)
    # Check explicit redirect language
    m = _DATA_HARVESTING.search(text)
    if m:
        return ScamSignal(
            name="data_harvesting",
            category=SignalCategory.RED_FLAG,
            weight=0.85,
            confidence=0.78,
            detail="Posting redirects applicants to an external form/site to collect personal data",
            evidence=m.group(0),
        )
    # Also catch bare Google Forms / Typeform links without redirect phrasing
    link_m = re.search(
        r"(forms\.gle|docs\.google\.com/forms|typeform\.com|airtable\.com/shr)",
        text,
        re.IGNORECASE,
    )
    if link_m:
        return ScamSignal(
            name="data_harvesting",
            category=SignalCategory.RED_FLAG,
            weight=0.85,
            confidence=0.70,
            detail="External data-collection form linked in job description",
            evidence=link_m.group(0),
        )
    return None


def check_compensation_red_flags(job: JobPosting) -> ScamSignal | None:
    m = _COMPENSATION_RED_FLAGS.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="compensation_red_flags",
        category=SignalCategory.WARNING,
        weight=0.55,
        confidence=0.65,
        detail="Compensation structure raises red flags (commission-only, unpaid training, 1099-only)",
        evidence=m.group(0),
    )


def check_company_name_suspicious(job: JobPosting) -> ScamSignal | None:
    name = (job.company or "").strip()
    if not name:
        return None

    # All-caps company name
    if name == name.upper() and len(name) > 3 and re.search(r"[A-Z]{4,}", name):
        return ScamSignal(
            name="company_name_suspicious",
            category=SignalCategory.WARNING,
            weight=0.5,
            confidence=0.50,
            detail="Company name is entirely uppercase — unusual for legitimate organisations",
            evidence=name,
        )

    # Single-word + generic suffix pattern (e.g. "Horizon Solutions", "Apex Global")
    m = _SUSPICIOUS_COMPANY_SUFFIX.match(name)
    if m:
        parts = name.split()
        if len(parts) <= 3:
            return ScamSignal(
                name="company_name_suspicious",
                category=SignalCategory.WARNING,
                weight=0.5,
                confidence=0.45,
                detail="Company name matches common scam pattern (generic word + Solutions/Global/etc.)",
                evidence=name,
            )

    return None


# ---------------------------------------------------------------------------
# Knowledge-base pattern matching (wires knowledge.py patterns into scoring)
# ---------------------------------------------------------------------------

_kb_compiled_cache: tuple[list[tuple[ScamPattern, re.Pattern | None]], str] | None = None


def _get_compiled_patterns(db_path: str = "") -> list[tuple[ScamPattern, re.Pattern | None]]:
    """Load active patterns from the DB and compile their regexes (cached)."""
    global _kb_compiled_cache
    cache_key = db_path

    if _kb_compiled_cache is not None and _kb_compiled_cache[1] == cache_key:
        return _kb_compiled_cache[0]

    compiled: list[tuple[ScamPattern, re.Pattern | None]] = []
    try:
        from sentinel.db import SentinelDB
        from sentinel.knowledge import KnowledgeBase

        db = SentinelDB(path=db_path) if db_path else SentinelDB()
        kb = KnowledgeBase(db=db)
        kb.seed_default_patterns()
        patterns = kb.get_active_patterns()
        for pat in patterns:
            rx = None
            if pat.regex and pat.regex.strip():
                with contextlib.suppress(re.error):
                    rx = re.compile(pat.regex)
            compiled.append((pat, rx))
        db.close()
    except Exception:
        logger.debug("Could not load knowledge-base patterns", exc_info=True)
        compiled = []

    _kb_compiled_cache = (compiled, cache_key)
    return compiled


def _reset_kb_cache() -> None:
    """Clear the compiled-patterns cache (used by tests)."""
    global _kb_compiled_cache
    _kb_compiled_cache = None


def check_knowledge_patterns(job: JobPosting, *, db_path: str = "") -> list[ScamSignal]:
    """Match job text against all active knowledge-base patterns.

    Returns a list of ScamSignal (one per matching pattern).
    """
    text = _full_text(job).lower()
    compiled = _get_compiled_patterns(db_path=db_path)
    signals: list[ScamSignal] = []

    for pat, rx in compiled:
        matched = False
        evidence = ""

        # Try regex first
        if rx is not None:
            m = rx.search(_full_text(job))
            if m:
                matched = True
                evidence = m.group(0)

        # Fall back to keyword matching (at least 2 keyword hits for a match)
        if not matched and pat.keywords:
            hits = [kw for kw in pat.keywords if kw.lower() in text]
            if len(hits) >= 2:
                matched = True
                evidence = "; ".join(hits[:3])

        if matched:
            # Derive weight from the pattern's Bayesian score, with a floor
            weight = max(0.55, pat.bayesian_score)
            try:
                category = pat.category if isinstance(pat.category, SignalCategory) else SignalCategory(pat.category)
            except ValueError:
                category = SignalCategory.RED_FLAG

            signals.append(
                ScamSignal(
                    name=f"kb_{pat.pattern_id}",
                    category=category,
                    weight=weight,
                    confidence=0.70,
                    detail=f"Knowledge-base pattern: {pat.name}",
                    evidence=evidence,
                )
            )

    return signals


# ---------------------------------------------------------------------------
# New fraud type signals
# ---------------------------------------------------------------------------

_PIG_BUTCHERING = re.compile(
    r"\b(cryptocurrency trader|digital asset (manager|analyst|trader)|"
    r"DeFi (analyst|trader|specialist)|liquidity provider|"
    r"investment opportunity.{0,60}(guaranteed|high) return|"
    r"forex (trader|analyst|manager)|crypto exchange (analyst|specialist)|"
    r"blockchain investment (analyst|manager))\b",
    re.IGNORECASE,
)


def check_pig_butchering_job(job: JobPosting) -> ScamSignal | None:
    """Detect pig butchering job scams -- fake crypto/investment roles."""
    m = _PIG_BUTCHERING.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="pig_butchering_job",
        category=SignalCategory.RED_FLAG,
        weight=0.88,
        confidence=0.80,
        detail="Posting matches pig butchering scam pattern (fake crypto/investment role)",
        evidence=m.group(0),
    )


_SURVEY_CLICKFARM = re.compile(
    r"\b(online survey.{0,40}(earn|paid|money)|"
    r"product reviewer.{0,40}(earn|paid|money|from home)|"
    r"social media evaluator|"
    r"paid per click|"
    r"earn per review|"
    r"get paid.{0,30}(surveys?|reviews?|clicks?)|"
    r"review products.{0,30}(earn|paid|keep))\b",
    re.IGNORECASE,
)


def check_survey_clickfarm(job: JobPosting) -> ScamSignal | None:
    """Detect survey/click-farm scams."""
    m = _SURVEY_CLICKFARM.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="survey_clickfarm",
        category=SignalCategory.WARNING,
        weight=0.75,
        confidence=0.72,
        detail="Posting matches survey/click-farm scam pattern",
        evidence=m.group(0),
    )


_VISA_KEYWORDS = re.compile(
    r"\b(guaranteed visa|guaranteed H-?1B|"
    r"H-?1B sponsor.{0,40}(fee|payment|deposit|cost)|"
    r"immigration fee|visa processing fee|"
    r"work permit fee|visa (application|processing).{0,40}(pay|fee|cost|deposit))\b",
    re.IGNORECASE,
)


def check_visa_sponsorship_scam(job: JobPosting) -> ScamSignal | None:
    """Detect visa sponsorship scams charging upfront fees."""
    m = _VISA_KEYWORDS.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="visa_sponsorship_scam",
        category=SignalCategory.RED_FLAG,
        weight=0.82,
        confidence=0.78,
        detail="Posting promises visa sponsorship with upfront fees -- legitimate employers never charge",
        evidence=m.group(0),
    )


_GOVT_IMPERSONATION = re.compile(
    r"\b(Department of Defense|DOD|FBI|CIA|DHS|NATO|"
    r"Department of Homeland Security|NSA|Secret Service|"
    r"Federal Bureau of Investigation|Central Intelligence Agency)\b",
    re.IGNORECASE,
)

_PERSONAL_INFO_REQUEST_GOV = re.compile(
    r"\b(provide (your )?(SSN|social security|passport|bank account|date of birth)|"
    r"send (your )?(SSN|passport|bank|personal (info|information|details))|"
    r"submit (your )?(personal|banking|financial) (info|information|details))\b",
    re.IGNORECASE,
)


def check_government_impersonation(job: JobPosting) -> ScamSignal | None:
    """Detect government/military impersonation scams."""
    text = _full_text(job)
    govt_m = _GOVT_IMPERSONATION.search(text)
    if not govt_m:
        return None
    # Must also request personal info to trigger -- real govt jobs don't do this
    info_m = _PERSONAL_INFO_REQUEST_GOV.search(text)
    if not info_m:
        return None
    # Additional signal: no company LinkedIn or using personal email
    has_company = bool(job.company_linkedin_url.strip())
    weight = 0.88 if not has_company else 0.72
    return ScamSignal(
        name="government_impersonation",
        category=SignalCategory.RED_FLAG,
        weight=weight,
        confidence=0.82,
        detail=f"Posting claims government/military affiliation ({govt_m.group(0)}) while requesting personal info",
        evidence=f"{govt_m.group(0)} + {info_m.group(0)}",
    )


_KNOWN_BRANDS = re.compile(
    r"\b(Google|Amazon|Microsoft|Apple|Meta|Facebook|Netflix|Tesla|"
    r"Goldman Sachs|JPMorgan|McKinsey|Deloitte|PwC|EY|KPMG|"
    r"Coca.Cola|Nike|Disney|Boeing|Lockheed Martin|Raytheon)\b",
    re.IGNORECASE,
)


def check_fake_staffing_agency(job: JobPosting) -> ScamSignal | None:
    """Detect fake staffing agencies using well-known brand names."""
    text = _full_text(job)
    brand_m = _KNOWN_BRANDS.search(text)
    if not brand_m:
        return None
    company = (job.company or "").strip().lower()
    brand_lower = brand_m.group(0).lower()
    # If the poster IS the brand, it is legitimate
    if brand_lower in company:
        return None
    # Must also lack a company LinkedIn page
    if job.company_linkedin_url.strip():
        return None
    return ScamSignal(
        name="fake_staffing_agency",
        category=SignalCategory.WARNING,
        weight=0.68,
        confidence=0.60,
        detail=f"Posting references {brand_m.group(0)} but is posted by unknown '{job.company}' with no LinkedIn page",
        evidence=f"brand={brand_m.group(0)}, poster={job.company}",
    )


_EVOLVED_MLM = re.compile(
    r"\b(brand ambassador.{0,60}(earn|income|commission|own boss)|"
    r"independent business owner|"
    r"wellness consultant.{0,60}(earn|income|opportunity)|"
    r"starter inventory|"
    r"authorized distributor.{0,60}(earn|income|opportunity)|"
    r"health and wellness.{0,40}(opportunity|income|earn)|"
    r"purchase (your |a )?starter (kit|package|inventory))\b",
    re.IGNORECASE,
)


def check_evolved_mlm(job: JobPosting) -> ScamSignal | None:
    """Detect evolved MLM schemes using modern language."""
    m = _EVOLVED_MLM.search(_full_text(job))
    if not m:
        return None
    return ScamSignal(
        name="evolved_mlm",
        category=SignalCategory.RED_FLAG,
        weight=0.78,
        confidence=0.74,
        detail="Posting uses evolved MLM language (brand ambassador, wellness consultant, starter inventory)",
        evidence=m.group(0),
    )


_CONTACT_SUSPICIOUS = re.compile(
    r"\b(apply (via|through|on|at) (telegram|whatsapp)|"
    r"contact (us |me )?(on|via|at|through) (telegram|whatsapp)|"
    r"(telegram|whatsapp) (only|to apply|for (details|info|more))|"
    r"send (message|msg|text|DM) (on|via|to) (telegram|whatsapp))\b",
    re.IGNORECASE,
)

_CELL_ONLY_CONTACT = re.compile(
    r"\b(call|text|contact|reach) (me|us) (at|on) \(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    re.IGNORECASE,
)


def check_contact_channel_suspicious(job: JobPosting) -> ScamSignal | None:
    """Detect suspicious contact channels (Telegram/WhatsApp only, personal cell)."""
    text = _full_text(job)
    m = _CONTACT_SUSPICIOUS.search(text)
    if m:
        return ScamSignal(
            name="contact_channel_suspicious",
            category=SignalCategory.WARNING,
            weight=0.65,
            confidence=0.68,
            detail="Posting directs applicants to Telegram/WhatsApp for application",
            evidence=m.group(0),
        )
    m = _CELL_ONLY_CONTACT.search(text)
    if m and not job.company_linkedin_url.strip():
        return ScamSignal(
            name="contact_channel_suspicious",
            category=SignalCategory.WARNING,
            weight=0.55,
            confidence=0.55,
            detail="Sole contact method is a personal phone number with no company LinkedIn",
            evidence=m.group(0),
        )
    return None


# ---------------------------------------------------------------------------
# New positive signals (reduce false positives)
# ---------------------------------------------------------------------------

_COMPANY_WEBSITE = re.compile(
    r"\b(https?://)?([a-z0-9][-a-z0-9]*\.)+[a-z]{2,}\b",
    re.IGNORECASE,
)

_ATS_SYSTEMS = re.compile(
    r"\b(greenhouse|lever|workday|taleo|icims|smartrecruiters|"
    r"brassring|jobvite|bamboohr|ashby|jazz ?hr|applicant tracking|"
    r"apply (on|at|through|via) (our |the )?(careers?|jobs?) (page|portal|site|website)|"
    r"careers?\.(company|com|org|io|co)|"
    r"/careers?/|/jobs?/)\b",
    re.IGNORECASE,
)


def check_verified_company_website(job: JobPosting) -> ScamSignal | None:
    """Positive signal: company has a website matching the posting domain."""
    text = _full_text(job)
    company = (job.company or "").strip().lower()
    if not company or len(company) < 3:
        return None

    # Look for URLs in the description
    urls = _COMPANY_WEBSITE.findall(text)
    if not urls:
        return None

    # Check if any URL contains the company name (simplified)
    company_word = re.sub(r"[^a-z0-9]", "", company)
    for url_parts in urls:
        # url_parts is a tuple from findall groups; join and check
        full = "".join(url_parts).lower()
        if company_word in re.sub(r"[^a-z0-9]", "", full):
            return ScamSignal(
                name="verified_company_website",
                category=SignalCategory.POSITIVE,
                weight=0.32,
                confidence=0.60,
                detail=f"Posting references a website matching company name '{job.company}'",
                evidence=full,
            )
    return None


def check_professional_application_process(job: JobPosting) -> ScamSignal | None:
    """Positive signal: company uses a professional ATS or careers page."""
    text = _full_text(job)
    m = _ATS_SYSTEMS.search(text)
    if not m:
        return None
    return ScamSignal(
        name="professional_application_process",
        category=SignalCategory.POSITIVE,
        weight=0.30,
        confidence=0.62,
        detail="Posting references a professional application process (ATS or company careers page)",
        evidence=m.group(0),
    )


# ---------------------------------------------------------------------------
# Known scam entity detection helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings (iterative, O(n*m))."""
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
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def _fuzzy_scam_match(name: str, scam_names: list[str], threshold: int = 3) -> str | None:
    """Return the closest scam name within *threshold* edits, or None."""
    name_lower = name.lower()
    best_name: str | None = None
    best_dist = threshold + 1
    for sname in scam_names:
        dist = _levenshtein(name_lower, sname.lower())
        if dist <= threshold and dist < best_dist:
            best_dist = dist
            best_name = sname
    return best_name


@lru_cache(maxsize=4)
def _load_scam_entity_names() -> list[str]:
    """Load all non-empty scam entity names from DB (cached per process)."""
    try:
        from sentinel.db import SentinelDB
        with SentinelDB() as db:
            entities = db.get_scam_entities()
        return [e["name"] for e in entities if e.get("name")]
    except Exception:
        return []


@lru_cache(maxsize=4)
def _load_scam_domains() -> set[str]:
    """Load all non-empty scam domains from DB (cached per process)."""
    try:
        from sentinel.db import SentinelDB
        with SentinelDB() as db:
            entities = db.get_scam_entities()
        return {e["domain"].lower() for e in entities if e.get("domain")}
    except Exception:
        return set()


def _extract_domain(text: str) -> str | None:
    """Extract a bare domain (e.g. 'amazon-jobs.net') from text, or None."""
    m = re.search(
        r"(?:https?://)?(?:www\.)?([a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9\-]+)+)",
        text,
    )
    if m:
        return m.group(1).lower()
    return None


def check_known_scam_entity(job: JobPosting) -> ScamSignal | None:
    """Check company name and domain against the known scam entity database.

    Performs:
    1. Exact name match (case-insensitive) against scam_entities
    2. Exact domain match against scam_entities
    3. Fuzzy name match (Levenshtein ≤ 3) against all scam entity names
    """
    company = (job.company or "").strip()
    full_text = _full_text(job)

    # 1. Exact name match
    if company:
        try:
            from sentinel.db import SentinelDB
            with SentinelDB() as db:
                if db.is_known_scam_entity(name=company):
                    return ScamSignal(
                        name="known_scam_entity",
                        category=SignalCategory.RED_FLAG,
                        weight=0.92,
                        confidence=0.90,
                        detail=f"Company '{company}' matches a known scam entity in the database",
                        evidence=company,
                    )
        except Exception:
            pass

    # 2. Domain match — check company name and any URL in the posting
    scam_domains = _load_scam_domains()
    if scam_domains:
        domain_candidate = _extract_domain(full_text)
        if domain_candidate and domain_candidate in scam_domains:
            return ScamSignal(
                name="known_scam_entity",
                category=SignalCategory.RED_FLAG,
                weight=0.92,
                confidence=0.88,
                detail=f"Domain '{domain_candidate}' matches a known scam domain",
                evidence=domain_candidate,
            )

    # 3. Fuzzy match on company name
    if company:
        scam_names = _load_scam_entity_names()
        match = _fuzzy_scam_match(company, scam_names, threshold=3)
        if match:
            return ScamSignal(
                name="known_scam_entity",
                category=SignalCategory.RED_FLAG,
                weight=0.80,
                confidence=0.70,
                detail=f"Company name '{company}' is suspiciously similar to known scam entity '{match}'",
                evidence=f"{company} ≈ {match}",
            )

    return None



# ---------------------------------------------------------------------------
# Graph-based signals — cluster membership, hub posting, Sybil detection
# ---------------------------------------------------------------------------

def _extract_graph_signals(
    job: "JobPosting",
    graph=None,
    profiler=None,
) -> list["ScamSignal"]:
    """Extract ScamSignals from graph analytics (ScamNetworkGraph + RecruiterProfiler).

    Both *graph* and *profiler* are optional; if None the function returns an
    empty list so the pipeline degrades gracefully when graph data is absent.
    """
    signals: list[ScamSignal] = []

    try:
        # ---- ScamNetworkGraph checks ----------------------------------------
        if graph is not None:
            from sentinel.graph import ScamNetworkGraph  # noqa: F401 (type check)

            # Add this job to the graph and get its posting_id
            posting_id = graph.add_posting(job)

            clusters = graph.get_clusters()
            for cluster in clusters:
                if posting_id not in cluster.node_ids:
                    continue

                # Any cluster with ≥2 members is suspicious
                cluster_size = len(cluster.node_ids)
                if cluster_size >= 2:
                    # Stronger signal for larger / cross-platform clusters
                    if len(cluster.platforms) > 1:
                        signals.append(ScamSignal(
                            name="cross_platform_posting",
                            category=SignalCategory.RED_FLAG,
                            weight=0.75,
                            confidence=0.72,
                            detail=(
                                f"Job appears in a cross-platform scam cluster "                                f"(cluster #{cluster.cluster_id}, {cluster_size} postings, "                                f"platforms: {', '.join(cluster.platforms)})"
                            ),
                            evidence=(
                                f"cluster_id={cluster.cluster_id}, "                                f"platforms={cluster.platforms}, "                                f"shared_features={cluster.shared_features}"
                            ),
                        ))
                    else:
                        signals.append(ScamSignal(
                            name="graph_cluster_member",
                            category=SignalCategory.RED_FLAG,
                            weight=0.65,
                            confidence=0.68,
                            detail=(
                                f"Job belongs to a scam-posting cluster "                                f"(cluster #{cluster.cluster_id}, {cluster_size} connected postings)"
                            ),
                            evidence=(
                                f"cluster_id={cluster.cluster_id}, "                                f"size={cluster_size}, "                                f"shared_features={cluster.shared_features}"
                            ),
                        ))

                # Hub detection — is this posting the hub of the cluster?
                if posting_id == cluster.hub_node and cluster.hub_degree >= 2:
                    signals.append(ScamSignal(
                        name="graph_hub_posting",
                        category=SignalCategory.RED_FLAG,
                        weight=0.80,
                        confidence=0.75,
                        detail=(
                            f"Job is the hub node in a scam cluster "                            f"(degree {cluster.hub_degree} connections)"
                        ),
                        evidence=(
                            f"hub_degree={cluster.hub_degree}, "                            f"cluster_id={cluster.cluster_id}"
                        ),
                    ))
                break  # each posting belongs to exactly one cluster

    except Exception:
        logger.debug("Graph cluster signal extraction failed", exc_info=True)

    try:
        # ---- RecruiterProfiler checks ----------------------------------------
        if profiler is not None:
            rid = job.recruiter_name or job.company or "unknown"
            flags = profiler.get_flags(rid)

            if flags:
                # Sybil detection: if this recruiter id is in any Sybil group
                sybil_groups = profiler.detect_sybils()
                in_sybil_group = any(rid in group for group in sybil_groups)

                if in_sybil_group:
                    signals.append(ScamSignal(
                        name="sybil_recruiter",
                        category=SignalCategory.RED_FLAG,
                        weight=0.78,
                        confidence=0.70,
                        detail=(
                            f"Recruiter '{rid}' is behaviourally similar to other "                            "flagged recruiters — likely a Sybil account"
                        ),
                        evidence=f"recruiter_id={rid}, flags={flags[:3]}",
                    ))
                else:
                    # Anomalous flags without Sybil group — moderate warning
                    flag_str = "; ".join(flags[:3])
                    signals.append(ScamSignal(
                        name="recruiter_anomaly_flags",
                        category=SignalCategory.WARNING,
                        weight=0.55,
                        confidence=0.60,
                        detail=f"Recruiter '{rid}' has anomalous behavioural flags: {flag_str}",
                        evidence=f"recruiter_id={rid}, flags={flags[:3]}",
                    ))

    except Exception:
        logger.debug("RecruiterProfiler signal extraction failed", exc_info=True)

    return signals

# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

ALL_SIGNALS = [
    # Red flags
    check_upfront_payment,
    check_personal_info_request,
    check_guaranteed_income,
    check_suspicious_email_domain,
    check_crypto_payment,
    check_no_company_presence,
    # Warnings
    check_salary_anomaly,
    check_vague_description,
    check_no_qualifications,
    check_urgency_language,
    check_wfh_unrealistic_pay,
    check_low_recruiter_connections,
    check_new_recruiter_account,
    # Ghost job
    check_stale_posting,
    check_repost_pattern,
    check_talent_pool_language,
    check_high_applicant_count,
    check_role_title_generic,
    # Structural
    check_grammar_quality,
    check_suspicious_links,
    check_ai_generated_content,
    # Positive
    check_established_company,
    check_detailed_requirements,
    check_verified_company_website,
    check_professional_application_process,
    # AI-informed scam detection
    check_phone_anomaly,
    check_interview_bypass,
    check_mlm_language,
    check_reshipping_scam,
    check_data_harvesting,
    check_compensation_red_flags,
    check_company_name_suspicious,
    check_known_scam_entity,
    # New fraud type signals
    check_pig_butchering_job,
    check_survey_clickfarm,
    check_visa_sponsorship_scam,
    check_government_impersonation,
    check_fake_staffing_agency,
    check_evolved_mlm,
    check_contact_channel_suspicious,
]

# Signals that require a db argument — called separately in extract_signals_with_db
DB_SIGNALS = [
    check_high_posting_velocity,
    check_cross_posting_duplicate,
]


def extract_signals(job: JobPosting) -> list[ScamSignal]:
    """Run all stateless signals against a job posting.

    Applies TextNormalizer before extraction so that Unicode confusables,
    leetspeak, zero-width chars, and homoglyphs are resolved first.
    Evasion signals (``evasion_attempt``, ``unicode_anomaly``,
    ``misspelled_scam_keyword``) are appended if obfuscation is detected.
    """
    # --- Normalization pass -------------------------------------------
    raw_text = _full_text(job)
    normalized_text = _NORMALIZER.normalize(raw_text)

    if normalized_text != raw_text:
        # Rebuild a normalized copy of the job for pattern matching.
        # We only override description and title so other fields stay intact.
        norm_desc = _NORMALIZER.normalize(job.description)
        norm_title = _NORMALIZER.normalize(job.title)
        from dataclasses import replace as _dc_replace
        job = _dc_replace(job, description=norm_desc, title=norm_title)

    # --- Standard signal extraction on (possibly normalized) job ------
    signals: list[ScamSignal] = []
    for check_fn in ALL_SIGNALS:
        signal = check_fn(job)
        if signal is not None:
            signals.append(signal)

    # --- Evasion detection on the *original* vs normalized text -------
    evasion_signals = _EVASION_DETECTOR.detect_evasion_attempts(raw_text, normalized_text)
    signals.extend(evasion_signals)

    return signals


def extract_signals_with_kb(job: JobPosting, *, db_path: str = "") -> list[ScamSignal]:
    """Run all stateless signals plus knowledge-base pattern matching."""
    signals = extract_signals(job)
    kb_signals = check_knowledge_patterns(job, db_path=db_path)
    signals.extend(kb_signals)
    return signals


def extract_signals_with_db(job: JobPosting, db) -> list[ScamSignal]:
    """Run all signals, including DB-backed velocity/dedup checks."""
    signals = extract_signals(job)
    for check_fn in DB_SIGNALS:
        signal = check_fn(job, db=db)
        if signal is not None:
            signals.append(signal)
    return signals


def extract_signals_with_graph(
    job: JobPosting,
    *,
    graph=None,
    profiler=None,
) -> list[ScamSignal]:
    """Run all stateless signals plus graph-based cluster/recruiter signals.

    Parameters
    ----------
    job:
        The job posting to analyse.
    graph:
        An optional :class: instance that has
        already been populated with other postings.  The current *job* will be
        added to the graph and cluster membership will be checked.
    profiler:
        An optional :class: instance that has
        already recorded historical recruiter activity for the job's recruiter.

    Both parameters default to *None*; when omitted the function behaves
    identically to :func:.
    """
    signals = extract_signals(job)
    graph_signals = _extract_graph_signals(job, graph=graph, profiler=profiler)
    signals.extend(graph_signals)
    return signals
