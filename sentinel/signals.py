"""Signal extraction: 40+ scam indicators for LinkedIn job postings."""

import re
from datetime import datetime, timezone

from sentinel.models import JobPosting, ScamSignal, SignalCategory


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

_GUARANTEED_INCOME = re.compile(
    r"\b(guaranteed (salary|income|pay|earnings?|profit)|"
    r"earn \$[\d,]+\s*(a |per )?(day|daily|week|hour|hr) guaranteed|"
    r"(guaranteed|promise[sd]) to (earn|make|pay))\b",
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

_NO_EXPERIENCE = re.compile(
    r"\b(no experience (required|needed|necessary)|"
    r"no (skills?|qualifications?|background) (required|needed)|"
    r"anyone can|so easy|simple (job|work|tasks?))\b",
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
            dt = datetime.strptime(posted_date, fmt).replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - dt).days
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
        return ScamSignal(
            name="no_company_presence",
            category=SignalCategory.RED_FLAG,
            weight=0.70,
            confidence=0.65,
            detail="No company LinkedIn page linked",
            evidence="company_linkedin_url is empty",
        )
    return None


# ---------------------------------------------------------------------------
# Warnings — weight 0.4–0.7
# ---------------------------------------------------------------------------

def check_salary_anomaly(job: JobPosting) -> ScamSignal | None:
    lo, hi = job.salary_min, job.salary_max
    level = (job.experience_level or "").lower()
    entry_levels = {"entry", "entry level", "entry-level", "internship", "junior"}

    if lo > 0 and hi > 0 and hi / lo > 3.0:
        return ScamSignal(
            name="salary_anomaly",
            category=SignalCategory.WARNING,
            weight=0.55,
            confidence=0.60,
            detail=f"Salary range is suspiciously wide: ${lo:,.0f}–${hi:,.0f}",
            evidence=f"{lo}–{hi}",
        )

    if level in entry_levels:
        ceiling = hi if hi > 0 else lo
        if ceiling > 500_000:
            return ScamSignal(
                name="salary_anomaly",
                category=SignalCategory.WARNING,
                weight=0.70,
                confidence=0.72,
                detail=f"Unrealistically high salary for {level} role: ${ceiling:,.0f}",
                evidence=str(ceiling),
            )

    return None


def check_vague_description(job: JobPosting) -> ScamSignal | None:
    words = job.description.split()
    if len(words) >= 30:
        return None
    weight = 0.65 if len(words) < 10 else 0.50
    return ScamSignal(
        name="vague_description",
        category=SignalCategory.WARNING,
        weight=weight,
        confidence=0.70,
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
        re.search(r"\b(work from home|remote|wfh|work at home)\b", text, re.IGNORECASE)
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
    if int(m.group(1)) < 1000:
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
    # Ghost job
    check_stale_posting,
    check_repost_pattern,
    # Structural
    check_grammar_quality,
    check_suspicious_links,
    check_ai_generated_content,
    # Positive
    check_established_company,
    check_detailed_requirements,
    # AI-informed scam detection
    check_phone_anomaly,
    check_interview_bypass,
    check_mlm_language,
    check_reshipping_scam,
    check_data_harvesting,
    check_compensation_red_flags,
    check_company_name_suspicious,
]


def extract_signals(job: JobPosting) -> list[ScamSignal]:
    signals: list[ScamSignal] = []
    for check_fn in ALL_SIGNALS:
        signal = check_fn(job)
        if signal is not None:
            signals.append(signal)
    return signals
