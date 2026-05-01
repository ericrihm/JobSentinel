"""Scam data ingestion pipeline: fetch, normalize, and seed known-scam job data.

Sources:
- FTC aggregate complaint data (seed dataset of common patterns)
- Kaggle EMSCAD/EMFJ CSV (Employment Scam Aegean Dataset)
- Programmatic scam pattern knowledge base

Usage:
    from sentinel.scam_data import ScamDataCollector
    from sentinel.db import SentinelDB

    db = SentinelDB()
    collector = ScamDataCollector()
    result = collector.seed_database(db)
    print(result)  # {'patterns_seeded': 25, 'patterns_new': 20, ...}
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.db import SentinelDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# ScamDataCollector
# ---------------------------------------------------------------------------


class ScamDataCollector:
    """Fetches, normalizes, and ingests known-scam job data from multiple sources."""

    # Default path to look for a local FTC data CSV
    DEFAULT_FTC_CSV = os.path.join(
        os.path.expanduser("~"), ".sentinel", "ftc_scam_data.csv"
    )

    # ---------------------------------------------------------------------------
    # 1. FTC data
    # ---------------------------------------------------------------------------

    def fetch_ftc_data(self, limit: int = 100) -> list[dict]:
        """Fetch FTC complaint data about job scams.

        First tries to load from a local CSV at DEFAULT_FTC_CSV (or FTC_CSV_PATH
        environment variable).  Falls back to a curated seed dataset of the top
        reported FTC job-scam patterns when no CSV is present.

        Each returned dict is shaped to match the jobs table schema plus an
        ``is_scam=True`` field.
        """
        csv_path = os.environ.get("FTC_CSV_PATH", self.DEFAULT_FTC_CSV)
        if Path(csv_path).exists():
            try:
                return self._load_ftc_csv(csv_path, limit=limit)
            except Exception:
                logger.warning("Failed to load FTC CSV at %s, using seed data", csv_path, exc_info=True)

        return self._ftc_seed_data(limit=limit)

    def _load_ftc_csv(self, path: str, limit: int = 100) -> list[dict]:
        """Parse a local FTC complaint CSV into job-posting dicts."""
        jobs: list[dict] = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                # FTC CSVs vary in column naming; handle common variants
                title = (
                    row.get("Job Title")
                    or row.get("job_title")
                    or row.get("Position")
                    or "Unknown Position"
                )
                company = (
                    row.get("Company")
                    or row.get("company_name")
                    or row.get("Business Name")
                    or "Unknown Company"
                )
                description = (
                    row.get("Description")
                    or row.get("description")
                    or row.get("Complaint")
                    or row.get("complaint_text")
                    or ""
                )
                location = row.get("Location") or row.get("location") or row.get("State") or ""
                jobs.append(
                    _make_job_dict(
                        title=title,
                        company=company,
                        description=description,
                        location=location,
                        source="ftc_csv",
                        is_scam=True,
                    )
                )
        logger.info("Loaded %d records from FTC CSV %s", len(jobs), path)
        return jobs

    def _ftc_seed_data(self, limit: int = 100) -> list[dict]:
        """Return a curated seed set of the top FTC-reported job scam patterns.

        Based on FTC Consumer Sentinel Network data reports and IC3 annual
        reports on employment/job-related fraud.
        """
        seed: list[dict] = [
            _make_job_dict(
                title="Work From Home Data Entry Specialist",
                company="Global Solutions LLC",
                description=(
                    "No experience required! Earn $25-$50/hour working from home. "
                    "You must purchase a starter kit ($150) before your first shift. "
                    "Send payment via Zelle or CashApp. Guaranteed income daily. "
                    "Apply now — limited spots available!"
                ),
                location="Remote",
                source="ftc_seed",
                is_scam=True,
                scam_category="advance_fee",
            ),
            _make_job_dict(
                title="Mystery Shopper Evaluator",
                company="National Retail Evaluators Inc",
                description=(
                    "We will send you a check for $2,800 to complete a mystery shopping "
                    "assignment at Walmart and MoneyGram. Deposit the check, keep $200 "
                    "as your fee, and wire the remaining $2,600 to our processing center "
                    "via Western Union within 24 hours. No experience needed."
                ),
                location="United States",
                source="ftc_seed",
                is_scam=True,
                scam_category="fake_check",
            ),
            _make_job_dict(
                title="Package Reshipping Coordinator",
                company="ShipFast International",
                description=(
                    "We need US-based shipping managers to receive packages at home and "
                    "reship them to our international clients. You will keep 10% of each "
                    "package value as your fee. Packages arrive weekly. No background "
                    "check required. Start immediately. Paid via Bitcoin or wire transfer."
                ),
                location="Work from home",
                source="ftc_seed",
                is_scam=True,
                scam_category="reshipping",
            ),
            _make_job_dict(
                title="Financial Recovery Agent",
                company="Apex Recovery Group",
                description=(
                    "Work from home processing financial transactions. You will receive "
                    "transfers to your personal bank account, take your 5% commission, "
                    "and forward the remainder. No experience required. Bank account and "
                    "routing number required at time of application. Daily payments."
                ),
                location="Remote",
                source="ftc_seed",
                is_scam=True,
                scam_category="money_mule",
            ),
            _make_job_dict(
                title="Brand Ambassador & Network Marketing Rep",
                company="Infinite Wellness Partners",
                description=(
                    "Join our ground-floor opportunity! Sell our health supplements and "
                    "build your downline. Earn unlimited income. Purchase your starter "
                    "business kit ($299) to get started. The more people you recruit, the "
                    "more you earn. Be your own boss — financial freedom is one step away!"
                ),
                location="Anywhere",
                source="ftc_seed",
                is_scam=True,
                scam_category="mlm_pyramid",
            ),
            _make_job_dict(
                title="Cryptocurrency Investment Analyst",
                company="CryptoWealth Trading Group",
                description=(
                    "We are recruiting crypto trading specialists. Learn our proprietary "
                    "system to generate $500-$2,000 per week. Buy Bitcoin or Ethereum and "
                    "we will show you how to double it in days. Training fee: $500 via "
                    "CashApp. Guaranteed profits. DM us on Telegram: t.me/cryptowealth99"
                ),
                location="Remote",
                source="ftc_seed",
                is_scam=True,
                scam_category="crypto_investment",
            ),
            _make_job_dict(
                title="Federal Government Job Placement Specialist",
                company="US Federal Employment Agency",
                description=(
                    "We have exclusive access to government job listings. For a one-time "
                    "processing fee of $75, we will submit your application to top federal "
                    "agencies. Social Security Number required for security clearance "
                    "pre-screening. 98% placement rate. Apply today!"
                ),
                location="Washington DC",
                source="ftc_seed",
                is_scam=True,
                scam_category="government_impersonation",
            ),
            _make_job_dict(
                title="Online Survey Specialist",
                company="SurveyPros Network",
                description=(
                    "Earn $5-$10 per survey completed from home. Unlimited earning "
                    "potential. To unlock premium surveys, activate your membership for "
                    "only $29.99/month. Click on the forms.gle link below to get started. "
                    "Payment via gift card only. No experience required."
                ),
                location="Remote",
                source="ftc_seed",
                is_scam=True,
                scam_category="survey_clickfarm",
            ),
            _make_job_dict(
                title="Personal Assistant to CEO",
                company="TechVenture Global",
                description=(
                    "Busy executive seeks personal assistant. Flexible hours, work from "
                    "home. Duties include managing errands and financial transactions. "
                    "You will receive checks to cash and keep a portion as your salary. "
                    "Must provide full name, address, date of birth, and bank account info "
                    "to begin. $800/week guaranteed."
                ),
                location="Remote",
                source="ftc_seed",
                is_scam=True,
                scam_category="identity_theft",
            ),
            _make_job_dict(
                title="Remote Customer Service Rep — No Experience Needed",
                company="Premier Staffing Solutions",
                description=(
                    "Earn $800 a week guaranteed processing orders from home. "
                    "No interview required — you are hired! Send us your driver's license "
                    "and social security number to complete your background check. "
                    "Training fee of $49.99 required. Start today!"
                ),
                location="United States (Remote)",
                source="ftc_seed",
                is_scam=True,
                scam_category="advance_fee",
            ),
        ]
        return seed[:limit]

    # ---------------------------------------------------------------------------
    # 2. Kaggle EMSCAD/EMFJ CSV loader
    # ---------------------------------------------------------------------------

    def load_kaggle_emfj(self, csv_path: str) -> list[dict]:
        """Parse the Kaggle Employment Scam Aegean Dataset (EMSCAD) CSV.

        Expected columns (from the EMFJ/EMSCAD dataset):
            title, location, department, salary_range, company_profile,
            description, requirements, benefits, telecommuting,
            has_company_logo, has_questions, employment_type,
            required_experience, required_education, industry, function,
            fraudulent

        Returns a list of dicts shaped for the jobs table, with ``is_scam``
        set from the ``fraudulent`` column.
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"EMFJ CSV not found: {csv_path}")

        jobs: list[dict] = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fraudulent = str(row.get("fraudulent", "0")).strip()
                    is_scam = fraudulent in ("1", "true", "True", "TRUE", "yes", "Yes")

                    # Build composite description from available text columns
                    parts = []
                    for field in ("company_profile", "description", "requirements", "benefits"):
                        val = str(row.get(field, "") or "").strip()
                        if val:
                            parts.append(val)
                    description = " ".join(parts)

                    # Parse salary range — EMFJ uses strings like "$50,000-$70,000"
                    salary_min, salary_max = _parse_salary_range(
                        str(row.get("salary_range", "") or "")
                    )

                    job = _make_job_dict(
                        title=str(row.get("title", "") or ""),
                        company="",  # EMFJ does not have a company column
                        description=description,
                        location=str(row.get("location", "") or ""),
                        salary_min=salary_min,
                        salary_max=salary_max,
                        source="kaggle_emfj",
                        is_scam=is_scam,
                    )
                    # Preserve EMFJ-specific fields as metadata
                    job["employment_type"] = str(row.get("employment_type", "") or "")
                    job["industry"] = str(row.get("industry", "") or "")
                    job["experience_level"] = str(row.get("required_experience", "") or "")
                    job["telecommuting"] = str(row.get("telecommuting", "0")).strip() in (
                        "1", "true", "True",
                    )
                    jobs.append(job)
                except Exception:
                    logger.debug("EMFJ: failed to parse row", exc_info=True)
                    continue

        scam_count = sum(1 for j in jobs if j.get("is_scam"))
        logger.info(
            "Loaded %d EMFJ jobs (%d scam, %d legitimate)",
            len(jobs),
            scam_count,
            len(jobs) - scam_count,
        )
        return jobs

    # ---------------------------------------------------------------------------
    # 3. Scam pattern knowledge base
    # ---------------------------------------------------------------------------

    def generate_scam_patterns(self) -> list[dict]:
        """Generate a comprehensive seed knowledge base of known scam patterns.

        Returns at least 20 pattern dicts, each with:
            pattern_id, name, description, category, keywords, regex,
            typical_weight, alpha, beta, status
        """
        return [
            # -------- Payment / Fee scams --------
            {
                "pattern_id": "advance_fee_payment",
                "name": "Advance Fee Payment",
                "description": (
                    "Job posting requires candidate to pay an upfront fee before starting "
                    "work (starter kit, training materials, background check fee, etc.). "
                    "Top FTC complaint category for job fraud."
                ),
                "category": "red_flag",
                "keywords": [
                    "starter kit fee", "training fee", "upfront fee", "pay deposit",
                    "background check fee", "registration fee", "application fee",
                    "processing fee", "pay before you start",
                ],
                "regex": (
                    r"\b(starter kit fee|training fee|upfront (fee|cost|payment)|"
                    r"pay (a |the )?(deposit|fee)|background check fee|"
                    r"registration fee|application fee|processing fee|"
                    r"purchase (your |a )?equipment|buy equipment)\b"
                ),
                "typical_weight": 0.95,
                "alpha": 9.0,
                "beta": 1.0,
            },
            {
                "pattern_id": "equipment_purchase_required",
                "name": "Equipment Purchase Required",
                "description": (
                    "Candidate is asked to purchase work equipment (laptop, phone, tools) "
                    "from a specific vendor or via wire transfer, often using a fake check "
                    "sent to the victim."
                ),
                "category": "red_flag",
                "keywords": [
                    "purchase equipment", "buy your laptop", "buy equipment", "order supplies",
                    "equipment purchase", "order your starter kit",
                ],
                "regex": (
                    r"\b(purchase (your |a |the )?(equipment|laptop|computer|supplies|tools)|"
                    r"buy (your )?(equipment|laptop|computer|supplies)|"
                    r"order (your )?(equipment|starter kit|supplies))\b"
                ),
                "typical_weight": 0.93,
                "alpha": 8.0,
                "beta": 1.0,
            },
            # -------- Fake check scams --------
            {
                "pattern_id": "fake_check_overpayment",
                "name": "Fake Check / Overpayment Scam",
                "description": (
                    "Scammer sends a fraudulent check, asks victim to deposit it and wire "
                    "back a portion. Common in mystery shopper, personal assistant, and "
                    "work-from-home postings. Bank eventually reverses the deposit."
                ),
                "category": "red_flag",
                "keywords": [
                    "we will send you a check", "deposit check", "wire the difference",
                    "money order", "western union", "moneygram", "send back",
                    "overpayment", "wire remaining",
                ],
                "regex": (
                    r"\b(send (you |a )?(check|money order|cashier.s check)|"
                    r"wire (the )?(remaining|difference|back|funds?)|"
                    r"western union|moneygram|deposit.{0,30}(check|funds?)|"
                    r"keep.{0,20}(as your (fee|commission)))\b"
                ),
                "typical_weight": 0.93,
                "alpha": 8.0,
                "beta": 1.0,
            },
            {
                "pattern_id": "mystery_shopper_check",
                "name": "Mystery Shopper Fake Check",
                "description": (
                    "Classic FTC-flagged scam: victim hired as 'mystery shopper' and sent "
                    "a fraudulent check to 'test' a wire transfer service. "
                    "Victim loses the wired amount when check bounces."
                ),
                "category": "red_flag",
                "keywords": [
                    "mystery shopper", "secret shopper", "shop evaluator",
                    "retail evaluator", "store evaluator",
                ],
                "regex": r"\b(mystery shopper|secret shopper|shop(ping)? evaluator|retail evaluator)\b",
                "typical_weight": 0.88,
                "alpha": 7.0,
                "beta": 1.0,
            },
            # -------- Data harvesting / Identity theft --------
            {
                "pattern_id": "identity_data_harvest",
                "name": "Identity Data Harvesting",
                "description": (
                    "Posting requests sensitive personal identifiers (SSN, bank account, "
                    "passport, driver's license) during the application or before any "
                    "interview, consistent with identity theft recruitment."
                ),
                "category": "red_flag",
                "keywords": [
                    "social security number", "SSN", "bank account number", "routing number",
                    "passport copy", "driver's license", "date of birth", "full address",
                    "credit card number",
                ],
                "regex": (
                    r"\b(social security( number)?|SSN|bank account( number)?|"
                    r"routing number|passport (number|copy)|drivers? licen[sc]e|"
                    r"date of birth|credit card( number)?|debit card)\b"
                ),
                "typical_weight": 0.92,
                "alpha": 8.5,
                "beta": 1.0,
            },
            {
                "pattern_id": "fake_background_check_data",
                "name": "Fake Background Check Data Collection",
                "description": (
                    "Scammer requests personal documents under the guise of a background "
                    "check, often with no legitimate company verification possible. "
                    "Used primarily for identity theft."
                ),
                "category": "red_flag",
                "keywords": [
                    "background check required", "submit identification", "provide SSN",
                    "send documents", "upload ID", "national ID",
                ],
                "regex": (
                    r"\b(submit (your )?(identification|documents?|ID|passport)|"
                    r"provide (your )?(SSN|social security|bank details)|"
                    r"upload (your )?(ID|documents?|passport)|"
                    r"send (your )?(documents?|ID|passport))\b"
                ),
                "typical_weight": 0.85,
                "alpha": 7.0,
                "beta": 1.5,
            },
            # -------- Reshipping / Money mule --------
            {
                "pattern_id": "reshipping_scam",
                "name": "Package Reshipping Scam",
                "description": (
                    "Victim receives stolen goods and reshipsthrough them to another "
                    "address. Often framed as 'package handler', 'shipping coordinator', "
                    "or 'quality control' role. Victim may be prosecuted for handling "
                    "stolen merchandise."
                ),
                "category": "red_flag",
                "keywords": [
                    "reship", "reshipping", "forward packages", "package handler",
                    "shipping coordinator", "receive and reship", "ship packages from home",
                ],
                "regex": (
                    r"\b(re-?ship(ping)?|forward packages|receive (and )?reship|"
                    r"ship packages (from|at) home|package handler|"
                    r"shipping coordinator)\b"
                ),
                "typical_weight": 0.91,
                "alpha": 8.0,
                "beta": 1.0,
            },
            {
                "pattern_id": "money_mule_transfer",
                "name": "Money Mule / Financial Agent",
                "description": (
                    "Victim is asked to receive money in their personal bank account and "
                    "forward it to another account, keeping a small commission. This is "
                    "money laundering and can result in criminal charges for the victim."
                ),
                "category": "red_flag",
                "keywords": [
                    "receive transfers", "forward funds", "financial agent",
                    "payment processor", "receive to your account", "keep a percentage",
                    "transfer agent", "money transfer agent",
                ],
                "regex": (
                    r"\b(receive (funds?|transfers?|money|payments?) (to|in|into) "
                    r"(your )?(bank|personal) account|forward (the )?(funds?|money|"
                    r"transfers?)|money transfer agent|financial agent|"
                    r"keep (a |your )?(commission|percentage|cut))\b"
                ),
                "typical_weight": 0.93,
                "alpha": 8.5,
                "beta": 1.0,
            },
            # -------- MLM / Pyramid schemes --------
            {
                "pattern_id": "mlm_pyramid_scheme",
                "name": "MLM / Pyramid Scheme Recruitment",
                "description": (
                    "Multi-level marketing or pyramid scheme disguised as a job. "
                    "Emphasises recruiting others over selling products. "
                    "Income primarily from recruitment fees, not actual sales."
                ),
                "category": "red_flag",
                "keywords": [
                    "build your downline", "recruit others", "network marketing",
                    "multi-level marketing", "MLM", "direct sales", "unlimited income",
                    "be your own boss", "ground floor opportunity", "passive income",
                    "financial freedom", "starter business kit",
                ],
                "regex": (
                    r"\b(build (your |a )?downline|recruit (others|friends|family)|"
                    r"multi.?level marketing|MLM|network marketing|"
                    r"ground.?floor opportunity|passive income stream|"
                    r"unlimited (earning|income) potential|starter (business )?kit)\b"
                ),
                "typical_weight": 0.80,
                "alpha": 7.0,
                "beta": 2.0,
            },
            # -------- Crypto / Investment recruitment --------
            {
                "pattern_id": "crypto_investment_scam",
                "name": "Crypto / Investment Recruitment Scam",
                "description": (
                    "Job posting recruits 'crypto traders', 'investment analysts', or "
                    "'financial educators' whose real purpose is to solicit crypto "
                    "deposits from the victim or their network. Often linked to "
                    "pig-butchering operations."
                ),
                "category": "red_flag",
                "keywords": [
                    "crypto trading", "bitcoin investment", "ethereum", "DeFi",
                    "trading signals", "double your investment", "guaranteed profits",
                    "crypto recruiter", "investment platform", "trading platform",
                ],
                "regex": (
                    r"\b(crypto(currency)? trad(ing|er)|bitcoin investment|"
                    r"ethereum trad(ing|er)|double your (investment|money|crypto)|"
                    r"guaranteed (profit|return|yield)|trading signal(s)?|"
                    r"DeFi (yield|platform|investment))\b"
                ),
                "typical_weight": 0.88,
                "alpha": 7.5,
                "beta": 1.5,
            },
            {
                "pattern_id": "pig_butchering_hybrid",
                "name": "Pig Butchering (Crypto Romance + Job Hybrid)",
                "description": (
                    "Sophisticated scam combining fake romantic or professional connection "
                    "with job offer on a fraudulent crypto trading platform. "
                    "Victim is coached to invest increasing amounts before platform "
                    "disappears. Often originates via WhatsApp, Telegram, or LinkedIn DM."
                ),
                "category": "red_flag",
                "keywords": [
                    "trading mentor", "crypto mentor", "I can teach you to trade",
                    "my trading strategy", "investment tutor", "WhatsApp me",
                    "contact me on Telegram", "exclusive trading group",
                ],
                "regex": (
                    r"\b(trading (mentor|coach|tutor)|I (can |will )?teach you to trade|"
                    r"exclusive trading (group|platform|community)|"
                    r"contact me on (telegram|whatsapp|signal)|"
                    r"(telegram|whatsapp)[\s:]+@?[A-Za-z0-9_]+)\b"
                ),
                "typical_weight": 0.90,
                "alpha": 8.0,
                "beta": 1.0,
            },
            # -------- Government impersonation --------
            {
                "pattern_id": "government_job_impersonation",
                "name": "Government Job Impersonation",
                "description": (
                    "Scammer impersonates a government agency or implies exclusive access "
                    "to government job listings in exchange for a fee. Often uses official-"
                    "sounding names and seals/logos to appear legitimate."
                ),
                "category": "red_flag",
                "keywords": [
                    "federal government job", "government employment agency",
                    "clearance jobs", "USAJOBS fee", "government job guarantee",
                    "security clearance processing fee",
                ],
                "regex": (
                    r"\b(federal (government )?jobs? (placement|agency|guarantee)|"
                    r"government (employment )?agency fee|"
                    r"security clearance (processing )?fee|"
                    r"exclusive (access to )?(government|federal) (jobs?|positions?))\b"
                ),
                "typical_weight": 0.87,
                "alpha": 7.5,
                "beta": 1.5,
            },
            # -------- Fake staffing agencies --------
            {
                "pattern_id": "fake_staffing_agency",
                "name": "Fake Staffing Agency",
                "description": (
                    "Fraudulent staffing or recruitment agency that collects fees, "
                    "harvests CVs, or promises jobs at well-known companies. "
                    "Legitimate staffing agencies never charge job seekers."
                ),
                "category": "warning",
                "keywords": [
                    "placement fee", "registration fee", "resume submission fee",
                    "guaranteed placement", "job guarantee fee",
                    "staffing fee", "recruiter fee",
                ],
                "regex": (
                    r"\b((placement|registration|submission|recruiter|staffing) fee|"
                    r"guaranteed (placement|job|position) fee|"
                    r"pay to (register|apply|submit)|"
                    r"job (guarantee|placement) (fee|cost))\b"
                ),
                "typical_weight": 0.82,
                "alpha": 7.0,
                "beta": 2.0,
            },
            # -------- Survey / click farms --------
            {
                "pattern_id": "survey_clickfarm",
                "name": "Survey / Click Farm",
                "description": (
                    "Victim is recruited to complete paid surveys, click ads, or watch "
                    "videos but must pay for a membership or 'survey access' to earn. "
                    "Earnings never materialise or are paid only in gift cards."
                ),
                "category": "warning",
                "keywords": [
                    "paid surveys", "survey specialist", "click ads for money",
                    "watch videos for money", "survey membership", "unlock surveys",
                    "earn per survey", "gift card payment",
                ],
                "regex": (
                    r"\b(paid survey(s)?|earn (per|per completed) survey|"
                    r"click (ads|links) (for|to earn)|watch videos (for|to earn)|"
                    r"survey (membership|access|unlock)|"
                    r"pay(ment)? (via|by|in) gift card(s)?)\b"
                ),
                "typical_weight": 0.75,
                "alpha": 6.0,
                "beta": 2.0,
            },
            # -------- Generic red-flag signals --------
            {
                "pattern_id": "guaranteed_unrealistic_income",
                "name": "Guaranteed / Unrealistic Income Claims",
                "description": (
                    "Posting guarantees specific high earnings with no skill or experience "
                    "requirements. Legitimate employers do not guarantee income."
                ),
                "category": "red_flag",
                "keywords": [
                    "guaranteed income", "guaranteed salary", "guaranteed $",
                    "earn $500 per day", "earn $1000 per week", "make $5000 a month",
                    "guaranteed earnings",
                ],
                "regex": (
                    r"\b(guaranteed (salary|income|earnings?|pay|profit)|"
                    r"earn \$[\d,]+\s*(a |per )?(day|daily|week|hour|hr) guaranteed|"
                    r"(guaranteed|promise[sd]) to (earn|make|pay)|"
                    r"make \$[\d,]+ (a |per )?(day|week) (guaranteed|easily))\b"
                ),
                "typical_weight": 0.88,
                "alpha": 7.5,
                "beta": 1.5,
            },
            {
                "pattern_id": "no_experience_high_pay",
                "name": "No Experience Required + High Pay Mismatch",
                "description": (
                    "Posting claims no experience is needed while advertising "
                    "above-market pay rates. This combination is a classic scam indicator."
                ),
                "category": "warning",
                "keywords": [
                    "no experience required", "no experience needed", "no skills required",
                    "anyone can do it", "so easy", "simple tasks",
                ],
                "regex": (
                    r"\b(no experience (required|needed|necessary)|"
                    r"no (skills?|qualifications?|background) (required|needed)|"
                    r"anyone can (do this|apply|qualify)|so easy to (earn|make|do)|"
                    r"simple (tasks|job|work))\b"
                ),
                "typical_weight": 0.70,
                "alpha": 6.0,
                "beta": 3.0,
            },
            {
                "pattern_id": "urgent_limited_slots",
                "name": "Urgency / Limited Slot Pressure",
                "description": (
                    "High-pressure language creates artificial urgency to prevent victims "
                    "from researching the opportunity. Common in all scam categories."
                ),
                "category": "warning",
                "keywords": [
                    "apply now", "limited spots", "hiring immediately", "act now",
                    "positions filling fast", "only 3 spots left", "urgent hiring",
                    "don't miss this", "limited openings",
                ],
                "regex": (
                    r"\b(apply (now|immediately|today|asap)|"
                    r"limited (spots?|openings?|positions?)|"
                    r"hiring (immediately|now|today|asap)|"
                    r"urgent(ly)? (hiring|needed)|positions? (filling|fill) fast|"
                    r"don'?t (miss|wait)|act (now|fast|quickly)|"
                    r"only \d+ (spots?|seats?|positions?) (left|remaining|available))\b"
                ),
                "typical_weight": 0.65,
                "alpha": 5.0,
                "beta": 3.0,
            },
            {
                "pattern_id": "personal_email_contact",
                "name": "Personal Email Domain Contact",
                "description": (
                    "Recruiter or hiring contact uses a personal/free email domain "
                    "(Gmail, Yahoo, Hotmail) rather than a corporate domain. "
                    "Legitimate employers communicate via corporate email."
                ),
                "category": "warning",
                "keywords": [
                    "@gmail.com", "@yahoo.com", "@hotmail.com", "@outlook.com",
                    "@aol.com", "@protonmail.com",
                ],
                "regex": (
                    r"@(gmail|yahoo|hotmail|outlook|aol|icloud|protonmail|"
                    r"mail|ymail|live|msn|me|googlemail)\."
                ),
                "typical_weight": 0.72,
                "alpha": 6.0,
                "beta": 3.0,
            },
            {
                "pattern_id": "crypto_payment_method",
                "name": "Cryptocurrency / Informal Payment Method",
                "description": (
                    "Salary or fees to be paid via Bitcoin, gift cards, Zelle, CashApp, "
                    "or Western Union. Legitimate employers pay via payroll/ACH."
                ),
                "category": "red_flag",
                "keywords": [
                    "bitcoin", "crypto", "gift card", "western union", "moneygram",
                    "zelle", "cashapp", "venmo", "wire transfer", "wire me",
                ],
                "regex": (
                    r"\b(bitcoin|btc|ethereum|eth|crypto(currency)?|gift card(s)?|"
                    r"western union|moneygram|wire transfer|wire (me|us|the)|"
                    r"zelle|cashapp|cash app|venmo)\b"
                ),
                "typical_weight": 0.85,
                "alpha": 8.0,
                "beta": 2.0,
            },
            {
                "pattern_id": "suspicious_url_shortener",
                "name": "Suspicious Link / Messaging App Contact",
                "description": (
                    "Job posting includes URL shorteners, Google Forms links, or directs "
                    "applicants to Telegram/WhatsApp — all used to evade platform "
                    "moderation and harvesting."
                ),
                "category": "warning",
                "keywords": [
                    "bit.ly", "tinyurl", "t.me", "telegram", "whatsapp",
                    "forms.gle", "typeform", "wa.me",
                ],
                "regex": (
                    r"(bit\.ly/|tinyurl\.com/|t\.me/|telegram\.me/|"
                    r"wa\.me/|whatsapp\.com/|forms\.gle/|typeform\.com)"
                ),
                "typical_weight": 0.68,
                "alpha": 5.5,
                "beta": 3.0,
            },
            {
                "pattern_id": "typosquat_domain",
                "name": "Typosquatting / Fake Company Domain",
                "description": (
                    "Company website uses a domain that mimics a well-known brand by "
                    "adding hyphens, words like 'jobs', 'careers', 'hiring', or using "
                    "non-standard TLDs (.net, .info instead of .com)."
                ),
                "category": "red_flag",
                "keywords": [
                    "amazon-jobs.net", "google-careers.net", "meta-jobs.net",
                    "microsoft-hiring.net", "linkedin-jobs.net",
                ],
                "regex": (
                    r"\b(amazon|google|meta|microsoft|apple|netflix|linkedin|"
                    r"indeed|glassdoor)-(jobs?|careers?|hiring|work|apply)"
                    r"\.(net|info|org|biz|co)\b"
                ),
                "typical_weight": 0.87,
                "alpha": 7.5,
                "beta": 1.5,
            },
            {
                "pattern_id": "too_good_salary",
                "name": "Unrealistically High Salary for Role",
                "description": (
                    "Advertised salary is significantly above market rate for the "
                    "stated role and experience level. Common tactic to attract victims "
                    "before requesting advance fees or personal information."
                ),
                "category": "warning",
                "keywords": [
                    "$5000 per week", "$20000 per month", "$500 per day",
                    "make thousands", "six figures from home",
                ],
                "regex": (
                    r"\b(\$[\d,]+\s*(per|a)\s*(hour|day|week) (guaranteed|easily)|"
                    r"earn (thousands|hundreds) (per|a) (day|week|hour)|"
                    r"six figures? (working |from )?home)\b"
                ),
                "typical_weight": 0.73,
                "alpha": 6.0,
                "beta": 3.0,
            },
            {
                "pattern_id": "vague_description_no_skills",
                "name": "Vague Job Description Without Required Skills",
                "description": (
                    "Description is extremely short, uses generic filler phrases, and "
                    "lists no actual job requirements, technical skills, or qualifications. "
                    "Scam postings are intentionally vague to attract a wide net."
                ),
                "category": "warning",
                "keywords": [
                    "various tasks", "help with projects", "general duties",
                    "assist as needed", "flexible work", "miscellaneous tasks",
                ],
                "regex": (
                    r"\b(various tasks?|general (duties|tasks?)|"
                    r"assist (as needed|with projects?)|"
                    r"miscellaneous (tasks?|duties)|flexible (work|hours|tasks?))\b"
                ),
                "typical_weight": 0.55,
                "alpha": 4.5,
                "beta": 4.0,
            },
        ]

    # ---------------------------------------------------------------------------
    # 4. Seed database
    # ---------------------------------------------------------------------------

    def seed_database(self, db: "SentinelDB") -> dict:
        """Populate the SentinelDB with patterns from generate_scam_patterns().

        Patterns are saved with status='active' and initial Bayesian priors
        derived from their typical_weight.

        Returns a dict with seeding statistics.
        """
        patterns = self.generate_scam_patterns()
        seeded = 0
        new = 0
        updated = 0
        errors = 0
        now = _now_iso()

        for p in patterns:
            try:
                # Check if pattern already exists
                existing = _get_existing_pattern(db, p["pattern_id"])

                keywords = p.get("keywords", [])
                import json as _json
                keywords_json = _json.dumps(keywords)

                pattern_row = {
                    "pattern_id": p["pattern_id"],
                    "name": p["name"],
                    "description": p["description"],
                    "category": p.get("category", "red_flag"),
                    "regex": p.get("regex", ""),
                    "keywords_json": keywords_json,
                    "alpha": p.get("alpha", 1.0),
                    "beta": p.get("beta", 1.0),
                    "observations": p.get("observations", 0),
                    "true_positives": p.get("true_positives", 0),
                    "false_positives": p.get("false_positives", 0),
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                }

                db.save_pattern(pattern_row)
                seeded += 1
                if existing is None:
                    new += 1
                else:
                    updated += 1
            except Exception:
                logger.warning("Failed to seed pattern %s", p.get("pattern_id"), exc_info=True)
                errors += 1

        result = {
            "patterns_seeded": seeded,
            "patterns_new": new,
            "patterns_updated": updated,
            "patterns_errors": errors,
            "seeded_at": now,
        }
        logger.info(
            "Seed complete: %d patterns (%d new, %d updated, %d errors)",
            seeded, new, updated, errors,
        )
        return result

    # ---------------------------------------------------------------------------
    # 5. Ingest labeled data
    # ---------------------------------------------------------------------------

    def ingest_labeled_data(self, db: "SentinelDB", data: list[dict]) -> dict:
        """Ingest a list of labeled job dicts and use results to update pattern weights.

        Each dict must have an ``is_scam`` field (bool or 0/1).  The method:
        1. Builds a JobPosting from each dict.
        2. Runs extract_signals + score_signals.
        3. Compares prediction vs label.
        4. Updates pattern weights via the flywheel (learn_from_report).
        5. Saves job to DB with computed scam_score.
        6. Returns accuracy metrics.
        """
        from sentinel.models import JobPosting, UserReport
        from sentinel.signals import extract_signals
        from sentinel.scorer import score_signals, build_result
        from sentinel.flywheel import DetectionFlywheel

        flywheel = DetectionFlywheel(db=db)

        total = 0
        correct = 0
        true_positives = 0
        true_negatives = 0
        false_positives_count = 0
        false_negatives = 0
        errors = 0

        for item in data:
            try:
                is_scam_label = bool(item.get("is_scam", False))

                job = JobPosting(
                    url=item.get("url") or _synthetic_url(item),
                    title=str(item.get("title", "") or ""),
                    company=str(item.get("company", "") or ""),
                    location=str(item.get("location", "") or ""),
                    description=str(item.get("description", "") or ""),
                    salary_min=float(item.get("salary_min", 0.0) or 0.0),
                    salary_max=float(item.get("salary_max", 0.0) or 0.0),
                    posted_date=str(item.get("posted_date", "") or ""),
                    experience_level=str(item.get("experience_level", "") or ""),
                    employment_type=str(item.get("employment_type", "") or ""),
                    industry=str(item.get("industry", "") or ""),
                    is_remote=bool(item.get("telecommuting", False)),
                    source=str(item.get("source", "labeled_data") or "labeled_data"),
                )

                signals = extract_signals(job)
                scam_score, confidence = score_signals(signals)
                result = build_result(job, signals, scam_score, confidence)

                # Prediction: scam if score >= 0.5
                predicted_scam = scam_score >= 0.5

                total += 1
                if predicted_scam == is_scam_label:
                    correct += 1
                    if is_scam_label:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if predicted_scam and not is_scam_label:
                        false_positives_count += 1
                    else:
                        false_negatives += 1

                # Update pattern weights via flywheel
                report = UserReport(
                    url=job.url,
                    is_scam=is_scam_label,
                    reason=f"labeled_data:{item.get('source', 'unknown')}",
                    reported_at=_now_iso(),
                    our_prediction=scam_score,
                    was_correct=(predicted_scam == is_scam_label),
                )
                flywheel.learn_from_report(report, result)

                # Persist job to DB
                import json as _json
                signals_json = _json.dumps([
                    {"name": s.name, "weight": s.weight, "detail": s.detail}
                    for s in signals
                ])
                db.save_job({
                    "url": job.url,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "description": job.description,
                    "salary_min": job.salary_min,
                    "salary_max": job.salary_max,
                    "scam_score": scam_score,
                    "confidence": confidence,
                    "risk_level": result.risk_level.value,
                    "analyzed_at": _now_iso(),
                    "signal_count": len(signals),
                    "signals_json": signals_json,
                    "user_reported": 1,
                    "user_verdict": "scam" if is_scam_label else "legitimate",
                })

            except Exception:
                logger.warning("Failed to process labeled item", exc_info=True)
                errors += 1
                continue

        accuracy = correct / total if total > 0 else 0.0
        precision = (
            true_positives / (true_positives + false_positives_count)
            if (true_positives + false_positives_count) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives_count,
            "false_negatives": false_negatives,
            "errors": errors,
        }
        logger.info(
            "Ingested %d labeled jobs: acc=%.3f prec=%.3f rec=%.3f f1=%.3f",
            total, accuracy, precision, recall, f1,
        )
        return metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_job_dict(
    title: str,
    company: str,
    description: str,
    location: str = "",
    salary_min: float = 0.0,
    salary_max: float = 0.0,
    source: str = "",
    is_scam: bool = True,
    scam_category: str = "",
    **extra: object,
) -> dict:
    """Build a normalized job dict matching the jobs table schema + is_scam flag."""
    url = _synthetic_url(
        {"title": title, "company": company, "source": source}
    )
    return {
        "url": url,
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "source": source,
        "is_scam": is_scam,
        "scam_category": scam_category,
        "posted_date": "",
        "scam_score": 0.0,
        "confidence": 0.0,
        "risk_level": "",
        "analyzed_at": _now_iso(),
        "signal_count": 0,
        "signals_json": "[]",
        "user_reported": 0,
        "user_verdict": "scam" if is_scam else "",
        **extra,
    }


def _synthetic_url(item: dict) -> str:
    """Generate a stable synthetic URL for items that lack one."""
    slug = f"{item.get('source','')}-{item.get('title','')}-{item.get('company','')}"
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:12]
    return f"sentinel://synthetic/{digest}"


def _parse_salary_range(salary_str: str) -> tuple[float, float]:
    """Parse a salary range string like '$50,000-$70,000' into (min, max) floats."""
    if not salary_str:
        return 0.0, 0.0
    # Remove currency symbols, spaces, commas
    cleaned = re.sub(r"[£€$,\s]", "", salary_str)
    # Match one or two numbers separated by a dash/hyphen
    match = re.match(r"([\d.]+)(?:[-–]([\d.]+))?", cleaned)
    if not match:
        return 0.0, 0.0
    try:
        lo = float(match.group(1))
        hi = float(match.group(2)) if match.group(2) else lo
        return lo, hi
    except (ValueError, TypeError):
        return 0.0, 0.0


def _get_existing_pattern(db: "SentinelDB", pattern_id: str) -> dict | None:
    """Return the pattern row if it exists in the DB, else None."""
    try:
        row = db.conn.execute(
            "SELECT * FROM patterns WHERE pattern_id = ?", (pattern_id,)
        ).fetchone()
        return dict(row) if row is not None else None
    except Exception:
        return None
