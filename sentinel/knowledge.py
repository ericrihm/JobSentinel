"""Knowledge base for scam patterns, verified companies, and user reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sentinel.db import SentinelDB
from sentinel.models import ScamPattern, SignalCategory, UserReport


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Default seed patterns
# ---------------------------------------------------------------------------

_DEFAULT_PATTERNS: list[dict] = [
    {
        "pattern_id": "fake_check_scam",
        "name": "Fake Check / Overpayment Scam",
        "description": "Employer sends a fake check, asks victim to cash it and wire back a portion.",
        "category": "red_flag",
        "regex": r"(?i)(send\s+check|deposit\s+check|cash\s+the\s+check|overpayment|wire\s+(back|transfer))",
        "keywords": ["send check", "deposit check", "wire back", "overpayment", "cash the check"],
    },
    {
        "pattern_id": "reshipping_scam",
        "name": "Reshipping / Package Forwarding Scam",
        "description": "Job requires receiving packages at home and re-shipping them — a money mule scheme.",
        "category": "red_flag",
        "regex": r"(?i)(reship|re-ship|package\s+forward|receive\s+(packages?|shipments?)\s+at\s+home|shipping\s+manager\s+from\s+home)",
        "keywords": ["reship", "receive packages at home", "forward packages", "shipping agent from home"],
    },
    {
        "pattern_id": "data_entry_scam",
        "name": "High-Pay Data Entry Scam",
        "description": "Vague 'data entry' role promising unusually high hourly rates with no experience required.",
        "category": "red_flag",
        "regex": r"(?i)(data\s+entry).{0,80}(\$[5-9]\d|\$\d{3,})\s*(per\s+hour|/hr|hourly)",
        "keywords": ["data entry", "no experience required", "work from home data entry", "earn up to"],
    },
    {
        "pattern_id": "mlm_recruitment",
        "name": "MLM / Network Marketing Recruitment",
        "description": "Job is actually recruitment for a multi-level marketing scheme disguised as employment.",
        "category": "red_flag",
        "regex": r"(?i)(network\s+marketing|multi.?level|unlimited\s+earning\s+potential|build\s+your\s+(own\s+)?team|recruit\s+(others|friends)|downline)",
        "keywords": ["network marketing", "multi-level", "unlimited earning potential", "build your team", "downline", "recruit others"],
    },
    {
        "pattern_id": "mystery_shopper",
        "name": "Mystery Shopper Scam",
        "description": "Fake mystery shopping job that uses fake check / upfront payment tactics.",
        "category": "red_flag",
        "regex": r"(?i)(mystery\s+shopper|secret\s+shopper).{0,200}(check|wire|payment|advance)",
        "keywords": ["mystery shopper", "secret shopper", "evaluate stores", "shop and get paid"],
    },
    {
        "pattern_id": "envelope_stuffing",
        "name": "Envelope Stuffing / Craft Assembly",
        "description": "Classic work-from-home scam: stuff envelopes or assemble items for pay.",
        "category": "red_flag",
        "regex": r"(?i)(stuff(ing)?\s+envelopes|envelope\s+stuffing|assemble\s+(crafts|products|items)\s+at\s+home|craft\s+assembly)",
        "keywords": ["stuff envelopes", "envelope stuffing", "assemble at home", "craft assembly from home"],
    },
    {
        "pattern_id": "assembly_at_home",
        "name": "Product Assembly at Home",
        "description": "Pay-to-play scheme where workers buy a kit upfront and are never paid for finished goods.",
        "category": "red_flag",
        "regex": r"(?i)(assemble\s+(products?|items?|crafts?|jewelry|toys?)\s+at\s+home|home\s+assembly\s+work|work\s+from\s+home.{0,60}assembly)",
        "keywords": ["assemble products at home", "home assembly work", "assembly kit", "buy starter kit"],
    },
    {
        "pattern_id": "pyramid_scheme",
        "name": "Pyramid / Ponzi Scheme",
        "description": "Returns depend primarily on recruitment rather than product/service sales.",
        "category": "red_flag",
        "regex": r"(?i)(pyramid|ponzi|earn\s+by\s+recruit|get\s+paid\s+to\s+recruit|investment\s+opportunity.{0,60}guaranteed\s+return)",
        "keywords": ["pyramid", "earn by recruiting", "get paid to recruit", "investment opportunity guaranteed"],
    },
    {
        "pattern_id": "upfront_payment_required",
        "name": "Upfront Payment / Training Fee",
        "description": "Legitimate employers never ask candidates to pay for training, materials, or background checks.",
        "category": "red_flag",
        "regex": r"(?i)(pay\s+(for|a)\s+(training|background\s+check|starter\s+kit|materials|certification|access\s+fee)|registration\s+fee\s+required|refundable\s+deposit)",
        "keywords": ["pay for training", "starter kit fee", "registration fee", "background check fee", "refundable deposit"],
    },
    {
        "pattern_id": "personal_info_upfront",
        "name": "SSN / Bank Info Before Interview",
        "description": "Requests for SSN, bank account, or passport number before any formal interview.",
        "category": "red_flag",
        "regex": r"(?i)(social\s+security\s+number|SSN|bank\s+account\s+(number|details)|passport\s+number|direct\s+deposit\s+info).{0,120}(apply|application|before\s+interview|to\s+get\s+started)",
        "keywords": ["provide SSN", "bank account number to apply", "passport copy to apply", "direct deposit to start"],
    },
    {
        "pattern_id": "guaranteed_income",
        "name": "Guaranteed Income Claims",
        "description": "Promises of specific guaranteed earnings, especially unrealistically high ones.",
        "category": "red_flag",
        "regex": r"(?i)(guarantee[sd]?\s+(\$[\d,]+|\d+k?)\s*(per\s+(week|month|day|hour))?|earn\s+up\s+to\s+\$[\d,]+\s+guaranteed|you\s+will\s+earn)",
        "keywords": ["guaranteed income", "earn guaranteed", "you will make", "earn up to guaranteed"],
    },
    {
        "pattern_id": "crypto_payment",
        "name": "Cryptocurrency / Wire Transfer Payment",
        "description": "Compensation or job duties involve cryptocurrency or wire transfers — classic money mule pattern.",
        "category": "red_flag",
        "regex": r"(?i)(paid\s+in\s+(bitcoin|crypto|ethereum|usdt)|cryptocurrency\s+(payment|wallet)|wire\s+transfer.{0,60}job|send\s+crypto|bitcoin\s+wallet)",
        "keywords": ["paid in bitcoin", "crypto wallet", "wire transfer job", "send cryptocurrency"],
    },
    {
        "pattern_id": "fake_company_email",
        "name": "Suspicious Email Domain",
        "description": "Corporate recruiter using a personal email (gmail/yahoo/hotmail) for official hiring.",
        "category": "red_flag",
        "regex": r"(?i)contact.{0,40}(@gmail\.com|@yahoo\.com|@hotmail\.com|@outlook\.com|@aol\.com).{0,60}(apply|questions|hiring)",
        "keywords": ["gmail recruiter", "yahoo hr", "contact us at gmail", "apply via personal email"],
    },
    {
        "pattern_id": "urgency_pressure",
        "name": "Urgency / Pressure Tactics",
        "description": "Artificial urgency designed to prevent applicants from doing due diligence.",
        "category": "warning",
        "regex": r"(?i)(apply\s+(now|immediately|today|asap)|limited\s+(spots?|positions?|openings?)|hiring\s+(now|immediately|urgently|asap)|offer\s+expires|must\s+respond\s+within)",
        "keywords": ["apply now", "limited spots", "hiring immediately", "offer expires", "must respond within 24 hours"],
    },
    {
        "pattern_id": "vague_description",
        "name": "Extremely Vague Job Description",
        "description": "Description lacks any concrete responsibilities, tools, or qualifications.",
        "category": "warning",
        "regex": r"",
        "keywords": ["various tasks", "general duties", "assist with projects", "flexible responsibilities", "other duties as assigned"],
    },
    {
        "pattern_id": "unrealistic_salary",
        "name": "Unrealistically High Salary",
        "description": "Advertised pay is significantly above market for the stated role and location.",
        "category": "warning",
        "regex": r"(?i)(\$[1-9]\d{2,}\s*(per\s+hour|/hr|hourly)).{0,80}(no\s+experience|entry\s+level|anyone\s+can)",
        "keywords": ["$100/hr no experience", "earn hundreds daily", "six figures working from home"],
    },
    {
        "pattern_id": "ghost_job_repost",
        "name": "Ghost Job / Chronic Repost",
        "description": "Same role reposted repeatedly with no apparent hires — likely a pipeline-building ghost job.",
        "category": "ghost_job",
        "regex": r"",
        "keywords": ["always hiring", "continuously recruiting", "open pipeline", "pool of candidates"],
    },
    {
        "pattern_id": "no_qualifications",
        "name": "No Skills or Qualifications Listed",
        "description": "Legitimate jobs list specific requirements. No requirements signals a scam or ghost posting.",
        "category": "warning",
        "regex": r"(?i)(no\s+(experience|qualifications?|skills?|degree|education)\s+(required|needed|necessary)|anyone\s+can\s+(apply|do\s+this|qualify))",
        "keywords": ["no experience required", "no qualifications needed", "anyone can apply", "no skills necessary"],
    },
    {
        "pattern_id": "work_from_home_unrealistic",
        "name": "Unrealistic Work-From-Home Pay",
        "description": "Remote work promising unusually high pay for low-skill tasks.",
        "category": "warning",
        "regex": r"(?i)(work\s+from\s+home).{0,120}(\$[5-9]\d{2,}|\$[1-9]\d{3,})\s*(per\s+(week|day)|weekly|daily)",
        "keywords": ["work from home $1000 a day", "make money from home guaranteed", "earn $500 daily from home"],
    },
    {
        "pattern_id": "job_offer_no_interview",
        "name": "Job Offer Without Interview",
        "description": "Instant job offers with no interview are a strong scam signal.",
        "category": "red_flag",
        "regex": r"(?i)(hired\s+immediately|no\s+interview\s+(required|needed)|instant(ly)?\s+(hired|approved)|you('re|re|are)\s+(already\s+)?hired)",
        "keywords": ["hired immediately", "no interview required", "instantly hired", "you're already hired"],
    },
]


class KnowledgeBase:
    """High-level interface to scam patterns, verified companies, and user reports."""

    def __init__(self, db: SentinelDB | None = None) -> None:
        self.db = db or SentinelDB()

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def add_pattern(self, pattern: ScamPattern) -> None:
        """Persist a ScamPattern to the database."""
        self.db.save_pattern(
            {
                "pattern_id": pattern.pattern_id,
                "name": pattern.name,
                "description": pattern.description,
                "category": pattern.category.value if hasattr(pattern.category, "value") else pattern.category,
                "regex": pattern.regex,
                "keywords": pattern.keywords,
                "alpha": pattern.alpha,
                "beta": pattern.beta,
                "observations": pattern.observations,
                "true_positives": pattern.true_positives,
                "false_positives": pattern.false_positives,
                "status": pattern.status,
                "created_at": pattern.created_at or _now_iso(),
                "updated_at": pattern.updated_at or _now_iso(),
            }
        )

    def get_active_patterns(self) -> list[ScamPattern]:
        """Return all active patterns as ScamPattern dataclass instances."""
        rows = self.db.get_patterns(status="active")
        patterns: list[ScamPattern] = []
        for row in rows:
            category_str = row.get("category", "red_flag")
            try:
                category = SignalCategory(category_str)
            except ValueError:
                category = SignalCategory.RED_FLAG

            keywords_raw = row.get("keywords") or row.get("keywords_json", "[]")
            if isinstance(keywords_raw, str):
                try:
                    keywords = json.loads(keywords_raw)
                except (json.JSONDecodeError, TypeError):
                    keywords = []
            else:
                keywords = keywords_raw

            patterns.append(
                ScamPattern(
                    pattern_id=row["pattern_id"],
                    name=row["name"],
                    description=row.get("description", ""),
                    category=category,
                    regex=row.get("regex", ""),
                    keywords=keywords,
                    alpha=row.get("alpha", 1.0),
                    beta=row.get("beta", 1.0),
                    observations=row.get("observations", 0),
                    true_positives=row.get("true_positives", 0),
                    false_positives=row.get("false_positives", 0),
                    status=row.get("status", "active"),
                    created_at=row.get("created_at", ""),
                    updated_at=row.get("updated_at", ""),
                )
            )
        return patterns

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def report_scam(
        self,
        url: str,
        is_scam: bool,
        reason: str = "",
        our_prediction: float = 0.0,
    ) -> None:
        """Record a user-submitted verdict for a job URL."""
        was_correct: bool
        if is_scam:
            was_correct = our_prediction >= 0.5
        else:
            was_correct = our_prediction < 0.5

        self.db.save_report(
            {
                "url": url,
                "is_scam": is_scam,
                "reason": reason,
                "our_prediction": our_prediction,
                "was_correct": was_correct,
                "reported_at": _now_iso(),
            }
        )

    def get_known_scam_urls(self) -> set[str]:
        """Return URLs that users have confirmed as scams."""
        rows = self.db.get_reports(limit=10_000)
        return {r["url"] for r in rows if r.get("is_scam") == 1}

    # ------------------------------------------------------------------
    # Companies
    # ------------------------------------------------------------------

    def is_known_safe_company(self, company: str) -> bool:
        """Return True if the company is in the DB and marked verified."""
        record = self.db.get_company(company)
        if record is None:
            return False
        return bool(record.get("is_verified"))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over analysed job postings."""
        return self.db.search_jobs(query, limit=limit)

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    def get_accuracy_stats(self) -> dict:
        """Return prediction accuracy broken down by scam vs legitimate."""
        reports = self.db.get_reports(limit=10_000)
        total = len(reports)
        if total == 0:
            return {
                "total_reports": 0,
                "correct": 0,
                "accuracy": 0.0,
                "scam_reports": 0,
                "legitimate_reports": 0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
            }

        true_positives = sum(
            1 for r in reports if r.get("is_scam") == 1 and r.get("was_correct") == 1
        )
        false_positives = sum(
            1 for r in reports if r.get("is_scam") == 0 and r.get("was_correct") == 0
        )
        true_negatives = sum(
            1 for r in reports if r.get("is_scam") == 0 and r.get("was_correct") == 1
        )
        false_negatives = sum(
            1 for r in reports if r.get("is_scam") == 1 and r.get("was_correct") == 0
        )
        correct = true_positives + true_negatives
        scam_reports = sum(1 for r in reports if r.get("is_scam") == 1)
        legitimate_reports = total - scam_reports

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        return {
            "total_reports": total,
            "correct": correct,
            "accuracy": round(correct / total, 4),
            "scam_reports": scam_reports,
            "legitimate_reports": legitimate_reports,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_default_patterns(self) -> None:
        """Seed the DB with default scam patterns if the patterns table is empty."""
        existing = self.db.get_patterns(status="active")
        # Also check deprecated/candidate so we don't re-seed a pruned DB
        existing_ids = {p["pattern_id"] for p in existing}
        existing_ids |= {p["pattern_id"] for p in self.db.get_patterns(status="deprecated")}
        existing_ids |= {p["pattern_id"] for p in self.db.get_patterns(status="candidate")}

        now = _now_iso()
        added = 0
        for raw in _DEFAULT_PATTERNS:
            if raw["pattern_id"] in existing_ids:
                continue
            self.db.save_pattern(
                {
                    **raw,
                    "keywords_json": json.dumps(raw.get("keywords", [])),
                    "alpha": 2.0,   # start with mild prior toward positive
                    "beta": 1.0,
                    "observations": 0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "status": "active",
                    "created_at": now,
                    "updated_at": now,
                }
            )
            added += 1
