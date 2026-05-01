"""Economic Ground Truth Validator — cross-references job claims against real-world economic data.

Five components:
  MarketRateValidator  — salary ranges vs role/location; "too good to be true" detection
  CompanyEconomics     — company size/age/funding vs claimed headcount
  BenefitsAnalyzer     — parse and sanity-check benefit packages
  GeographicValidator  — HQ vs posting location, MLM patterns, area code checks
  IndustryBenchmark    — percentile ranking and drift detection for salaries

All stdlib.  Imports from sentinel.models and sentinel.db.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional

from sentinel.models import JobPosting, ScamSignal, SignalCategory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference data — static tables with last-updated timestamps
# ---------------------------------------------------------------------------

# ISO 8601 date: when these figures were last reviewed
_DATA_UPDATED = "2025-08-01"

# ---------------------------------------------------------------------------
# SALARY DATA
# role_category -> experience_level -> (p10, p25, p50, p75, p90, p99)
# All figures in USD, US national median.
# Sources: BLS Occupational Employment Statistics, Levels.fyi, LinkedIn Salary
# ---------------------------------------------------------------------------

SALARY_RANGES: dict[str, dict[str, tuple[int, int, int, int, int, int]]] = {
    # (p10, p25, p50, p75, p90, p99)
    "software_engineer": {
        "entry":  (60_000,  75_000,  92_000, 115_000, 135_000, 175_000),
        "mid":    (95_000, 110_000, 135_000, 165_000, 200_000, 280_000),
        "senior": (140_000, 165_000, 195_000, 235_000, 280_000, 400_000),
        "exec":   (200_000, 250_000, 320_000, 420_000, 550_000, 900_000),
    },
    "data_scientist": {
        "entry":  (65_000,  80_000,  98_000, 122_000, 148_000, 200_000),
        "mid":    (95_000, 115_000, 140_000, 175_000, 215_000, 300_000),
        "senior": (140_000, 165_000, 195_000, 235_000, 285_000, 400_000),
        "exec":   (190_000, 240_000, 305_000, 400_000, 520_000, 800_000),
    },
    "data_analyst": {
        "entry":  (42_000,  55_000,  68_000,  85_000, 102_000, 135_000),
        "mid":    (62_000,  78_000,  96_000, 118_000, 142_000, 185_000),
        "senior": (88_000, 108_000, 130_000, 158_000, 188_000, 240_000),
        "exec":   (130_000, 165_000, 205_000, 260_000, 320_000, 480_000),
    },
    "machine_learning_engineer": {
        "entry":  (75_000,  92_000, 115_000, 140_000, 170_000, 225_000),
        "mid":    (110_000, 135_000, 165_000, 200_000, 245_000, 340_000),
        "senior": (160_000, 195_000, 235_000, 285_000, 345_000, 500_000),
        "exec":   (220_000, 280_000, 360_000, 460_000, 600_000, 950_000),
    },
    "devops_engineer": {
        "entry":  (58_000,  72_000,  90_000, 112_000, 135_000, 175_000),
        "mid":    (90_000, 108_000, 130_000, 158_000, 192_000, 260_000),
        "senior": (130_000, 158_000, 188_000, 225_000, 270_000, 380_000),
        "exec":   (180_000, 220_000, 280_000, 360_000, 460_000, 700_000),
    },
    "product_manager": {
        "entry":  (65_000,  80_000,  98_000, 120_000, 145_000, 190_000),
        "mid":    (95_000, 118_000, 145_000, 178_000, 218_000, 300_000),
        "senior": (140_000, 170_000, 210_000, 258_000, 315_000, 450_000),
        "exec":   (185_000, 235_000, 300_000, 390_000, 510_000, 800_000),
    },
    "marketing_manager": {
        "entry":  (38_000,  48_000,  60_000,  76_000,  92_000, 125_000),
        "mid":    (58_000,  72_000,  90_000, 112_000, 138_000, 185_000),
        "senior": (85_000, 105_000, 130_000, 162_000, 198_000, 268_000),
        "exec":   (130_000, 165_000, 210_000, 270_000, 350_000, 550_000),
    },
    "sales": {
        "entry":  (32_000,  42_000,  55_000,  72_000,  92_000, 135_000),
        "mid":    (50_000,  65_000,  85_000, 112_000, 145_000, 210_000),
        "senior": (75_000, 100_000, 130_000, 170_000, 220_000, 350_000),
        "exec":   (120_000, 160_000, 215_000, 290_000, 390_000, 650_000),
    },
    "customer_service": {
        "entry":  (26_000,  32_000,  40_000,  50_000,  62_000,  82_000),
        "mid":    (34_000,  42_000,  52_000,  65_000,  80_000, 105_000),
        "senior": (44_000,  55_000,  68_000,  85_000, 105_000, 138_000),
        "exec":   (65_000,  85_000, 108_000, 138_000, 175_000, 250_000),
    },
    "human_resources": {
        "entry":  (36_000,  45_000,  56_000,  70_000,  86_000, 115_000),
        "mid":    (52_000,  65_000,  80_000,  99_000, 122_000, 162_000),
        "senior": (75_000,  95_000, 118_000, 148_000, 182_000, 248_000),
        "exec":   (115_000, 148_000, 188_000, 245_000, 320_000, 510_000),
    },
    "accounting": {
        "entry":  (38_000,  48_000,  60_000,  75_000,  92_000, 125_000),
        "mid":    (56_000,  70_000,  88_000, 110_000, 135_000, 180_000),
        "senior": (82_000, 104_000, 130_000, 162_000, 200_000, 270_000),
        "exec":   (130_000, 168_000, 215_000, 280_000, 368_000, 600_000),
    },
    "project_manager": {
        "entry":  (45_000,  56_000,  70_000,  88_000, 108_000, 145_000),
        "mid":    (68_000,  84_000, 104_000, 128_000, 158_000, 210_000),
        "senior": (95_000, 120_000, 148_000, 182_000, 222_000, 300_000),
        "exec":   (140_000, 178_000, 225_000, 290_000, 378_000, 600_000),
    },
    "nursing": {
        "entry":  (48_000,  58_000,  70_000,  86_000, 105_000, 140_000),
        "mid":    (62_000,  75_000,  92_000, 112_000, 138_000, 182_000),
        "senior": (82_000, 100_000, 122_000, 150_000, 184_000, 245_000),
        "exec":   (110_000, 140_000, 178_000, 228_000, 300_000, 480_000),
    },
    "teacher": {
        "entry":  (32_000,  38_000,  46_000,  56_000,  68_000,  88_000),
        "mid":    (40_000,  49_000,  60_000,  74_000,  90_000, 118_000),
        "senior": (52_000,  64_000,  78_000,  96_000, 118_000, 155_000),
        "exec":   (72_000,  92_000, 116_000, 148_000, 192_000, 290_000),
    },
    "warehouse": {
        "entry":  (24_000,  30_000,  38_000,  48_000,  60_000,  80_000),
        "mid":    (32_000,  40_000,  50_000,  62_000,  76_000, 100_000),
        "senior": (42_000,  52_000,  64_000,  80_000,  98_000, 130_000),
        "exec":   (60_000,  76_000,  96_000, 122_000, 158_000, 240_000),
    },
    "graphic_designer": {
        "entry":  (32_000,  40_000,  50_000,  62_000,  76_000, 100_000),
        "mid":    (46_000,  58_000,  72_000,  90_000, 110_000, 148_000),
        "senior": (65_000,  82_000, 102_000, 126_000, 156_000, 210_000),
        "exec":   (95_000, 122_000, 155_000, 198_000, 258_000, 400_000),
    },
    "legal": {
        "entry":  (48_000,  60_000,  76_000,  96_000, 120_000, 165_000),
        "mid":    (78_000, 100_000, 128_000, 165_000, 210_000, 295_000),
        "senior": (120_000, 158_000, 205_000, 270_000, 360_000, 560_000),
        "exec":   (185_000, 250_000, 340_000, 465_000, 640_000, 1_100_000),
    },
    "admin_assistant": {
        "entry":  (28_000,  35_000,  43_000,  54_000,  66_000,  88_000),
        "mid":    (36_000,  45_000,  56_000,  70_000,  86_000, 115_000),
        "senior": (48_000,  60_000,  74_000,  92_000, 114_000, 152_000),
        "exec":   (65_000,  82_000, 102_000, 128_000, 162_000, 240_000),
    },
}

# ---------------------------------------------------------------------------
# Regional cost-of-living multipliers (applied to national median)
# Key: lowercase city/region substring or state abbreviation
# ---------------------------------------------------------------------------

COL_MULTIPLIERS: dict[str, float] = {
    # High-cost metros
    "san francisco":   1.45,
    "san jose":        1.40,
    "palo alto":       1.42,
    "mountain view":   1.40,
    "sunnyvale":       1.40,
    "santa clara":     1.38,
    "seattle":         1.25,
    "new york":        1.32,
    "nyc":             1.32,
    "manhattan":       1.40,
    "brooklyn":        1.28,
    "boston":          1.28,
    "cambridge":       1.30,
    "washington dc":   1.22,
    "washington, dc":  1.22,
    "los angeles":     1.25,
    "santa monica":    1.28,
    "venice, ca":      1.26,
    "austin":          1.10,
    "denver":          1.12,
    "chicago":         1.10,
    "miami":           1.12,
    "portland":        1.12,
    "san diego":       1.20,
    # Mid-cost metros
    "atlanta":         1.00,
    "dallas":          1.00,
    "houston":         0.98,
    "phoenix":         0.98,
    "minneapolis":     1.02,
    "salt lake city":  1.02,
    "charlotte":       0.98,
    "raleigh":         1.00,
    "nashville":       1.00,
    "columbus":        0.95,
    "indianapolis":    0.94,
    "kansas city":     0.93,
    "memphis":         0.90,
    "st. louis":       0.92,
    # Low-cost
    "rural":           0.72,
    "remote":          1.00,  # remote jobs use national median
    "anywhere":        1.00,
}

# Fall-back state-level multipliers (two-letter state abbreviations)
_STATE_COL: dict[str, float] = {
    "CA": 1.30, "NY": 1.28, "WA": 1.22, "MA": 1.25, "AK": 1.15,
    "HI": 1.20, "NJ": 1.22, "CT": 1.18, "MD": 1.15, "CO": 1.10,
    "OR": 1.10, "IL": 1.05, "VA": 1.08, "TX": 0.98, "FL": 1.00,
    "GA": 0.98, "NC": 0.97, "OH": 0.93, "MI": 0.93, "PA": 0.98,
    "MN": 1.00, "AZ": 0.97, "IN": 0.91, "KS": 0.90, "MO": 0.91,
    "TN": 0.92, "AL": 0.88, "AR": 0.86, "MS": 0.84, "WV": 0.83,
}

# ---------------------------------------------------------------------------
# Role-category classification patterns  (title → category key)
# ---------------------------------------------------------------------------

_ROLE_PATTERNS: list[tuple[str, str]] = [
    # most-specific first
    (r"\b(machine learning|ml engineer|mlops|ai engineer)\b", "machine_learning_engineer"),
    (r"\b(devops|site reliability|sre|platform engineer|infrastructure engineer)\b", "devops_engineer"),
    (r"\b(software|developer|programmer|engineer|swe|sde|backend|frontend|full.?stack|coder)\b", "software_engineer"),
    (r"\b(data scient\w*|machine learning scientist)\b", "data_scientist"),
    (r"\b(data analyst|business analyst|bi analyst|analytics)\b", "data_analyst"),
    (r"\b(product manager|program manager)\b", "product_manager"),
    (r"\b(marketing)\b", "marketing_manager"),
    (r"\b(sales|account executive|account manager|bdr|sdr|business development)\b", "sales"),
    (r"\b(customer service|customer support|call center|help desk|client service)\b", "customer_service"),
    (r"\b(human resources|hr |recruiter|talent acquisition|people ops)\b", "human_resources"),
    (r"\b(account\w*|bookkeep\w*|auditor|controller|cpa|financial analyst)\b", "accounting"),
    (r"\b(project manager|pmp|scrum master|agile coach)\b", "project_manager"),
    (r"\b(nurse|nursing|rn |lpn |cna |clinical)\b", "nursing"),
    (r"\b(teacher|educator|instructor|professor|tutor|curriculum)\b", "teacher"),
    (r"\b(warehouse|forklift|picker|packer|logistics|supply chain|fulfillment)\b", "warehouse"),
    (r"\b(graphic designer|visual designer|ux designer|ui designer|illustrator)\b", "graphic_designer"),
    (r"\b(attorney|lawyer|paralegal|counsel|legal)\b", "legal"),
    (r"\b(admin|administrative|executive assistant|office manager|receptionist)\b", "admin_assistant"),
]
_ROLE_RE = [(re.compile(p, re.IGNORECASE), cat) for p, cat in _ROLE_PATTERNS]


def classify_role(title: str) -> str | None:
    """Map a job title to a salary-range category key, or None if unknown."""
    for pattern, cat in _ROLE_RE:
        if pattern.search(title):
            return cat
    return None


# ---------------------------------------------------------------------------
# Experience-level normalisation
# ---------------------------------------------------------------------------

_LEVEL_ENTRY = re.compile(
    r"\b(entry.level|entry|intern|internship|junior|jr\.?|"
    r"new.grad|early.career|0.?[–\-]2.?years?)\b",
    re.IGNORECASE,
)
_LEVEL_SENIOR = re.compile(
    r"\b(senior|sr\.?|lead|principal|staff|distinguished|fellow|"
    r"director|vp|vice.president|svp|evp|c.level|cto|ceo|coo|cfo)\b",
    re.IGNORECASE,
)
_LEVEL_EXEC = re.compile(
    r"\b(chief|executive|president|cxo|c-suite|managing.director|"
    r"vp of|head of|general.manager)\b",
    re.IGNORECASE,
)


def normalize_experience_level(raw: str) -> str:
    """Return 'entry', 'mid', 'senior', or 'exec'."""
    if _LEVEL_EXEC.search(raw):
        return "exec"
    if _LEVEL_SENIOR.search(raw):
        return "senior"
    if _LEVEL_ENTRY.search(raw):
        return "entry"
    return "mid"


# ---------------------------------------------------------------------------
# Location → CoL multiplier
# ---------------------------------------------------------------------------

def get_col_multiplier(location: str) -> float:
    """Return the cost-of-living multiplier for a location string."""
    loc_lower = location.lower()
    # Exact substring match in COL_MULTIPLIERS (longest key wins)
    best_key = ""
    best_mult = 1.0
    for key, mult in COL_MULTIPLIERS.items():
        if key in loc_lower and len(key) > len(best_key):
            best_key = key
            best_mult = mult
    if best_key:
        return best_mult
    # Fall back to state abbreviation (", TX" pattern)
    m = re.search(r",?\s*([A-Z]{2})\b", location)
    if m:
        state = m.group(1)
        return _STATE_COL.get(state, 1.0)
    return 1.0


# ===========================================================================
# 1. MarketRateValidator
# ===========================================================================

@dataclass
class SalaryValidationResult:
    """Outcome from MarketRateValidator."""
    is_suspicious: bool = False
    signals: list[ScamSignal] = field(default_factory=list)
    role_category: str | None = None
    experience_level: str = "mid"
    col_multiplier: float = 1.0
    expected_p50: int | None = None
    expected_p90: int | None = None
    percentile_estimate: int | None = None  # 0-100 or None if unknown


class MarketRateValidator:
    """Validates salary claims against BLS/market-rate reference data."""

    # Thresholds
    TOO_HIGH_MULTIPLIER = 2.0   # > 2x p90 → suspicious
    VERY_HIGH_MULTIPLIER = 3.0  # > 3x p90 → strong red flag
    TOO_LOW_MULTIPLIER = 0.35   # < 35% of p25 → suspicious (possible bait/switcher)

    # "Too good to be true" combinations (remote + entry + implausible salary)
    _TGTB_THRESHOLD = 150_000   # entry-level remote over this is suspicious
    _IMPOSSIBLE_INTERN_THRESHOLD = 120_000  # intern + this salary is impossible

    def validate(self, job: JobPosting) -> SalaryValidationResult:
        """Full salary validation; returns a SalaryValidationResult with signals."""
        result = SalaryValidationResult()

        if not job.salary_min and not job.salary_max:
            return result  # No salary data — nothing to flag

        salary = job.salary_max or job.salary_min

        # Determine role and experience
        role = classify_role(job.title)
        exp_raw = f"{job.experience_level} {job.title}"
        level = normalize_experience_level(exp_raw)
        col = get_col_multiplier(job.location)

        result.role_category = role
        result.experience_level = level
        result.col_multiplier = col

        if role and level in SALARY_RANGES.get(role, {}):
            p10, p25, p50, p75, p90, p99 = SALARY_RANGES[role][level]
            adj_p50 = int(p50 * col)
            adj_p90 = int(p90 * col)
            adj_p99 = int(p99 * col)
            adj_p25 = int(p25 * col)
            adj_p10 = int(p10 * col)

            result.expected_p50 = adj_p50
            result.expected_p90 = adj_p90

            # Estimate percentile
            result.percentile_estimate = self._estimate_percentile(
                salary, (adj_p10, adj_p25, adj_p50, adj_p90, adj_p99)
            )

            sig = self._check_market_rate(job, salary, role, level, adj_p25, adj_p90, adj_p99)
            if sig:
                result.signals.append(sig)
                result.is_suspicious = True
        else:
            # No category match — fall back to generic wide-range check
            sig = self._check_generic(job)
            if sig:
                result.signals.append(sig)
                result.is_suspicious = True

        # "Too good to be true" composite check
        tgtb_sig = self._check_too_good_to_be_true(job, salary, level)
        if tgtb_sig:
            result.signals.append(tgtb_sig)
            result.is_suspicious = True

        # Impossible combination check
        impossible_sig = self._check_impossible_combination(job, salary, level)
        if impossible_sig:
            result.signals.append(impossible_sig)
            result.is_suspicious = True

        return result

    # ------------------------------------------------------------------

    def _estimate_percentile(self, salary: float, benchmarks: tuple) -> int:
        """Rough percentile estimate from (p10, p25, p50, p90, p99)."""
        p10, p25, p50, p90, p99 = benchmarks
        if salary <= p10:
            return 5
        if salary <= p25:
            return int(10 + 15 * (salary - p10) / max(1, p25 - p10))
        if salary <= p50:
            return int(25 + 25 * (salary - p25) / max(1, p50 - p25))
        if salary <= p90:
            return int(50 + 40 * (salary - p50) / max(1, p90 - p50))
        if salary <= p99:
            return int(90 + 9 * (salary - p90) / max(1, p99 - p90))
        return 99

    def _check_market_rate(
        self,
        job: JobPosting,
        salary: float,
        role: str,
        level: str,
        adj_p25: int,
        adj_p90: int,
        adj_p99: int,
    ) -> ScamSignal | None:
        location_label = job.location or "US national"

        if salary > adj_p90 * self.VERY_HIGH_MULTIPLIER:
            return ScamSignal(
                name="salary_too_high",
                category=SignalCategory.RED_FLAG,
                weight=0.88,
                confidence=0.85,
                detail=(
                    f"Salary ${salary:,.0f} is >{self.VERY_HIGH_MULTIPLIER:.0f}x P90 "
                    f"(${adj_p90:,}) for {level} {role.replace('_', ' ')} "
                    f"in {location_label}"
                ),
                evidence=f"role={role}, level={level}, adj_p90=${adj_p90:,}",
            )

        if salary > adj_p90 * self.TOO_HIGH_MULTIPLIER:
            return ScamSignal(
                name="salary_too_high",
                category=SignalCategory.WARNING,
                weight=0.72,
                confidence=0.78,
                detail=(
                    f"Salary ${salary:,.0f} is >{self.TOO_HIGH_MULTIPLIER:.0f}x P90 "
                    f"(${adj_p90:,}) for {level} {role.replace('_', ' ')} "
                    f"in {location_label}"
                ),
                evidence=f"role={role}, level={level}, adj_p90=${adj_p90:,}",
            )

        if adj_p25 > 0 and salary < adj_p25 * self.TOO_LOW_MULTIPLIER:
            return ScamSignal(
                name="salary_too_low",
                category=SignalCategory.WARNING,
                weight=0.60,
                confidence=0.65,
                detail=(
                    f"Salary ${salary:,.0f} is <{self.TOO_LOW_MULTIPLIER:.0%} of P25 "
                    f"(${adj_p25:,}) for {level} {role.replace('_', ' ')} "
                    f"in {location_label} — possible bait-and-switch"
                ),
                evidence=f"role={role}, level={level}, adj_p25=${adj_p25:,}",
            )

        return None

    def _check_generic(self, job: JobPosting) -> ScamSignal | None:
        """Fallback: flag extreme salary range width when category is unknown."""
        lo = job.salary_min or 0
        hi = job.salary_max or 0
        if lo > 0 and hi > 0 and hi > lo * 4:
            return ScamSignal(
                name="salary_range_implausible",
                category=SignalCategory.WARNING,
                weight=0.55,
                confidence=0.60,
                detail=f"Salary range ${lo:,.0f}–${hi:,.0f} is implausibly wide (>{4}x spread)",
                evidence=f"min={lo}, max={hi}",
            )
        return None

    def _check_too_good_to_be_true(
        self, job: JobPosting, salary: float, level: str
    ) -> ScamSignal | None:
        """Flag remote + entry-level + high salary combinations."""
        if not job.is_remote:
            return None
        if level != "entry":
            return None
        if salary >= self._TGTB_THRESHOLD:
            return ScamSignal(
                name="too_good_to_be_true",
                category=SignalCategory.RED_FLAG,
                weight=0.82,
                confidence=0.80,
                detail=(
                    f"Remote + entry-level + ${salary:,.0f}/yr exceeds the "
                    f"${self._TGTB_THRESHOLD:,} threshold — classic too-good-to-be-true pattern"
                ),
                evidence=f"is_remote=True, level=entry, salary=${salary:,.0f}",
            )
        return None

    def _check_impossible_combination(
        self, job: JobPosting, salary: float, level: str
    ) -> ScamSignal | None:
        """Flag intern/entry salary + impossible base pay."""
        title_lower = job.title.lower()
        is_intern = any(w in title_lower for w in ("intern", "internship", "co-op", "coop"))
        if is_intern and salary >= self._IMPOSSIBLE_INTERN_THRESHOLD:
            return ScamSignal(
                name="impossible_combination",
                category=SignalCategory.RED_FLAG,
                weight=0.90,
                confidence=0.88,
                detail=(
                    f"Intern role with ${salary:,.0f}/yr base — no legitimate "
                    f"internship pays this (threshold: ${self._IMPOSSIBLE_INTERN_THRESHOLD:,})"
                ),
                evidence=f"title={job.title!r}, salary=${salary:,.0f}",
            )

        # No experience needed + high salary
        desc_lower = job.description.lower()
        no_exp_phrases = (
            "no experience required", "no experience needed",
            "no experience necessary", "anyone can",
        )
        if any(p in desc_lower for p in no_exp_phrases) and salary >= 200_000:
            return ScamSignal(
                name="impossible_combination",
                category=SignalCategory.RED_FLAG,
                weight=0.86,
                confidence=0.84,
                detail=(
                    f"'No experience required' + ${salary:,.0f}/yr base — "
                    "legitimate high-paying roles always require qualifications"
                ),
                evidence=f"salary=${salary:,.0f}, no_experience_phrase=True",
            )
        return None


# ===========================================================================
# 2. CompanyEconomics
# ===========================================================================

# Expected employee count ranges by funding stage and age
# (min_employees, max_employees)
_FUNDING_STAGE_EMPLOYEES: dict[str, tuple[int, int]] = {
    "pre-seed":   (1, 15),
    "seed":       (2, 50),
    "series_a":   (10, 150),
    "series_b":   (30, 500),
    "series_c":   (100, 1_500),
    "series_d":   (300, 5_000),
    "late_stage": (500, 20_000),
    "public":     (200, 500_000),
    "private":    (1, 500_000),
    "bootstrapped": (1, 200),
}

# Industry-specific hiring patterns
# "continuous" = tech/startup always hiring; "seasonal" = retail/agriculture
INDUSTRY_HIRING_PATTERNS: dict[str, str] = {
    "technology":       "continuous",
    "software":         "continuous",
    "finance":          "continuous",
    "healthcare":       "continuous",
    "retail":           "seasonal",
    "agriculture":      "seasonal",
    "education":        "seasonal",   # school-year hiring
    "hospitality":      "seasonal",
    "construction":     "project_based",
    "consulting":       "project_based",
    "government":       "cycle_based",
    "nonprofit":        "grant_based",
    "media":            "continuous",
    "manufacturing":    "steady",
    "logistics":        "seasonal",
}

# Revenue per employee benchmarks by industry (USD)
# (p25, p50, p75) — used for sanity checks if revenue data is available
REVENUE_PER_EMPLOYEE: dict[str, tuple[int, int, int]] = {
    "software":     (200_000, 350_000,  600_000),
    "technology":   (180_000, 320_000,  580_000),
    "finance":      (250_000, 450_000, 800_000),
    "healthcare":   (120_000, 200_000, 380_000),
    "retail":       (80_000,  150_000, 280_000),
    "manufacturing":(90_000,  160_000, 280_000),
    "consulting":   (120_000, 200_000, 380_000),
    "media":        (100_000, 180_000, 320_000),
    "logistics":    (90_000,  155_000, 270_000),
    "hospitality":  (50_000,  90_000,  165_000),
}


@dataclass
class CompanyEconomicsResult:
    """Outcome from CompanyEconomics analysis."""
    is_suspicious: bool = False
    signals: list[ScamSignal] = field(default_factory=list)


class CompanyEconomics:
    """Validates company size, age, and funding against claimed characteristics."""

    # Minimum age (days) for a company to plausibly have many employees
    _AGE_THRESHOLDS: list[tuple[int, int]] = [
        # (max_age_days, max_plausible_employees)
        (30,  5),
        (90,  20),
        (180, 60),
        (365, 200),
    ]

    def validate(
        self,
        job: JobPosting,
        company_founded_year: int | None = None,
        claimed_employees: int | None = None,
        funding_stage: str | None = None,
        annual_revenue: float | None = None,
    ) -> CompanyEconomicsResult:
        result = CompanyEconomicsResult()

        # Determine claimed employee count from posting
        emp_count = claimed_employees or self._parse_company_size(job.company_size)

        # Age-based check
        if company_founded_year and emp_count:
            age_sig = self._check_age_vs_size(company_founded_year, emp_count, job.company)
            if age_sig:
                result.signals.append(age_sig)
                result.is_suspicious = True

        # Funding stage vs employee count
        if funding_stage and emp_count:
            stage_sig = self._check_funding_vs_size(funding_stage, emp_count, job.company)
            if stage_sig:
                result.signals.append(stage_sig)
                result.is_suspicious = True

        # Revenue per employee sanity check
        if annual_revenue and emp_count and emp_count > 0:
            rev_sig = self._check_revenue_per_employee(
                annual_revenue, emp_count, job.industry or "technology"
            )
            if rev_sig:
                result.signals.append(rev_sig)
                result.is_suspicious = True

        # Hiring pattern mismatch
        pattern_sig = self._check_hiring_pattern(job)
        if pattern_sig:
            result.signals.append(pattern_sig)
            result.is_suspicious = True

        return result

    # ------------------------------------------------------------------

    def _parse_company_size(self, size_str: str) -> int | None:
        """Parse LinkedIn company_size strings like '1-10', '51-200', '10001+'."""
        if not size_str:
            return None
        m = re.search(r"(\d[\d,]*)\+?", size_str.replace(",", ""))
        if m:
            return int(m.group(1))
        return None

    def _check_age_vs_size(
        self, founded_year: int, employee_count: int, company_name: str
    ) -> ScamSignal | None:
        current_year = datetime.now(UTC).year
        age_years = max(0, current_year - founded_year)
        age_days = age_years * 365

        for max_age_days, max_employees in self._AGE_THRESHOLDS:
            if age_days <= max_age_days and employee_count > max_employees:
                return ScamSignal(
                    name="company_age_size_mismatch",
                    category=SignalCategory.RED_FLAG,
                    weight=0.80,
                    confidence=0.75,
                    detail=(
                        f"Company founded {founded_year} ({age_years}y old) "
                        f"claims {employee_count:,} employees — "
                        f"implausible for a <{max_age_days//30}mo-old company"
                    ),
                    evidence=f"company={company_name!r}, founded={founded_year}, employees={employee_count}",
                )
        return None

    def _check_funding_vs_size(
        self, funding_stage: str, employee_count: int, company_name: str
    ) -> ScamSignal | None:
        stage_key = funding_stage.lower().replace(" ", "_").replace("-", "_")
        limits = _FUNDING_STAGE_EMPLOYEES.get(stage_key)
        if not limits:
            return None
        min_emp, max_emp = limits
        if employee_count > max_emp * 2:
            return ScamSignal(
                name="funding_stage_size_mismatch",
                category=SignalCategory.WARNING,
                weight=0.65,
                confidence=0.70,
                detail=(
                    f"{funding_stage} company claims {employee_count:,} employees — "
                    f"typical range is {min_emp}–{max_emp}"
                ),
                evidence=f"company={company_name!r}, stage={funding_stage}, employees={employee_count}",
            )
        return None

    def _check_revenue_per_employee(
        self, revenue: float, employees: int, industry: str
    ) -> ScamSignal | None:
        rev_per_emp = revenue / employees
        ind_lower = industry.lower()
        benchmark = REVENUE_PER_EMPLOYEE.get(ind_lower, REVENUE_PER_EMPLOYEE.get("technology"))
        if not benchmark:
            return None
        p25, p50, p75 = benchmark
        if rev_per_emp < p25 * 0.2:
            return ScamSignal(
                name="revenue_per_employee_anomaly",
                category=SignalCategory.WARNING,
                weight=0.60,
                confidence=0.65,
                detail=(
                    f"Revenue per employee ${rev_per_emp:,.0f} is far below "
                    f"the {industry} P25 (${p25:,}) — figures may be fabricated"
                ),
                evidence=f"rev_per_emp=${rev_per_emp:,.0f}, industry_p25=${p25:,}",
            )
        if rev_per_emp > p75 * 10:
            return ScamSignal(
                name="revenue_per_employee_anomaly",
                category=SignalCategory.WARNING,
                weight=0.58,
                confidence=0.60,
                detail=(
                    f"Revenue per employee ${rev_per_emp:,.0f} is implausibly high "
                    f"vs {industry} P75 (${p75:,})"
                ),
                evidence=f"rev_per_emp=${rev_per_emp:,.0f}, industry_p75=${p75:,}",
            )
        return None

    def _check_hiring_pattern(self, job: JobPosting) -> ScamSignal | None:
        """Flag always-hiring language for seasonal/project industries."""
        desc_lower = job.description.lower()
        always_hiring = any(
            phrase in desc_lower
            for phrase in ("always hiring", "always looking", "open pipeline", "continuous hiring")
        )
        if not always_hiring:
            return None
        industry_lower = (job.industry or "").lower()
        pattern = INDUSTRY_HIRING_PATTERNS.get(industry_lower, "continuous")
        if pattern in ("seasonal", "project_based", "grant_based"):
            return ScamSignal(
                name="hiring_pattern_mismatch",
                category=SignalCategory.WARNING,
                weight=0.58,
                confidence=0.62,
                detail=(
                    f"'Always hiring' language is unusual for {job.industry} "
                    f"which follows a {pattern} hiring pattern"
                ),
                evidence=f"industry={job.industry!r}, pattern={pattern}",
            )
        return None


# ===========================================================================
# 3. BenefitsAnalyzer
# ===========================================================================

# Benefits mapped to (keyword_patterns, tier)
# tier: "basic", "standard", "premium", "luxury"
_BENEFIT_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # (pattern, benefit_name, tier)
    (re.compile(r"\b(health insurance|medical (insurance|coverage|plan))\b", re.I), "health_insurance", "basic"),
    (re.compile(r"\b(dental( (insurance|coverage|plan))?)\b", re.I), "dental", "standard"),
    (re.compile(r"\b(vision( (insurance|coverage|plan))?)\b", re.I), "vision", "standard"),
    (re.compile(r"\b(401k|401\(k\)|retirement (plan|account)|pension)\b", re.I), "retirement", "standard"),
    (re.compile(r"\b(pto|paid time off|vacation days?|paid vacation)\b", re.I), "pto", "standard"),
    (re.compile(r"\b(unlimited pto|unlimited paid time off|unlimited vacation)\b", re.I), "unlimited_pto", "premium"),
    (re.compile(r"\b(parental leave|maternity leave|paternity leave|family leave)\b", re.I), "parental_leave", "premium"),
    (re.compile(r"\b(stock options?|equity|espp|rsu|restricted stock)\b", re.I), "equity", "premium"),
    (re.compile(r"\b(annual bonus|performance bonus|signing bonus|yearly bonus)\b", re.I), "bonus", "premium"),
    (re.compile(r"\b(company (car|vehicle)|car allowance|auto allowance)\b", re.I), "company_car", "luxury"),
    (re.compile(r"\b(profit.?sharing)\b", re.I), "profit_sharing", "premium"),
    (re.compile(r"\b(gym membership|fitness (reimbursement|stipend|allowance))\b", re.I), "gym", "premium"),
    (re.compile(r"\b(remote work|work from home|wfh|flexible (schedule|hours|working))\b", re.I), "remote_flexible", "standard"),
    (re.compile(r"\b(tuition reimbursement|education (reimbursement|assistance)|learning stipend)\b", re.I), "tuition", "premium"),
    (re.compile(r"\b(life insurance)\b", re.I), "life_insurance", "standard"),
    (re.compile(r"\b(disability insurance|short.term disability|long.term disability)\b", re.I), "disability", "standard"),
    (re.compile(r"\b(employee (discount|perks)|staff discount)\b", re.I), "employee_discount", "basic"),
    (re.compile(r"\b(free (lunch|meals?|snacks?|food)|catered (lunch|meals?))\b", re.I), "free_food", "premium"),
    (re.compile(r"\b(relocation (assistance|package|bonus))\b", re.I), "relocation", "standard"),
    (re.compile(r"\b(mental health (benefit|support|coverage)|therapy (benefit|stipend))\b", re.I), "mental_health", "premium"),
    (re.compile(r"\b(child ?care|daycare|childcare (benefit|stipend))\b", re.I), "childcare", "premium"),
    (re.compile(r"\b(home office (stipend|allowance|budget)|equipment (stipend|allowance))\b", re.I), "home_office", "standard"),
    (re.compile(r"\b(travel (allowance|reimbursement|stipend))\b", re.I), "travel_allowance", "standard"),
    (re.compile(r"\b(pet insurance)\b", re.I), "pet_insurance", "luxury"),
    (re.compile(r"\b(sabbatical)\b", re.I), "sabbatical", "luxury"),
]

# Industry norms for benefit tiers at each experience level
# (min_expected_standard_count, suspicious_luxury_count_at_entry)
_INDUSTRY_BENEFIT_NORMS: dict[str, dict[str, int]] = {
    "technology": {"entry_premium_ok": 2, "mid_premium_ok": 4, "senior_premium_ok": 6},
    "healthcare": {"entry_premium_ok": 2, "mid_premium_ok": 3, "senior_premium_ok": 5},
    "finance":    {"entry_premium_ok": 2, "mid_premium_ok": 4, "senior_premium_ok": 6},
    "retail":     {"entry_premium_ok": 0, "mid_premium_ok": 1, "senior_premium_ok": 3},
    "education":  {"entry_premium_ok": 1, "mid_premium_ok": 2, "senior_premium_ok": 3},
    "default":    {"entry_premium_ok": 1, "mid_premium_ok": 3, "senior_premium_ok": 5},
}

# Threshold for benefit keyword stuffing
_STUFFING_THRESHOLD = 15


@dataclass
class BenefitsResult:
    """Outcome from BenefitsAnalyzer."""
    found_benefits: list[str] = field(default_factory=list)
    tiers: dict[str, list[str]] = field(default_factory=dict)
    signals: list[ScamSignal] = field(default_factory=list)
    is_suspicious: bool = False


class BenefitsAnalyzer:
    """Parses and validates benefit packages in job descriptions."""

    def analyze(self, job: JobPosting) -> BenefitsResult:
        result = BenefitsResult()
        text = f"{job.description} {job.title}"
        found: list[tuple[str, str]] = []  # (benefit_name, tier)
        seen_benefits: set[str] = set()

        for pattern, benefit_name, tier in _BENEFIT_PATTERNS:
            if benefit_name not in seen_benefits and pattern.search(text):
                found.append((benefit_name, tier))
                seen_benefits.add(benefit_name)

        result.found_benefits = [b for b, _ in found]

        # Group by tier
        by_tier: dict[str, list[str]] = {}
        for b, tier in found:
            by_tier.setdefault(tier, []).append(b)
        result.tiers = by_tier

        # Keyword stuffing check
        if len(found) >= _STUFFING_THRESHOLD:
            result.signals.append(ScamSignal(
                name="benefit_keyword_stuffing",
                category=SignalCategory.WARNING,
                weight=0.60,
                confidence=0.65,
                detail=(
                    f"Job lists {len(found)} benefits — padding with {len(found)} "
                    f"benefit claims is a common scam distraction technique "
                    f"(threshold: {_STUFFING_THRESHOLD})"
                ),
                evidence=f"benefit_count={len(found)}",
            ))
            result.is_suspicious = True

        # Unrealistic entry-level package
        level = normalize_experience_level(f"{job.experience_level} {job.title}")
        if level == "entry":
            luxury_benefits = by_tier.get("luxury", [])
            premium_benefits = by_tier.get("premium", [])
            industry_key = (job.industry or "").lower()
            norms = _INDUSTRY_BENEFIT_NORMS.get(industry_key, _INDUSTRY_BENEFIT_NORMS["default"])
            allowed = norms["entry_premium_ok"]

            if luxury_benefits and "company_car" in luxury_benefits:
                result.signals.append(ScamSignal(
                    name="unrealistic_benefits",
                    category=SignalCategory.WARNING,
                    weight=0.65,
                    confidence=0.68,
                    detail="Entry-level role offers company car — extremely rare for legitimate postings",
                    evidence=f"luxury_benefits={luxury_benefits}",
                ))
                result.is_suspicious = True

            if len(premium_benefits) > allowed + 3:
                result.signals.append(ScamSignal(
                    name="unrealistic_benefits",
                    category=SignalCategory.WARNING,
                    weight=0.58,
                    confidence=0.62,
                    detail=(
                        f"Entry-level posting offers {len(premium_benefits)} premium benefits "
                        f"(equity, unlimited PTO, bonuses, etc.) — exceeds industry norms"
                    ),
                    evidence=f"premium_benefits={premium_benefits}",
                ))
                result.is_suspicious = True

        return result

    def parse_benefits(self, text: str) -> list[str]:
        """Return a list of benefit names found in *text*."""
        found = []
        seen: set[str] = set()
        for pattern, name, _ in _BENEFIT_PATTERNS:
            if name not in seen and pattern.search(text):
                found.append(name)
                seen.add(name)
        return found


# ===========================================================================
# 4. GeographicValidator
# ===========================================================================

# Area code → US state mapping (sample, covering major area codes)
_AREA_CODE_STATE: dict[str, str] = {
    "201": "NJ", "202": "DC", "203": "CT", "205": "AL", "206": "WA",
    "207": "ME", "208": "ID", "209": "CA", "210": "TX", "212": "NY",
    "213": "CA", "214": "TX", "215": "PA", "216": "OH", "217": "IL",
    "218": "MN", "219": "IN", "224": "IL", "225": "LA", "228": "MS",
    "229": "GA", "231": "MI", "234": "OH", "239": "FL", "240": "MD",
    "248": "MI", "251": "AL", "252": "NC", "253": "WA", "254": "TX",
    "256": "AL", "260": "IN", "262": "WI", "267": "PA", "269": "MI",
    "270": "KY", "272": "PA", "276": "VA", "281": "TX", "301": "MD",
    "302": "DE", "303": "CO", "304": "WV", "305": "FL", "307": "WY",
    "308": "NE", "309": "IL", "310": "CA", "312": "IL", "313": "MI",
    "314": "MO", "315": "NY", "316": "KS", "317": "IN", "318": "LA",
    "319": "IA", "320": "MN", "321": "FL", "323": "CA", "325": "TX",
    "330": "OH", "331": "IL", "334": "AL", "336": "NC", "337": "LA",
    "339": "MA", "346": "TX", "347": "NY", "351": "MA", "352": "FL",
    "360": "WA", "361": "TX", "364": "KY", "369": "CA", "380": "OH",
    "385": "UT", "386": "FL", "401": "RI", "402": "NE", "404": "GA",
    "405": "OK", "406": "MT", "407": "FL", "408": "CA", "409": "TX",
    "410": "MD", "412": "PA", "413": "MA", "414": "WI", "415": "CA",
    "417": "MO", "419": "OH", "423": "TN", "424": "CA", "425": "WA",
    "430": "TX", "432": "TX", "434": "VA", "435": "UT", "440": "OH",
    "442": "CA", "443": "MD", "447": "IL", "458": "OR", "463": "IN",
    "469": "TX", "470": "GA", "475": "CT", "478": "GA", "479": "AR",
    "480": "AZ", "484": "PA", "501": "AR", "502": "KY", "503": "OR",
    "504": "LA", "505": "NM", "507": "MN", "508": "MA", "509": "WA",
    "510": "CA", "512": "TX", "513": "OH", "515": "IA", "516": "NY",
    "517": "MI", "518": "NY", "520": "AZ", "530": "CA", "531": "NE",
    "534": "WI", "539": "OK", "540": "VA", "541": "OR", "551": "NJ",
    "559": "CA", "561": "FL", "562": "CA", "563": "IA", "567": "OH",
    "570": "PA", "571": "VA", "573": "MO", "574": "IN", "575": "NM",
    "580": "OK", "585": "NY", "586": "MI", "601": "MS", "602": "AZ",
    "603": "NH", "605": "SD", "606": "KY", "607": "NY", "608": "WI",
    "609": "NJ", "610": "PA", "612": "MN", "614": "OH", "615": "TN",
    "616": "MI", "617": "MA", "618": "IL", "619": "CA", "620": "KS",
    "623": "AZ", "626": "CA", "628": "CA", "629": "TN", "630": "IL",
    "631": "NY", "636": "MO", "641": "IA", "646": "NY", "650": "CA",
    "651": "MN", "657": "CA", "660": "MO", "661": "CA", "662": "MS",
    "667": "MD", "669": "CA", "671": "GU", "678": "GA", "681": "WV",
    "682": "TX", "701": "ND", "702": "NV", "703": "VA", "704": "NC",
    "706": "GA", "707": "CA", "708": "IL", "712": "IA", "713": "TX",
    "714": "CA", "715": "WI", "716": "NY", "717": "PA", "718": "NY",
    "719": "CO", "720": "CO", "724": "PA", "725": "NV", "727": "FL",
    "731": "TN", "732": "NJ", "734": "MI", "737": "TX", "740": "OH",
    "743": "NC", "747": "CA", "754": "FL", "757": "VA", "760": "CA",
    "762": "GA", "763": "MN", "765": "IN", "769": "MS", "770": "GA",
    "772": "FL", "773": "IL", "774": "MA", "775": "NV", "779": "IL",
    "781": "MA", "785": "KS", "786": "FL", "801": "UT", "802": "VT",
    "803": "SC", "804": "VA", "805": "CA", "806": "TX", "808": "HI",
    "810": "MI", "812": "IN", "813": "FL", "814": "PA", "815": "IL",
    "816": "MO", "817": "TX", "818": "CA", "820": "CA", "828": "NC",
    "830": "TX", "831": "CA", "832": "TX", "835": "PA", "843": "SC",
    "845": "NY", "847": "IL", "848": "NJ", "850": "FL", "854": "SC",
    "857": "MA", "858": "CA", "859": "KY", "860": "CT", "862": "NJ",
    "863": "FL", "864": "SC", "865": "TN", "870": "AR", "872": "IL",
    "878": "PA", "901": "TN", "903": "TX", "904": "FL", "906": "MI",
    "907": "AK", "908": "NJ", "909": "CA", "910": "NC", "912": "GA",
    "913": "KS", "914": "NY", "915": "TX", "916": "CA", "917": "NY",
    "918": "OK", "919": "NC", "920": "WI", "925": "CA", "928": "AZ",
    "929": "NY", "930": "IN", "931": "TN", "936": "TX", "937": "OH",
    "940": "TX", "941": "FL", "945": "TX", "947": "MI", "949": "CA",
    "951": "CA", "952": "MN", "954": "FL", "956": "TX", "959": "CT",
    "970": "CO", "971": "OR", "972": "TX", "973": "NJ", "978": "MA",
    "979": "TX", "980": "NC", "984": "NC", "985": "LA", "989": "MI",
}

# MLM indicator: "must be near [small/specific town]" patterns
_MLM_GEO_PATTERNS = re.compile(
    r"\b(must (live|reside|be located?) (near|in|within \d+ miles? of)|"
    r"local (candidates?|applicants?) only|"
    r"must be within \d+ miles?|"
    r"prefer (local|nearby) candidates?)\b",
    re.IGNORECASE,
)

# Country-specific entity suffixes to catch geographic impossibilities
_UK_ENTITY_SUFFIXES = re.compile(r"\b(Ltd\.?|Limited|PLC|LLP)\b", re.IGNORECASE)
_US_ENTITY_SUFFIXES = re.compile(r"\b(Inc\.?|LLC|Corp\.?|L\.L\.C\.?)\b", re.IGNORECASE)

# Suspicious posting hour range (3am–5am local, per UTC offset)
_SUSPICIOUS_HOUR_RANGE = (3, 5)  # 3am to 5am


@dataclass
class GeographicResult:
    """Outcome from GeographicValidator."""
    signals: list[ScamSignal] = field(default_factory=list)
    is_suspicious: bool = False


class GeographicValidator:
    """Validates geographic consistency of job postings."""

    def validate(self, job: JobPosting, posted_hour_utc: int | None = None) -> GeographicResult:
        result = GeographicResult()

        # MLM "must be near small town" check
        mlm_sig = self._check_mlm_geographic(job)
        if mlm_sig:
            result.signals.append(mlm_sig)
            result.is_suspicious = True

        # Geographic impossibility: UK entity posting US-only jobs
        geo_imp_sig = self._check_geographic_impossibility(job)
        if geo_imp_sig:
            result.signals.append(geo_imp_sig)
            result.is_suspicious = True

        # Phone area code vs stated location
        phone_sig = self._check_phone_area_code(job)
        if phone_sig:
            result.signals.append(phone_sig)
            result.is_suspicious = True

        # Suspicious posting time
        if posted_hour_utc is not None:
            time_sig = self._check_posting_time(job, posted_hour_utc)
            if time_sig:
                result.signals.append(time_sig)
                result.is_suspicious = True

        return result

    # ------------------------------------------------------------------

    def _check_mlm_geographic(self, job: JobPosting) -> ScamSignal | None:
        text = f"{job.description} {job.location}"
        m = _MLM_GEO_PATTERNS.search(text)
        if not m:
            return None
        if not job.is_remote:
            return None  # "local only" is normal for on-site jobs
        return ScamSignal(
            name="mlm_geographic_restriction",
            category=SignalCategory.WARNING,
            weight=0.68,
            confidence=0.65,
            detail=(
                "Remote job requires proximity to a specific location — "
                "common MLM/pyramid scheme recruiting tactic"
            ),
            evidence=m.group(0),
        )

    def _check_geographic_impossibility(self, job: JobPosting) -> ScamSignal | None:
        desc = job.description
        company = job.company
        location = job.location.lower()

        # UK-registered entity but US-only posting
        has_uk_suffix = _UK_ENTITY_SUFFIXES.search(company)
        is_us_only = (
            re.search(r"\b(USA|United States|US only|US citizen)\b", desc, re.I)
            and not re.search(r"\b(UK|United Kingdom|London|England|British)\b", location, re.I)
        )
        if has_uk_suffix and is_us_only:
            return ScamSignal(
                name="geographic_impossibility",
                category=SignalCategory.WARNING,
                weight=0.62,
                confidence=0.60,
                detail=(
                    f"Company '{company}' appears UK-registered (Ltd/PLC/LLP) "
                    "but job is listed as US-only"
                ),
                evidence=f"company={company!r}, location={job.location!r}",
            )

        # London office mentioned but company is US-only LLC
        has_us_suffix = _US_ENTITY_SUFFIXES.search(company)
        mentions_london = re.search(r"\b(London office|London HQ|based in London)\b", desc, re.I)
        is_us_registered = bool(re.search(r"\b(US|USA|United States)\b", location, re.I))
        if has_us_suffix and mentions_london and is_us_registered:
            return ScamSignal(
                name="geographic_impossibility",
                category=SignalCategory.WARNING,
                weight=0.60,
                confidence=0.60,
                detail=(
                    f"US LLC claims London office — inconsistent registration type"
                ),
                evidence=f"company={company!r}",
            )

        return None

    def _check_phone_area_code(self, job: JobPosting) -> ScamSignal | None:
        """Extract phone numbers from description and check area codes vs location."""
        text = job.description
        m = re.search(r"\b(\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
        if not m:
            return None
        area_code = m.group(1)
        phone_state = _AREA_CODE_STATE.get(area_code)
        if not phone_state:
            return None  # Unknown area code — can't validate

        location = job.location
        # Extract state from location string
        state_m = re.search(r",?\s*([A-Z]{2})\b", location)
        if not state_m:
            return None
        stated_state = state_m.group(1)

        if phone_state != stated_state and not job.is_remote:
            return ScamSignal(
                name="phone_area_code_mismatch",
                category=SignalCategory.WARNING,
                weight=0.62,
                confidence=0.60,
                detail=(
                    f"Phone area code {area_code} belongs to {phone_state} "
                    f"but job is listed in {stated_state}"
                ),
                evidence=f"area_code={area_code}, phone_state={phone_state}, stated_state={stated_state}",
            )
        return None

    def _check_posting_time(self, job: JobPosting, posted_hour_utc: int) -> ScamSignal | None:
        """Flag postings created between 3am–5am local if not a global company."""
        start, end = _SUSPICIOUS_HOUR_RANGE
        if not (start <= posted_hour_utc <= end):
            return None
        # Global companies legitimately post at odd hours (international offices)
        company_size = job.company_size or ""
        is_large = any(x in company_size for x in ("10001", "5001", "1001"))
        if is_large:
            return None
        return ScamSignal(
            name="suspicious_posting_time",
            category=SignalCategory.WARNING,
            weight=0.50,
            confidence=0.45,
            detail=(
                f"Job posted at {posted_hour_utc:02d}:00 UTC — postings between "
                f"{start}am–{end}am from small companies are statistically suspicious"
            ),
            evidence=f"posted_hour_utc={posted_hour_utc}, company_size={company_size!r}",
        )


# ===========================================================================
# 5. IndustryBenchmark
# ===========================================================================

# Static benchmark store with timestamps
# Structure: industry -> role -> level -> {p25, p50, p75, p90, sample_size, updated}
INDUSTRY_BENCHMARKS: dict[str, dict[str, dict[str, dict]]] = {
    "technology": {
        "software_engineer": {
            "entry":  {"p25": 80_000, "p50": 98_000,  "p75": 120_000, "p90": 145_000, "sample_size": 45_000, "updated": "2025-08-01"},
            "mid":    {"p25": 118_000, "p50": 145_000, "p75": 178_000, "p90": 215_000, "sample_size": 62_000, "updated": "2025-08-01"},
            "senior": {"p25": 170_000, "p50": 205_000, "p75": 248_000, "p90": 300_000, "sample_size": 38_000, "updated": "2025-08-01"},
        },
        "data_scientist": {
            "entry":  {"p25": 85_000, "p50": 105_000, "p75": 128_000, "p90": 155_000, "sample_size": 18_000, "updated": "2025-08-01"},
            "mid":    {"p25": 120_000, "p50": 148_000, "p75": 182_000, "p90": 220_000, "sample_size": 22_000, "updated": "2025-08-01"},
            "senior": {"p25": 165_000, "p50": 200_000, "p75": 242_000, "p90": 292_000, "sample_size": 14_000, "updated": "2025-08-01"},
        },
        "product_manager": {
            "entry":  {"p25": 90_000, "p50": 110_000, "p75": 135_000, "p90": 162_000, "sample_size": 12_000, "updated": "2025-08-01"},
            "mid":    {"p25": 130_000, "p50": 160_000, "p75": 195_000, "p90": 238_000, "sample_size": 15_000, "updated": "2025-08-01"},
            "senior": {"p25": 175_000, "p50": 215_000, "p75": 260_000, "p90": 318_000, "sample_size": 9_500,  "updated": "2025-08-01"},
        },
    },
    "finance": {
        "accounting": {
            "entry":  {"p25": 50_000, "p50": 62_000,  "p75": 78_000,  "p90": 95_000,  "sample_size": 28_000, "updated": "2025-08-01"},
            "mid":    {"p25": 72_000, "p50": 90_000,  "p75": 112_000, "p90": 138_000, "sample_size": 32_000, "updated": "2025-08-01"},
            "senior": {"p25": 100_000, "p50": 128_000, "p75": 162_000, "p90": 200_000, "sample_size": 18_000, "updated": "2025-08-01"},
        },
    },
    "healthcare": {
        "nursing": {
            "entry":  {"p25": 58_000, "p50": 70_000,  "p75": 86_000,  "p90": 105_000, "sample_size": 55_000, "updated": "2025-08-01"},
            "mid":    {"p25": 72_000, "p50": 88_000,  "p75": 108_000, "p90": 132_000, "sample_size": 68_000, "updated": "2025-08-01"},
            "senior": {"p25": 92_000, "p50": 112_000, "p75": 138_000, "p90": 170_000, "sample_size": 42_000, "updated": "2025-08-01"},
        },
    },
    "retail": {
        "customer_service": {
            "entry":  {"p25": 28_000, "p50": 34_000,  "p75": 42_000,  "p90": 52_000,  "sample_size": 95_000, "updated": "2025-08-01"},
            "mid":    {"p25": 36_000, "p50": 44_000,  "p75": 55_000,  "p90": 68_000,  "sample_size": 72_000, "updated": "2025-08-01"},
            "senior": {"p25": 46_000, "p50": 58_000,  "p75": 72_000,  "p90": 88_000,  "sample_size": 38_000, "updated": "2025-08-01"},
        },
        "warehouse": {
            "entry":  {"p25": 26_000, "p50": 32_000,  "p75": 40_000,  "p90": 50_000,  "sample_size": 120_000, "updated": "2025-08-01"},
            "mid":    {"p25": 34_000, "p50": 42_000,  "p75": 52_000,  "p90": 64_000,  "sample_size": 85_000,  "updated": "2025-08-01"},
        },
    },
}

# Drift thresholds: if observed median deviates from stored p50 by this fraction,
# flag for benchmark update
_DRIFT_THRESHOLD_FRACTION = 0.15  # 15%


@dataclass
class BenchmarkResult:
    """Outcome from IndustryBenchmark.rank()."""
    percentile: int | None = None          # 0–100
    label: str = "unknown"                 # e.g. "95th percentile"
    industry: str = ""
    role: str = ""
    level: str = ""
    benchmark: dict | None = None
    signals: list[ScamSignal] = field(default_factory=list)
    is_suspicious: bool = False


class IndustryBenchmark:
    """Percentile ranking and drift detection for salary benchmarks."""

    def rank(self, job: JobPosting) -> BenchmarkResult:
        """Rank the job's salary against industry benchmarks."""
        result = BenchmarkResult()

        if not job.salary_max and not job.salary_min:
            return result

        salary = job.salary_max or job.salary_min
        role = classify_role(job.title)
        level = normalize_experience_level(f"{job.experience_level} {job.title}")
        industry = (job.industry or "technology").lower()

        result.role = role or ""
        result.level = level
        result.industry = industry

        # Look up benchmark
        industry_data = INDUSTRY_BENCHMARKS.get(industry, {})
        role_data = industry_data.get(role or "", {})
        bench = role_data.get(level)

        if not bench:
            # Cross-reference with national data
            if role and level in SALARY_RANGES.get(role, {}):
                p10, p25, p50, p75, p90, p99 = SALARY_RANGES[role][level]
                bench = {"p25": p25, "p50": p50, "p75": p75, "p90": p90}
            else:
                return result

        result.benchmark = bench
        p25 = bench.get("p25", 0)
        p50 = bench.get("p50", 0)
        p75 = bench.get("p75", 0)
        p90 = bench.get("p90", 0)

        # Percentile estimate
        pct = self._estimate_percentile(salary, p25, p50, p75, p90)
        result.percentile = pct
        result.label = self._percentile_label(pct)

        # Flag extreme percentile
        if pct >= 95:
            result.signals.append(ScamSignal(
                name="salary_extreme_percentile",
                category=SignalCategory.WARNING,
                weight=0.65,
                confidence=0.70,
                detail=(
                    f"Salary ${salary:,.0f} is in the {pct}th percentile "
                    f"for {level} {role or 'unknown'} in {industry} — "
                    f"only 5% of legitimate postings reach this level"
                ),
                evidence=f"percentile={pct}, benchmark_p90=${p90:,}",
            ))
            result.is_suspicious = True

        return result

    def check_drift(
        self,
        industry: str,
        role: str,
        level: str,
        observed_median: float,
    ) -> ScamSignal | None:
        """Return a drift signal if observed_median deviates from stored benchmark."""
        bench_p50 = (
            INDUSTRY_BENCHMARKS
            .get(industry, {})
            .get(role, {})
            .get(level, {})
            .get("p50")
        )
        if not bench_p50:
            # Fall back to SALARY_RANGES
            if role and level in SALARY_RANGES.get(role, {}):
                bench_p50 = SALARY_RANGES[role][level][2]  # p50 is index 2
        if not bench_p50:
            return None

        drift = abs(observed_median - bench_p50) / bench_p50
        if drift > _DRIFT_THRESHOLD_FRACTION:
            direction = "above" if observed_median > bench_p50 else "below"
            return ScamSignal(
                name="benchmark_drift",
                category=SignalCategory.WARNING,
                weight=0.50,
                confidence=0.55,
                detail=(
                    f"Observed {industry}/{role}/{level} median ${observed_median:,.0f} "
                    f"is {drift:.0%} {direction} stored benchmark ${bench_p50:,} — "
                    f"benchmark may need updating"
                ),
                evidence=f"drift={drift:.2%}, stored_p50=${bench_p50:,}, observed=${observed_median:,.0f}",
            )
        return None

    def get_benchmark(self, industry: str, role: str, level: str) -> dict | None:
        """Return stored benchmark dict or None."""
        return (
            INDUSTRY_BENCHMARKS
            .get(industry, {})
            .get(role, {})
            .get(level)
        )

    # ------------------------------------------------------------------

    def _estimate_percentile(self, salary: float, p25: int, p50: int, p75: int, p90: int) -> int:
        if salary <= p25:
            return max(1, int(25 * salary / max(1, p25)))
        if salary <= p50:
            return int(25 + 25 * (salary - p25) / max(1, p50 - p25))
        if salary <= p75:
            return int(50 + 25 * (salary - p50) / max(1, p75 - p50))
        if salary <= p90:
            return int(75 + 15 * (salary - p75) / max(1, p90 - p75))
        return min(99, int(90 + 9 * (salary - p90) / max(1, p90)))

    def _percentile_label(self, pct: int) -> str:
        if pct <= 10:
            return f"{pct}th percentile (below market)"
        if pct <= 25:
            return f"{pct}th percentile (below average)"
        if pct <= 50:
            return f"{pct}th percentile (average)"
        if pct <= 75:
            return f"{pct}th percentile (above average)"
        if pct <= 90:
            return f"{pct}th percentile (well above average)"
        return f"{pct}th percentile (exceptional — verify)"


# ===========================================================================
# Convenience: run all validators at once
# ===========================================================================

@dataclass
class EconomicValidationResult:
    """Aggregated result from all five economic validators."""
    salary: SalaryValidationResult = field(default_factory=SalaryValidationResult)
    company: CompanyEconomicsResult = field(default_factory=CompanyEconomicsResult)
    benefits: BenefitsResult = field(default_factory=BenefitsResult)
    geography: GeographicResult = field(default_factory=GeographicResult)
    benchmark: BenchmarkResult = field(default_factory=BenchmarkResult)

    @property
    def all_signals(self) -> list[ScamSignal]:
        signals: list[ScamSignal] = []
        signals.extend(self.salary.signals)
        signals.extend(self.company.signals)
        signals.extend(self.benefits.signals)
        signals.extend(self.geography.signals)
        signals.extend(self.benchmark.signals)
        return signals

    @property
    def is_suspicious(self) -> bool:
        return any([
            self.salary.is_suspicious,
            self.company.is_suspicious,
            self.benefits.is_suspicious,
            self.geography.is_suspicious,
            self.benchmark.is_suspicious,
        ])


def validate_economics(
    job: JobPosting,
    company_founded_year: int | None = None,
    claimed_employees: int | None = None,
    funding_stage: str | None = None,
    annual_revenue: float | None = None,
    posted_hour_utc: int | None = None,
) -> EconomicValidationResult:
    """Run all economic validators and return an aggregated result."""
    result = EconomicValidationResult()
    result.salary = MarketRateValidator().validate(job)
    result.company = CompanyEconomics().validate(
        job,
        company_founded_year=company_founded_year,
        claimed_employees=claimed_employees,
        funding_stage=funding_stage,
        annual_revenue=annual_revenue,
    )
    result.benefits = BenefitsAnalyzer().analyze(job)
    result.geography = GeographicValidator().validate(job, posted_hour_utc=posted_hour_utc)
    result.benchmark = IndustryBenchmark().rank(job)
    return result
