"""Corporate Fraud Handbook integration — Wells/ACFE fraud detection adapted for job scams.

Adapts concepts from Joseph T. Wells' *Corporate Fraud Handbook: Prevention and
Detection* (ACFE) to employment fraud detection.  Key frameworks mapped:

- **Fraud Triangle** (Cressey): Pressure, Opportunity, Rationalization scoring
- **ACFE Fraud Tree**: Occupational fraud taxonomy mapped to job scam categories
- **Benford's Law**: First-digit distribution analysis on salary figures
- **Behavioral red flags**: Linguistic indicators of deception in postings
- **Proactive detection**: Ratio analysis, anomaly scoring, stylometry

Each class exposes a ``score(job)`` or ``analyze(values)`` method returning a
dict compatible with the rest of the Sentinel scoring pipeline.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

from sentinel.models import JobPosting, ScamSignal, SignalCategory
from sentinel.research import ResearchTopic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benford's Law expected first-digit probabilities
# ---------------------------------------------------------------------------

BENFORD_EXPECTED: dict[int, float] = {
    d: math.log10(1 + 1 / d) for d in range(1, 10)
}


# ---------------------------------------------------------------------------
# Fraud Triangle Scorer
# ---------------------------------------------------------------------------

# Pressure indicators — language suggesting desperation / too-good-to-be-true
_PRESSURE_PATTERNS = re.compile(
    r"\b(earn fast|quick money|fast cash|immediate (income|pay|earning)|"
    r"financial freedom|debt free|pay off (your )?(debt|bills|loans)|"
    r"replace your (income|salary|job)|"
    r"tired of (your )?(9.to.5|job|boss|commute)|"
    r"escape the rat race|be your own boss|"
    r"struggling financially|need money (now|fast|urgently)|"
    r"desperate for (work|income|a job)|"
    r"(earn|make) money (from|at) home)\b",
    re.IGNORECASE,
)

# Opportunity indicators — unusually low barriers to entry
_OPPORTUNITY_PATTERNS = re.compile(
    r"\b(no (experience|degree|resume|background check|interview) (required|needed|necessary)|"
    r"anyone can (do|apply|qualify|start)|"
    r"start (immediately|today|right now|same day)|"
    r"instant (hire|hiring|approval)|"
    r"no (training|certification) (required|needed)|"
    r"open to (all|everyone|anybody)|"
    r"guaranteed (position|hire|acceptance|placement)|"
    r"automatic(ally)? (hire[sd]?|accept(ed)?))\b",
    re.IGNORECASE,
)

# Rationalization indicators — language that pre-emptively justifies or normalizes
_RATIONALIZATION_PATTERNS = re.compile(
    r"\b(this is (not|100%|totally|completely) (legit|legitimate|legal|real)|"
    r"(we are|this is) a (real|legitimate|legal|registered) (company|business|opportunity)|"
    r"not a (scam|pyramid|ponzi|fraud|hoax)|"
    r"guaranteed (safe|legit|legitimate|secure)|"
    r"100% (safe|secure|guaranteed|legitimate|legal|risk.free)|"
    r"(fully|officially) (licensed|registered|certified|accredited)|"
    r"thousands (have|of people) (already|successfully)|"
    r"(don'?t|do not) (worry|be afraid|hesitate)|"
    r"trust(ed)? (by|company|employer|opportunity)|"
    r"no risk (involved|whatsoever|at all))\b",
    re.IGNORECASE,
)


class FraudTriangleScorer:
    """Score job postings against Cressey's Fraud Triangle model.

    The three legs:
    - **Pressure**: Does the posting exploit financial desperation?
    - **Opportunity**: Does it promise unusually low barriers?
    - **Rationalization**: Does it pre-emptively justify legitimacy?

    A posting that hits all three legs is far more likely to be fraudulent.
    """

    PRESSURE_WEIGHT = 0.30
    OPPORTUNITY_WEIGHT = 0.35
    RATIONALIZATION_WEIGHT = 0.35

    def score(self, job: JobPosting) -> dict:
        """Return a dict with per-leg scores and a composite triangle score."""
        text = f"{job.title} {job.description}"

        pressure_hits = _PRESSURE_PATTERNS.findall(text)
        opportunity_hits = _OPPORTUNITY_PATTERNS.findall(text)
        rationalization_hits = _RATIONALIZATION_PATTERNS.findall(text)

        # Normalize each leg to 0-1 (saturates at 3+ hits)
        pressure_score = min(len(pressure_hits) / 3.0, 1.0)
        opportunity_score = min(len(opportunity_hits) / 3.0, 1.0)
        rationalization_score = min(len(rationalization_hits) / 2.0, 1.0)

        composite = (
            self.PRESSURE_WEIGHT * pressure_score
            + self.OPPORTUNITY_WEIGHT * opportunity_score
            + self.RATIONALIZATION_WEIGHT * rationalization_score
        )

        # Amplify if ALL three legs are present (multiplicative boost)
        legs_present = sum([
            pressure_score > 0,
            opportunity_score > 0,
            rationalization_score > 0,
        ])
        if legs_present == 3:
            composite = min(composite * 1.3, 1.0)

        return {
            "pressure_score": round(pressure_score, 3),
            "opportunity_score": round(opportunity_score, 3),
            "rationalization_score": round(rationalization_score, 3),
            "legs_present": legs_present,
            "composite_score": round(composite, 3),
            "pressure_hits": [h if isinstance(h, str) else h[0] for h in pressure_hits[:5]],
            "opportunity_hits": [h if isinstance(h, str) else h[0] for h in opportunity_hits[:5]],
            "rationalization_hits": [h if isinstance(h, str) else h[0] for h in rationalization_hits[:5]],
        }

    def to_signal(self, job: JobPosting) -> ScamSignal | None:
        """Convert a Fraud Triangle score into a ScamSignal if significant."""
        result = self.score(job)
        if result["composite_score"] < 0.25:
            return None
        weight = min(0.90, 0.50 + result["composite_score"] * 0.45)
        return ScamSignal(
            name="fraud_triangle",
            category=SignalCategory.RED_FLAG if result["legs_present"] >= 3 else SignalCategory.WARNING,
            weight=round(weight, 2),
            confidence=round(min(result["composite_score"] + 0.1, 0.95), 2),
            detail=(
                f"Fraud Triangle: {result['legs_present']}/3 legs present "
                f"(P={result['pressure_score']:.0%}, O={result['opportunity_score']:.0%}, "
                f"R={result['rationalization_score']:.0%})"
            ),
            evidence=f"composite={result['composite_score']:.3f}",
        )


# ---------------------------------------------------------------------------
# Benford's Law Analyzer
# ---------------------------------------------------------------------------

class BenfordAnalyzer:
    """Analyze salary digit distributions against Benford's Law.

    Natural salary figures follow predictable first-digit distributions.
    Fabricated or template-generated salary figures often deviate.
    """

    def __init__(self, significance_threshold: float = 0.25) -> None:
        self.significance_threshold = significance_threshold

    @staticmethod
    def first_digit(n: float) -> int | None:
        """Extract the first non-zero digit from a number."""
        if n <= 0:
            return None
        s = f"{abs(n):.0f}"
        for ch in s:
            if ch != "0" and ch != ".":
                return int(ch)
        return None

    def analyze(self, values: list[float]) -> dict:
        """Analyze a list of salary values against Benford's expected distribution.

        Returns a dict with observed vs expected distributions, chi-squared
        statistic, and a conformity score (0 = perfect Benford, higher = deviation).
        """
        digits = []
        for v in values:
            d = self.first_digit(v)
            if d is not None:
                digits.append(d)

        n = len(digits)
        if n < 5:
            return {
                "sample_size": n,
                "sufficient_data": False,
                "chi_squared": 0.0,
                "conformity_score": 0.0,
                "anomalous": False,
                "observed": {},
                "expected": dict(BENFORD_EXPECTED),
            }

        counter = Counter(digits)
        observed: dict[int, float] = {}
        for d in range(1, 10):
            observed[d] = counter.get(d, 0) / n

        # Chi-squared goodness of fit
        chi_sq = 0.0
        for d in range(1, 10):
            expected_count = BENFORD_EXPECTED[d] * n
            observed_count = counter.get(d, 0)
            if expected_count > 0:
                chi_sq += (observed_count - expected_count) ** 2 / expected_count

        # Mean Absolute Deviation from Benford
        mad = sum(abs(observed.get(d, 0) - BENFORD_EXPECTED[d]) for d in range(1, 10)) / 9

        # Conformity score: 0 = perfect, 1 = maximally deviant
        # Normalize chi-squared by degrees of freedom (8)
        conformity_score = min(chi_sq / (8 * n * 0.05), 1.0) if n > 0 else 0.0

        return {
            "sample_size": n,
            "sufficient_data": True,
            "chi_squared": round(chi_sq, 4),
            "mean_absolute_deviation": round(mad, 6),
            "conformity_score": round(conformity_score, 4),
            "anomalous": conformity_score > self.significance_threshold,
            "observed": {d: round(v, 4) for d, v in observed.items()},
            "expected": {d: round(v, 4) for d, v in BENFORD_EXPECTED.items()},
        }

    def analyze_posting(self, job: JobPosting) -> dict:
        """Analyze a single posting's salary figures against Benford patterns.

        For a single posting, we check whether the salary digits exhibit
        suspicious patterns (e.g., round numbers, psychological pricing).
        """
        values = []
        if job.salary_min > 0:
            values.append(job.salary_min)
        if job.salary_max > 0:
            values.append(job.salary_max)

        flags: list[str] = []

        for v in values:
            s = f"{v:.0f}"
            # Check for suspiciously round numbers (all zeros after first digit)
            if len(s) >= 4 and s[1:] == "0" * (len(s) - 1):
                flags.append(f"${v:,.0f} is perfectly round")
            # Check for psychological pricing ($X9,999 or $X4,999)
            if s.endswith("999") or s.endswith("9999"):
                flags.append(f"${v:,.0f} uses psychological pricing")
            # Check for repeated digits
            if len(set(s)) == 1 and len(s) >= 3:
                flags.append(f"${v:,.0f} has all repeated digits")

        # Ratio analysis: wide range is suspicious
        ratio = 0.0
        if job.salary_min > 0 and job.salary_max > 0:
            ratio = job.salary_max / job.salary_min
            if ratio > 5.0:
                flags.append(f"Max/min ratio is {ratio:.1f}x (>5x is suspicious)")

        return {
            "values": values,
            "flags": flags,
            "suspicious": len(flags) > 0,
            "salary_ratio": round(ratio, 2) if ratio > 0 else None,
        }


# ---------------------------------------------------------------------------
# Linguistic Forensics (Stylometry)
# ---------------------------------------------------------------------------

class LinguisticForensics:
    """Basic stylometry to detect machine-generated or template-based scam postings.

    Analyzes:
    - Sentence length distribution (variance, mean)
    - Vocabulary richness (type-token ratio)
    - Readability scores (Flesch-Kincaid approximation)
    - Punctuation patterns
    - Repetition indicators
    """

    # Sentence boundary pattern
    _SENTENCE_SPLIT = re.compile(r"[.!?]+\s+|[\n]{2,}")
    _WORD_SPLIT = re.compile(r"\b[a-zA-Z]+\b")

    def analyze(self, text: str) -> dict:
        """Compute stylometric features of a text."""
        if not text or len(text.strip()) < 20:
            return {
                "sufficient_text": False,
                "word_count": 0,
                "sentence_count": 0,
                "readability_score": 0.0,
                "vocabulary_richness": 0.0,
                "avg_sentence_length": 0.0,
                "sentence_length_variance": 0.0,
                "repetition_score": 0.0,
                "exclamation_density": 0.0,
                "caps_density": 0.0,
                "anomaly_flags": [],
            }

        words = self._WORD_SPLIT.findall(text)
        word_count = len(words)
        if word_count == 0:
            return {
                "sufficient_text": False,
                "word_count": 0,
                "sentence_count": 0,
                "readability_score": 0.0,
                "vocabulary_richness": 0.0,
                "avg_sentence_length": 0.0,
                "sentence_length_variance": 0.0,
                "repetition_score": 0.0,
                "exclamation_density": 0.0,
                "caps_density": 0.0,
                "anomaly_flags": [],
            }

        # Sentence analysis
        sentences = [s.strip() for s in self._SENTENCE_SPLIT.split(text) if s.strip()]
        sentence_count = max(len(sentences), 1)
        sentence_lengths = [len(self._WORD_SPLIT.findall(s)) for s in sentences]
        avg_sentence_length = statistics.mean(sentence_lengths) if sentence_lengths else 0.0
        sentence_length_variance = (
            statistics.variance(sentence_lengths) if len(sentence_lengths) >= 2 else 0.0
        )

        # Vocabulary richness: type-token ratio
        words_lower = [w.lower() for w in words]
        unique_words = set(words_lower)
        vocabulary_richness = len(unique_words) / word_count if word_count > 0 else 0.0

        # Syllable estimation for Flesch-Kincaid
        total_syllables = sum(self._estimate_syllables(w) for w in words)
        # Flesch Reading Ease
        if sentence_count > 0 and word_count > 0:
            readability_score = (
                206.835
                - 1.015 * (word_count / sentence_count)
                - 84.6 * (total_syllables / word_count)
            )
        else:
            readability_score = 0.0

        # Punctuation density
        exclamation_count = text.count("!")
        exclamation_density = exclamation_count / word_count if word_count > 0 else 0.0

        # CAPS density: fraction of words that are ALL-CAPS (len >= 2)
        caps_words = sum(1 for w in words if w == w.upper() and len(w) >= 2)
        caps_density = caps_words / word_count if word_count > 0 else 0.0

        # Repetition: how many words appear more than expected
        word_freq = Counter(words_lower)
        top_freq = word_freq.most_common(10)
        # Exclude common English stopwords for repetition scoring
        stopwords = {
            "the", "a", "an", "and", "or", "is", "are", "was", "were", "be",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "it", "this", "that", "you", "we", "our", "your", "will", "can",
        }
        content_repeats = [
            (w, c) for w, c in top_freq
            if w not in stopwords and c >= 3 and len(w) >= 4
        ]
        repetition_score = min(len(content_repeats) / 5.0, 1.0)

        # Anomaly flags
        anomaly_flags: list[str] = []
        if vocabulary_richness < 0.3 and word_count >= 50:
            anomaly_flags.append("low_vocabulary_richness")
        if sentence_length_variance < 2.0 and sentence_count >= 5:
            anomaly_flags.append("uniform_sentence_length")
        if exclamation_density > 0.05:
            anomaly_flags.append("high_exclamation_density")
        if caps_density > 0.10:
            anomaly_flags.append("high_caps_density")
        if repetition_score > 0.4:
            anomaly_flags.append("high_content_repetition")
        if readability_score > 80 and word_count >= 50:
            anomaly_flags.append("overly_simple_language")
        if avg_sentence_length < 5 and sentence_count >= 3:
            anomaly_flags.append("very_short_sentences")

        return {
            "sufficient_text": True,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "readability_score": round(readability_score, 2),
            "vocabulary_richness": round(vocabulary_richness, 4),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "sentence_length_variance": round(sentence_length_variance, 2),
            "repetition_score": round(repetition_score, 4),
            "exclamation_density": round(exclamation_density, 4),
            "caps_density": round(caps_density, 4),
            "anomaly_flags": anomaly_flags,
        }

    def to_signal(self, job: JobPosting) -> ScamSignal | None:
        """Convert a stylometric analysis into a ScamSignal if anomalies are found."""
        result = self.analyze(job.description)
        if not result["sufficient_text"]:
            return None
        flags = result["anomaly_flags"]
        if len(flags) < 2:
            return None
        weight = min(0.70, 0.35 + len(flags) * 0.08)
        return ScamSignal(
            name="linguistic_anomaly",
            category=SignalCategory.STRUCTURAL,
            weight=round(weight, 2),
            confidence=round(min(0.45 + len(flags) * 0.08, 0.85), 2),
            detail=f"Linguistic forensics: {len(flags)} anomaly flag(s) — {', '.join(flags)}",
            evidence=f"vocab_richness={result['vocabulary_richness']:.3f}, "
                     f"readability={result['readability_score']:.1f}",
        )

    @staticmethod
    def _estimate_syllables(word: str) -> int:
        """Rough syllable count for English words."""
        word = word.lower().strip()
        if len(word) <= 2:
            return 1
        # Remove trailing silent e
        if word.endswith("e") and not word.endswith("le"):
            word = word[:-1]
        # Count vowel groups
        count = len(re.findall(r"[aeiouy]+", word))
        return max(count, 1)


# ---------------------------------------------------------------------------
# ACFE Fraud Tree — job scam taxonomy mapping
# ---------------------------------------------------------------------------

@dataclass
class FraudTreeNode:
    """A node in the ACFE Fraud Tree mapped to job scam context."""
    category: str
    subcategory: str
    job_scam_mapping: str
    detection_keywords: list[str] = field(default_factory=list)
    risk_weight: float = 0.5


FRAUD_TREE_MAPPING: list[FraudTreeNode] = [
    # Asset Misappropriation -> Fee/advance theft from job seekers
    FraudTreeNode(
        category="asset_misappropriation",
        subcategory="billing_schemes",
        job_scam_mapping="Fake equipment purchase / training fee billing",
        detection_keywords=["training fee", "equipment fee", "registration fee", "starter kit",
                           "purchase equipment", "buy your own"],
        risk_weight=0.90,
    ),
    FraudTreeNode(
        category="asset_misappropriation",
        subcategory="skimming",
        job_scam_mapping="Undisclosed deductions from promised pay",
        detection_keywords=["deduction from pay", "processing fee deducted",
                           "administrative charge", "platform fee"],
        risk_weight=0.75,
    ),
    FraudTreeNode(
        category="asset_misappropriation",
        subcategory="payroll_schemes",
        job_scam_mapping="Fake payroll requiring bank account access",
        detection_keywords=["direct deposit setup", "payroll processing",
                           "bank routing number", "void check"],
        risk_weight=0.80,
    ),
    FraudTreeNode(
        category="asset_misappropriation",
        subcategory="expense_reimbursement",
        job_scam_mapping="Fake check / overpayment reimbursement scams",
        detection_keywords=["reimburse", "overpayment", "send back difference",
                           "deposit check", "wire back excess"],
        risk_weight=0.92,
    ),
    # Corruption -> Fake recruiter / conflict-of-interest schemes
    FraudTreeNode(
        category="corruption",
        subcategory="conflicts_of_interest",
        job_scam_mapping="Fake recruiter steering candidates to paid services",
        detection_keywords=["exclusive partner", "preferred vendor", "required training provider",
                           "mandatory certification program", "must use our platform"],
        risk_weight=0.70,
    ),
    FraudTreeNode(
        category="corruption",
        subcategory="bribery_kickbacks",
        job_scam_mapping="Job placement fee / pay-for-position schemes",
        detection_keywords=["placement fee", "job guarantee fee", "priority processing",
                           "fast-track placement", "guaranteed interview fee"],
        risk_weight=0.85,
    ),
    FraudTreeNode(
        category="corruption",
        subcategory="economic_extortion",
        job_scam_mapping="Threats of lost opportunity to extract payment",
        detection_keywords=["offer expires", "last chance", "position will be filled",
                           "lose your spot", "pay now or miss out"],
        risk_weight=0.80,
    ),
    # Financial Statement Fraud -> Misrepresentation of company / role
    FraudTreeNode(
        category="financial_statement_fraud",
        subcategory="revenue_overstatement",
        job_scam_mapping="Inflated company revenue / size claims",
        detection_keywords=["billion dollar company", "fortune 500 client",
                           "market leader", "fastest growing"],
        risk_weight=0.45,
    ),
    FraudTreeNode(
        category="financial_statement_fraud",
        subcategory="asset_overvaluation",
        job_scam_mapping="Inflated compensation / benefit claims",
        detection_keywords=["unlimited earning", "six figure income", "earn up to",
                           "top earners make", "uncapped commission"],
        risk_weight=0.65,
    ),
]


# ---------------------------------------------------------------------------
# Research topics — FRAUD_HANDBOOK_TOPICS
# ---------------------------------------------------------------------------

FRAUD_HANDBOOK_TOPICS: list[ResearchTopic] = [
    ResearchTopic(
        area="fraud_triangle_pressure",
        priority=0.85,
        reason="Research financial/psychological pressure tactics used in job scams "
               "(Cressey's Fraud Triangle - Pressure leg)",
    ),
    ResearchTopic(
        area="fraud_triangle_opportunity",
        priority=0.85,
        reason="Research how job scams create perceived low-barrier opportunity "
               "(Cressey's Fraud Triangle - Opportunity leg)",
    ),
    ResearchTopic(
        area="fraud_triangle_rationalization",
        priority=0.80,
        reason="Research pre-emptive legitimacy language in job scams "
               "(Cressey's Fraud Triangle - Rationalization leg)",
    ),
    ResearchTopic(
        area="benford_salary_analysis",
        priority=0.70,
        reason="Research Benford's Law deviations in scam job salary distributions "
               "vs legitimate postings",
    ),
    ResearchTopic(
        area="acfe_billing_schemes_jobs",
        priority=0.90,
        reason="Research ACFE billing scheme patterns mapped to job posting fraud "
               "(fake equipment, training fees)",
    ),
    ResearchTopic(
        area="stylometry_scam_rings",
        priority=0.75,
        reason="Research linguistic fingerprinting techniques to identify scam ring "
               "authorship across multiple fake job postings",
    ),
    ResearchTopic(
        area="behavioral_red_flags_recruiters",
        priority=0.80,
        reason="Research ACFE behavioral red flags adapted for fake recruiter "
               "profile detection",
    ),
    ResearchTopic(
        area="overpayment_check_fraud",
        priority=0.88,
        reason="Research check fraud / overpayment scam patterns in employment "
               "context (ACFE asset misappropriation mapping)",
    ),
    ResearchTopic(
        area="corruption_placement_fees",
        priority=0.78,
        reason="Research ACFE corruption schemes mapped to job placement fee "
               "and pay-for-position scams",
    ),
    ResearchTopic(
        area="economic_viability_modeling",
        priority=0.72,
        reason="Research economic modeling to detect non-viable compensation "
               "offers (financial statement fraud analogy)",
    ),
]


# ---------------------------------------------------------------------------
# Deep research prompts — FRAUD_HANDBOOK_PROMPTS
# ---------------------------------------------------------------------------

FRAUD_HANDBOOK_PROMPTS: list[dict] = [
    {
        "topic": "fraud_triangle_pressure_language",
        "prompt_text": (
            "Using Cressey's Fraud Triangle model (Pressure leg), identify the "
            "specific linguistic patterns that employment scammers use to exploit "
            "financial desperation in job seekers. Provide: (1) exact phrases and "
            "sentence templates that appeal to financial pressure, (2) how these "
            "differ from legitimate job postings that mention benefits, (3) regex "
            "patterns to detect pressure exploitation, (4) false-positive "
            "mitigation strategies for legitimate postings that mention compensation."
        ),
        "expected_patterns": ["pressure_phrases", "regex_patterns", "exclusion_rules"],
        "priority": 0.90,
    },
    {
        "topic": "fraud_triangle_opportunity_barriers",
        "prompt_text": (
            "Using Cressey's Fraud Triangle (Opportunity leg), analyze how fraudulent "
            "job postings create perceived low-barrier opportunities. What specific "
            "language patterns distinguish 'no experience needed' scams from legitimate "
            "entry-level postings? Provide concrete detection rules comparing: legitimate "
            "entry-level (still lists SOME requirements) vs scam (lists NO requirements "
            "and promises immediate hire). Include at least 10 distinguishing features."
        ),
        "expected_patterns": ["opportunity_indicators", "legitimacy_markers", "comparison_rules"],
        "priority": 0.88,
    },
    {
        "topic": "fraud_triangle_rationalization_defensive",
        "prompt_text": (
            "Using Cressey's Fraud Triangle (Rationalization leg), identify how "
            "scam job postings pre-emptively defend their legitimacy. Legitimate "
            "companies never say 'this is not a scam'. What specific defensive "
            "language patterns indicate fraud? Provide: (1) a taxonomy of "
            "rationalization phrases in job scams, (2) psychological manipulation "
            "techniques (Sykes & Matza neutralization theory) adapted for employment "
            "fraud, (3) detection patterns with confidence weights."
        ),
        "expected_patterns": ["rationalization_phrases", "neutralization_techniques", "confidence_weights"],
        "priority": 0.85,
    },
    {
        "topic": "benford_law_salary_analysis",
        "prompt_text": (
            "Apply Benford's Law to employment fraud detection. Analyze how the "
            "first-digit distribution of salary figures in fraudulent job postings "
            "differs from legitimate ones. Consider: (1) do scam salaries cluster "
            "around psychological price points ($99,999, $50,000, $100,000)? "
            "(2) what is the expected Benford distribution for legitimate salary "
            "ranges? (3) what digit-level anomalies indicate fabricated compensation? "
            "(4) how to combine Benford analysis with ratio analysis (max/min salary "
            "ratio) for stronger detection."
        ),
        "expected_patterns": ["digit_distributions", "ratio_thresholds", "combined_scores"],
        "priority": 0.75,
    },
    {
        "topic": "acfe_asset_misappropriation_job_scams",
        "prompt_text": (
            "Map the ACFE Fraud Tree's asset misappropriation branch to employment "
            "scam patterns. For each sub-scheme (skimming, cash larceny, billing, "
            "payroll, expense reimbursement), identify the job scam equivalent: "
            "(1) fake equipment purchase schemes, (2) training fee billing, "
            "(3) advance fee fraud, (4) check overpayment scams. Provide specific "
            "linguistic markers, typical dollar amounts, and detection regex patterns "
            "for each mapped scheme."
        ),
        "expected_patterns": ["scheme_mapping", "dollar_thresholds", "regex_patterns"],
        "priority": 0.92,
    },
    {
        "topic": "corruption_schemes_recruitment",
        "prompt_text": (
            "Adapt the ACFE Fraud Tree's corruption branch (conflicts of interest, "
            "bribery, kickbacks, extortion) to recruitment fraud. How do fake "
            "recruiters and staffing agencies use corruption-like schemes? Identify: "
            "(1) pay-for-position scams, (2) mandatory training provider kickbacks, "
            "(3) fake background check fee extortion, (4) placement guarantee fee "
            "schemes. Provide detection patterns and distinguishing features from "
            "legitimate staffing agency practices."
        ),
        "expected_patterns": ["corruption_mapping", "fee_patterns", "legitimacy_markers"],
        "priority": 0.82,
    },
    {
        "topic": "financial_statement_fraud_company_misrep",
        "prompt_text": (
            "Apply the ACFE's financial statement fraud framework to detect company "
            "misrepresentation in job postings. How do scam postings inflate company "
            "credentials (revenue overstatement analogy) or fabricate benefits "
            "(asset overvaluation analogy)? Provide: (1) specific claim patterns "
            "that are verifiable vs fabricated, (2) red flags for inflated company "
            "size/revenue claims, (3) detection rules for benefit packages that "
            "exceed industry norms for the stated company size."
        ),
        "expected_patterns": ["misrepresentation_patterns", "verifiable_claims", "benefit_anomalies"],
        "priority": 0.78,
    },
    {
        "topic": "behavioral_red_flags_posting_patterns",
        "prompt_text": (
            "Adapt the ACFE's behavioral red flag research (84% of fraudsters "
            "display at least one behavioral indicator) to job posting analysis. "
            "The top ACFE behavioral red flags are: living beyond means, financial "
            "difficulties, close vendor association, control issues, defensiveness. "
            "Map each to job posting behaviors: (1) what posting metadata patterns "
            "indicate 'control issues' (same poster, many companies)? (2) what "
            "language patterns indicate 'defensiveness'? (3) what patterns indicate "
            "'close association' manipulation (fake testimonials, insider language)?"
        ),
        "expected_patterns": ["metadata_patterns", "language_indicators", "behavioral_mapping"],
        "priority": 0.80,
    },
    {
        "topic": "proactive_ratio_analysis",
        "prompt_text": (
            "Apply the ACFE's proactive fraud detection ratio analysis to job "
            "posting data. What ratios in job postings are diagnostic of fraud? "
            "Consider: (1) salary-to-requirement ratio (high pay, low requirements), "
            "(2) description-length-to-specificity ratio, (3) urgency-word frequency "
            "ratios, (4) positive-claim to concrete-detail ratios, (5) company-size "
            "to posting-volume ratios. For each ratio, provide the threshold values "
            "that distinguish legitimate from fraudulent postings."
        ),
        "expected_patterns": ["ratio_definitions", "threshold_values", "combined_scoring"],
        "priority": 0.85,
    },
    {
        "topic": "stylometry_scam_ring_detection",
        "prompt_text": (
            "How can forensic stylometry (authorship analysis) detect scam ring "
            "operations? Scam rings produce many postings with similar linguistic "
            "fingerprints. Identify: (1) which stylometric features (sentence length "
            "distribution, vocabulary richness, type-token ratio, readability) best "
            "distinguish individual scam authors, (2) what feature thresholds indicate "
            "template-generated vs human-written postings, (3) how to compute a "
            "'template probability' score from stylometric features."
        ),
        "expected_patterns": ["stylometric_features", "template_thresholds", "authorship_clustering"],
        "priority": 0.77,
    },
    {
        "topic": "economic_viability_scoring",
        "prompt_text": (
            "Design an economic viability model for job postings. A legitimate "
            "business must be able to afford the compensation it offers. What "
            "signals indicate that a posted salary is economically non-viable for "
            "the claimed company? Consider: (1) salary vs company size mismatch, "
            "(2) compensation vs industry norms, (3) benefit packages that exceed "
            "company class, (4) revenue-per-employee feasibility. Provide scoring "
            "rules with threshold values."
        ),
        "expected_patterns": ["viability_rules", "mismatch_thresholds", "industry_benchmarks"],
        "priority": 0.73,
    },
    {
        "topic": "interview_investigation_techniques",
        "prompt_text": (
            "Adapt the ACFE's fraud interview and investigation techniques to "
            "automated job posting analysis. The ACFE teaches structured interview "
            "methods to detect deception. What deception indicators from interview "
            "science apply to written text? Consider: (1) hedging language, "
            "(2) passive voice overuse, (3) pronoun distancing, (4) excessive detail "
            "in irrelevant areas, (5) vagueness in critical details (compensation, "
            "responsibilities). Provide regex patterns and scoring weights."
        ),
        "expected_patterns": ["deception_indicators", "regex_patterns", "scoring_weights"],
        "priority": 0.70,
    },
    {
        "topic": "fraud_prevention_trust_signals",
        "prompt_text": (
            "Using the ACFE's fraud prevention program framework, identify the "
            "positive trust signals that indicate a job posting is from a company "
            "with strong anti-fraud controls. What elements of a legitimate posting "
            "demonstrate: (1) proper corporate governance (structured application "
            "process, ATS usage), (2) transparent compensation (specific ranges, "
            "not vague promises), (3) verifiable company identity (domain-matched "
            "email, LinkedIn presence), (4) professional HR practices? These signals "
            "should REDUCE the scam score."
        ),
        "expected_patterns": ["trust_signals", "governance_indicators", "score_reduction_rules"],
        "priority": 0.68,
    },
    {
        "topic": "data_analytics_anomaly_detection",
        "prompt_text": (
            "Apply the ACFE's proactive data analytics framework to job posting "
            "metadata. What statistical anomalies in posting metadata indicate "
            "fraud? Consider: (1) posting time-of-day distributions, (2) geographic "
            "inconsistencies, (3) company age vs posting volume, (4) recruiter "
            "account age vs connection count, (5) description text similarity "
            "clustering. For each anomaly type, specify the statistical test and "
            "threshold values."
        ),
        "expected_patterns": ["statistical_tests", "threshold_values", "metadata_anomalies"],
        "priority": 0.80,
    },
    {
        "topic": "neutralization_techniques_job_fraud",
        "prompt_text": (
            "Apply Sykes & Matza's neutralization theory (referenced in Wells' "
            "Corporate Fraud Handbook) to job posting fraud. The five techniques "
            "of neutralization are: denial of responsibility, denial of injury, "
            "denial of victim, condemnation of condemners, appeal to higher loyalties. "
            "How do job scam postings use each technique? Provide: (1) specific "
            "language patterns for each technique, (2) detection regex, (3) how to "
            "distinguish genuine from manipulative use."
        ),
        "expected_patterns": ["neutralization_mapping", "language_patterns", "detection_rules"],
        "priority": 0.72,
    },
    {
        "topic": "network_analysis_recruiter_fraud",
        "prompt_text": (
            "Apply network analysis concepts from fraud investigation to recruiter "
            "and company relationship mapping. How can graph-based analysis of "
            "recruiter-company-posting relationships reveal scam networks? Consider: "
            "(1) same recruiter posting for many unrelated companies, (2) companies "
            "sharing identical descriptions, (3) recruiter accounts with suspicious "
            "connection patterns, (4) shell company detection via shared metadata. "
            "What graph metrics (centrality, clustering coefficient) are diagnostic?"
        ),
        "expected_patterns": ["graph_metrics", "network_indicators", "shell_detection_rules"],
        "priority": 0.76,
    },
    {
        "topic": "supply_chain_industry_mismatch",
        "prompt_text": (
            "Apply supply-chain fraud detection (ACFE) to detect industry/role "
            "mismatches in job postings. Does the company's claimed industry match "
            "the roles they are posting? Consider: (1) a 'tech company' posting "
            "only customer service roles, (2) a 'financial services firm' with no "
            "finance-specific requirements, (3) industry-role compatibility scoring, "
            "(4) how to build an industry-role affinity matrix for detection."
        ),
        "expected_patterns": ["industry_role_matrix", "mismatch_scoring", "compatibility_rules"],
        "priority": 0.65,
    },
    {
        "topic": "temporal_pattern_analysis",
        "prompt_text": (
            "Apply the ACFE's temporal fraud detection patterns to job posting "
            "behavior. Fraudulent postings often follow distinct temporal patterns: "
            "(1) batch posting (many postings within minutes), (2) holiday/weekend "
            "posting (when oversight is reduced), (3) rapid reposting after "
            "removal, (4) posting velocity spikes. What temporal features should be "
            "extracted, and what thresholds indicate fraud vs legitimate hiring surges?"
        ),
        "expected_patterns": ["temporal_features", "batch_thresholds", "velocity_metrics"],
        "priority": 0.74,
    },
]


# ---------------------------------------------------------------------------
# Fraud Diamond Scorer (Wolfe & Hermanson 2004)
# ---------------------------------------------------------------------------

# Capability indicators — signs that the scam author has the skills/position
# to execute fraud (the fourth element beyond the Fraud Triangle).

_CAPABILITY_POSITION_PATTERNS = re.compile(
    r"\b(hiring manager|head of (hr|recruitment|talent)|"
    r"vp of (hr|operations|recruitment)|director of (hiring|talent|hr)|"
    r"ceo|cfo|founder|managing director|chief .* officer|"
    r"authorized (representative|agent)|official (recruiter|representative)|"
    r"senior (recruiter|hiring|talent))\b",
    re.IGNORECASE,
)

_CAPABILITY_COERCION_PATTERNS = re.compile(
    r"\b(refer (a |your )?friend|referral bonus|team.?building|"
    r"recruit (others|members|your (team|network))|"
    r"bring (your |a )?(friend|team|partner)|"
    r"multi.?level|downline|upline|network (marketing|opportunity)|"
    r"build (your |a )?team|mentor (others|new (members|recruits)))\b",
    re.IGNORECASE,
)

_CAPABILITY_CONFIDENCE_PATTERNS = re.compile(
    r"\b(proven (system|method|track record|results)|"
    r"guaranteed (results|success|income|returns)|"
    r"industry.?leading|world.?class|unmatched|unrivaled|"
    r"(top|best|premier|elite) (company|opportunity|program)|"
    r"exclusive (opportunity|program|offer|access)|"
    r"hand.?picked|specially selected|chosen (few|candidates))\b",
    re.IGNORECASE,
)


class FraudDiamondScorer:
    """Score job postings against the Fraud Diamond model (Wolfe & Hermanson 2004).

    Extends Cressey's Fraud Triangle by adding **Capability** — the fourth element
    that captures whether the fraudster has the skills, position, and traits needed
    to execute and conceal the fraud.

    The six capability components:
    1. Position/Function — authority claims in the posting
    2. Brains/Intelligence — sophistication of social engineering
    3. Confidence/Ego — bold, absolute claims
    4. Coercion Skills — multi-level/referral recruitment language
    5. Effective Lying — internally consistent fabricated details
    6. Immunity to Stress — persistent re-posting after takedowns

    Research shows auditors using the Diamond assess fraud risk 17% higher
    than those using only the Triangle.
    """

    PRESSURE_WEIGHT = 0.25
    OPPORTUNITY_WEIGHT = 0.25
    RATIONALIZATION_WEIGHT = 0.25
    CAPABILITY_WEIGHT = 0.25

    def __init__(self) -> None:
        self._triangle = FraudTriangleScorer()

    def score(self, job: JobPosting) -> dict:
        """Return per-element scores and a composite diamond score."""
        # Get the triangle scores first
        tri = self._triangle.score(job)
        text = f"{job.title} {job.description}"

        # Capability sub-scores
        position_hits = _CAPABILITY_POSITION_PATTERNS.findall(text)
        coercion_hits = _CAPABILITY_COERCION_PATTERNS.findall(text)
        confidence_hits = _CAPABILITY_CONFIDENCE_PATTERNS.findall(text)

        position_score = min(len(position_hits) / 2.0, 1.0)
        coercion_score = min(len(coercion_hits) / 2.0, 1.0)
        confidence_score = min(len(confidence_hits) / 2.0, 1.0)

        # Aggregate capability: weighted combination of sub-factors
        capability_score = min(
            (position_score * 0.30 + coercion_score * 0.35 + confidence_score * 0.35),
            1.0,
        )

        # Composite diamond score
        composite = (
            self.PRESSURE_WEIGHT * tri["pressure_score"]
            + self.OPPORTUNITY_WEIGHT * tri["opportunity_score"]
            + self.RATIONALIZATION_WEIGHT * tri["rationalization_score"]
            + self.CAPABILITY_WEIGHT * capability_score
        )

        # Amplify when all 4 elements present (1.35x boost)
        elements_present = sum([
            tri["pressure_score"] > 0,
            tri["opportunity_score"] > 0,
            tri["rationalization_score"] > 0,
            capability_score > 0,
        ])
        if elements_present == 4:
            composite = min(composite * 1.35, 1.0)
        elif elements_present == 3:
            composite = min(composite * 1.15, 1.0)

        return {
            "pressure_score": tri["pressure_score"],
            "opportunity_score": tri["opportunity_score"],
            "rationalization_score": tri["rationalization_score"],
            "capability_score": round(capability_score, 3),
            "capability_position": round(position_score, 3),
            "capability_coercion": round(coercion_score, 3),
            "capability_confidence": round(confidence_score, 3),
            "elements_present": elements_present,
            "composite_score": round(composite, 3),
            "triangle_composite": tri["composite_score"],
        }

    def to_signal(self, job: JobPosting) -> ScamSignal | None:
        """Convert a Fraud Diamond score into a ScamSignal if significant."""
        result = self.score(job)
        if result["composite_score"] < 0.25:
            return None
        weight = min(0.92, 0.50 + result["composite_score"] * 0.45)
        return ScamSignal(
            name="fraud_diamond",
            category=(
                SignalCategory.RED_FLAG
                if result["elements_present"] >= 4
                else SignalCategory.WARNING
            ),
            weight=round(weight, 2),
            confidence=round(min(result["composite_score"] + 0.1, 0.95), 2),
            detail=(
                f"Fraud Diamond: {result['elements_present']}/4 elements "
                f"(P={result['pressure_score']:.0%}, O={result['opportunity_score']:.0%}, "
                f"R={result['rationalization_score']:.0%}, C={result['capability_score']:.0%})"
            ),
            evidence=f"composite={result['composite_score']:.3f}",
        )


# ---------------------------------------------------------------------------
# Neutralization Technique Detector (Sykes & Matza 1957, extended)
# ---------------------------------------------------------------------------

_NEUTRALIZATION_PATTERNS: dict[str, re.Pattern] = {
    "denial_of_responsibility": re.compile(
        r"\b(the (market|economy|industry) (forced|requires|demands)|"
        r"(due to|because of) (market|economic|industry) (conditions|changes|demands)|"
        r"circumstances (beyond|outside) (our )?(control|influence)|"
        r"(we|they) had no (choice|option|alternative)|"
        r"(forced|compelled|required) (to|by))\b",
        re.IGNORECASE,
    ),
    "denial_of_injury": re.compile(
        r"\b((fully |100% )?refundable|no risk (involved|whatsoever|at all)|"
        r"risk.?free|money.?back guarantee|zero risk|"
        r"nothing to lose|can'?t lose|no harm|"
        r"(small|minimal|modest) (investment|fee|cost)|"
        r"you (won'?t|will not) lose (anything|money|a (dime|cent)))\b",
        re.IGNORECASE,
    ),
    "denial_of_victim": re.compile(
        r"\b(only (serious|committed|dedicated) (candidates|applicants|people)|"
        r"not for (everyone|the lazy|tire kickers|quitters)|"
        r"(weed|filter|screen) out .*(uncommitted|unserious|lazy)|"
        r"if you'?re not (serious|committed|ready)|"
        r"(serious|real) (inquiries|applicants|candidates) only)\b",
        re.IGNORECASE,
    ),
    "condemnation_of_condemners": re.compile(
        r"\b((don'?t|do not) (let|listen to) (the )?(haters|skeptics|naysayers|negativity|doubters)|"
        r"(ignore|disregard) (the )?(negativity|critics|skeptics|haters)|"
        r"(haters|skeptics|naysayers) (will|always) (say|tell you)|"
        r"(people|others) (said|told me) (it was|it'?s) (impossible|a scam)|"
        r"(prove|proved) (the |them |everyone )?(wrong|doubters wrong))\b",
        re.IGNORECASE,
    ),
    "appeal_to_higher_loyalties": re.compile(
        r"\b((join|become part of) (our|the|a) (family|tribe|community|movement)|"
        r"(help|support|serve) (your )?(family|community|country|people)|"
        r"(for|to help) (your )?(kids|children|family|loved ones)|"
        r"(give|provide) (your family|your kids|them) (a better|the best)|"
        r"(make|build) a (better|brighter) (future|life|world))\b",
        re.IGNORECASE,
    ),
    "defense_of_necessity": re.compile(
        r"\b((required|mandatory|necessary) (by law|by regulation|for compliance)|"
        r"(legal|regulatory) requirement|legally (required|mandated)|"
        r"(have to|must) (charge|collect|require) (this|a) (fee|payment)|"
        r"(covers|for) (mandatory|required) (background|training|certification|processing))\b",
        re.IGNORECASE,
    ),
    "claim_of_normality": re.compile(
        r"\b((standard|normal|common|typical|usual) (practice|procedure|industry|in the industry)|"
        r"(all|every|most) (companies|employers|businesses) (do|require|charge) (this|the same)|"
        r"(industry|market) standard|nothing (unusual|out of the ordinary)|"
        r"(just|simply) (how|the way) (it|things|business) (works|is done))\b",
        re.IGNORECASE,
    ),
    "claim_of_entitlement": re.compile(
        r"\b(you (deserve|earned|are worth|owe (it to )?(yourself|your family))|"
        r"(treat|reward) yourself|you'?ve (earned|worked for) (this|it)|"
        r"(time|chance) (to|for) (live|enjoy|reward)|"
        r"(finally|at last) (get|have|enjoy) what you (deserve|earned))\b",
        re.IGNORECASE,
    ),
}


class NeutralizationDetector:
    """Detect Sykes & Matza's neutralization techniques in job posting text.

    Identifies eight neutralization techniques (5 original + 3 extended for
    corporate fraud) that scammers use to pre-emptively justify their fraud
    and overcome victims' moral objections.

    Each technique maps to the Rationalization leg of the Fraud Triangle/Diamond.
    """

    # Correlation weight: how strongly each technique correlates with fraud
    TECHNIQUE_WEIGHTS: dict[str, float] = {
        "denial_of_responsibility": 0.55,
        "denial_of_injury": 0.75,
        "denial_of_victim": 0.70,
        "condemnation_of_condemners": 0.85,
        "appeal_to_higher_loyalties": 0.50,
        "defense_of_necessity": 0.65,
        "claim_of_normality": 0.60,
        "claim_of_entitlement": 0.45,
    }

    def detect(self, text: str) -> dict:
        """Analyze text for neutralization techniques.

        Returns a dict with per-technique hits and a composite score.
        """
        if not text or len(text.strip()) < 20:
            return {
                "techniques_detected": [],
                "technique_count": 0,
                "composite_score": 0.0,
                "details": {},
            }

        detected: list[str] = []
        details: dict[str, list[str]] = {}

        for technique, pattern in _NEUTRALIZATION_PATTERNS.items():
            hits = pattern.findall(text)
            if hits:
                detected.append(technique)
                # Flatten: hits may be tuples from groups
                flat = []
                for h in hits[:3]:
                    flat.append(h if isinstance(h, str) else h[0])
                details[technique] = flat

        # Composite: weighted sum of detected techniques, capped at 1.0
        composite = 0.0
        if detected:
            total_weight = sum(self.TECHNIQUE_WEIGHTS[t] for t in detected)
            max_possible = sum(self.TECHNIQUE_WEIGHTS.values())
            composite = min(total_weight / max_possible, 1.0)

            # Amplify if 3+ techniques detected (strong indicator)
            if len(detected) >= 3:
                composite = min(composite * 1.3, 1.0)

        return {
            "techniques_detected": detected,
            "technique_count": len(detected),
            "composite_score": round(composite, 3),
            "details": details,
        }

    def to_signal(self, job: JobPosting) -> ScamSignal | None:
        """Convert neutralization detection into a ScamSignal."""
        result = self.detect(f"{job.title} {job.description}")
        if result["technique_count"] < 1:
            return None
        weight = min(0.85, 0.40 + result["technique_count"] * 0.10)
        return ScamSignal(
            name="neutralization_technique",
            category=(
                SignalCategory.RED_FLAG
                if result["technique_count"] >= 3
                else SignalCategory.WARNING
            ),
            weight=round(weight, 2),
            confidence=round(min(0.40 + result["technique_count"] * 0.12, 0.90), 2),
            detail=(
                f"Neutralization: {result['technique_count']} technique(s) — "
                + ", ".join(result["techniques_detected"][:4])
            ),
            evidence=f"composite={result['composite_score']:.3f}",
        )


# ---------------------------------------------------------------------------
# Behavioral Red Flag Scorer (ACFE RTTN 2024)
# ---------------------------------------------------------------------------

_BEHAVIORAL_RED_FLAG_PATTERNS: dict[str, re.Pattern] = {
    "living_beyond_means": re.compile(
        r"\b(luxury (lifestyle|living|car|vacation|home)|"
        r"(drive|own) (a )?(lamborghini|ferrari|porsche|bmw|mercedes)|"
        r"(financial|time) freedom|live (the |your )?dream|"
        r"(quit|leave|fire) your (boss|job|9.to.5)|"
        r"(millionaire|wealthy|rich) (lifestyle|mindset|club)|"
        r"(work|earn|travel) from (anywhere|the beach|paradise)|"
        r"passive (income|revenue|earnings))\b",
        re.IGNORECASE,
    ),
    "financial_difficulties_targeting": re.compile(
        r"\b((struggling|drowning) (with|in) (debt|bills|finances)|"
        r"(can'?t|unable to) (pay|afford|make ends meet)|"
        r"(behind on|late on) (rent|mortgage|bills|payments)|"
        r"(paycheck to paycheck|living paycheck)|"
        r"need (extra|more|quick|fast) (money|cash|income)|"
        r"(unemployment|laid off|fired|let go|downsized))\b",
        re.IGNORECASE,
    ),
    "vendor_association": re.compile(
        r"\b((exclusive|preferred|official) (partner(ship)?|vendor|provider|affiliate)|"
        r"(must|required to) (use|buy from|purchase through) (our|the|this)|"
        r"(only available|exclusively) through (us|our (partner|platform))|"
        r"(partnered|teamed up|working) with (a )?select (few|group|vendors))\b",
        re.IGNORECASE,
    ),
    "control_issues": re.compile(
        r"\b((do not|don'?t) (contact|call|email|reach out to) (the company|hr|anyone else)|"
        r"(only|exclusively) (through|via|contact) (me|this (email|number|channel))|"
        r"(I am|I'?m) (the|your) (only|sole|exclusive|direct) (contact|point of contact|recruiter)|"
        r"(keep|this (is|must be|should be)) (confidential|private|between us|secret))\b",
        re.IGNORECASE,
    ),
    "defensiveness": re.compile(
        r"\b((why|how dare) (would|do) you (question|doubt|ask)|"
        r"(I|we) (don'?t|do not) (need to|have to) (prove|explain|justify)|"
        r"(stop|quit|enough with the) (asking|questioning|doubting)|"
        r"take it or leave it|"
        r"(this|the) offer (won'?t|will not) (last|be available) (forever|long|much longer))\b",
        re.IGNORECASE,
    ),
    "bullying_intimidation": re.compile(
        r"\b((you'?ll|you will) (regret|miss out|be sorry)|"
        r"(don'?t|do not) (waste|miss) this (chance|opportunity)|"
        r"(last|final) (chance|warning|opportunity|call)|"
        r"(act|apply|respond|decide) (now|today|immediately) or (lose|miss|forfeit)|"
        r"(your loss|too bad (for you)?|you snooze you lose))\b",
        re.IGNORECASE,
    ),
    "wheeler_dealer": re.compile(
        r"\b((special|exclusive|insider|limited) (deal|offer|discount|price|opportunity)|"
        r"(one.?time|today.?only|limited.?time) (offer|deal|opportunity|price)|"
        r"(normally|usually|regular price) \$?\d+.* (but|now|today|for you) \$?\d+|"
        r"(act|sign up|register) (now|today) (and|to) (get|receive|save)|"
        r"(early bird|founder|charter) (rate|pricing|deal|access))\b",
        re.IGNORECASE,
    ),
}

# ACFE RTTN 2024 prevalence (used as base weights)
_RED_FLAG_PREVALENCE: dict[str, float] = {
    "living_beyond_means": 0.39,
    "financial_difficulties_targeting": 0.27,
    "vendor_association": 0.18,
    "control_issues": 0.15,
    "defensiveness": 0.12,
    "bullying_intimidation": 0.11,
    "wheeler_dealer": 0.08,
}


class BehavioralRedFlagScorer:
    """Score job postings against ACFE behavioral red flag indicators.

    The ACFE 2024 Report to the Nations found that 85% of fraud perpetrators
    displayed at least one behavioral red flag before detection. This scorer
    maps the eight most common behavioral indicators to job posting text patterns.

    Each red flag is weighted by its ACFE-documented prevalence (39% for living
    beyond means down to 8% for wheeler-dealer attitude).
    """

    def score(self, job: JobPosting) -> dict:
        """Score a job posting for behavioral red flags.

        Returns a dict with per-flag detections and a composite score.
        """
        text = f"{job.title} {job.description}"
        if not text.strip():
            return {
                "flags_detected": [],
                "flag_count": 0,
                "composite_score": 0.0,
                "details": {},
            }

        detected: list[str] = []
        details: dict[str, list[str]] = {}

        for flag, pattern in _BEHAVIORAL_RED_FLAG_PATTERNS.items():
            hits = pattern.findall(text)
            if hits:
                detected.append(flag)
                flat = []
                for h in hits[:3]:
                    flat.append(h if isinstance(h, str) else h[0])
                details[flag] = flat

        # Composite: prevalence-weighted sum of detected flags
        composite = 0.0
        if detected:
            weighted_sum = sum(_RED_FLAG_PREVALENCE.get(f, 0.10) for f in detected)
            max_possible = sum(_RED_FLAG_PREVALENCE.values())
            composite = min(weighted_sum / max_possible, 1.0)

            # Amplify if 3+ flags detected (strong compound indicator)
            if len(detected) >= 3:
                composite = min(composite * 1.4, 1.0)

        return {
            "flags_detected": detected,
            "flag_count": len(detected),
            "composite_score": round(composite, 3),
            "details": details,
        }

    def to_signal(self, job: JobPosting) -> ScamSignal | None:
        """Convert behavioral red flag scoring into a ScamSignal."""
        result = self.score(job)
        if result["flag_count"] < 2:
            return None
        weight = min(0.85, 0.40 + result["flag_count"] * 0.10)
        return ScamSignal(
            name="behavioral_red_flags",
            category=(
                SignalCategory.RED_FLAG
                if result["flag_count"] >= 3
                else SignalCategory.WARNING
            ),
            weight=round(weight, 2),
            confidence=round(min(0.40 + result["flag_count"] * 0.10, 0.85), 2),
            detail=(
                f"ACFE Behavioral Red Flags: {result['flag_count']} flag(s) — "
                + ", ".join(result["flags_detected"][:4])
            ),
            evidence=f"composite={result['composite_score']:.3f}",
        )


# ---------------------------------------------------------------------------
# Fraud Ratio Analyzer (ACFE Proactive Detection)
# ---------------------------------------------------------------------------

class FraudRatioAnalyzer:
    """Compute diagnostic ratios from ACFE proactive detection methodology.

    Implements quantitative ratio analysis for job posting fraud detection.
    Each ratio compares an aspect of the posting to expected norms, flagging
    deviations as fraud indicators.

    Organizations using proactive data analytics experience 50% lower fraud
    losses (ACFE RTTN 2024).
    """

    # Threshold configuration: (warn_threshold, red_flag_threshold)
    RATIO_THRESHOLDS: dict[str, tuple[float, float]] = {
        "salary_range_width": (3.0, 5.0),           # max/min salary ratio
        "exclamation_to_period": (0.3, 0.6),         # ! to . ratio
        "caps_word_density": (0.08, 0.15),           # fraction of ALL-CAPS words
        "urgency_word_density": (0.02, 0.05),        # urgency words per total words
        "vague_to_specific": (2.0, 4.0),             # vague phrases vs specific details
        "pay_to_requirements": (3.0, 8.0),           # salary per requirement listed
        "personal_info_density": (0.01, 0.03),       # sensitive info requests per word
    }

    _URGENCY_WORDS = re.compile(
        r"\b(immediately|urgent(ly)?|asap|right now|hurry|limited (time|spots|positions)|"
        r"act (now|fast|quickly)|don'?t (wait|delay|hesitate)|today only|"
        r"(apply|start|respond|act) (now|today|immediately))\b",
        re.IGNORECASE,
    )

    _VAGUE_PHRASES = re.compile(
        r"\b(various (tasks|duties|responsibilities)|"
        r"other (duties|tasks|responsibilities) as (assigned|needed)|"
        r"flexible (hours|schedule|duties)|"
        r"general (tasks|duties|office work|responsibilities)|"
        r"administrative (tasks|duties|support)|"
        r"day.to.day (operations|tasks|activities))\b",
        re.IGNORECASE,
    )

    _SPECIFIC_DETAILS = re.compile(
        r"\b(\d+ years? (of )?(experience|work)|"
        r"(bachelor|master|phd|degree) (in|of)|"
        r"(proficient|experienced|skilled) (in|with) \w+|"
        r"(python|java|sql|excel|tableau|salesforce|aws|azure|gcp)|"
        r"(certified|certification|license|licensed) (in|as|for))\b",
        re.IGNORECASE,
    )

    _PERSONAL_INFO_REQUESTS = re.compile(
        r"\b(social security|ssn|bank (account|routing)|credit card|"
        r"date of birth|dob|passport (number|copy)|"
        r"driver'?s? licen[sc]e (number|copy)|"
        r"mother'?s? maiden|tax (id|identification)|"
        r"send (us |me )?(your |a )?(photo|picture|headshot|selfie))\b",
        re.IGNORECASE,
    )

    def analyze(self, job: JobPosting) -> dict:
        """Compute fraud-diagnostic ratios for a job posting.

        Returns a dict with individual ratios, flags, and a composite anomaly score.
        """
        text = f"{job.title} {job.description}"
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        word_count = max(len(words), 1)

        ratios: dict[str, float | None] = {}
        flags: list[str] = []

        # 1. Salary range width ratio
        if job.salary_min > 0 and job.salary_max > 0:
            ratio = job.salary_max / job.salary_min
            ratios["salary_range_width"] = round(ratio, 2)
            w, r = self.RATIO_THRESHOLDS["salary_range_width"]
            if ratio >= r:
                flags.append(f"salary_range_width={ratio:.1f}x (red flag: >{r}x)")
            elif ratio >= w:
                flags.append(f"salary_range_width={ratio:.1f}x (warning: >{w}x)")
        else:
            ratios["salary_range_width"] = None

        # 2. Exclamation to period ratio
        excl_count = text.count("!")
        period_count = max(text.count("."), 1)
        excl_ratio = excl_count / period_count
        ratios["exclamation_to_period"] = round(excl_ratio, 3)
        w, r = self.RATIO_THRESHOLDS["exclamation_to_period"]
        if excl_ratio >= r:
            flags.append(f"exclamation_to_period={excl_ratio:.2f} (red flag: >{r})")
        elif excl_ratio >= w:
            flags.append(f"exclamation_to_period={excl_ratio:.2f} (warning: >{w})")

        # 3. CAPS word density
        caps_words = sum(1 for w in words if w == w.upper() and len(w) >= 2)
        caps_density = caps_words / word_count
        ratios["caps_word_density"] = round(caps_density, 4)
        w, r = self.RATIO_THRESHOLDS["caps_word_density"]
        if caps_density >= r:
            flags.append(f"caps_word_density={caps_density:.3f} (red flag: >{r})")
        elif caps_density >= w:
            flags.append(f"caps_word_density={caps_density:.3f} (warning: >{w})")

        # 4. Urgency word density
        urgency_hits = self._URGENCY_WORDS.findall(text)
        urgency_density = len(urgency_hits) / word_count
        ratios["urgency_word_density"] = round(urgency_density, 4)
        w, r = self.RATIO_THRESHOLDS["urgency_word_density"]
        if urgency_density >= r:
            flags.append(f"urgency_word_density={urgency_density:.3f} (red flag: >{r})")
        elif urgency_density >= w:
            flags.append(f"urgency_word_density={urgency_density:.3f} (warning: >{w})")

        # 5. Vague to specific ratio
        vague_count = len(self._VAGUE_PHRASES.findall(text))
        specific_count = max(len(self._SPECIFIC_DETAILS.findall(text)), 1)
        vague_ratio = vague_count / specific_count
        ratios["vague_to_specific"] = round(vague_ratio, 2)
        w, r = self.RATIO_THRESHOLDS["vague_to_specific"]
        if vague_ratio >= r:
            flags.append(f"vague_to_specific={vague_ratio:.1f} (red flag: >{r})")
        elif vague_ratio >= w:
            flags.append(f"vague_to_specific={vague_ratio:.1f} (warning: >{w})")

        # 6. Personal info request density
        pi_hits = self._PERSONAL_INFO_REQUESTS.findall(text)
        pi_density = len(pi_hits) / word_count
        ratios["personal_info_density"] = round(pi_density, 4)
        w, r = self.RATIO_THRESHOLDS["personal_info_density"]
        if pi_density >= r:
            flags.append(f"personal_info_density={pi_density:.4f} (red flag: >{r})")
        elif pi_density >= w:
            flags.append(f"personal_info_density={pi_density:.4f} (warning: >{w})")

        # Composite anomaly score: fraction of ratios that exceed thresholds
        total_ratios = len(self.RATIO_THRESHOLDS)
        warning_count = len(flags)
        composite = min(warning_count / total_ratios, 1.0)

        return {
            "ratios": ratios,
            "flags": flags,
            "flag_count": len(flags),
            "composite_score": round(composite, 3),
        }

    def to_signal(self, job: JobPosting) -> ScamSignal | None:
        """Convert ratio analysis into a ScamSignal."""
        result = self.analyze(job)
        if result["flag_count"] < 2:
            return None
        weight = min(0.80, 0.35 + result["flag_count"] * 0.10)
        return ScamSignal(
            name="fraud_ratio_anomaly",
            category=(
                SignalCategory.RED_FLAG
                if result["flag_count"] >= 4
                else SignalCategory.WARNING
            ),
            weight=round(weight, 2),
            confidence=round(min(0.40 + result["flag_count"] * 0.10, 0.85), 2),
            detail=(
                f"ACFE Ratio Analysis: {result['flag_count']} anomalous ratio(s) — "
                + "; ".join(result["flags"][:3])
            ),
            evidence=f"composite={result['composite_score']:.3f}",
        )


# ---------------------------------------------------------------------------
# Scheme Lifecycle Analyzer
# ---------------------------------------------------------------------------

@dataclass
class LifecyclePhase:
    """A phase in the fraud scheme lifecycle."""
    name: str
    description: str
    typical_duration_days: tuple[int, int]  # (min, max)
    indicators: list[str] = field(default_factory=list)
    risk_level: float = 0.5


SCHEME_LIFECYCLE_PHASES: list[LifecyclePhase] = [
    LifecyclePhase(
        name="seeding",
        description="Creating fake company profiles and building social proof",
        typical_duration_days=(7, 28),
        indicators=[
            "new company profile",
            "few employees listed",
            "recently created domain",
            "stock photo logo",
            "generic company description",
        ],
        risk_level=0.4,
    ),
    LifecyclePhase(
        name="casting",
        description="Posting jobs across platforms, targeting demographics",
        typical_duration_days=(1, 7),
        indicators=[
            "multiple similar postings",
            "cross-platform duplication",
            "broad targeting",
            "high posting velocity",
            "varied job titles same description",
        ],
        risk_level=0.5,
    ),
    LifecyclePhase(
        name="hooking",
        description="Engaging victims, building rapport and trust",
        typical_duration_days=(1, 14),
        indicators=[
            "rapid response to applicants",
            "informal interview process",
            "excessive friendliness",
            "building urgency",
            "personal contact requests",
        ],
        risk_level=0.6,
    ),
    LifecyclePhase(
        name="extraction",
        description="Collecting fees, personal data, or unpaid labor",
        typical_duration_days=(1, 3),
        indicators=[
            "payment request",
            "sensitive data request",
            "equipment purchase required",
            "training fee",
            "bank account setup",
        ],
        risk_level=0.9,
    ),
    LifecyclePhase(
        name="concealment",
        description="Covering tracks, deleting profiles and evidence",
        typical_duration_days=(0, 1),
        indicators=[
            "profile deletion",
            "posting removal",
            "contact information change",
            "company name change",
            "domain redirect",
        ],
        risk_level=0.8,
    ),
    LifecyclePhase(
        name="adaptation",
        description="Modifying tactics based on detection avoidance learning",
        typical_duration_days=(7, 28),
        indicators=[
            "similar but modified postings",
            "new company name same pattern",
            "evolved language patterns",
            "different platform same scam",
            "keyword substitution",
        ],
        risk_level=0.7,
    ),
]

# Phase detection keywords (in-text indicators)
_PHASE_KEYWORDS: dict[str, re.Pattern] = {
    "seeding": re.compile(
        r"\b(new(ly)? (established|founded|created|launched)|"
        r"(start.?up|startup|emerging|growing) (company|business|firm)|"
        r"(recently|just) (launched|opened|started|incorporated)|"
        r"(expanding|growing) (rapidly|fast|quickly))\b",
        re.IGNORECASE,
    ),
    "casting": re.compile(
        r"\b((hiring|looking for|seeking) (multiple|many|several|numerous|dozens)|"
        r"(mass|bulk|large.scale|nationwide) (hiring|recruitment)|"
        r"(many|multiple|several) (positions|openings|vacancies) (available|open)|"
        r"(immediate|urgent) (openings|positions|vacancies))\b",
        re.IGNORECASE,
    ),
    "extraction": re.compile(
        r"\b((pay|send|wire|transfer|submit) .*(fee|payment|deposit|charge)|"
        r"(training|registration|application|processing|background) fee|"
        r"(provide|submit|send) .*(bank|ssn|social security|credit card|routing)|"
        r"(purchase|buy|order) .*(equipment|kit|materials|software|license))\b",
        re.IGNORECASE,
    ),
    "concealment": re.compile(
        r"\b((do not|don'?t) (share|post|distribute|forward) this|"
        r"(confidential|private|secret) (opportunity|position|offer)|"
        r"(limited|exclusive) (time|access|group)|"
        r"(delete|remove|destroy) .*(after|when|once) (reading|reviewing))\b",
        re.IGNORECASE,
    ),
}


class SchemeLifecycleAnalyzer:
    """Analyze job postings for fraud scheme lifecycle phase indicators.

    Based on the Fraud Management Lifecycle Theory (Wilhelm 2004) and adapted
    for employment scam detection. Maps six lifecycle phases (seeding, casting,
    hooking, extraction, concealment, adaptation) to detectable posting characteristics.

    Early-phase detection enables intervention before victims are harmed.
    """

    def analyze(self, job: JobPosting) -> dict:
        """Determine which lifecycle phase(s) a posting exhibits.

        Returns a dict with detected phases, confidence, and the dominant phase.
        """
        text = f"{job.title} {job.description}"
        if not text.strip():
            return {
                "phases_detected": [],
                "dominant_phase": None,
                "phase_scores": {},
                "composite_risk": 0.0,
            }

        phase_scores: dict[str, float] = {}
        for phase_name, pattern in _PHASE_KEYWORDS.items():
            hits = pattern.findall(text)
            score = min(len(hits) / 2.0, 1.0)
            if score > 0:
                phase_scores[phase_name] = round(score, 3)

        phases_detected = list(phase_scores.keys())

        # Determine dominant phase (highest score)
        dominant_phase = None
        if phase_scores:
            dominant_phase = max(phase_scores, key=phase_scores.get)  # type: ignore[arg-type]

        # Composite risk: weighted by phase risk levels
        composite_risk = 0.0
        if phase_scores:
            phase_lookup = {p.name: p for p in SCHEME_LIFECYCLE_PHASES}
            weighted_sum = sum(
                score * phase_lookup[name].risk_level
                for name, score in phase_scores.items()
                if name in phase_lookup
            )
            max_weighted = sum(p.risk_level for p in SCHEME_LIFECYCLE_PHASES)
            composite_risk = min(weighted_sum / max_weighted, 1.0)

            # Amplify if multiple phases detected (multi-phase = active operation)
            if len(phases_detected) >= 2:
                composite_risk = min(composite_risk * 1.3, 1.0)

        return {
            "phases_detected": phases_detected,
            "dominant_phase": dominant_phase,
            "phase_scores": phase_scores,
            "composite_risk": round(composite_risk, 3),
        }

    def to_signal(self, job: JobPosting) -> ScamSignal | None:
        """Convert lifecycle analysis into a ScamSignal."""
        result = self.analyze(job)
        if not result["phases_detected"]:
            return None
        # Only signal if extraction or concealment phase detected, or 2+ phases
        high_risk_phases = {"extraction", "concealment"}
        has_high_risk = bool(set(result["phases_detected"]) & high_risk_phases)
        if not has_high_risk and len(result["phases_detected"]) < 2:
            return None

        weight = min(0.85, 0.45 + len(result["phases_detected"]) * 0.10)
        return ScamSignal(
            name="scheme_lifecycle",
            category=(
                SignalCategory.RED_FLAG
                if has_high_risk
                else SignalCategory.WARNING
            ),
            weight=round(weight, 2),
            confidence=round(min(0.40 + result["composite_risk"] * 0.5, 0.88), 2),
            detail=(
                f"Scheme Lifecycle: phase(s) {', '.join(result['phases_detected'])} "
                f"(dominant={result['dominant_phase']})"
            ),
            evidence=f"composite_risk={result['composite_risk']:.3f}",
        )


# ---------------------------------------------------------------------------
# Convenience: extract all fraud-handbook signals from a job posting
# ---------------------------------------------------------------------------

# Module-level singletons
_FRAUD_TRIANGLE_SCORER = FraudTriangleScorer()
_FRAUD_DIAMOND_SCORER = FraudDiamondScorer()
_BENFORD_ANALYZER = BenfordAnalyzer()
_LINGUISTIC_FORENSICS = LinguisticForensics()
_NEUTRALIZATION_DETECTOR = NeutralizationDetector()
_BEHAVIORAL_RED_FLAG_SCORER = BehavioralRedFlagScorer()
_FRAUD_RATIO_ANALYZER = FraudRatioAnalyzer()
_SCHEME_LIFECYCLE_ANALYZER = SchemeLifecycleAnalyzer()


def extract_fraud_handbook_signals(job: JobPosting) -> list[ScamSignal]:
    """Run all Fraud Handbook-derived analyses on a job posting.

    Returns a list of ScamSignals (may be empty if nothing triggers).
    """
    signals: list[ScamSignal] = []

    # Fraud Triangle
    triangle_signal = _FRAUD_TRIANGLE_SCORER.to_signal(job)
    if triangle_signal is not None:
        signals.append(triangle_signal)

    # Fraud Diamond (extends Triangle with Capability)
    diamond_signal = _FRAUD_DIAMOND_SCORER.to_signal(job)
    if diamond_signal is not None:
        signals.append(diamond_signal)

    # Linguistic Forensics
    linguistic_signal = _LINGUISTIC_FORENSICS.to_signal(job)
    if linguistic_signal is not None:
        signals.append(linguistic_signal)

    # Neutralization Techniques
    neutralization_signal = _NEUTRALIZATION_DETECTOR.to_signal(job)
    if neutralization_signal is not None:
        signals.append(neutralization_signal)

    # Behavioral Red Flags
    behavioral_signal = _BEHAVIORAL_RED_FLAG_SCORER.to_signal(job)
    if behavioral_signal is not None:
        signals.append(behavioral_signal)

    # Fraud Ratio Analysis
    ratio_signal = _FRAUD_RATIO_ANALYZER.to_signal(job)
    if ratio_signal is not None:
        signals.append(ratio_signal)

    # Scheme Lifecycle
    lifecycle_signal = _SCHEME_LIFECYCLE_ANALYZER.to_signal(job)
    if lifecycle_signal is not None:
        signals.append(lifecycle_signal)

    # Fraud Tree keyword matching
    text = f"{job.title} {job.description}".lower()
    for node in FRAUD_TREE_MAPPING:
        hits = [kw for kw in node.detection_keywords if kw.lower() in text]
        if len(hits) >= 2:
            weight = min(node.risk_weight, 0.92)
            signals.append(ScamSignal(
                name=f"fraud_tree_{node.subcategory}",
                category=SignalCategory.RED_FLAG if weight >= 0.75 else SignalCategory.WARNING,
                weight=round(weight, 2),
                confidence=round(min(0.5 + len(hits) * 0.1, 0.90), 2),
                detail=f"ACFE Fraud Tree [{node.category}/{node.subcategory}]: {node.job_scam_mapping}",
                evidence="; ".join(hits[:3]),
            ))

    return signals
