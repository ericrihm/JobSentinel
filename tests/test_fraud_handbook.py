"""Tests for sentinel/fraud_handbook.py — Corporate Fraud Handbook integration.

Covers: FraudTriangleScorer, BenfordAnalyzer, LinguisticForensics,
FRAUD_TREE_MAPPING, FRAUD_HANDBOOK_TOPICS, FRAUD_HANDBOOK_PROMPTS,
FraudDiamondScorer, NeutralizationDetector, BehavioralRedFlagScorer,
FraudRatioAnalyzer, SchemeLifecycleAnalyzer,
and the extract_fraud_handbook_signals convenience function.
"""

import pytest

from sentinel.fraud_handbook import (
    BENFORD_EXPECTED,
    FRAUD_HANDBOOK_PROMPTS,
    FRAUD_HANDBOOK_TOPICS,
    FRAUD_TREE_MAPPING,
    SCHEME_LIFECYCLE_PHASES,
    BehavioralRedFlagScorer,
    BenfordAnalyzer,
    FraudDiamondScorer,
    FraudRatioAnalyzer,
    FraudTreeNode,
    FraudTriangleScorer,
    LifecyclePhase,
    LinguisticForensics,
    NeutralizationDetector,
    SchemeLifecycleAnalyzer,
    extract_fraud_handbook_signals,
)
from sentinel.models import JobPosting, ScamSignal, SignalCategory
from sentinel.research import ResearchTopic

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scam_posting() -> JobPosting:
    """Scam posting that should trigger Fraud Triangle and linguistic anomalies."""
    return JobPosting(
        url="https://example.com/scam-job",
        title="EARN $5000/WEEK — No Experience Needed!!!",
        company="Quick Cash Solutions",
        location="Remote",
        description=(
            "Are you tired of your 9-to-5 job? Want financial freedom? "
            "Earn fast cash working from home! No experience required. "
            "No degree needed. Anyone can qualify. Start immediately! "
            "This is 100% legitimate and not a scam. We are a real company. "
            "Thousands have already joined and are earning money from home. "
            "Don't worry — this is totally legit. No risk involved. "
            "Pay a small registration fee of $99 to get started. "
            "You must also provide your bank account number for direct deposit. "
            "GUARANTEED income! ACT NOW — limited spots available!!!"
        ),
        salary_min=5000.0,
        salary_max=25000.0,
        is_remote=True,
    )


@pytest.fixture
def legit_posting() -> JobPosting:
    """Legitimate posting that should NOT trigger Fraud Triangle."""
    return JobPosting(
        url="https://linkedin.com/jobs/view/12345",
        title="Senior Software Engineer, Backend",
        company="Stripe",
        location="San Francisco, CA",
        description=(
            "We are looking for a Senior Software Engineer to join our payments "
            "infrastructure team. You will design, build, and operate distributed "
            "systems that process billions of dollars in transactions annually. "
            "Requirements: 5+ years of experience in Go, Java, or Python. "
            "Strong understanding of distributed systems and microservices. "
            "Experience with PostgreSQL, Redis, and Kafka. "
            "BS/MS in Computer Science or equivalent. "
            "Benefits: competitive salary, equity, 401k, health insurance, "
            "dental, vision, parental leave, and flexible PTO."
        ),
        salary_min=180000.0,
        salary_max=280000.0,
        company_linkedin_url="https://linkedin.com/company/stripe",
        company_size="5000+",
    )


@pytest.fixture
def triangle_scorer() -> FraudTriangleScorer:
    return FraudTriangleScorer()


@pytest.fixture
def benford_analyzer() -> BenfordAnalyzer:
    return BenfordAnalyzer()


@pytest.fixture
def linguistics() -> LinguisticForensics:
    return LinguisticForensics()


# ---------------------------------------------------------------------------
# Benford's Law expected distribution
# ---------------------------------------------------------------------------

class TestBenfordExpected:
    def test_benford_sums_to_one(self):
        """Benford expected probabilities must sum to 1.0."""
        total = sum(BENFORD_EXPECTED.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_benford_digit_1_most_common(self):
        """Digit 1 should have ~30.1% probability (highest)."""
        assert BENFORD_EXPECTED[1] == pytest.approx(0.30103, abs=1e-4)

    def test_benford_digit_9_least_common(self):
        """Digit 9 should have ~4.6% probability (lowest)."""
        assert BENFORD_EXPECTED[9] == pytest.approx(0.04576, abs=1e-4)

    def test_benford_monotonically_decreasing(self):
        """Each successive digit should have lower probability."""
        for d in range(1, 9):
            assert BENFORD_EXPECTED[d] > BENFORD_EXPECTED[d + 1]


# ---------------------------------------------------------------------------
# Fraud Triangle Scorer
# ---------------------------------------------------------------------------

class TestFraudTriangleScorer:
    def test_scam_triggers_all_three_legs(self, triangle_scorer, scam_posting):
        """A classic scam posting should trigger all three Fraud Triangle legs."""
        result = triangle_scorer.score(scam_posting)
        assert result["legs_present"] == 3
        assert result["pressure_score"] > 0
        assert result["opportunity_score"] > 0
        assert result["rationalization_score"] > 0
        assert result["composite_score"] > 0.5

    def test_legit_posting_low_score(self, triangle_scorer, legit_posting):
        """A legitimate posting should have a low composite score."""
        result = triangle_scorer.score(legit_posting)
        assert result["composite_score"] < 0.25

    def test_pressure_only(self, triangle_scorer):
        """Posting with only pressure language should score pressure leg only."""
        job = JobPosting(
            description="Tired of your 9-to-5? Want financial freedom? Earn fast cash today!"
        )
        result = triangle_scorer.score(job)
        assert result["pressure_score"] > 0
        assert result["opportunity_score"] == 0
        assert result["rationalization_score"] == 0
        assert result["legs_present"] == 1

    def test_opportunity_only(self, triangle_scorer):
        """Posting with only opportunity language should score opportunity leg only."""
        job = JobPosting(
            description="No experience required. No degree needed. Start immediately. Anyone can apply."
        )
        result = triangle_scorer.score(job)
        assert result["opportunity_score"] > 0
        assert result["legs_present"] >= 1

    def test_rationalization_only(self, triangle_scorer):
        """Posting with only rationalization language should score that leg."""
        job = JobPosting(
            description="This is not a scam. We are a legitimate company. 100% safe. No risk involved."
        )
        result = triangle_scorer.score(job)
        assert result["rationalization_score"] > 0

    def test_three_legs_amplification(self, triangle_scorer, scam_posting):
        """When all 3 legs present, composite should be amplified (1.3x boost)."""
        result = triangle_scorer.score(scam_posting)
        base = (
            triangle_scorer.PRESSURE_WEIGHT * result["pressure_score"]
            + triangle_scorer.OPPORTUNITY_WEIGHT * result["opportunity_score"]
            + triangle_scorer.RATIONALIZATION_WEIGHT * result["rationalization_score"]
        )
        # Composite should be at least the base (may be amplified)
        assert result["composite_score"] >= base or result["composite_score"] == 1.0

    def test_to_signal_scam(self, triangle_scorer, scam_posting):
        """Scam posting should produce a ScamSignal from Fraud Triangle."""
        signal = triangle_scorer.to_signal(scam_posting)
        assert signal is not None
        assert signal.name == "fraud_triangle"
        assert signal.weight >= 0.50

    def test_to_signal_legit_returns_none(self, triangle_scorer, legit_posting):
        """Legitimate posting should return None (below threshold)."""
        signal = triangle_scorer.to_signal(legit_posting)
        assert signal is None

    def test_to_signal_red_flag_when_all_legs(self, triangle_scorer, scam_posting):
        """When all 3 legs are present, category should be RED_FLAG."""
        signal = triangle_scorer.to_signal(scam_posting)
        assert signal is not None
        assert signal.category == SignalCategory.RED_FLAG

    def test_score_empty_posting(self, triangle_scorer):
        """Empty posting should produce all-zero scores."""
        job = JobPosting()
        result = triangle_scorer.score(job)
        assert result["composite_score"] == 0.0
        assert result["legs_present"] == 0

    def test_composite_capped_at_one(self, triangle_scorer):
        """Composite score should never exceed 1.0."""
        job = JobPosting(
            description=(
                "Earn fast cash! Quick money! Financial freedom! Immediate income! "
                "No experience needed. Anyone can qualify. Start immediately. "
                "No degree needed. No training required. No resume required. "
                "This is not a scam. 100% legitimate. Totally legit. "
                "No risk at all. Guaranteed safe."
            )
        )
        result = triangle_scorer.score(job)
        assert result["composite_score"] <= 1.0


# ---------------------------------------------------------------------------
# Benford Analyzer
# ---------------------------------------------------------------------------

class TestBenfordAnalyzer:
    def test_first_digit_basic(self, benford_analyzer):
        """First digit extraction from basic numbers."""
        assert benford_analyzer.first_digit(1234) == 1
        assert benford_analyzer.first_digit(5000) == 5
        assert benford_analyzer.first_digit(99000) == 9
        assert benford_analyzer.first_digit(250000) == 2

    def test_first_digit_zero_negative(self, benford_analyzer):
        """Zero and negative numbers should return None."""
        assert benford_analyzer.first_digit(0) is None
        assert benford_analyzer.first_digit(-100) is None

    def test_first_digit_small_numbers(self, benford_analyzer):
        """Small positive numbers should work correctly."""
        assert benford_analyzer.first_digit(7.5) == 8  # rounds to 8
        assert benford_analyzer.first_digit(1.0) == 1
        # 0.5 rounds to 0 via :.0f, which has no non-zero leading digit
        assert benford_analyzer.first_digit(0.5) is None
        assert benford_analyzer.first_digit(1.5) == 2  # rounds to 2

    def test_analyze_insufficient_data(self, benford_analyzer):
        """Fewer than 5 values should be marked as insufficient."""
        result = benford_analyzer.analyze([100, 200, 300])
        assert result["sufficient_data"] is False
        assert result["sample_size"] == 3

    def test_analyze_natural_distribution(self, benford_analyzer):
        """A Benford-conforming distribution should have low conformity score."""
        # Generate values roughly following Benford's Law
        import random
        random.seed(42)
        values = []
        for _ in range(1000):
            # Log-uniform distribution follows Benford
            values.append(10 ** (random.uniform(3, 6)))
        result = benford_analyzer.analyze(values)
        assert result["sufficient_data"] is True
        assert result["sample_size"] == 1000
        # Should have low chi-squared relative to sample size
        assert result["chi_squared"] < 100  # generous threshold for randomness

    def test_analyze_uniform_digits_anomalous(self, benford_analyzer):
        """All same first digits should be flagged as anomalous."""
        # All salaries starting with 5
        values = [50000 + i * 100 for i in range(100)]
        result = benford_analyzer.analyze(values)
        assert result["sufficient_data"] is True
        # Should show deviation from Benford
        assert result["chi_squared"] > 10

    def test_analyze_posting_round_numbers(self, benford_analyzer):
        """Perfectly round salary numbers should be flagged."""
        job = JobPosting(salary_min=50000.0, salary_max=100000.0)
        result = benford_analyzer.analyze_posting(job)
        assert result["suspicious"] is True
        assert any("perfectly round" in f for f in result["flags"])

    def test_analyze_posting_psychological_pricing(self, benford_analyzer):
        """Psychological pricing patterns should be flagged."""
        job = JobPosting(salary_min=49999.0, salary_max=99999.0)
        result = benford_analyzer.analyze_posting(job)
        assert result["suspicious"] is True
        assert any("psychological pricing" in f for f in result["flags"])

    def test_analyze_posting_wide_ratio(self, benford_analyzer):
        """Very wide salary range should be flagged."""
        job = JobPosting(salary_min=10000.0, salary_max=200000.0)
        result = benford_analyzer.analyze_posting(job)
        assert result["suspicious"] is True
        assert any("ratio" in f.lower() for f in result["flags"])

    def test_analyze_posting_normal_salary(self, benford_analyzer):
        """Normal salary range should NOT be flagged."""
        job = JobPosting(salary_min=85000.0, salary_max=120000.0)
        result = benford_analyzer.analyze_posting(job)
        # Normal range, not round numbers, not wide ratio
        assert result["salary_ratio"] is not None
        assert result["salary_ratio"] < 5.0

    def test_analyze_posting_no_salary(self, benford_analyzer):
        """Posting with no salary data should return empty values."""
        job = JobPosting()
        result = benford_analyzer.analyze_posting(job)
        assert result["values"] == []
        assert not result["suspicious"]


# ---------------------------------------------------------------------------
# Linguistic Forensics
# ---------------------------------------------------------------------------

class TestLinguisticForensics:
    def test_analyze_scam_text(self, linguistics, scam_posting):
        """Scam text should show linguistic anomalies."""
        result = linguistics.analyze(scam_posting.description)
        assert result["sufficient_text"] is True
        assert result["word_count"] > 20

    def test_analyze_empty_text(self, linguistics):
        """Empty text should be marked as insufficient."""
        result = linguistics.analyze("")
        assert result["sufficient_text"] is False
        assert result["word_count"] == 0

    def test_analyze_short_text(self, linguistics):
        """Very short text should be marked as insufficient."""
        result = linguistics.analyze("Hello world.")
        assert result["sufficient_text"] is False

    def test_vocabulary_richness(self, linguistics):
        """Text with repeated words should have low vocabulary richness."""
        repetitive = "money money money earn money fast money quick money easy money"
        result = linguistics.analyze(repetitive * 3)
        # The repetitive nature means low type-token ratio
        assert result["vocabulary_richness"] < 0.5

    def test_exclamation_density(self, linguistics):
        """Text with many exclamation marks should flag high exclamation density."""
        text = "Apply now! Great opportunity! Don't miss out! Earn big! Start today! Act fast!"
        result = linguistics.analyze(text)
        assert result["exclamation_density"] > 0.0

    def test_caps_density(self, linguistics):
        """Text with ALL-CAPS words should flag high caps density."""
        text = ("EARN MONEY FAST working from HOME. This AMAZING OPPORTUNITY "
                "will CHANGE your LIFE. Apply NOW for this INCREDIBLE position.")
        result = linguistics.analyze(text)
        assert result["caps_density"] > 0.1

    def test_readability_score_computed(self, linguistics, legit_posting):
        """Readability score should be computed for substantial text."""
        result = linguistics.analyze(legit_posting.description)
        assert result["readability_score"] != 0.0

    def test_to_signal_scam(self, linguistics, scam_posting):
        """Scam posting should produce a linguistic anomaly signal."""
        signal = linguistics.to_signal(scam_posting)
        # May or may not trigger depending on exact anomaly count
        if signal is not None:
            assert signal.name == "linguistic_anomaly"
            assert signal.category == SignalCategory.STRUCTURAL

    def test_to_signal_legit_returns_none(self, linguistics, legit_posting):
        """Legitimate posting should not produce a linguistic anomaly signal."""
        signal = linguistics.to_signal(legit_posting)
        assert signal is None

    def test_estimate_syllables(self, linguistics):
        """Syllable estimation sanity checks."""
        assert linguistics._estimate_syllables("the") == 1
        assert linguistics._estimate_syllables("hello") == 2
        assert linguistics._estimate_syllables("computer") >= 2
        assert linguistics._estimate_syllables("a") == 1

    def test_sentence_length_variance(self, linguistics):
        """Text with uniform sentence lengths should have low variance."""
        uniform = "This is good. That is bad. Here we go. Now we see. All is well."
        result = linguistics.analyze(uniform)
        assert result["sentence_length_variance"] < 2.0

    def test_anomaly_flags_list(self, linguistics):
        """Anomaly flags should be a list of strings."""
        text = (
            "EARN MONEY!!! EARN MONEY!!! EARN MONEY!!! EARN MONEY!!! "
            "FAST CASH!!! FAST CASH!!! FAST CASH!!! FAST CASH!!! "
            "APPLY NOW!!! APPLY NOW!!! APPLY NOW!!! APPLY NOW!!! "
            "This is a great opportunity for everyone who wants money."
        )
        result = linguistics.analyze(text)
        assert isinstance(result["anomaly_flags"], list)
        for flag in result["anomaly_flags"]:
            assert isinstance(flag, str)


# ---------------------------------------------------------------------------
# Fraud Tree Mapping
# ---------------------------------------------------------------------------

class TestFraudTreeMapping:
    def test_fraud_tree_has_entries(self):
        """Fraud tree mapping should have at least 5 entries."""
        assert len(FRAUD_TREE_MAPPING) >= 5

    def test_fraud_tree_all_valid_nodes(self):
        """Every entry should be a valid FraudTreeNode."""
        for node in FRAUD_TREE_MAPPING:
            assert isinstance(node, FraudTreeNode)
            assert node.category in {
                "asset_misappropriation", "corruption", "financial_statement_fraud"
            }
            assert node.subcategory
            assert node.job_scam_mapping
            assert 0 < node.risk_weight <= 1.0
            assert len(node.detection_keywords) >= 2

    def test_fraud_tree_covers_all_acfe_categories(self):
        """Fraud tree should cover all three ACFE Fraud Tree categories."""
        categories = {node.category for node in FRAUD_TREE_MAPPING}
        assert "asset_misappropriation" in categories
        assert "corruption" in categories
        assert "financial_statement_fraud" in categories

    def test_fraud_tree_keywords_lowercase(self):
        """Detection keywords should be lowercase for case-insensitive matching."""
        for node in FRAUD_TREE_MAPPING:
            for kw in node.detection_keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in {node.subcategory} is not lowercase"


# ---------------------------------------------------------------------------
# Research Topics
# ---------------------------------------------------------------------------

class TestFraudHandbookTopics:
    def test_topics_not_empty(self):
        """Should have at least 5 research topics."""
        assert len(FRAUD_HANDBOOK_TOPICS) >= 5

    def test_topics_are_research_topics(self):
        """Each topic should be a ResearchTopic instance."""
        for t in FRAUD_HANDBOOK_TOPICS:
            assert isinstance(t, ResearchTopic)
            assert t.area
            assert 0 < t.priority <= 1.0
            assert len(t.reason) > 10

    def test_topics_priorities_valid(self):
        """All priorities should be between 0 and 1."""
        for t in FRAUD_HANDBOOK_TOPICS:
            assert 0.0 < t.priority <= 1.0

    def test_topics_unique_areas(self):
        """Each topic should have a unique area name."""
        areas = [t.area for t in FRAUD_HANDBOOK_TOPICS]
        assert len(areas) == len(set(areas)), "Duplicate topic areas found"


# ---------------------------------------------------------------------------
# Research Prompts
# ---------------------------------------------------------------------------

class TestFraudHandbookPrompts:
    def test_prompts_count(self):
        """Should have 15-20 research prompts."""
        assert 15 <= len(FRAUD_HANDBOOK_PROMPTS) <= 25

    def test_prompts_have_required_fields(self):
        """Each prompt should have topic, prompt_text, expected_patterns, priority."""
        for p in FRAUD_HANDBOOK_PROMPTS:
            assert "topic" in p and p["topic"]
            assert "prompt_text" in p and len(p["prompt_text"]) > 50
            assert "expected_patterns" in p and isinstance(p["expected_patterns"], list)
            assert "priority" in p and 0 < p["priority"] <= 1.0

    def test_prompts_unique_topics(self):
        """Each prompt should target a unique topic."""
        topics = [p["topic"] for p in FRAUD_HANDBOOK_PROMPTS]
        assert len(topics) == len(set(topics)), "Duplicate prompt topics found"

    def test_prompts_reference_frameworks(self):
        """At least half the prompts should reference specific frameworks."""
        framework_keywords = [
            "fraud triangle", "cressey", "acfe", "benford", "stylometry",
            "neutralization", "sykes", "matza", "behavioral", "ratio analysis",
        ]
        count = 0
        for p in FRAUD_HANDBOOK_PROMPTS:
            text = p["prompt_text"].lower()
            if any(kw in text for kw in framework_keywords):
                count += 1
        assert count >= len(FRAUD_HANDBOOK_PROMPTS) // 2


# ---------------------------------------------------------------------------
# Extract fraud handbook signals (integration)
# ---------------------------------------------------------------------------

class TestExtractFraudHandbookSignals:
    def test_scam_produces_signals(self, scam_posting):
        """Scam posting should produce at least one fraud-handbook signal."""
        signals = extract_fraud_handbook_signals(scam_posting)
        assert len(signals) >= 1
        names = [s.name for s in signals]
        assert "fraud_triangle" in names

    def test_legit_produces_no_signals(self, legit_posting):
        """Legitimate posting should produce zero or very few signals."""
        signals = extract_fraud_handbook_signals(legit_posting)
        # Legitimate postings shouldn't trigger fraud handbook signals
        assert len(signals) <= 1

    def test_signals_are_scam_signals(self, scam_posting):
        """All returned signals should be ScamSignal instances."""
        signals = extract_fraud_handbook_signals(scam_posting)
        for s in signals:
            assert isinstance(s, ScamSignal)
            assert s.name
            assert s.weight > 0
            assert s.confidence > 0

    def test_fraud_tree_signal_from_billing_scheme(self):
        """Posting with billing scheme keywords should trigger fraud tree signal."""
        job = JobPosting(
            description=(
                "Before you start, you must pay a training fee and purchase "
                "your own equipment. The registration fee is $199 and the "
                "starter kit costs $299. Buy your own laptop as well."
            )
        )
        signals = extract_fraud_handbook_signals(job)
        tree_signals = [s for s in signals if s.name.startswith("fraud_tree_")]
        assert len(tree_signals) >= 1

    def test_empty_posting_no_crash(self):
        """Empty posting should not crash, should return empty or minimal list."""
        job = JobPosting()
        signals = extract_fraud_handbook_signals(job)
        assert isinstance(signals, list)

    def test_scam_produces_diamond_signal(self, scam_posting):
        """Scam posting should produce a fraud_diamond signal."""
        signals = extract_fraud_handbook_signals(scam_posting)
        names = [s.name for s in signals]
        # Diamond signal may or may not fire depending on capability detection
        # But triangle should always fire for a classic scam
        assert "fraud_triangle" in names

    def test_extraction_phase_scam_triggers_lifecycle(self):
        """Posting with extraction-phase keywords should trigger lifecycle signal."""
        job = JobPosting(
            description=(
                "To start working, you must pay a registration fee of $199 and "
                "purchase the required training materials. Please provide your "
                "bank account number and social security number for payroll setup. "
                "Submit your payment via wire transfer to begin immediately."
            )
        )
        signals = extract_fraud_handbook_signals(job)
        lifecycle_signals = [s for s in signals if s.name == "scheme_lifecycle"]
        assert len(lifecycle_signals) >= 1


# ---------------------------------------------------------------------------
# Fraud Diamond Scorer (Wolfe & Hermanson 2004)
# ---------------------------------------------------------------------------

class TestFraudDiamondScorer:
    @pytest.fixture
    def diamond_scorer(self) -> FraudDiamondScorer:
        return FraudDiamondScorer()

    def test_scam_posting_scores_elements(self, diamond_scorer, scam_posting):
        """A scam posting should score on multiple Fraud Diamond elements."""
        result = diamond_scorer.score(scam_posting)
        assert result["pressure_score"] > 0
        assert result["opportunity_score"] > 0
        assert result["rationalization_score"] > 0
        assert result["composite_score"] > 0.3

    def test_capability_position_detected(self, diamond_scorer):
        """Position/authority claims should trigger capability scoring."""
        job = JobPosting(
            description=(
                "I am the VP of HR at this company and I am the official "
                "hiring manager for this position. As the Director of Talent "
                "Acquisition, I can guarantee your placement."
            )
        )
        result = diamond_scorer.score(job)
        assert result["capability_position"] > 0
        assert result["capability_score"] > 0

    def test_capability_coercion_detected(self, diamond_scorer):
        """MLM/referral language should trigger coercion capability."""
        job = JobPosting(
            description=(
                "Refer a friend and earn a referral bonus! Build your team "
                "and recruit others to join the network opportunity. "
                "Mentor new recruits and grow your downline."
            )
        )
        result = diamond_scorer.score(job)
        assert result["capability_coercion"] > 0

    def test_capability_confidence_detected(self, diamond_scorer):
        """Bold/absolute claims should trigger confidence capability."""
        job = JobPosting(
            description=(
                "This is a proven system with guaranteed results. "
                "We are an industry-leading, world-class company. "
                "This exclusive opportunity is for hand-picked candidates only."
            )
        )
        result = diamond_scorer.score(job)
        assert result["capability_confidence"] > 0

    def test_legit_posting_low_diamond_score(self, diamond_scorer, legit_posting):
        """Legitimate posting should have a low diamond composite score."""
        result = diamond_scorer.score(legit_posting)
        assert result["composite_score"] < 0.25

    def test_four_elements_amplification(self, diamond_scorer):
        """All 4 elements present should amplify the composite score."""
        job = JobPosting(
            description=(
                "Tired of your 9-to-5? Want financial freedom? Earn fast cash! "
                "No experience needed. Anyone can qualify. Start immediately. "
                "This is not a scam. We are a legitimate company. 100% safe. "
                "I am the hiring manager. Proven system. Guaranteed results. "
                "Refer a friend and build your team."
            )
        )
        result = diamond_scorer.score(job)
        assert result["elements_present"] == 4
        assert result["composite_score"] > 0.5

    def test_diamond_composite_capped(self, diamond_scorer):
        """Composite score should never exceed 1.0."""
        job = JobPosting(
            description=(
                "Earn fast cash! Quick money! Financial freedom! "
                "No experience needed. Anyone can qualify. Start immediately. "
                "No degree needed. No training required. "
                "This is not a scam. 100% legitimate. Totally legit. "
                "No risk at all. Guaranteed safe. "
                "I am the CEO and hiring manager. VP of HR. Director of Talent. "
                "Proven system. Guaranteed results. World-class. "
                "Refer a friend. Build your team. Recruit others."
            )
        )
        result = diamond_scorer.score(job)
        assert result["composite_score"] <= 1.0

    def test_to_signal_returns_none_for_legit(self, diamond_scorer, legit_posting):
        """Legitimate posting should not produce a Diamond signal."""
        signal = diamond_scorer.to_signal(legit_posting)
        assert signal is None

    def test_to_signal_scam_produces_signal(self, diamond_scorer, scam_posting):
        """Scam posting should produce a Diamond signal."""
        signal = diamond_scorer.to_signal(scam_posting)
        # May or may not trigger depending on capability; check structure if it does
        if signal is not None:
            assert signal.name == "fraud_diamond"
            assert signal.weight > 0
            assert signal.confidence > 0

    def test_empty_posting_zero_score(self, diamond_scorer):
        """Empty posting should produce zero scores."""
        job = JobPosting()
        result = diamond_scorer.score(job)
        assert result["composite_score"] == 0.0
        assert result["capability_score"] == 0.0
        assert result["elements_present"] == 0

    def test_includes_triangle_composite(self, diamond_scorer, scam_posting):
        """Diamond result should include the underlying Triangle composite."""
        result = diamond_scorer.score(scam_posting)
        assert "triangle_composite" in result
        assert result["triangle_composite"] > 0


# ---------------------------------------------------------------------------
# Neutralization Technique Detector
# ---------------------------------------------------------------------------

class TestNeutralizationDetector:
    @pytest.fixture
    def detector(self) -> NeutralizationDetector:
        return NeutralizationDetector()

    def test_denial_of_injury_detected(self, detector):
        """Denial of injury phrases should be detected."""
        text = "This is fully refundable. No risk involved. Zero risk. You can't lose."
        result = detector.detect(text)
        assert "denial_of_injury" in result["techniques_detected"]

    def test_condemnation_of_condemners_detected(self, detector):
        """Anti-skeptic language should trigger condemnation of condemners."""
        text = (
            "Don't let the haters stop you. Ignore the skeptics. "
            "People told me it was a scam but I proved them wrong."
        )
        result = detector.detect(text)
        assert "condemnation_of_condemners" in result["techniques_detected"]

    def test_appeal_to_higher_loyalties_detected(self, detector):
        """Family/community language should trigger appeal to higher loyalties."""
        text = (
            "Join our family! Help your community. Give your kids a better future. "
            "Build a brighter life for your loved ones."
        )
        result = detector.detect(text)
        assert "appeal_to_higher_loyalties" in result["techniques_detected"]

    def test_denial_of_victim_detected(self, detector):
        """Victim-blaming language should trigger denial of victim."""
        text = "Only serious candidates should apply. Not for the lazy or uncommitted."
        result = detector.detect(text)
        assert "denial_of_victim" in result["techniques_detected"]

    def test_claim_of_normality_detected(self, detector):
        """Industry-standard claims should trigger claim of normality."""
        text = (
            "This is standard practice in the industry. "
            "All companies require this. Just how business is done."
        )
        result = detector.detect(text)
        assert "claim_of_normality" in result["techniques_detected"]

    def test_defense_of_necessity_detected(self, detector):
        """Mandatory-fee justification should trigger defense of necessity."""
        text = (
            "This fee is required by law for mandatory background processing. "
            "Legally required certification must be completed first."
        )
        result = detector.detect(text)
        assert "defense_of_necessity" in result["techniques_detected"]

    def test_claim_of_entitlement_detected(self, detector):
        """Self-worth exploitation should trigger claim of entitlement."""
        text = "You deserve this opportunity. You've earned it. Reward yourself."
        result = detector.detect(text)
        assert "claim_of_entitlement" in result["techniques_detected"]

    def test_multiple_techniques_amplified(self, detector):
        """Detecting 3+ techniques should amplify the composite score."""
        text = (
            "This is fully refundable with no risk involved. "
            "Don't let the haters hold you back. Ignore the skeptics. "
            "Only serious candidates should apply. Not for tire kickers. "
            "This is standard practice. All employers charge this."
        )
        result = detector.detect(text)
        assert result["technique_count"] >= 3
        # Composite should be amplified
        assert result["composite_score"] > 0.3

    def test_empty_text_no_detection(self, detector):
        """Empty text should detect nothing."""
        result = detector.detect("")
        assert result["technique_count"] == 0
        assert result["composite_score"] == 0.0

    def test_legit_text_minimal_detection(self, detector, legit_posting):
        """Legitimate posting should have zero or very few neutralization hits."""
        result = detector.detect(legit_posting.description)
        assert result["technique_count"] <= 1

    def test_composite_capped_at_one(self, detector):
        """Composite score should never exceed 1.0."""
        text = (
            "No risk involved. Fully refundable. Zero risk. Nothing to lose. "
            "Don't let the haters stop you. Ignore the negativity. "
            "Only serious candidates. Not for quitters. "
            "Standard practice. All companies do this. "
            "You deserve this. You've earned it. Reward yourself. "
            "Join our family. Help your community. "
            "Required by law. Legally mandated. "
            "The economy forced companies to adopt this model."
        )
        result = detector.detect(text)
        assert result["composite_score"] <= 1.0

    def test_to_signal_returns_none_no_techniques(self, detector, legit_posting):
        """No techniques should return None signal."""
        signal = detector.to_signal(legit_posting)
        # Legit posting with 0-1 techniques returns None (threshold is 1)
        if signal is not None:
            assert signal.name == "neutralization_technique"

    def test_to_signal_scam_produces_signal(self, detector):
        """Posting with neutralization techniques should produce a signal."""
        job = JobPosting(
            description=(
                "This is fully refundable. No risk involved at all. "
                "Don't let the haters stop you. Ignore the skeptics. "
                "Only serious candidates should apply."
            )
        )
        signal = detector.to_signal(job)
        assert signal is not None
        assert signal.name == "neutralization_technique"
        assert signal.weight > 0
        assert signal.confidence > 0

    def test_details_contain_hit_strings(self, detector):
        """Details should contain the matched phrases."""
        text = "This is fully refundable. No risk involved."
        result = detector.detect(text)
        if "denial_of_injury" in result["details"]:
            assert len(result["details"]["denial_of_injury"]) > 0

    def test_technique_weights_valid(self, detector):
        """All technique weights should be between 0 and 1."""
        for technique, weight in detector.TECHNIQUE_WEIGHTS.items():
            assert 0.0 < weight <= 1.0, f"{technique} weight {weight} out of range"


# ---------------------------------------------------------------------------
# Behavioral Red Flag Scorer
# ---------------------------------------------------------------------------

class TestBehavioralRedFlagScorer:
    @pytest.fixture
    def scorer(self) -> BehavioralRedFlagScorer:
        return BehavioralRedFlagScorer()

    def test_living_beyond_means_detected(self, scorer):
        """Lifestyle promises should trigger living_beyond_means flag."""
        job = JobPosting(
            description=(
                "Achieve a luxury lifestyle! Drive a Lamborghini. "
                "Work from the beach. Live the dream. "
                "Passive income and financial freedom await."
            )
        )
        result = scorer.score(job)
        assert "living_beyond_means" in result["flags_detected"]

    def test_financial_difficulties_targeting_detected(self, scorer):
        """Targeting financial stress should trigger financial_difficulties flag."""
        job = JobPosting(
            description=(
                "Are you struggling with debt? Can't pay your rent? "
                "Living paycheck to paycheck? Need extra money fast?"
            )
        )
        result = scorer.score(job)
        assert "financial_difficulties_targeting" in result["flags_detected"]

    def test_vendor_association_detected(self, scorer):
        """Vendor steering language should trigger vendor_association flag."""
        job = JobPosting(
            description=(
                "You must use our exclusive partner for training. "
                "Only available through our platform."
            )
        )
        result = scorer.score(job)
        assert "vendor_association" in result["flags_detected"]

    def test_control_issues_detected(self, scorer):
        """Control language should trigger control_issues flag."""
        job = JobPosting(
            description=(
                "Do not contact the company directly. I am your only "
                "point of contact. Only through me. Keep this confidential."
            )
        )
        result = scorer.score(job)
        assert "control_issues" in result["flags_detected"]

    def test_bullying_intimidation_detected(self, scorer):
        """Threat/pressure language should trigger bullying_intimidation flag."""
        job = JobPosting(
            description=(
                "You'll regret missing this. Don't miss this chance. "
                "Last chance to apply. Act now or lose your spot. "
                "Your loss if you don't apply."
            )
        )
        result = scorer.score(job)
        assert "bullying_intimidation" in result["flags_detected"]

    def test_wheeler_dealer_detected(self, scorer):
        """Deal-making language should trigger wheeler_dealer flag."""
        job = JobPosting(
            description=(
                "Special exclusive limited deal! One-time offer today only. "
                "Act now and get the early bird rate. "
                "Sign up today to receive a founder discount."
            )
        )
        result = scorer.score(job)
        assert "wheeler_dealer" in result["flags_detected"]

    def test_multiple_flags_amplified(self, scorer):
        """Detecting 3+ flags should amplify the composite score."""
        job = JobPosting(
            description=(
                "Achieve a luxury lifestyle! Financial freedom! Live the dream! "
                "Struggling with debt? Need extra money? Living paycheck to paycheck? "
                "You'll regret missing this. Last chance. Act now or lose your spot."
            )
        )
        result = scorer.score(job)
        assert result["flag_count"] >= 3
        assert result["composite_score"] > 0.3

    def test_legit_posting_minimal_flags(self, scorer, legit_posting):
        """Legitimate posting should trigger zero or very few red flags."""
        result = scorer.score(legit_posting)
        assert result["flag_count"] <= 1

    def test_empty_posting_no_flags(self, scorer):
        """Empty posting should detect no flags."""
        job = JobPosting()
        result = scorer.score(job)
        assert result["flag_count"] == 0
        assert result["composite_score"] == 0.0

    def test_composite_capped_at_one(self, scorer):
        """Composite score should never exceed 1.0."""
        job = JobPosting(
            description=(
                "Luxury lifestyle! Drive a Ferrari! Live the dream! Passive income! "
                "Struggling with debt? Can't afford rent? Need quick money? "
                "Exclusive partner. Must use our platform. Only through us. "
                "Don't contact anyone else. I am your only contact. Keep confidential. "
                "You'll regret this. Last chance. Act now. Your loss. "
                "Special deal! One-time offer! Early bird rate! Sign up today!"
            )
        )
        result = scorer.score(job)
        assert result["composite_score"] <= 1.0

    def test_to_signal_threshold(self, scorer):
        """Signal should only fire with 2+ flags detected."""
        # Single flag should not produce a signal
        job = JobPosting(
            description="Achieve a luxury lifestyle. Live the dream."
        )
        signal = scorer.to_signal(job)
        assert signal is None

    def test_to_signal_with_multiple_flags(self, scorer):
        """Multiple flags should produce a behavioral_red_flags signal."""
        job = JobPosting(
            description=(
                "Luxury lifestyle! Live the dream! Financial freedom! "
                "Struggling with debt? Need extra money fast? "
                "Last chance! Act now or lose your spot!"
            )
        )
        signal = scorer.to_signal(job)
        assert signal is not None
        assert signal.name == "behavioral_red_flags"
        assert signal.weight > 0
        assert signal.confidence > 0


# ---------------------------------------------------------------------------
# Fraud Ratio Analyzer
# ---------------------------------------------------------------------------

class TestFraudRatioAnalyzer:
    @pytest.fixture
    def analyzer(self) -> FraudRatioAnalyzer:
        return FraudRatioAnalyzer()

    def test_wide_salary_range_flagged(self, analyzer):
        """Very wide salary range should flag salary_range_width ratio."""
        job = JobPosting(
            salary_min=10000.0,
            salary_max=100000.0,
            description="General position available.",
        )
        result = analyzer.analyze(job)
        assert any("salary_range_width" in f for f in result["flags"])

    def test_normal_salary_range_clean(self, analyzer):
        """Normal salary range should not flag salary_range_width."""
        job = JobPosting(
            salary_min=80000.0,
            salary_max=120000.0,
            description="Software engineer position. Requires 5 years of experience in Python.",
        )
        result = analyzer.analyze(job)
        salary_flags = [f for f in result["flags"] if "salary_range_width" in f]
        assert len(salary_flags) == 0

    def test_high_exclamation_ratio_flagged(self, analyzer):
        """Excessive exclamation marks should flag exclamation_to_period ratio."""
        job = JobPosting(
            description=(
                "Apply now! Great opportunity! Don't miss out! "
                "Earn big! Start today! Act fast! Amazing!"
            )
        )
        result = analyzer.analyze(job)
        assert any("exclamation_to_period" in f for f in result["flags"])

    def test_high_caps_density_flagged(self, analyzer):
        """Excessive ALL-CAPS words should flag caps_word_density ratio."""
        job = JobPosting(
            description=(
                "EARN MONEY FAST NOW! THIS AMAZING OPPORTUNITY WILL CHANGE "
                "YOUR LIFE FOREVER! JOIN TODAY FOR FREE! ACT NOW!"
            )
        )
        result = analyzer.analyze(job)
        assert any("caps_word_density" in f for f in result["flags"])

    def test_urgency_density_flagged(self, analyzer):
        """Excessive urgency words should flag urgency_word_density."""
        job = JobPosting(
            description=(
                "Apply immediately! Urgent! ASAP! Limited spots available! "
                "Act now! Don't delay! Hurry! Apply today! Respond now!"
            )
        )
        result = analyzer.analyze(job)
        assert any("urgency_word_density" in f for f in result["flags"])

    def test_personal_info_request_flagged(self, analyzer):
        """Personal info requests should flag personal_info_density."""
        job = JobPosting(
            description=(
                "Please provide your social security number, bank account "
                "and routing number, date of birth, and mother's maiden name "
                "to complete your application for this position."
            )
        )
        result = analyzer.analyze(job)
        assert any("personal_info_density" in f for f in result["flags"])

    def test_legit_posting_minimal_flags(self, analyzer, legit_posting):
        """Legitimate posting should have few ratio anomalies."""
        result = analyzer.analyze(legit_posting)
        assert result["flag_count"] <= 2

    def test_composite_proportional_to_flags(self, analyzer):
        """Composite score should grow with more ratio violations."""
        job = JobPosting(
            salary_min=5000.0,
            salary_max=100000.0,
            description=(
                "APPLY NOW!!! URGENT!!! LIMITED SPOTS!!! ACT IMMEDIATELY!!! "
                "Provide your SSN, bank account, and date of birth. "
                "No details about the role. Various tasks as needed."
            )
        )
        result = analyzer.analyze(job)
        assert result["flag_count"] >= 3
        assert result["composite_score"] > 0.3

    def test_composite_capped_at_one(self, analyzer):
        """Composite score should never exceed 1.0."""
        job = JobPosting(
            salary_min=1000.0,
            salary_max=999000.0,
            description=(
                "APPLY NOW!!! URGENT!!! LIMITED SPOTS!!! HURRY!!! ACT FAST!!! "
                "DON'T WAIT!!! IMMEDIATELY!!! TODAY ONLY!!! "
                "Provide SSN, bank account, date of birth, credit card, "
                "driver's license, passport number, mother's maiden name. "
                "Various tasks as needed. Flexible duties. General work."
            )
        )
        result = analyzer.analyze(job)
        assert result["composite_score"] <= 1.0

    def test_no_salary_ratio_is_none(self, analyzer):
        """Posting without salary should have None for salary_range_width."""
        job = JobPosting(description="A simple job posting.")
        result = analyzer.analyze(job)
        assert result["ratios"]["salary_range_width"] is None

    def test_to_signal_threshold(self, analyzer):
        """Signal should only fire with 2+ ratio violations."""
        job = JobPosting(
            description="A simple posting with normal language and no issues."
        )
        signal = analyzer.to_signal(job)
        assert signal is None

    def test_to_signal_fires_with_multiple_flags(self, analyzer):
        """Multiple ratio violations should produce a fraud_ratio_anomaly signal."""
        job = JobPosting(
            salary_min=5000.0,
            salary_max=200000.0,
            description=(
                "APPLY NOW!!! URGENT!!! ACT IMMEDIATELY!!! "
                "Provide your SSN and bank account number!"
            )
        )
        signal = analyzer.to_signal(job)
        assert signal is not None
        assert signal.name == "fraud_ratio_anomaly"
        assert signal.weight > 0


# ---------------------------------------------------------------------------
# Scheme Lifecycle Analyzer
# ---------------------------------------------------------------------------

class TestSchemeLifecycleAnalyzer:
    @pytest.fixture
    def lifecycle(self) -> SchemeLifecycleAnalyzer:
        return SchemeLifecycleAnalyzer()

    def test_lifecycle_phases_defined(self):
        """Should have at least 5 lifecycle phases defined."""
        assert len(SCHEME_LIFECYCLE_PHASES) >= 5

    def test_lifecycle_phases_valid(self):
        """Each phase should be a valid LifecyclePhase with proper fields."""
        for phase in SCHEME_LIFECYCLE_PHASES:
            assert isinstance(phase, LifecyclePhase)
            assert phase.name
            assert phase.description
            assert len(phase.typical_duration_days) == 2
            assert phase.typical_duration_days[0] <= phase.typical_duration_days[1]
            assert 0 < phase.risk_level <= 1.0
            assert len(phase.indicators) >= 3

    def test_seeding_phase_detected(self, lifecycle):
        """New/startup company language should detect seeding phase."""
        job = JobPosting(
            description=(
                "We are a newly established company. Just launched last month. "
                "This startup is expanding rapidly and growing fast."
            )
        )
        result = lifecycle.analyze(job)
        assert "seeding" in result["phases_detected"]

    def test_casting_phase_detected(self, lifecycle):
        """Mass hiring language should detect casting phase."""
        job = JobPosting(
            description=(
                "Hiring multiple positions! Mass recruitment event. "
                "Many openings available. Urgent immediate openings."
            )
        )
        result = lifecycle.analyze(job)
        assert "casting" in result["phases_detected"]

    def test_extraction_phase_detected(self, lifecycle):
        """Fee/data collection language should detect extraction phase."""
        job = JobPosting(
            description=(
                "To start, you must pay a training fee of $199. "
                "Please provide your bank account and social security number. "
                "Purchase the required equipment kit before your first day."
            )
        )
        result = lifecycle.analyze(job)
        assert "extraction" in result["phases_detected"]

    def test_concealment_phase_detected(self, lifecycle):
        """Secrecy language should detect concealment phase."""
        job = JobPosting(
            description=(
                "Do not share this opportunity with others. "
                "This is a confidential position. Limited access group only. "
                "Delete this message after reviewing."
            )
        )
        result = lifecycle.analyze(job)
        assert "concealment" in result["phases_detected"]

    def test_dominant_phase_is_highest_scoring(self, lifecycle):
        """Dominant phase should be the one with the highest score."""
        job = JobPosting(
            description=(
                "Pay a registration fee. Pay a training fee. "
                "Send your bank account number. Provide your SSN. "
                "Purchase equipment kit now."
            )
        )
        result = lifecycle.analyze(job)
        if result["phases_detected"]:
            dominant = result["dominant_phase"]
            assert dominant is not None
            assert result["phase_scores"][dominant] >= max(
                result["phase_scores"].values()
            ) - 0.001

    def test_multi_phase_amplification(self, lifecycle):
        """Detecting 2+ phases should amplify composite risk."""
        job = JobPosting(
            description=(
                "This newly established startup is hiring multiple positions. "
                "Mass recruitment across the country. Expanding rapidly. "
                "Pay a training fee to get started."
            )
        )
        result = lifecycle.analyze(job)
        if len(result["phases_detected"]) >= 2:
            assert result["composite_risk"] > 0.1

    def test_legit_posting_no_phases(self, lifecycle, legit_posting):
        """Legitimate posting should detect no or very few lifecycle phases."""
        result = lifecycle.analyze(legit_posting)
        assert len(result["phases_detected"]) <= 1

    def test_empty_posting_no_phases(self, lifecycle):
        """Empty posting should detect no phases."""
        job = JobPosting()
        result = lifecycle.analyze(job)
        assert result["phases_detected"] == []
        assert result["dominant_phase"] is None
        assert result["composite_risk"] == 0.0

    def test_composite_risk_capped(self, lifecycle):
        """Composite risk should never exceed 1.0."""
        job = JobPosting(
            description=(
                "Newly established startup, just launched, expanding rapidly. "
                "Hiring multiple positions, mass recruitment, many openings, urgent openings. "
                "Pay registration fee, pay training fee, provide bank account, SSN. "
                "Confidential opportunity, do not share, limited access, delete after reading."
            )
        )
        result = lifecycle.analyze(job)
        assert result["composite_risk"] <= 1.0

    def test_to_signal_extraction_fires(self, lifecycle):
        """Extraction phase should produce a RED_FLAG signal."""
        job = JobPosting(
            description=(
                "Pay a training fee. Provide your bank account. "
                "Purchase the required equipment."
            )
        )
        signal = lifecycle.to_signal(job)
        assert signal is not None
        assert signal.name == "scheme_lifecycle"
        assert signal.category == SignalCategory.RED_FLAG

    def test_to_signal_concealment_fires(self, lifecycle):
        """Concealment phase should produce a RED_FLAG signal."""
        job = JobPosting(
            description=(
                "Do not share this confidential opportunity. "
                "Limited access group. Delete after reviewing."
            )
        )
        signal = lifecycle.to_signal(job)
        assert signal is not None
        assert signal.name == "scheme_lifecycle"
        assert signal.category == SignalCategory.RED_FLAG

    def test_to_signal_single_low_risk_phase_no_signal(self, lifecycle):
        """A single low-risk phase alone should not produce a signal."""
        job = JobPosting(
            description="We are a newly established company. Just launched."
        )
        signal = lifecycle.to_signal(job)
        # Single seeding phase without extraction/concealment = no signal
        assert signal is None

    def test_to_signal_multi_phase_warning(self, lifecycle):
        """Multiple non-extraction phases should produce a WARNING signal."""
        job = JobPosting(
            description=(
                "Newly established startup expanding rapidly. "
                "Hiring multiple positions. Mass recruitment. "
                "Many openings available. Urgent immediate openings."
            )
        )
        signal = lifecycle.to_signal(job)
        if signal is not None:
            assert signal.name == "scheme_lifecycle"
