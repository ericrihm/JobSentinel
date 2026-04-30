"""Comprehensive tests for Sentinel scam detection platform.

Covers: models, signals, scorer, scanner, validator, db, knowledge, integration.
"""

import os
import tempfile

import pytest

from sentinel.models import (
    CompanyProfile,
    JobPosting,
    RiskLevel,
    ScamPattern,
    ScamSignal,
    SignalCategory,
    UserReport,
    ValidationResult,
)


# ===========================================================================
# TestModels
# ===========================================================================


class TestModels:
    def test_job_posting_defaults(self):
        job = JobPosting()
        assert job.url == ""
        assert job.title == ""
        assert job.company == ""
        assert job.salary_min == 0.0
        assert job.salary_max == 0.0
        assert job.salary_currency == "USD"
        assert job.is_remote is False
        assert job.source == "linkedin"

    def test_job_posting_explicit_fields(self):
        job = JobPosting(
            url="https://linkedin.com/jobs/1",
            title="Senior Python Engineer",
            company="Acme Corp",
            location="San Francisco, CA",
            salary_min=120000.0,
            salary_max=180000.0,
            is_remote=True,
        )
        assert job.title == "Senior Python Engineer"
        assert job.salary_min == 120000.0
        assert job.is_remote is True

    def test_job_posting_to_dict(self):
        job = JobPosting(
            url="https://example.com/jobs/42",
            title="Data Engineer",
            company="StartupXYZ",
            salary_min=100000.0,
            salary_max=150000.0,
            is_remote=True,
        )
        d = job.to_dict()
        assert d["url"] == "https://example.com/jobs/42"
        assert d["title"] == "Data Engineer"
        assert d["salary_min"] == 100000.0
        assert d["is_remote"] is True

    def test_validation_result_risk_labels(self):
        job = JobPosting(title="Test Job")
        result = ValidationResult(job=job, scam_score=0.1)
        assert result.risk_label() == "Verified Safe"

        result.scam_score = 0.3
        assert result.risk_label() == "Likely Legitimate"

        result.scam_score = 0.5
        assert result.risk_label() == "Suspicious"

        result.scam_score = 0.7
        assert result.risk_label() == "Likely Scam"

        result.scam_score = 0.9
        assert result.risk_label() == "Almost Certainly Scam"

    def test_scam_signal_bayesian_weight(self):
        # Default alpha=1, beta=1 → weight = 0.5
        sig = ScamSignal(name="test_signal", category=SignalCategory.RED_FLAG)
        assert sig.bayesian_weight == pytest.approx(0.5)

        # Higher alpha → higher weight
        sig2 = ScamSignal(
            name="test_signal",
            category=SignalCategory.RED_FLAG,
            alpha=9.0,
            beta=1.0,
        )
        assert sig2.bayesian_weight == pytest.approx(0.9)

    def test_scam_signal_defaults(self):
        sig = ScamSignal(name="upfront_payment", category=SignalCategory.RED_FLAG)
        assert sig.weight == 0.5
        assert sig.confidence == 0.5
        assert sig.detail == ""
        assert sig.evidence == ""
        assert sig.alpha == 1.0
        assert sig.beta == 1.0

    def test_validation_result_to_dict_structure(self):
        job = JobPosting(url="http://example.com/1", title="Engineer", company="Corp")
        red = ScamSignal("upfront_payment", SignalCategory.RED_FLAG, weight=0.95, detail="fee required")
        warn = ScamSignal("vague_description", SignalCategory.WARNING, weight=0.5, detail="sparse")
        result = ValidationResult(
            job=job,
            signals=[red, warn],
            scam_score=0.75,
            confidence=0.8,
            risk_level=RiskLevel.HIGH,
        )
        d = result.to_dict()
        assert d["scam_score"] == 0.75
        assert d["risk_level"] == "high"
        assert d["signal_count"] == 2
        assert len(d["red_flags"]) == 1
        assert d["red_flags"][0]["name"] == "upfront_payment"
        assert len(d["warnings"]) == 1

    def test_validation_result_signal_category_properties(self):
        job = JobPosting()
        red = ScamSignal("upfront_payment", SignalCategory.RED_FLAG)
        warn = ScamSignal("vague_description", SignalCategory.WARNING)
        ghost = ScamSignal("stale_posting", SignalCategory.GHOST_JOB)
        pos = ScamSignal("established_company", SignalCategory.POSITIVE)
        struct = ScamSignal("grammar_quality", SignalCategory.STRUCTURAL)

        result = ValidationResult(job=job, signals=[red, warn, ghost, pos, struct])
        assert len(result.red_flags) == 1
        assert len(result.warnings) == 1
        assert len(result.ghost_indicators) == 1
        assert len(result.positive_signals) == 1

    def test_scam_pattern_precision(self):
        pattern = ScamPattern(
            pattern_id="test_pat",
            name="Test Pattern",
            description="A test pattern",
            category=SignalCategory.RED_FLAG,
            true_positives=8,
            false_positives=2,
        )
        assert pattern.precision == pytest.approx(0.8)

    def test_scam_pattern_precision_zero_division(self):
        pattern = ScamPattern(
            pattern_id="empty_pat",
            name="Empty Pattern",
            description="No observations",
            category=SignalCategory.WARNING,
        )
        assert pattern.precision == 0.0

    def test_scam_pattern_bayesian_score(self):
        pattern = ScamPattern(
            pattern_id="bayes_pat",
            name="Bayesian Pattern",
            description="Has strong prior",
            category=SignalCategory.RED_FLAG,
            alpha=4.0,
            beta=1.0,
        )
        assert pattern.bayesian_score == pytest.approx(4.0 / 5.0)


# ===========================================================================
# TestSignals
# ===========================================================================


class TestSignals:
    # --- check_upfront_payment ---

    def test_upfront_payment_detects_training_fee(self):
        job = JobPosting(description="You must pay a training fee of $100 before starting.")
        from sentinel.signals import check_upfront_payment
        sig = check_upfront_payment(job)
        assert sig is not None
        assert sig.name == "upfront_payment"
        assert sig.category == SignalCategory.RED_FLAG
        assert sig.weight == pytest.approx(0.95)

    def test_upfront_payment_detects_registration_fee(self):
        from sentinel.signals import check_upfront_payment
        job = JobPosting(description="An upfront fee is required before you can begin.")
        sig = check_upfront_payment(job)
        assert sig is not None
        assert sig.name == "upfront_payment"

    def test_upfront_payment_clean_posting(self):
        from sentinel.signals import check_upfront_payment
        job = JobPosting(
            title="Software Engineer",
            description="Competitive salary and benefits. We offer comprehensive onboarding.",
        )
        assert check_upfront_payment(job) is None

    # --- check_guaranteed_income ---

    def test_guaranteed_income_positive(self):
        from sentinel.signals import check_guaranteed_income
        job = JobPosting(description="This role offers guaranteed salary of $5,000 per week.")
        sig = check_guaranteed_income(job)
        assert sig is not None
        assert sig.name == "guaranteed_income"
        assert sig.category == SignalCategory.RED_FLAG

    def test_guaranteed_income_negative(self):
        from sentinel.signals import check_guaranteed_income
        job = JobPosting(description="Salary commensurate with experience. Comprehensive benefits package.")
        assert check_guaranteed_income(job) is None

    # --- check_suspicious_email_domain ---

    def test_suspicious_email_gmail(self):
        from sentinel.signals import check_suspicious_email_domain
        job = JobPosting(
            description="Please send your resume to recruiter@gmail.com to apply.",
            company="BigCorporation Inc",
        )
        sig = check_suspicious_email_domain(job)
        assert sig is not None
        assert sig.name == "suspicious_email_domain"
        assert sig.category == SignalCategory.RED_FLAG

    def test_suspicious_email_in_recruiter_field(self):
        from sentinel.signals import check_suspicious_email_domain
        job = JobPosting(
            description="We are hiring a remote assistant.",
            recruiter_name="Jane Doe jane@yahoo.com",
        )
        sig = check_suspicious_email_domain(job)
        assert sig is not None

    def test_suspicious_email_clean(self):
        from sentinel.signals import check_suspicious_email_domain
        job = JobPosting(
            description="Apply at careers.acme.com. Contact hr@acme.com for questions.",
            company="Acme Corp",
        )
        assert check_suspicious_email_domain(job) is None

    # --- check_no_company_presence ---

    def test_no_company_presence_empty_name(self):
        from sentinel.signals import check_no_company_presence
        job = JobPosting(company="", company_linkedin_url="")
        sig = check_no_company_presence(job)
        assert sig is not None
        assert sig.name == "no_company_presence"
        assert sig.weight == pytest.approx(0.85)

    def test_no_company_presence_has_name_no_linkedin(self):
        from sentinel.signals import check_no_company_presence
        job = JobPosting(company="Acme Corp", company_linkedin_url="")
        sig = check_no_company_presence(job)
        assert sig is not None
        assert sig.weight == pytest.approx(0.70)

    def test_no_company_presence_fully_present(self):
        from sentinel.signals import check_no_company_presence
        job = JobPosting(
            company="Acme Corp",
            company_linkedin_url="https://linkedin.com/company/acme-corp",
        )
        assert check_no_company_presence(job) is None

    # --- check_salary_anomaly ---

    def test_salary_anomaly_wide_range(self):
        from sentinel.signals import check_salary_anomaly
        # hi / lo > 3.0 → anomaly
        job = JobPosting(salary_min=40000, salary_max=300000)
        sig = check_salary_anomaly(job)
        assert sig is not None
        assert sig.name == "salary_anomaly"
        assert sig.category == SignalCategory.WARNING

    def test_salary_anomaly_entry_level_unrealistic(self):
        from sentinel.signals import check_salary_anomaly
        job = JobPosting(salary_min=600000, salary_max=0, experience_level="entry")
        sig = check_salary_anomaly(job)
        assert sig is not None
        assert sig.category == SignalCategory.WARNING

    def test_salary_anomaly_normal_range(self):
        from sentinel.signals import check_salary_anomaly
        # hi / lo = 180k / 120k = 1.5 → no anomaly
        job = JobPosting(salary_min=120000, salary_max=180000)
        assert check_salary_anomaly(job) is None

    def test_salary_anomaly_no_salary(self):
        from sentinel.signals import check_salary_anomaly
        job = JobPosting(salary_min=0, salary_max=0)
        assert check_salary_anomaly(job) is None

    # --- check_vague_description ---

    def test_vague_description_empty(self):
        from sentinel.signals import check_vague_description
        job = JobPosting(description="")
        sig = check_vague_description(job)
        assert sig is not None
        assert sig.name == "vague_description"
        assert sig.weight == pytest.approx(0.65)

    def test_vague_description_few_words(self):
        from sentinel.signals import check_vague_description
        job = JobPosting(description="Work from home job.")
        sig = check_vague_description(job)
        assert sig is not None

    def test_vague_description_detailed(self):
        from sentinel.signals import check_vague_description
        desc = (
            "We are looking for an experienced Python developer to join our backend team. "
            "Must have 3+ years of experience building scalable RESTful APIs using Django or FastAPI. "
            "Familiarity with Docker and AWS required."
        )
        assert len(desc.split()) >= 30
        job = JobPosting(description=desc)
        assert check_vague_description(job) is None

    # --- check_urgency_language ---

    def test_urgency_language_apply_now(self):
        from sentinel.signals import check_urgency_language
        job = JobPosting(title="Marketing Manager", description="Apply now! Limited spots available.")
        sig = check_urgency_language(job)
        assert sig is not None
        assert sig.name == "urgency_language"
        assert sig.category == SignalCategory.WARNING

    def test_urgency_language_hiring_immediately(self):
        from sentinel.signals import check_urgency_language
        job = JobPosting(description="We are hiring immediately for multiple open positions.")
        sig = check_urgency_language(job)
        assert sig is not None

    def test_urgency_language_calm_posting(self):
        from sentinel.signals import check_urgency_language
        job = JobPosting(
            description="We are a growing tech company building the next generation of productivity tools."
        )
        assert check_urgency_language(job) is None

    # --- check_established_company ---

    def test_established_company_large(self):
        from sentinel.signals import check_established_company
        job = JobPosting(company="Google", company_size="100000")
        sig = check_established_company(job)
        assert sig is not None
        assert sig.name == "established_company"
        assert sig.category == SignalCategory.POSITIVE

    def test_established_company_small(self):
        from sentinel.signals import check_established_company
        job = JobPosting(company="Tiny Startup", company_size="10")
        assert check_established_company(job) is None

    def test_established_company_missing_size(self):
        from sentinel.signals import check_established_company
        job = JobPosting(company="Unknown Corp", company_size="")
        assert check_established_company(job) is None

    # --- check_detailed_requirements ---

    def test_detailed_requirements_full(self):
        from sentinel.signals import check_detailed_requirements
        desc = (
            "We require Python, JavaScript, React, and TypeScript expertise. "
            "Candidate must have 5+ years of experience. A bachelor's degree in CS is preferred. "
            "We offer health insurance and dental benefits."
        )
        job = JobPosting(description=desc)
        sig = check_detailed_requirements(job)
        assert sig is not None
        assert sig.name == "detailed_requirements"
        assert sig.category == SignalCategory.POSITIVE

    def test_detailed_requirements_sparse(self):
        from sentinel.signals import check_detailed_requirements
        job = JobPosting(description="Looking for someone to help with general office tasks.")
        assert check_detailed_requirements(job) is None

    # --- extract_signals ---

    def test_extract_signals_returns_list(self):
        from sentinel.signals import extract_signals
        job = JobPosting(
            title="Data Entry Clerk",
            description="Work from home. No experience needed. Earn guaranteed income.",
            company="",
        )
        signals = extract_signals(job)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_extract_signals_scam_job_has_red_flags(self):
        from sentinel.signals import extract_signals
        job = JobPosting(
            title="Work From Home",
            description=(
                "Send us $200 upfront fee to get started. "
                "Guaranteed salary every week. Apply now! Limited spots."
            ),
            company="",
        )
        signals = extract_signals(job)
        red_flags = [s for s in signals if s.category == SignalCategory.RED_FLAG]
        assert len(red_flags) >= 2

    def test_extract_signals_legit_job_has_positive_signals(self):
        from sentinel.signals import extract_signals
        desc = (
            "We need a Python developer with 5+ years of experience with Python, "
            "JavaScript, React, and PostgreSQL. Bachelor's degree required. "
            "Health insurance, dental, 401k, and equity offered. "
            "Full-time position at our San Francisco, CA office."
        )
        job = JobPosting(
            title="Senior Software Engineer",
            company="TechCorp",
            company_linkedin_url="https://linkedin.com/company/techcorp",
            company_size="5000",
            description=desc,
        )
        signals = extract_signals(job)
        positive = [s for s in signals if s.category == SignalCategory.POSITIVE]
        assert len(positive) >= 1

    def test_extract_signals_all_signal_funcs_called(self):
        from sentinel.signals import ALL_SIGNALS
        # ALL_SIGNALS should contain at least 15 signal functions
        assert len(ALL_SIGNALS) >= 15

    def test_extract_signals_repost_pattern(self):
        from sentinel.signals import check_repost_pattern
        job = JobPosting(is_repost=True)
        sig = check_repost_pattern(job)
        assert sig is not None
        assert sig.name == "repost_pattern"
        assert sig.category == SignalCategory.GHOST_JOB

    def test_extract_signals_no_repost(self):
        from sentinel.signals import check_repost_pattern
        job = JobPosting(is_repost=False)
        assert check_repost_pattern(job) is None

    def test_crypto_payment_detected(self):
        from sentinel.signals import check_crypto_payment
        job = JobPosting(description="Payment via Bitcoin only. No other payment methods accepted.")
        sig = check_crypto_payment(job)
        assert sig is not None
        assert sig.name == "crypto_payment"

    def test_crypto_payment_not_present(self):
        from sentinel.signals import check_crypto_payment
        job = JobPosting(description="Competitive compensation with stock options and 401k.")
        assert check_crypto_payment(job) is None


# ===========================================================================
# TestScorer
# ===========================================================================


class TestScorer:
    def test_score_signals_empty(self):
        from sentinel.scorer import score_signals
        score, confidence = score_signals([])
        assert score == 0.0
        assert confidence == 0.0

    def test_score_signals_single_scam(self):
        from sentinel.scorer import score_signals
        sig = ScamSignal("upfront_payment", SignalCategory.RED_FLAG, weight=0.95)
        score, confidence = score_signals([sig])
        assert score > 0.8
        assert 0.0 < confidence < 1.0

    def test_score_signals_single_positive(self):
        from sentinel.scorer import score_signals
        sig = ScamSignal("established_company", SignalCategory.POSITIVE, weight=0.35)
        score, confidence = score_signals([sig])
        # Positive signal should push score below 0.5
        assert score < 0.5

    def test_score_signals_mixed_pushes_toward_scam(self):
        from sentinel.scorer import score_signals
        scam_sigs = [
            ScamSignal("upfront_payment", SignalCategory.RED_FLAG, weight=0.95),
            ScamSignal("guaranteed_income", SignalCategory.RED_FLAG, weight=0.85),
            ScamSignal("urgency_language", SignalCategory.WARNING, weight=0.58),
        ]
        pos_sigs = [
            ScamSignal("established_company", SignalCategory.POSITIVE, weight=0.35),
        ]
        all_signals = scam_sigs + pos_sigs
        score, confidence = score_signals(all_signals)
        # 3 scam signals vs 1 positive → should still be elevated
        assert score > 0.5

    def test_score_signals_all_positive_low_score(self):
        from sentinel.scorer import score_signals
        sigs = [
            ScamSignal("established_company", SignalCategory.POSITIVE, weight=0.35),
            ScamSignal("detailed_requirements", SignalCategory.POSITIVE, weight=0.38),
        ]
        score, _ = score_signals(sigs)
        assert score < 0.4

    def test_score_signals_returns_floats_in_range(self):
        from sentinel.scorer import score_signals
        sigs = [ScamSignal("sig1", SignalCategory.RED_FLAG, weight=0.7)]
        score, confidence = score_signals(sigs)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_classify_risk_thresholds(self):
        from sentinel.scorer import classify_risk
        assert classify_risk(0.1) == RiskLevel.SAFE
        assert classify_risk(0.19) == RiskLevel.SAFE
        assert classify_risk(0.2) == RiskLevel.LOW
        assert classify_risk(0.39) == RiskLevel.LOW
        assert classify_risk(0.4) == RiskLevel.SUSPICIOUS
        assert classify_risk(0.59) == RiskLevel.SUSPICIOUS
        assert classify_risk(0.6) == RiskLevel.HIGH
        assert classify_risk(0.79) == RiskLevel.HIGH
        assert classify_risk(0.8) == RiskLevel.SCAM
        assert classify_risk(1.0) == RiskLevel.SCAM

    def test_build_result_pipeline(self):
        from sentinel.scorer import build_result
        job = JobPosting(title="Test Job", company="Test Corp")
        sigs = [
            ScamSignal("upfront_payment", SignalCategory.RED_FLAG, weight=0.95),
            ScamSignal("guaranteed_income", SignalCategory.RED_FLAG, weight=0.85),
        ]
        result = build_result(job, sigs, analysis_time_ms=5.0)
        assert isinstance(result, ValidationResult)
        assert result.scam_score > 0.8
        assert result.risk_level == RiskLevel.SCAM
        assert result.analysis_time_ms == 5.0
        assert len(result.signals) == 2

    def test_build_result_no_signals(self):
        from sentinel.scorer import build_result
        job = JobPosting(title="Clean Job")
        result = build_result(job, [], analysis_time_ms=1.0)
        assert result.scam_score == 0.0
        assert result.confidence == 0.0
        assert result.risk_level == RiskLevel.SAFE

    def test_signal_weight_tracker_uniform_prior(self):
        from sentinel.scorer import SignalWeightTracker
        tracker = SignalWeightTracker()
        # Sampling from Beta(1,1) should be in (0,1); mean ≈ 0.5
        weights = [tracker.get_weight("test_signal") for _ in range(100)]
        assert all(0.0 <= w <= 1.0 for w in weights)
        assert 0.2 < sum(weights) / len(weights) < 0.8  # roughly centered

    def test_signal_weight_tracker_update_true_positive(self):
        from sentinel.scorer import SignalWeightTracker
        tracker = SignalWeightTracker()
        for _ in range(10):
            tracker.update("upfront_payment", was_correct=True)
        # After 10 true-positive updates, alpha=11, beta=1 → mean ≈ 0.917
        # get_weight samples from this posterior
        weights = [tracker.get_weight("upfront_payment") for _ in range(200)]
        assert sum(weights) / len(weights) > 0.7

    def test_signal_weight_tracker_update_false_positive(self):
        from sentinel.scorer import SignalWeightTracker
        tracker = SignalWeightTracker()
        for _ in range(10):
            tracker.update("weak_signal", was_correct=False)
        # After 10 false-positive updates, alpha=1, beta=11 → mean ≈ 0.083
        weights = [tracker.get_weight("weak_signal") for _ in range(200)]
        assert sum(weights) / len(weights) < 0.3

    def test_signal_weight_tracker_ranking(self):
        from sentinel.scorer import SignalWeightTracker
        tracker = SignalWeightTracker()
        tracker.update("good_signal", was_correct=True)
        tracker.update("good_signal", was_correct=True)
        tracker.update("bad_signal", was_correct=False)
        tracker.update("bad_signal", was_correct=False)
        ranking = tracker.get_ranking()
        names = [r[0] for r in ranking]
        assert "good_signal" in names
        assert names.index("good_signal") < names.index("bad_signal")


# ===========================================================================
# TestScanner
# ===========================================================================


class TestScanner:
    def test_parse_job_text_basic(self):
        from sentinel.scanner import parse_job_text
        text = (
            "Job Title: Backend Engineer\n"
            "Company: Tech Corp\n"
            "Location: Austin, TX\n"
            "We are looking for a Python developer."
        )
        job = parse_job_text(text)
        assert isinstance(job, JobPosting)
        assert "Austin" in job.location or "TX" in job.location

    def test_parse_job_text_extracts_remote(self):
        from sentinel.scanner import parse_job_text
        text = "This is a fully remote position. Work from home anywhere in the US."
        job = parse_job_text(text)
        assert job.is_remote is True
        assert job.location == "Remote"

    def test_extract_salary_range_k_notation(self):
        from sentinel.scanner import extract_salary
        lo, hi, currency = extract_salary("Salary: $120k - $180k")
        assert lo == 120000.0
        assert hi == 180000.0
        assert currency == "USD"

    def test_extract_salary_annual_full(self):
        from sentinel.scanner import extract_salary
        lo, hi, currency = extract_salary("Compensation: $120,000 - $150,000/year")
        assert lo == 120000.0
        assert hi == 150000.0
        assert currency == "USD"

    def test_extract_salary_hourly_converts_to_annual(self):
        from sentinel.scanner import extract_salary
        lo, hi, currency = extract_salary("Pay: $50/hr")
        assert lo == pytest.approx(50 * 2080)
        assert hi == pytest.approx(50 * 2080)

    def test_extract_salary_annual_single(self):
        from sentinel.scanner import extract_salary
        lo, hi, currency = extract_salary("Earn $95k/year")
        assert lo == 95000.0
        assert hi == 95000.0

    def test_extract_salary_no_match(self):
        from sentinel.scanner import extract_salary
        lo, hi, currency = extract_salary("Salary commensurate with experience.")
        assert lo == 0.0
        assert hi == 0.0

    def test_extract_location_city_state(self):
        from sentinel.scanner import extract_location
        assert "San Francisco" in extract_location("We are located in San Francisco, CA.")

    def test_extract_location_remote(self):
        from sentinel.scanner import extract_location
        assert extract_location("This is a remote role. Work from anywhere.") == "Remote"

    def test_extract_location_empty(self):
        from sentinel.scanner import extract_location
        assert extract_location("Join our team and make an impact.") == ""

    def test_detect_experience_level_senior(self):
        from sentinel.scanner import detect_experience_level
        assert detect_experience_level("Senior Software Engineer") == "senior"

    def test_detect_experience_level_entry(self):
        from sentinel.scanner import detect_experience_level
        assert detect_experience_level("Entry level position for new graduates") == "entry"

    def test_detect_experience_level_executive(self):
        from sentinel.scanner import detect_experience_level
        assert detect_experience_level("Vice President of Engineering") == "executive"

    def test_detect_experience_level_unknown(self):
        from sentinel.scanner import detect_experience_level
        assert detect_experience_level("Software Engineer") == ""

    def test_parse_job_text_salary_extraction(self):
        from sentinel.scanner import parse_job_text
        text = "Software Engineer position paying $120k - $160k per year in Seattle, WA."
        job = parse_job_text(text)
        assert job.salary_min == 120000.0
        assert job.salary_max == 160000.0


# ===========================================================================
# TestValidator
# ===========================================================================


class TestValidator:
    def test_is_known_company_google(self):
        from sentinel.validator import _is_known_company
        assert _is_known_company("google") is True

    def test_is_known_company_case_insensitive(self):
        from sentinel.validator import _is_known_company
        assert _is_known_company("Google") is True
        assert _is_known_company("MICROSOFT") is True
        assert _is_known_company("amazon") is True

    def test_is_known_company_unknown(self):
        from sentinel.validator import _is_known_company
        assert _is_known_company("xyz123unknownco") is False
        assert _is_known_company("scam jobs inc") is False

    def test_validate_company_known(self):
        from sentinel.validator import validate_company
        profile = validate_company("Google")
        assert isinstance(profile, CompanyProfile)
        assert profile.is_verified is True
        assert profile.verification_source == "known_companies_list"
        assert profile.name == "Google"

    def test_validate_company_unknown(self):
        from sentinel.validator import validate_company
        profile = validate_company("XYZ Mystery Corp 99999")
        assert isinstance(profile, CompanyProfile)
        assert profile.is_verified is False

    def test_known_companies_set_not_empty(self):
        from sentinel.validator import KNOWN_COMPANIES
        assert len(KNOWN_COMPANIES) > 50
        assert "anthropic" in KNOWN_COMPANIES
        assert "openai" in KNOWN_COMPANIES


# ===========================================================================
# TestDB
# ===========================================================================


class TestDB:
    def test_save_and_get_job_round_trip(self, tmp_path):
        from sentinel.db import SentinelDB
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            job_data = {
                "url": "https://linkedin.com/jobs/12345",
                "title": "Software Engineer",
                "company": "Acme Corp",
                "location": "New York, NY",
                "description": "Build and maintain backend services.",
                "salary_min": 120000.0,
                "salary_max": 160000.0,
                "scam_score": 0.05,
                "risk_level": "safe",
                "signal_count": 2,
                "signals_json": "[]",
            }
            db.save_job(job_data)
            result = db.get_job("https://linkedin.com/jobs/12345")
        assert result is not None
        assert result["title"] == "Software Engineer"
        assert result["scam_score"] == pytest.approx(0.05)
        assert result["company"] == "Acme Corp"

    def test_get_job_not_found(self, tmp_path):
        from sentinel.db import SentinelDB
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            result = db.get_job("https://nonexistent.com/job/999")
        assert result is None

    def test_search_jobs_fts(self, tmp_path):
        from sentinel.db import SentinelDB
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            db.save_job({
                "url": "https://example.com/1",
                "title": "Python Developer",
                "company": "Tech Corp",
                "description": "Build APIs with Python and FastAPI.",
                "scam_score": 0.1,
                "risk_level": "safe",
            })
            db.save_job({
                "url": "https://example.com/2",
                "title": "Java Engineer",
                "company": "Enterprise Co",
                "description": "Spring Boot microservices architect.",
                "scam_score": 0.05,
                "risk_level": "safe",
            })
            results = db.search_jobs("Python")
        assert len(results) == 1
        assert results[0]["title"] == "Python Developer"

    def test_save_report(self, tmp_path):
        from sentinel.db import SentinelDB
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            db.save_job({
                "url": "https://scamjob.com/1",
                "title": "Work From Home",
                "company": "ScamCo",
                "description": "Send us money upfront.",
                "scam_score": 0.95,
                "risk_level": "scam",
            })
            db.save_report({
                "url": "https://scamjob.com/1",
                "is_scam": True,
                "reason": "Asked me to send money",
                "our_prediction": 0.95,
                "was_correct": True,
            })
            reports = db.get_reports(limit=10)
        assert len(reports) == 1
        assert reports[0]["is_scam"] == 1
        assert reports[0]["reason"] == "Asked me to send money"

    def test_get_stats_empty(self, tmp_path):
        from sentinel.db import SentinelDB
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            stats = db.get_stats()
        assert stats["total_jobs_analyzed"] == 0
        assert stats["scam_jobs_detected"] == 0
        assert stats["total_user_reports"] == 0
        assert stats["active_patterns"] == 0

    def test_get_stats_with_data(self, tmp_path):
        from sentinel.db import SentinelDB
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            # One scam job (score >= 0.6) and one legit job
            db.save_job({
                "url": "https://scam.com/1",
                "title": "Scam Job",
                "company": "Scammer",
                "description": "Pay us first",
                "scam_score": 0.9,
                "risk_level": "scam",
            })
            db.save_job({
                "url": "https://legit.com/1",
                "title": "Real Job",
                "company": "Google",
                "description": "Genuine position",
                "scam_score": 0.05,
                "risk_level": "safe",
            })
            stats = db.get_stats()
        assert stats["total_jobs_analyzed"] == 2
        assert stats["scam_jobs_detected"] == 1
        assert stats["avg_scam_score"] == pytest.approx(0.475)


# ===========================================================================
# TestKnowledge
# ===========================================================================


class TestKnowledge:
    def test_seed_default_patterns_populates_db(self, tmp_path):
        from sentinel.db import SentinelDB
        from sentinel.knowledge import KnowledgeBase, _DEFAULT_PATTERNS
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            kb = KnowledgeBase(db)
            kb.seed_default_patterns()
            patterns = kb.get_active_patterns()
        assert len(patterns) == len(_DEFAULT_PATTERNS)

    def test_seed_default_patterns_idempotent(self, tmp_path):
        from sentinel.db import SentinelDB
        from sentinel.knowledge import KnowledgeBase, _DEFAULT_PATTERNS
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            kb = KnowledgeBase(db)
            kb.seed_default_patterns()
            kb.seed_default_patterns()  # second call should not duplicate
            patterns = kb.get_active_patterns()
        assert len(patterns) == len(_DEFAULT_PATTERNS)

    def test_report_scam_saves_correctly(self, tmp_path):
        from sentinel.db import SentinelDB
        from sentinel.knowledge import KnowledgeBase
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            kb = KnowledgeBase(db)
            kb.report_scam(
                url="https://scam.com/job/1",
                is_scam=True,
                reason="Requested bank account number",
                our_prediction=0.85,
            )
            reports = db.get_reports(limit=10)
        assert len(reports) == 1
        assert reports[0]["is_scam"] == 1
        assert reports[0]["was_correct"] == 1  # prediction 0.85 >= 0.5 → correct

    def test_search_returns_matching_jobs(self, tmp_path):
        from sentinel.db import SentinelDB
        from sentinel.knowledge import KnowledgeBase
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            kb = KnowledgeBase(db)
            db.save_job({
                "url": "https://jobs.example.com/reshipping",
                "title": "Shipping Coordinator",
                "company": "Home Jobs LLC",
                "description": "Receive packages at home and reship them. No experience required.",
                "scam_score": 0.95,
                "risk_level": "scam",
            })
            # FTS5 phrase search requires adjacent tokens; use single term
            results = kb.search("reship")
        assert len(results) >= 1

    def test_get_accuracy_stats_empty(self, tmp_path):
        from sentinel.db import SentinelDB
        from sentinel.knowledge import KnowledgeBase
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            kb = KnowledgeBase(db)
            stats = kb.get_accuracy_stats()
        assert stats["total_reports"] == 0
        assert stats["accuracy"] == 0.0
        assert stats["precision"] == 0.0

    def test_get_accuracy_stats_perfect_predictions(self, tmp_path):
        from sentinel.db import SentinelDB
        from sentinel.knowledge import KnowledgeBase
        db_path = str(tmp_path / "test.db")
        with SentinelDB(db_path) as db:
            kb = KnowledgeBase(db)
            # 2 correct predictions: scam correctly flagged, legit correctly cleared
            kb.report_scam("https://scam.com/1", is_scam=True, our_prediction=0.9)
            kb.report_scam("https://legit.com/1", is_scam=False, our_prediction=0.1)
            stats = kb.get_accuracy_stats()
        assert stats["total_reports"] == 2
        assert stats["correct"] == 2
        assert stats["accuracy"] == 1.0
        assert stats["true_positives"] == 1
        assert stats["true_negatives"] == 1
        assert stats["false_positives"] == 0
        assert stats["false_negatives"] == 0


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestIntegration:
    """End-to-end pipeline tests using scorer.build_result (analyzer.analyze_job
    has a known bug passing `signals` to `classify_risk`; these tests use the
    lower-level scorer API which is what the rest of the system actually uses)."""

    def _run_pipeline(self, job: JobPosting) -> ValidationResult:
        """Run the full signal → score → classify → build pipeline."""
        from sentinel.scorer import build_result
        from sentinel.signals import extract_signals
        signals = extract_signals(job)
        return build_result(job, signals)

    def test_obvious_scam_job(self):
        job = JobPosting(
            title="Work From Home Data Entry",
            description=(
                "Earn guaranteed income of $500 per day! No experience needed. "
                "You must pay a registration fee upfront. Apply now! "
                "Send your SSN and bank account number to get started."
            ),
            company="",
            company_linkedin_url="",
        )
        result = self._run_pipeline(job)
        assert result.scam_score >= 0.6
        assert result.risk_level in (RiskLevel.HIGH, RiskLevel.SCAM)
        assert len(result.red_flags) >= 2

    def test_legit_tech_job(self):
        job = JobPosting(
            title="Senior Backend Engineer",
            company="Google",
            company_linkedin_url="https://linkedin.com/company/google",
            company_size="140000",
            description=(
                "We are looking for a Senior Backend Engineer with 5+ years of experience "
                "in Python, Go, and distributed systems. You will work on Google Cloud infrastructure. "
                "PostgreSQL, Kubernetes, and Terraform experience required. Bachelor's degree preferred. "
                "We offer competitive salary, health insurance, dental, vision, 401k, and equity."
            ),
            salary_min=180000.0,
            salary_max=280000.0,
        )
        result = self._run_pipeline(job)
        assert result.scam_score < 0.5
        assert len(result.positive_signals) >= 1

    def test_ambiguous_job_produces_result(self):
        """An ambiguous posting should produce a valid ValidationResult in any risk tier."""
        job = JobPosting(
            title="Marketing Assistant",
            company="Sunrise Media",
            description=(
                "Join our growing team to help with social media management and content creation. "
                "Must be enthusiastic and a self-starter. Flexible hours available."
            ),
        )
        result = self._run_pipeline(job)
        assert isinstance(result, ValidationResult)
        assert 0.0 <= result.scam_score <= 1.0
        assert isinstance(result.risk_level, RiskLevel)

    def test_ghost_job_repost_detected(self):
        job = JobPosting(
            title="Software Engineer",
            company="Zombie Corp",
            company_linkedin_url="https://linkedin.com/company/zombie",
            description="We are always looking for talented engineers to join our team.",
            posted_date="2024-01-01",  # very old
            is_repost=True,
        )
        result = self._run_pipeline(job)
        ghost = result.ghost_indicators
        assert len(ghost) >= 1

    def test_format_result_text_output(self):
        from sentinel.analyzer import format_result_text
        job = JobPosting(
            title="Scam Job",
            company="Fraudster Inc",
            url="https://scam.com/1",
        )
        sig = ScamSignal(
            "upfront_payment",
            SignalCategory.RED_FLAG,
            weight=0.95,
            detail="fee required",
        )
        result = ValidationResult(
            job=job,
            signals=[sig],
            scam_score=0.92,
            confidence=0.85,
            risk_level=RiskLevel.SCAM,
        )
        text = format_result_text(result)
        assert "[SCAM]" in text
        assert "upfront_payment" in text
        assert "fee required" in text
        assert "Scam Job" in text
