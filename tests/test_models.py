"""Tests for sentinel/models.py — core data model classes."""

import pytest
from sentinel.models import (
    JobPosting,
    ScamSignal,
    SignalCategory,
    RiskLevel,
    CompanyProfile,
    ValidationResult,
    ScamPattern,
    UserReport,
)


# ---------------------------------------------------------------------------
# RiskLevel enum
# ---------------------------------------------------------------------------

class TestRiskLevel:
    def test_all_values_accessible(self):
        assert RiskLevel.SAFE.value == "safe"
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.SUSPICIOUS.value == "suspicious"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.SCAM.value == "scam"

    def test_from_value(self):
        assert RiskLevel("scam") == RiskLevel.SCAM

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            RiskLevel("definitely_a_scam")


# ---------------------------------------------------------------------------
# SignalCategory enum
# ---------------------------------------------------------------------------

class TestSignalCategory:
    def test_all_categories_present(self):
        expected = {"red_flag", "warning", "ghost_job", "structural", "positive"}
        actual = {c.value for c in SignalCategory}
        assert expected == actual

    def test_from_value(self):
        assert SignalCategory("red_flag") == SignalCategory.RED_FLAG


# ---------------------------------------------------------------------------
# JobPosting dataclass
# ---------------------------------------------------------------------------

class TestJobPosting:
    def test_default_construction(self):
        job = JobPosting()
        assert job.url == ""
        assert job.salary_min == 0.0
        assert job.salary_max == 0.0
        assert job.is_remote is False
        assert job.is_repost is False
        assert job.source == "linkedin"
        assert job.applicant_count == 0
        assert job.recruiter_connections == 0

    def test_explicit_construction(self):
        job = JobPosting(
            url="https://example.com/job/123",
            title="Senior Engineer",
            company="Acme",
            location="Remote",
            salary_min=100_000.0,
            salary_max=160_000.0,
            is_remote=True,
        )
        assert job.title == "Senior Engineer"
        assert job.salary_min == pytest.approx(100_000.0)
        assert job.is_remote is True

    def test_to_dict_has_required_keys(self):
        job = JobPosting(url="https://u.com", title="Dev", company="Corp")
        d = job.to_dict()
        for key in ("url", "title", "company", "location",
                    "salary_min", "salary_max", "posted_date",
                    "experience_level", "employment_type", "is_remote"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_match(self):
        job = JobPosting(url="https://x.com", title="Analyst", is_remote=True,
                         salary_min=60_000.0, salary_max=80_000.0)
        d = job.to_dict()
        assert d["url"] == "https://x.com"
        assert d["title"] == "Analyst"
        assert d["is_remote"] is True
        assert d["salary_min"] == pytest.approx(60_000.0)

    def test_mutable_fields_independent(self):
        """Two JobPosting instances should not share mutable state."""
        a = JobPosting()
        b = JobPosting()
        a.title = "A"
        b.title = "B"
        assert a.title != b.title

    def test_zero_salary_defaults(self):
        job = JobPosting()
        assert job.salary_min == 0.0
        assert job.salary_max == 0.0

    def test_empty_string_fields_allowed(self):
        job = JobPosting(title="", company="", location="")
        assert job.title == ""
        d = job.to_dict()
        assert d["title"] == ""


# ---------------------------------------------------------------------------
# ScamSignal dataclass
# ---------------------------------------------------------------------------

class TestScamSignal:
    def test_minimal_construction(self):
        sig = ScamSignal(name="upfront_fee", category=SignalCategory.RED_FLAG)
        assert sig.name == "upfront_fee"
        assert sig.category == SignalCategory.RED_FLAG
        assert sig.weight == pytest.approx(0.5)
        assert sig.confidence == pytest.approx(0.5)
        assert sig.alpha == pytest.approx(1.0)
        assert sig.beta == pytest.approx(1.0)

    def test_bayesian_weight_default(self):
        sig = ScamSignal(name="s", category=SignalCategory.WARNING)
        # alpha=1, beta=1 → 0.5
        assert sig.bayesian_weight == pytest.approx(0.5)

    def test_bayesian_weight_high_alpha(self):
        sig = ScamSignal(name="s", category=SignalCategory.RED_FLAG, alpha=9.0, beta=1.0)
        assert sig.bayesian_weight == pytest.approx(0.9)

    def test_bayesian_weight_low_alpha(self):
        sig = ScamSignal(name="s", category=SignalCategory.POSITIVE, alpha=1.0, beta=9.0)
        assert sig.bayesian_weight == pytest.approx(0.1)

    def test_detail_and_evidence(self):
        sig = ScamSignal(
            name="guaranteed_income",
            category=SignalCategory.RED_FLAG,
            detail="Claims $5k/week guaranteed",
            evidence="Earn GUARANTEED $5,000 per week",
        )
        assert "guaranteed" in sig.detail.lower()
        assert "guaranteed" in sig.evidence.lower()

    def test_weight_bounds_not_enforced(self):
        """Dataclass does not enforce [0,1]; verify it stores raw values."""
        sig = ScamSignal(name="s", category=SignalCategory.RED_FLAG, weight=2.5)
        assert sig.weight == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# CompanyProfile dataclass
# ---------------------------------------------------------------------------

class TestCompanyProfile:
    def test_default_values(self):
        cp = CompanyProfile(name="TestCo")
        assert cp.name == "TestCo"
        assert cp.is_verified is False
        assert cp.employee_count == 0
        assert cp.glassdoor_rating == pytest.approx(0.0)
        assert cp.whois_age_days == 0
        assert cp.has_linkedin_page is False

    def test_full_construction(self):
        cp = CompanyProfile(
            name="Google",
            domain="google.com",
            employee_count=150_000,
            is_verified=True,
            whois_age_days=9000,
            glassdoor_rating=4.3,
            linkedin_url="https://www.linkedin.com/company/google",
        )
        assert cp.is_verified is True
        assert cp.employee_count == 150_000
        assert cp.glassdoor_rating == pytest.approx(4.3)


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------

def _make_result(scam_score=0.5, signals=None):
    job = JobPosting(url="https://t.example.com", title="Dev", company="Co")
    return ValidationResult(
        job=job,
        signals=signals or [],
        scam_score=scam_score,
        confidence=0.8,
        risk_level=RiskLevel.SUSPICIOUS,
    )


class TestValidationResult:
    def test_default_construction(self):
        job = JobPosting()
        result = ValidationResult(job=job)
        assert result.scam_score == pytest.approx(0.0)
        assert result.confidence == pytest.approx(0.0)
        assert result.risk_level == RiskLevel.LOW
        assert result.signals == []
        assert result.signal_attributions == []

    def test_signals_list_independent_across_instances(self):
        a = ValidationResult(job=JobPosting())
        b = ValidationResult(job=JobPosting())
        a.signals.append(ScamSignal(name="x", category=SignalCategory.RED_FLAG))
        assert len(b.signals) == 0

    def test_red_flags_property(self):
        sigs = [
            ScamSignal("rf1", SignalCategory.RED_FLAG),
            ScamSignal("w1",  SignalCategory.WARNING),
            ScamSignal("rf2", SignalCategory.RED_FLAG),
        ]
        result = _make_result(signals=sigs)
        red_flags = result.red_flags
        assert len(red_flags) == 2
        assert all(s.category == SignalCategory.RED_FLAG for s in red_flags)

    def test_warnings_property(self):
        sigs = [
            ScamSignal("w1", SignalCategory.WARNING),
            ScamSignal("rf", SignalCategory.RED_FLAG),
        ]
        result = _make_result(signals=sigs)
        assert len(result.warnings) == 1
        assert result.warnings[0].name == "w1"

    def test_ghost_indicators_property(self):
        sigs = [
            ScamSignal("g1", SignalCategory.GHOST_JOB),
            ScamSignal("g2", SignalCategory.GHOST_JOB),
            ScamSignal("p1", SignalCategory.POSITIVE),
        ]
        result = _make_result(signals=sigs)
        assert len(result.ghost_indicators) == 2

    def test_positive_signals_property(self):
        sigs = [
            ScamSignal("pos1", SignalCategory.POSITIVE),
            ScamSignal("neg1", SignalCategory.RED_FLAG),
        ]
        result = _make_result(signals=sigs)
        assert len(result.positive_signals) == 1
        assert result.positive_signals[0].name == "pos1"

    def test_empty_signals(self):
        result = _make_result(signals=[])
        assert result.red_flags == []
        assert result.warnings == []
        assert result.ghost_indicators == []
        assert result.positive_signals == []

    # risk_label thresholds

    def test_risk_label_verified_safe(self):
        assert _make_result(scam_score=0.0).risk_label() == "Verified Safe"
        assert _make_result(scam_score=0.19).risk_label() == "Verified Safe"

    def test_risk_label_likely_legitimate(self):
        assert _make_result(scam_score=0.2).risk_label() == "Likely Legitimate"
        assert _make_result(scam_score=0.39).risk_label() == "Likely Legitimate"

    def test_risk_label_suspicious(self):
        assert _make_result(scam_score=0.4).risk_label() == "Suspicious"
        assert _make_result(scam_score=0.59).risk_label() == "Suspicious"

    def test_risk_label_likely_scam(self):
        assert _make_result(scam_score=0.6).risk_label() == "Likely Scam"
        assert _make_result(scam_score=0.79).risk_label() == "Likely Scam"

    def test_risk_label_almost_certainly_scam(self):
        assert _make_result(scam_score=0.8).risk_label() == "Almost Certainly Scam"
        assert _make_result(scam_score=1.0).risk_label() == "Almost Certainly Scam"

    def test_to_dict_shape(self):
        sigs = [
            ScamSignal("rf1", SignalCategory.RED_FLAG, detail="bad"),
            ScamSignal("w1",  SignalCategory.WARNING,  detail="caution"),
        ]
        result = _make_result(scam_score=0.7, signals=sigs)
        result.ai_tier_used = "haiku"
        result.analysis_time_ms = 42.5
        d = result.to_dict()
        for key in (
            "job", "scam_score", "confidence", "risk_level", "risk_label",
            "red_flags", "warnings", "ghost_indicators", "positive_signals",
            "signal_count", "ai_tier_used", "analysis_time_ms",
        ):
            assert key in d, f"Missing key in to_dict: {key}"
        assert d["scam_score"] == pytest.approx(0.7, abs=0.001)
        assert d["signal_count"] == 2
        assert d["red_flags"][0]["name"] == "rf1"

    def test_to_dict_risk_level_is_string(self):
        result = _make_result()
        d = result.to_dict()
        assert isinstance(d["risk_level"], str)

    def test_to_dict_rounding(self):
        result = _make_result(scam_score=0.123456789)
        d = result.to_dict()
        # Should be rounded to 3 decimal places
        assert d["scam_score"] == pytest.approx(0.123, abs=0.0005)


# ---------------------------------------------------------------------------
# ScamPattern dataclass
# ---------------------------------------------------------------------------

class TestScamPattern:
    def test_precision_no_observations(self):
        p = ScamPattern(
            pattern_id="p1",
            name="Test",
            description="desc",
            category=SignalCategory.RED_FLAG,
        )
        assert p.precision == pytest.approx(0.0)

    def test_precision_with_observations(self):
        p = ScamPattern(
            pattern_id="p1",
            name="Test",
            description="desc",
            category=SignalCategory.RED_FLAG,
            true_positives=8,
            false_positives=2,
        )
        assert p.precision == pytest.approx(0.8)

    def test_bayesian_score_default(self):
        p = ScamPattern(
            pattern_id="p1",
            name="Test",
            description="desc",
            category=SignalCategory.RED_FLAG,
        )
        # alpha=1, beta=1 → 0.5
        assert p.bayesian_score == pytest.approx(0.5)

    def test_bayesian_score_biased(self):
        p = ScamPattern(
            pattern_id="p1",
            name="Test",
            description="desc",
            category=SignalCategory.RED_FLAG,
            alpha=4.0,
            beta=1.0,
        )
        assert p.bayesian_score == pytest.approx(0.8)

    def test_default_status(self):
        p = ScamPattern(
            pattern_id="px",
            name="X",
            description="d",
            category=SignalCategory.WARNING,
        )
        assert p.status == "active"

    def test_keywords_default_empty(self):
        p = ScamPattern(
            pattern_id="pk",
            name="K",
            description="d",
            category=SignalCategory.STRUCTURAL,
        )
        assert p.keywords == []

    def test_keywords_independent_across_instances(self):
        a = ScamPattern("a", "A", "d", SignalCategory.RED_FLAG)
        b = ScamPattern("b", "B", "d", SignalCategory.RED_FLAG)
        a.keywords.append("kw")
        assert "kw" not in b.keywords


# ---------------------------------------------------------------------------
# UserReport dataclass
# ---------------------------------------------------------------------------

class TestUserReport:
    def test_construction(self):
        report = UserReport(url="https://scam.example.com", is_scam=True, reason="Upfront fee")
        assert report.is_scam is True
        assert report.reason == "Upfront fee"
        assert report.our_prediction == pytest.approx(0.0)
        assert report.was_correct is False

    def test_default_is_correct_false(self):
        report = UserReport(url="https://legit.example.com", is_scam=False)
        assert report.was_correct is False

    def test_fully_specified(self):
        report = UserReport(
            url="https://j.example.com",
            is_scam=True,
            reason="Suspicious contact",
            reported_at="2026-04-01T10:00:00",
            our_prediction=0.9,
            was_correct=True,
        )
        assert report.our_prediction == pytest.approx(0.9)
        assert report.was_correct is True
        assert report.reported_at == "2026-04-01T10:00:00"
