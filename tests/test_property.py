"""Property-based tests for JobSentinel using Hypothesis.

These tests verify robustness invariants — no crashes, valid output ranges,
and type contracts — across the full space of arbitrary job posting inputs.
"""

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from sentinel.models import JobPosting, RiskLevel
from sentinel.signals import extract_signals, ALL_SIGNALS
from sentinel.scorer import score_signals, classify_risk, build_result

# Strategy for generating random JobPostings
job_strategy = st.builds(
    JobPosting,
    url=st.text(max_size=200),
    title=st.text(max_size=500),
    company=st.text(max_size=200),
    location=st.text(max_size=200),
    description=st.text(max_size=5000),
    salary_min=st.floats(min_value=0, max_value=1_000_000, allow_nan=False, allow_infinity=False),
    salary_max=st.floats(min_value=0, max_value=1_000_000, allow_nan=False, allow_infinity=False),
    posted_date=st.sampled_from(["", "2024-01-01", "2026-04-01", "3 days ago", "invalid"]),
    recruiter_connections=st.integers(min_value=0, max_value=10000),
    is_remote=st.booleans(),
    is_repost=st.booleans(),
)


class TestSignalExtractorRobustness:
    @given(job=job_strategy)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_extract_signals_never_raises(self, job):
        """No signal extractor should ever raise an exception."""
        signals = extract_signals(job)
        assert isinstance(signals, list)

    @given(job=job_strategy)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_all_signals_have_valid_category(self, job):
        """Every extracted signal has a valid SignalCategory."""
        from sentinel.models import SignalCategory
        signals = extract_signals(job)
        for s in signals:
            assert isinstance(s.category, SignalCategory)


class TestScorerRobustness:
    @given(job=job_strategy)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_score_always_in_range(self, job):
        """Score is always between 0 and 1."""
        signals = extract_signals(job)
        score, confidence = score_signals(signals)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0

    @given(job=job_strategy)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_classify_risk_always_valid(self, job):
        """classify_risk always returns a valid RiskLevel."""
        signals = extract_signals(job)
        score, _ = score_signals(signals)
        risk = classify_risk(score)
        assert isinstance(risk, RiskLevel)


class TestBuildResultRobustness:
    @given(job=job_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_build_result_always_returns_validation_result(self, job):
        """build_result never crashes and always returns a ValidationResult."""
        from sentinel.models import ValidationResult
        signals = extract_signals(job)
        result = build_result(job, signals)
        assert isinstance(result, ValidationResult)
        assert 0.0 <= result.scam_score <= 1.0


class TestIndividualSignals:
    @given(desc=st.text(max_size=3000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_no_individual_signal_crashes(self, desc):
        """Each individual signal function handles arbitrary text."""
        job = JobPosting(description=desc)
        for signal_fn in ALL_SIGNALS:
            result = signal_fn(job)
            assert result is None or hasattr(result, 'name')
