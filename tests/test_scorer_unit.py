"""Unit tests for sentinel/scorer.py — Bayesian scoring, risk classification, EnsembleScorer."""

import math
import pytest

from sentinel.models import (
    JobPosting,
    RiskLevel,
    ScamSignal,
    SignalCategory,
    ValidationResult,
)
from sentinel.scorer import (
    EnsembleResult,
    EnsembleScorer,
    SignalWeightTracker,
    _RISK_THRESHOLDS,
    _reset_learned_weights_cache,
    build_result,
    classify_risk,
    score_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _red_flag(name: str = "test_flag", weight: float = 0.9) -> ScamSignal:
    return ScamSignal(
        name=name,
        category=SignalCategory.RED_FLAG,
        weight=weight,
        confidence=0.8,
        detail="test red flag",
        evidence="test",
    )


def _positive(name: str = "test_positive", weight: float = 0.3) -> ScamSignal:
    return ScamSignal(
        name=name,
        category=SignalCategory.POSITIVE,
        weight=weight,
        confidence=0.8,
        detail="test positive signal",
        evidence="test",
    )


def _make_job(**kwargs) -> JobPosting:
    defaults = dict(
        url="https://example.com/job/1",
        title="Software Engineer",
        company="Acme Corp",
        location="Remote",
        description="Build distributed systems using Python and Kubernetes.",
        company_linkedin_url="https://linkedin.com/company/acme",
    )
    defaults.update(kwargs)
    return JobPosting(**defaults)


# ---------------------------------------------------------------------------
# score_signals()
# ---------------------------------------------------------------------------

class TestScoreSignals:
    def setup_method(self):
        _reset_learned_weights_cache()

    def test_empty_signals_returns_zero_zero(self):
        score, confidence = score_signals([], use_learned_weights=False)
        assert score == 0.0
        assert confidence == 0.0

    def test_red_flags_return_high_score(self):
        signals = [
            _red_flag("upfront_payment", weight=0.95),
            _red_flag("personal_info_request", weight=0.92),
            _red_flag("guaranteed_income", weight=0.85),
        ]
        score, confidence = score_signals(signals, use_learned_weights=False)
        assert score > 0.8, f"Expected high score with red flags, got {score}"
        assert confidence > 0.0

    def test_positive_signals_return_low_score(self):
        signals = [
            _positive("established_company", weight=0.35),
            _positive("detailed_requirements", weight=0.38),
            _positive("verified_website", weight=0.32),
        ]
        score, confidence = score_signals(signals, use_learned_weights=False)
        assert score < 0.5, f"Expected low score with positive signals, got {score}"
        assert confidence > 0.0

    def test_mixed_signals_return_moderate_score(self):
        signals = [
            _red_flag("upfront_payment", weight=0.9),
            _positive("established_company", weight=0.35),
            _red_flag("urgency", weight=0.6),
            _positive("detailed_requirements", weight=0.38),
        ]
        score, confidence = score_signals(signals, use_learned_weights=False)
        # With mixed signals, score should be moderate and confidence lower
        assert 0.0 < score < 1.0
        # Mixed signals should reduce the agreement factor → lower confidence
        # compared to all-red-flag scenario
        all_red_score, all_red_conf = score_signals(
            [_red_flag("a"), _red_flag("b"), _red_flag("c"), _red_flag("d")],
            use_learned_weights=False,
        )
        assert confidence < all_red_conf

    def test_score_bounds_are_0_to_1(self):
        many_red_flags = [_red_flag(f"flag_{i}", weight=0.99) for i in range(20)]
        score, confidence = score_signals(many_red_flags, use_learned_weights=False)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_single_strong_red_flag_raises_score(self):
        signals = [_red_flag("crypto_payment", weight=0.90)]
        score, confidence = score_signals(signals, use_learned_weights=False)
        assert score > 0.5

    def test_more_red_flags_increases_score(self):
        one_flag = [_red_flag("a", weight=0.8)]
        three_flags = [_red_flag("a", weight=0.8), _red_flag("b", weight=0.8), _red_flag("c", weight=0.8)]
        score_one, _ = score_signals(one_flag, use_learned_weights=False)
        score_three, _ = score_signals(three_flags, use_learned_weights=False)
        assert score_three > score_one


# ---------------------------------------------------------------------------
# classify_risk()
# ---------------------------------------------------------------------------

class TestClassifyRisk:
    def test_below_safe_threshold_is_safe(self):
        assert classify_risk(0.0) == RiskLevel.SAFE
        assert classify_risk(0.19) == RiskLevel.SAFE

    def test_at_safe_boundary_is_low(self):
        # 0.2 is the upper bound for SAFE; at 0.2 we should get LOW
        assert classify_risk(0.2) == RiskLevel.LOW

    def test_low_range(self):
        assert classify_risk(0.2) == RiskLevel.LOW
        assert classify_risk(0.39) == RiskLevel.LOW

    def test_at_low_boundary_is_suspicious(self):
        assert classify_risk(0.4) == RiskLevel.SUSPICIOUS

    def test_suspicious_range(self):
        assert classify_risk(0.4) == RiskLevel.SUSPICIOUS
        assert classify_risk(0.59) == RiskLevel.SUSPICIOUS

    def test_at_suspicious_boundary_is_high(self):
        assert classify_risk(0.6) == RiskLevel.HIGH

    def test_high_range(self):
        assert classify_risk(0.6) == RiskLevel.HIGH
        assert classify_risk(0.79) == RiskLevel.HIGH

    def test_at_high_boundary_is_scam(self):
        assert classify_risk(0.8) == RiskLevel.SCAM

    def test_above_high_threshold_is_scam(self):
        assert classify_risk(0.8) == RiskLevel.SCAM
        assert classify_risk(0.9) == RiskLevel.SCAM
        assert classify_risk(1.0) == RiskLevel.SCAM

    def test_exact_threshold_boundaries(self):
        """Verify each threshold boundary in _RISK_THRESHOLDS."""
        safe_thresh = _RISK_THRESHOLDS["safe"]        # 0.2
        low_thresh = _RISK_THRESHOLDS["low"]          # 0.4
        suspicious_thresh = _RISK_THRESHOLDS["suspicious"]  # 0.6
        high_thresh = _RISK_THRESHOLDS["high"]        # 0.8

        assert classify_risk(safe_thresh - 0.001) == RiskLevel.SAFE
        assert classify_risk(safe_thresh) == RiskLevel.LOW
        assert classify_risk(low_thresh - 0.001) == RiskLevel.LOW
        assert classify_risk(low_thresh) == RiskLevel.SUSPICIOUS
        assert classify_risk(suspicious_thresh - 0.001) == RiskLevel.SUSPICIOUS
        assert classify_risk(suspicious_thresh) == RiskLevel.HIGH
        assert classify_risk(high_thresh - 0.001) == RiskLevel.HIGH
        assert classify_risk(high_thresh) == RiskLevel.SCAM


# ---------------------------------------------------------------------------
# build_result()
# ---------------------------------------------------------------------------

class TestBuildResult:
    def setup_method(self):
        _reset_learned_weights_cache()

    def test_returns_validation_result(self):
        job = _make_job()
        signals = [_red_flag("upfront_payment", weight=0.95)]
        result = build_result(job, signals, analysis_time_ms=12.5)
        assert isinstance(result, ValidationResult)

    def test_result_has_all_fields_populated(self):
        job = _make_job()
        signals = [
            _red_flag("upfront_payment", weight=0.95),
            _red_flag("crypto_payment", weight=0.90),
        ]
        result = build_result(job, signals, analysis_time_ms=25.0)

        assert result.job is job
        assert result.signals is signals
        assert isinstance(result.scam_score, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.risk_level, RiskLevel)
        assert result.analysis_time_ms == 25.0
        assert isinstance(result.ensemble_disagreement, float)
        assert isinstance(result.needs_review, bool)

    def test_result_scam_score_matches_score_signals(self):
        job = _make_job()
        signals = [_red_flag("test", weight=0.9)]
        result = build_result(job, signals)
        expected_score, _ = score_signals(signals)
        assert result.scam_score == expected_score

    def test_result_risk_level_matches_classify_risk(self):
        job = _make_job()
        signals = [_red_flag("upfront_payment", weight=0.95)] * 5
        result = build_result(job, signals)
        expected_risk = classify_risk(result.scam_score)
        assert result.risk_level == expected_risk

    def test_empty_signals_give_safe_result(self):
        job = _make_job()
        result = build_result(job, [])
        assert result.scam_score == 0.0
        assert result.confidence == 0.0
        assert result.risk_level == RiskLevel.SAFE

    def test_signal_attributions_is_list(self):
        job = _make_job()
        signals = [_red_flag("a"), _positive("b")]
        result = build_result(job, signals)
        assert isinstance(result.signal_attributions, list)


# ---------------------------------------------------------------------------
# EnsembleScorer.score_ensemble()
# ---------------------------------------------------------------------------

class TestEnsembleScorer:
    def test_score_ensemble_returns_ensemble_result(self):
        scorer = EnsembleScorer()
        job = _make_job()
        signals = [_red_flag("upfront_payment", weight=0.95)]
        result = scorer.score_ensemble(None, job, signals)
        assert isinstance(result, EnsembleResult)

    def test_ensemble_result_has_disagreement(self):
        scorer = EnsembleScorer()
        job = _make_job()
        # Mixed signals should cause non-trivial disagreement
        signals = [
            _red_flag("a", weight=0.9),
            _positive("b", weight=0.35),
        ]
        result = scorer.score_ensemble(None, job, signals)
        assert hasattr(result, "disagreement")
        assert isinstance(result.disagreement, float)
        assert result.disagreement >= 0.0

    def test_ensemble_result_fields_are_populated(self):
        scorer = EnsembleScorer()
        job = _make_job()
        signals = [_red_flag("test", weight=0.85)]
        result = scorer.score_ensemble(None, job, signals)

        assert 0.0 <= result.primary_score <= 1.0
        assert 0.0 <= result.weighted_avg_score <= 1.0
        assert 0.0 <= result.majority_vote_score <= 1.0
        assert 0.0 <= result.ensemble_score <= 1.0
        assert isinstance(result.method_scores, dict)
        assert "primary" in result.method_scores
        assert "weighted_avg" in result.method_scores
        assert "majority_vote" in result.method_scores

    def test_high_disagreement_triggers_confidence_adjustment(self):
        """When std-dev > 0.2, confidence_adjustment should be -0.2."""
        scorer = EnsembleScorer()
        job = _make_job()
        # Use a mix that is likely to produce high disagreement:
        # one very strong red flag to push primary high,
        # plus positives to confuse the weighted_avg/majority_vote methods.
        signals = [
            _red_flag("a", weight=0.99),
            _red_flag("b", weight=0.99),
            _positive("c", weight=0.01),
            _positive("d", weight=0.01),
        ]
        result = scorer.score_ensemble(None, job, signals)
        # disagreement >= DISAGREEMENT_THRESHOLD → confidence_adjustment == -0.2
        if result.disagreement > EnsembleScorer.DISAGREEMENT_THRESHOLD:
            assert result.confidence_adjustment == -0.2
        else:
            assert result.confidence_adjustment == 0.0

    def test_no_signals_returns_zero_scores(self):
        scorer = EnsembleScorer()
        job = _make_job()
        result = scorer.score_ensemble(None, job, [])
        assert result.primary_score == 0.0
        assert result.weighted_avg_score == 0.0
        assert result.majority_vote_score == 0.0

    def test_update_method_accuracy_updates_posteriors(self):
        scorer = EnsembleScorer()
        initial = scorer.get_method_accuracy()["primary"]
        scorer.update_method_accuracy(None, "primary", was_correct=True)
        updated = scorer.get_method_accuracy()["primary"]
        # Correct observation should push accuracy upward
        assert updated > initial

    def test_update_method_accuracy_incorrect_lowers_accuracy(self):
        scorer = EnsembleScorer()
        # Start from default Beta(1,1) = 0.5
        scorer.update_method_accuracy(None, "primary", was_correct=False)
        after_wrong = scorer.get_method_accuracy()["primary"]
        assert after_wrong < 0.5

    def test_update_method_accuracy_unknown_name_is_ignored(self):
        scorer = EnsembleScorer()
        # Should not raise
        scorer.update_method_accuracy(None, "nonexistent_method", was_correct=True)
        # Posteriors for valid methods unchanged
        acc = scorer.get_method_accuracy()
        assert set(acc.keys()) == {"primary", "weighted_avg", "majority_vote"}

    def test_get_method_accuracy_returns_all_methods(self):
        scorer = EnsembleScorer()
        acc = scorer.get_method_accuracy()
        assert set(acc.keys()) == {"primary", "weighted_avg", "majority_vote"}
        for name, v in acc.items():
            assert 0.0 < v < 1.0, f"Method {name} accuracy {v} out of range"


# ---------------------------------------------------------------------------
# SignalWeightTracker — Thompson Sampling
# ---------------------------------------------------------------------------

class TestSignalWeightTracker:
    def test_get_weight_returns_value_in_0_1(self):
        tracker = SignalWeightTracker()
        for _ in range(50):
            w = tracker.get_weight("some_signal")
            assert 0.0 <= w <= 1.0, f"Thompson sample {w} out of [0,1]"

    def test_get_weight_unknown_signal_uses_uniform_prior(self):
        """Beta(1,1) is the uniform prior — samples should span the full [0,1] range."""
        tracker = SignalWeightTracker()
        samples = [tracker.get_weight("unseen") for _ in range(200)]
        # With Beta(1,1) the mean should be near 0.5
        mean = sum(samples) / len(samples)
        assert 0.2 < mean < 0.8, f"Uniform prior mean {mean} unexpectedly skewed"

    def test_many_correct_observations_push_weight_high(self):
        tracker = SignalWeightTracker()
        for _ in range(50):
            tracker.update("strong_signal", was_correct=True)
        # After 50 correct obs, Beta(51,1); mean ≈ 51/52 ≈ 0.98
        samples = [tracker.get_weight("strong_signal") for _ in range(100)]
        mean = sum(samples) / len(samples)
        assert mean > 0.8, f"Expected high mean after many correct obs, got {mean}"

    def test_many_incorrect_observations_push_weight_low(self):
        tracker = SignalWeightTracker()
        for _ in range(50):
            tracker.update("weak_signal", was_correct=False)
        samples = [tracker.get_weight("weak_signal") for _ in range(100)]
        mean = sum(samples) / len(samples)
        assert mean < 0.2, f"Expected low mean after many incorrect obs, got {mean}"

    def test_update_increments_alpha_on_correct(self):
        tracker = SignalWeightTracker()
        tracker.update("sig", was_correct=True)
        alpha, beta = tracker.get_posterior("sig")
        assert alpha == 2.0  # prior 1.0 + 1 correct
        assert beta == 1.0

    def test_update_increments_beta_on_incorrect(self):
        tracker = SignalWeightTracker()
        tracker.update("sig", was_correct=False)
        alpha, beta = tracker.get_posterior("sig")
        assert alpha == 1.0
        assert beta == 2.0  # prior 1.0 + 1 incorrect

    def test_get_ranking_returns_sorted_by_mean(self):
        tracker = SignalWeightTracker()
        for _ in range(20):
            tracker.update("good_signal", was_correct=True)
        for _ in range(20):
            tracker.update("bad_signal", was_correct=False)

        ranking = tracker.get_ranking()
        names = [r[0] for r in ranking]
        assert names.index("good_signal") < names.index("bad_signal")

    def test_save_and_load_roundtrip(self, tmp_path):
        tracker = SignalWeightTracker()
        tracker.update("sig_a", was_correct=True)
        tracker.update("sig_b", was_correct=False)

        path = str(tmp_path / "weights.json")
        tracker.save(path)

        tracker2 = SignalWeightTracker()
        tracker2.load(path)
        assert tracker2.get_posterior("sig_a") == tracker.get_posterior("sig_a")
        assert tracker2.get_posterior("sig_b") == tracker.get_posterior("sig_b")
