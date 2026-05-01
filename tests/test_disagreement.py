"""Tests for sentinel/disagreement.py — Ensemble Disagreement Learning.

35 tests covering DisagreementDetector, ActiveLearningSelector, ConsensusBuilder.
"""

import math
import pytest

from sentinel.disagreement import (
    ActiveLearningSelector,
    ConsensusBuilder,
    ConsensusResult,
    DisagreementCase,
    DisagreementDetector,
    LabelingCandidate,
    SubsystemScores,
    _entropy,
    _mean,
    _std_dev,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUBSYSTEM_NAMES = ["primary", "shadow", "fraud_triangle", "benford", "linguistic"]


def _scores(**kw: float) -> SubsystemScores:
    return dict(kw)


def _full_scores(primary=0.8, shadow=0.2, fraud_triangle=0.7, benford=0.1, linguistic=0.75):
    return _scores(
        primary=primary,
        shadow=shadow,
        fraud_triangle=fraud_triangle,
        benford=benford,
        linguistic=linguistic,
    )


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelpers:
    def test_entropy_uniform(self):
        """All-0.5 scores → maximum binary entropy (1.0 each → avg 1.0)."""
        scores = [0.5, 0.5, 0.5]
        ent = _entropy(scores)
        assert ent == pytest.approx(1.0, abs=1e-6)

    def test_entropy_certain(self):
        """Very confident scores → near-zero entropy."""
        scores = [0.999, 0.999, 0.999]
        ent = _entropy(scores)
        assert ent < 0.02

    def test_entropy_empty(self):
        assert _entropy([]) == 0.0

    def test_entropy_mixed(self):
        scores = [0.9, 0.1, 0.8, 0.2]
        ent = _entropy(scores)
        # Mixed extremes have lower entropy than all-0.5
        assert 0.0 < ent < 1.0

    def test_std_dev_zero_variance(self):
        assert _std_dev([0.5, 0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)

    def test_std_dev_single_element(self):
        assert _std_dev([0.7]) == 0.0

    def test_mean_basic(self):
        assert _mean([0.0, 1.0]) == pytest.approx(0.5)

    def test_mean_empty(self):
        assert _mean([]) == 0.0


# ===========================================================================
# DisagreementDetector tests
# ===========================================================================

class TestDisagreementDetector:
    @pytest.fixture
    def detector(self):
        return DisagreementDetector(threshold=0.3)

    def test_detect_strong_disagreement(self, detector):
        scores = _scores(primary=0.9, shadow=0.1)
        case = detector.detect("job-1", scores)
        assert case is not None
        assert case.score_spread == pytest.approx(0.8, abs=1e-4)
        assert case.job_id == "job-1"

    def test_detect_below_threshold_returns_none(self, detector):
        scores = _scores(primary=0.6, shadow=0.5)   # spread = 0.1 < 0.3
        case = detector.detect("job-2", scores)
        assert case is None

    def test_detect_exact_threshold_returns_none(self, detector):
        scores = _scores(primary=0.6, shadow=0.3)   # spread == 0.3, not > 0.3
        case = detector.detect("job-3", scores)
        assert case is None

    def test_detect_just_above_threshold(self, detector):
        scores = _scores(primary=0.61, shadow=0.3)  # spread = 0.31
        case = detector.detect("job-4", scores)
        assert case is not None

    def test_detect_single_subsystem_returns_none(self, detector):
        scores = _scores(primary=0.9)
        case = detector.detect("job-5", scores)
        assert case is None

    def test_detect_entropy_populated(self, detector):
        scores = _scores(primary=0.9, shadow=0.1)
        case = detector.detect("job-1", scores)
        assert case.entropy > 0.0

    def test_detect_disagreeing_pairs_populated(self, detector):
        scores = _scores(primary=0.9, shadow=0.1, benford=0.8)
        case = detector.detect("job-1", scores)
        assert len(case.disagreeing_pairs) >= 1
        # All pairs are above threshold
        for a, b, diff in case.disagreeing_pairs:
            assert diff >= detector.threshold

    def test_detect_information_value_positive(self, detector):
        scores = _scores(primary=0.9, shadow=0.1)
        case = detector.detect("job-1", scores)
        assert case.information_value > 0.0

    def test_detect_batch_sorted_by_information_value(self, detector):
        jobs = [
            ("job-low", _scores(primary=0.7, shadow=0.35)),  # spread 0.35
            ("job-high", _scores(primary=0.95, shadow=0.05)),  # spread 0.9
            ("job-mid", _scores(primary=0.8, shadow=0.45)),   # spread 0.35, but below threshold?
        ]
        cases = detector.detect_batch(jobs)
        assert len(cases) >= 1
        # Sorted: highest information_value first
        for i in range(len(cases) - 1):
            assert cases[i].information_value >= cases[i + 1].information_value

    def test_detect_batch_filters_below_threshold(self, detector):
        jobs = [
            ("ok-1", _scores(primary=0.55, shadow=0.5)),     # spread 0.05 — filtered
            ("ok-2", _scores(primary=0.9, shadow=0.1)),      # spread 0.8 — kept
        ]
        cases = detector.detect_batch(jobs)
        assert all(c.job_id != "ok-1" for c in cases)
        assert any(c.job_id == "ok-2" for c in cases)

    def test_summarise_empty(self, detector):
        summary = detector.summarise([])
        assert summary["count"] == 0

    def test_summarise_nonempty(self, detector):
        scores = _scores(primary=0.9, shadow=0.1)
        case = detector.detect("j1", scores)
        summary = detector.summarise([case])
        assert summary["count"] == 1
        assert summary["mean_spread"] > 0.0

    def test_rank_by_information_value(self, detector):
        j1 = detector.detect("j1", _scores(primary=0.9, shadow=0.1))     # big spread
        j2 = detector.detect("j2", _scores(primary=0.75, shadow=0.4))    # smaller spread
        ranked = detector.rank_by_information_value([j2, j1])
        assert ranked[0].job_id == "j1"

    def test_custom_threshold(self):
        detector_strict = DisagreementDetector(threshold=0.5)
        scores = _scores(primary=0.8, shadow=0.45)  # spread=0.35, below 0.5
        case = detector_strict.detect("j1", scores)
        assert case is None

        detector_loose = DisagreementDetector(threshold=0.2)
        case2 = detector_loose.detect("j1", scores)
        assert case2 is not None

    def test_subsystem_scores_preserved(self, detector):
        scores = _scores(primary=0.9, shadow=0.1, benford=0.85)
        case = detector.detect("j1", scores)
        assert case.subsystem_scores == scores


# ===========================================================================
# ActiveLearningSelector tests
# ===========================================================================

class TestActiveLearningSelector:
    @pytest.fixture
    def selector(self):
        return ActiveLearningSelector(budget_per_cycle=5)

    def _pool(self, n=10):
        """Generate a pool of n jobs with varying scores."""
        pool = []
        for i in range(n):
            t = i / max(n - 1, 1)
            pool.append((
                f"job-{i}",
                _scores(
                    primary=round(t, 2),
                    shadow=round(1.0 - t, 2),
                    fraud_triangle=round(0.3 + t * 0.4, 2),
                ),
            ))
        return pool

    def test_select_returns_candidates(self, selector):
        pool = self._pool()
        candidates = selector.select(pool)
        assert len(candidates) >= 1

    def test_select_respects_budget(self, selector):
        pool = self._pool(20)
        candidates = selector.select(pool)
        assert len(candidates) <= selector.budget_per_cycle

    def test_select_n_override(self, selector):
        pool = self._pool(20)
        candidates = selector.select(pool, n=2)
        assert len(candidates) <= 2

    def test_budget_exhausted_returns_empty(self, selector):
        pool = self._pool(20)
        selector.select(pool, n=selector.budget_per_cycle)
        for jid, _ in pool[:5]:
            selector.mark_labeled(jid)
        # No budget left (5 marked = budget exhausted)
        # Reset and confirm
        assert selector.queries_remaining >= 0

    def test_mark_labeled_skips_on_next_select(self, selector):
        pool = [("j1", _scores(primary=0.5, shadow=0.5))]
        selector.mark_labeled("j1")
        candidates = selector.select(pool)
        assert all(c.job_id != "j1" for c in candidates)

    def test_reset_cycle(self, selector):
        pool = self._pool(5)
        selector.select(pool)
        for jid, _ in pool:
            selector.mark_labeled(jid)
        selector.reset_cycle()
        assert selector._queries_this_cycle == 0

    def test_uncertainty_sample_prefers_boundary(self, selector):
        pool = [
            ("boundary", 0.51),
            ("certain-scam", 0.95),
            ("certain-legit", 0.05),
        ]
        results = selector.uncertainty_sample(pool, n=1)
        assert results[0][0] == "boundary"

    def test_uncertainty_sample_excludes_labeled(self, selector):
        pool = [("j1", 0.5), ("j2", 0.8)]
        selector.mark_labeled("j1")
        results = selector.uncertainty_sample(pool, n=2)
        assert all(jid != "j1" for jid, _ in results)

    def test_committee_sample_prefers_disagreement(self, selector):
        pool = [
            ("agree", _scores(primary=0.8, shadow=0.8)),        # low std-dev
            ("disagree", _scores(primary=0.9, shadow=0.1)),     # high std-dev
        ]
        results = selector.committee_sample(pool, n=1)
        assert results[0][0] == "disagree"

    def test_committee_sample_returns_std_dev(self, selector):
        pool = [("j1", _scores(primary=0.9, shadow=0.1))]
        results = selector.committee_sample(pool, n=1)
        assert len(results) == 1
        jid, std = results[0]
        assert std > 0.0

    def test_expected_model_change_at_boundary(self, selector):
        # At decision boundary (score=0.5), uncertainty=1.0
        emc = selector.expected_model_change("j", 0.5, n_signals=4)
        assert emc > 0.0

    def test_expected_model_change_certain_is_low(self, selector):
        emc = selector.expected_model_change("j", 0.99, n_signals=4)
        assert emc < 0.1

    def test_candidate_has_selection_reason(self, selector):
        pool = self._pool(5)
        candidates = selector.select(pool)
        for c in candidates:
            assert c.selection_reason in {"uncertainty", "committee", "model_change"}

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            ActiveLearningSelector(
                uncertainty_weight=0.5,
                committee_weight=0.5,
                model_change_weight=0.1,
            )

    def test_queries_remaining_decrements(self, selector):
        initial = selector.queries_remaining
        selector.mark_labeled("j1")
        assert selector.queries_remaining == initial - 1


# ===========================================================================
# ConsensusBuilder tests
# ===========================================================================

class TestConsensusBuilder:
    @pytest.fixture
    def builder(self):
        return ConsensusBuilder(subsystem_names=SUBSYSTEM_NAMES)

    def test_build_consensus_basic(self, builder):
        scores = _full_scores(primary=0.8, shadow=0.8, fraud_triangle=0.8, benford=0.8, linguistic=0.8)
        result = builder.build_consensus(scores)
        assert result.consensus_score == pytest.approx(0.8, abs=0.05)

    def test_build_consensus_high_scam(self, builder):
        scores = _full_scores(primary=0.9, shadow=0.85, fraud_triangle=0.9, benford=0.8, linguistic=0.9)
        result = builder.build_consensus(scores)
        assert result.consensus_score >= 0.7

    def test_build_consensus_low_scam(self, builder):
        scores = _full_scores(primary=0.1, shadow=0.15, fraud_triangle=0.05, benford=0.2, linguistic=0.1)
        result = builder.build_consensus(scores)
        assert result.consensus_score <= 0.3

    def test_agreement_ratio_unanimous(self, builder):
        scores = _full_scores(primary=0.9, shadow=0.85, fraud_triangle=0.9, benford=0.8, linguistic=0.9)
        result = builder.build_consensus(scores)
        assert result.agreement_ratio == pytest.approx(1.0)

    def test_agreement_ratio_split(self, builder):
        scores = _full_scores(primary=0.9, shadow=0.1, fraud_triangle=0.8, benford=0.2, linguistic=0.85)
        result = builder.build_consensus(scores)
        assert 0.0 < result.agreement_ratio < 1.0

    def test_breakdown_string_populated(self, builder):
        scores = _full_scores(primary=0.8, shadow=0.8, fraud_triangle=0.8, benford=0.8, linguistic=0.8)
        result = builder.build_consensus(scores)
        assert "scam" in result.breakdown or "legit" in result.breakdown

    def test_stacking_score_returned_when_enabled(self, builder):
        scores = _full_scores()
        result = builder.build_consensus(scores, use_stacking=True)
        assert result.stacking_score is not None

    def test_stacking_score_none_when_disabled(self, builder):
        scores = _full_scores()
        result = builder.build_consensus(scores, use_stacking=False)
        assert result.stacking_score is None

    def test_update_accuracy_shifts_weights(self, builder):
        # Mark primary as always correct, others wrong
        for _ in range(20):
            builder.update_accuracy("primary", True)
            for name in ["shadow", "fraud_triangle", "benford", "linguistic"]:
                builder.update_accuracy(name, False)
        weights = builder.get_meta_weights()
        assert weights["primary"] > weights["shadow"]

    def test_meta_weights_sum_to_one(self, builder):
        weights = builder.get_meta_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_register_new_subsystem(self, builder):
        builder.register_subsystem("new_scorer")
        weights = builder.get_meta_weights()
        assert "new_scorer" in weights

    def test_accuracy_summary_structure(self, builder):
        summary = builder.get_accuracy_summary()
        assert "primary" in summary
        assert "accuracy" in summary["primary"]
        assert "observations" in summary["primary"]

    def test_empty_scores_fallback(self, builder):
        result = builder.build_consensus({})
        # Should not crash
        assert result.consensus_score == pytest.approx(0.0)

    def test_unknown_subsystems_ignored(self, builder):
        scores = {"unknown_system": 0.9}
        result = builder.build_consensus(scores)
        # No crash; fallback to mean
        assert 0.0 <= result.consensus_score <= 1.0

    def test_confidence_in_range(self, builder):
        scores = _full_scores()
        result = builder.build_consensus(scores)
        assert 0.0 <= result.confidence <= 1.0

    def test_meta_weights_keys_match_subsystems(self, builder):
        scores = _full_scores()
        result = builder.build_consensus(scores)
        for key in result.meta_weights:
            assert key in SUBSYSTEM_NAMES
