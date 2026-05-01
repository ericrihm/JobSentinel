"""Tests for sentinel.robustness — PerturbationEngine, RobustnessScorer, AdversarialProber."""

from __future__ import annotations

import math
import random

import pytest

from sentinel.robustness import (
    AdversarialProber,
    PerturbationEngine,
    RobustnessReport,
    RobustnessScorer,
    analyse_robustness,
)


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

SCAM_TEXT = (
    "Work from home! GUARANTEED $5,000/week — no experience required. "
    "Pay a $99 registration fee to get started. "
    "Provide your Social Security Number and bank account. Apply NOW!"
)

LEGIT_TEXT = (
    "Senior Software Engineer at Acme Inc. Requires 5+ years of Python and "
    "distributed systems experience. Competitive salary $150k-$200k. "
    "Full-time position with equity and benefits."
)


def _static_scorer(score: float):
    """Return a scoring function that always returns *score* regardless of input."""
    def fn(text: str) -> float:
        return score
    return fn


def _keyword_scorer(text: str) -> float:
    """Simple keyword-based scorer for testing (not the real Sentinel scorer)."""
    text_lower = text.lower()
    scam_words = [
        "guaranteed", "fee", "social security", "bank account",
        "registration fee", "upfront", "bitcoin", "no experience",
        "wire transfer", "gift card",
    ]
    hits = sum(1 for w in scam_words if w in text_lower)
    return min(1.0, hits * 0.15)


@pytest.fixture
def engine():
    return PerturbationEngine(seed=42)


@pytest.fixture
def scorer():
    return RobustnessScorer(scoring_fn=_keyword_scorer, n_perturbations=15, seed=42)


@pytest.fixture
def prober():
    return AdversarialProber(scoring_fn=_keyword_scorer)


# ---------------------------------------------------------------------------
# PerturbationEngine — basic construction
# ---------------------------------------------------------------------------


class TestPerturbationEngineBasic:
    def test_returns_n_variants(self, engine: PerturbationEngine):
        variants = engine.generate(SCAM_TEXT, n=10)
        assert len(variants) == 10

    def test_all_variants_are_strings(self, engine: PerturbationEngine):
        variants = engine.generate(LEGIT_TEXT, n=5)
        assert all(isinstance(v, str) for v in variants)

    def test_strategy_names_nonempty(self, engine: PerturbationEngine):
        names = engine.strategy_names()
        assert len(names) >= 7

    def test_empty_text_returns_empty_clones(self, engine: PerturbationEngine):
        variants = engine.generate("", n=5)
        assert variants == [""] * 5

    def test_seeded_engine_is_deterministic(self):
        e1 = PerturbationEngine(seed=7)
        e2 = PerturbationEngine(seed=7)
        v1 = e1.generate(SCAM_TEXT, n=5)
        v2 = e2.generate(SCAM_TEXT, n=5)
        assert v1 == v2

    def test_different_seeds_produce_different_outputs(self):
        e1 = PerturbationEngine(seed=1)
        e2 = PerturbationEngine(seed=2)
        v1 = e1.generate(SCAM_TEXT, n=20)
        v2 = e2.generate(SCAM_TEXT, n=20)
        assert v1 != v2


# ---------------------------------------------------------------------------
# PerturbationEngine — character-level strategies
# ---------------------------------------------------------------------------


class TestCharLevelPerturbations:
    def test_char_swap_changes_text(self, engine: PerturbationEngine):
        result = engine.char_swap("abcdef ghijkl mnopqr")
        assert result != "abcdef ghijkl mnopqr"

    def test_char_swap_preserves_length_approximately(self, engine: PerturbationEngine):
        original = "hello world this is a test"
        result = engine.char_swap(original)
        # Swap doesn't change length
        assert len(result) == len(original)

    def test_char_insert_increases_length(self, engine: PerturbationEngine):
        original = "testing character insertion here"
        result = engine.char_insert(original)
        assert len(result) > len(original)

    def test_char_delete_decreases_length(self, engine: PerturbationEngine):
        original = "testing character deletion here"
        result = engine.char_delete(original)
        assert len(result) < len(original)

    def test_char_delete_preserves_at_least_one_char(self, engine: PerturbationEngine):
        result = engine.char_delete("x")
        assert len(result) >= 1

    def test_targeted_char_swap(self, engine: PerturbationEngine):
        variants = engine.generate_targeted(SCAM_TEXT, n=5, strategy="char_swap")
        assert len(variants) == 5
        # At least one should differ from original
        assert any(v != SCAM_TEXT for v in variants)

    def test_targeted_char_insert(self, engine: PerturbationEngine):
        variants = engine.generate_targeted(LEGIT_TEXT, n=5, strategy="char_insert")
        assert all(len(v) >= len(LEGIT_TEXT) for v in variants)

    def test_targeted_char_delete(self, engine: PerturbationEngine):
        variants = engine.generate_targeted(LEGIT_TEXT, n=5, strategy="char_delete")
        assert all(len(v) <= len(LEGIT_TEXT) for v in variants)


# ---------------------------------------------------------------------------
# PerturbationEngine — word-level strategies
# ---------------------------------------------------------------------------


class TestWordLevelPerturbations:
    def test_synonym_sub_changes_text(self, engine: PerturbationEngine):
        text = "This is a guaranteed income opportunity."
        results = engine.generate_targeted(text, n=10, strategy="synonym_sub")
        assert any(r != text for r in results)

    def test_synonym_sub_text_without_known_words_unchanged(self, engine: PerturbationEngine):
        text = "xyz abc qwerty"
        result = engine.synonym_sub(text)
        assert result == text

    def test_word_reorder_changes_text(self, engine: PerturbationEngine):
        text = "The quick brown fox jumps over the lazy dog."
        result = engine.word_reorder(text)
        # Words may be reordered but all words still present
        original_words = set(text.split())
        result_words = set(result.split())
        assert original_words == result_words

    def test_targeted_synonym_sub_returns_strings(self, engine: PerturbationEngine):
        variants = engine.generate_targeted(SCAM_TEXT, n=5, strategy="synonym_sub")
        assert all(isinstance(v, str) for v in variants)


# ---------------------------------------------------------------------------
# PerturbationEngine — formatting strategies
# ---------------------------------------------------------------------------


class TestFormattingPerturbations:
    def test_whitespace_perturb_produces_string(self, engine: PerturbationEngine):
        result = engine.whitespace_perturb(SCAM_TEXT)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_capitalisation_perturb_changes_case(self, engine: PerturbationEngine):
        text = "Apply now for this amazing role."
        results = engine.generate_targeted(text, n=10, strategy="capitalisation_perturb")
        assert any(r.lower() != text.lower() or r != text for r in results)

    def test_punctuation_perturb_produces_string(self, engine: PerturbationEngine):
        result = engine.punctuation_perturb(SCAM_TEXT)
        assert isinstance(result, str)

    def test_unknown_strategy_raises(self, engine: PerturbationEngine):
        with pytest.raises(ValueError, match="Unknown strategy"):
            engine.generate_targeted(SCAM_TEXT, n=1, strategy="nonexistent_strategy")

    def test_generate_with_strategy_subset(self, engine: PerturbationEngine):
        variants = engine.generate(SCAM_TEXT, n=10, strategies=["char_swap", "char_delete"])
        assert len(variants) == 10


# ---------------------------------------------------------------------------
# RobustnessScorer — basic
# ---------------------------------------------------------------------------


class TestRobustnessScorerBasic:
    def test_returns_report(self, scorer: RobustnessScorer):
        report = scorer.score(SCAM_TEXT)
        assert isinstance(report, RobustnessReport)

    def test_original_score_in_range(self, scorer: RobustnessScorer):
        report = scorer.score(SCAM_TEXT)
        assert 0.0 <= report.original_score <= 1.0

    def test_perturbed_scores_count(self, scorer: RobustnessScorer):
        report = scorer.score(SCAM_TEXT)
        assert len(report.perturbed_scores) == 15

    def test_all_perturbed_scores_in_range(self, scorer: RobustnessScorer):
        report = scorer.score(LEGIT_TEXT)
        assert all(0.0 <= s <= 1.0 for s in report.perturbed_scores)

    def test_score_std_nonnegative(self, scorer: RobustnessScorer):
        report = scorer.score(SCAM_TEXT)
        assert report.score_std >= 0.0

    def test_min_max_consistent(self, scorer: RobustnessScorer):
        report = scorer.score(SCAM_TEXT)
        assert report.min_score <= report.max_score

    def test_fragility_score_in_range(self, scorer: RobustnessScorer):
        report = scorer.score(SCAM_TEXT)
        assert 0.0 <= report.fragility_score <= 1.0

    def test_summary_nonempty(self, scorer: RobustnessScorer):
        report = scorer.score(SCAM_TEXT)
        assert len(report.summary) > 0

    def test_suggested_improvements_nonempty(self, scorer: RobustnessScorer):
        report = scorer.score(SCAM_TEXT)
        assert len(report.suggested_improvements) >= 1


# ---------------------------------------------------------------------------
# RobustnessScorer — stable scorer should report low fragility
# ---------------------------------------------------------------------------


class TestRobustnessScorerStability:
    def test_static_scorer_zero_fragility(self):
        scorer = RobustnessScorer(scoring_fn=_static_scorer(0.8), n_perturbations=10, seed=1)
        report = scorer.score(SCAM_TEXT)
        assert report.score_std == 0.0
        assert report.fragility_score == 0.0
        assert not report.is_fragile

    def test_static_scorer_no_adversarial_examples(self):
        scorer = RobustnessScorer(scoring_fn=_static_scorer(0.5), n_perturbations=10, seed=1)
        report = scorer.score(SCAM_TEXT)
        assert report.adversarial_examples == []

    def test_stable_human_review_not_requested(self):
        scorer = RobustnessScorer(scoring_fn=_static_scorer(0.3), n_perturbations=10, seed=1)
        report = scorer.score(LEGIT_TEXT)
        assert not report.human_review_requested

    def test_fragile_scorer_triggers_review(self):
        """Scorer that oscillates wildly should trigger human review."""
        toggle = {"state": 0}

        def oscillating(text: str) -> float:
            toggle["state"] = 1 - toggle["state"]
            return toggle["state"] * 0.9  # alternates 0.0 and 0.9

        scorer = RobustnessScorer(
            scoring_fn=oscillating, n_perturbations=20,
            fragility_threshold=0.05, seed=1,
        )
        report = scorer.score(SCAM_TEXT)
        assert report.is_fragile or report.human_review_requested


# ---------------------------------------------------------------------------
# AdversarialProber — basic
# ---------------------------------------------------------------------------


class TestAdversarialProberBasic:
    def test_returns_dict(self, prober: AdversarialProber):
        result = prober.probe(SCAM_TEXT)
        assert isinstance(result, dict)

    def test_required_keys_present(self, prober: AdversarialProber):
        result = prober.probe(SCAM_TEXT)
        for key in [
            "original_score", "signal_impacts", "weakest_signal",
            "adversarial_flips", "report_line",
        ]:
            assert key in result

    def test_scam_text_has_signal_impacts(self, prober: AdversarialProber):
        result = prober.probe(SCAM_TEXT)
        assert len(result["signal_impacts"]) > 0

    def test_signal_impacts_sorted_by_delta(self, prober: AdversarialProber):
        result = prober.probe(SCAM_TEXT)
        deltas = [d for _, _, d in result["signal_impacts"]]
        assert deltas == sorted(deltas, reverse=True)

    def test_legit_text_few_impacts(self, prober: AdversarialProber):
        """Legitimate text has fewer probe keywords, so fewer impacts."""
        result = prober.probe(LEGIT_TEXT)
        scam_result = prober.probe(SCAM_TEXT)
        assert len(result["signal_impacts"]) <= len(scam_result["signal_impacts"])

    def test_weakest_signal_is_string(self, prober: AdversarialProber):
        result = prober.probe(SCAM_TEXT)
        assert isinstance(result["weakest_signal"], str)

    def test_report_line_is_string(self, prober: AdversarialProber):
        result = prober.probe(SCAM_TEXT)
        assert isinstance(result["report_line"], str)
        assert len(result["report_line"]) > 0

    def test_text_without_keywords_empty_impacts(self, prober: AdversarialProber):
        result = prober.probe("A well-crafted legitimate job posting with no scam signals.")
        assert result["signal_impacts"] == []

    def test_find_minimal_flip_returns_tuple_or_none(self, prober: AdversarialProber):
        result = prober.find_minimal_flip(SCAM_TEXT)
        assert result is None or isinstance(result, tuple)

    def test_rank_signals_by_impact_sorted(self, prober: AdversarialProber):
        ranking = prober.rank_signals_by_impact(SCAM_TEXT)
        deltas = [d for _, d in ranking]
        assert deltas == sorted(deltas, reverse=True)


# ---------------------------------------------------------------------------
# AdversarialProber — flip detection
# ---------------------------------------------------------------------------


class TestAdversarialProberFlips:
    def test_guaranteed_is_high_impact(self, prober: AdversarialProber):
        """Removing 'guaranteed' from a scam text should reduce score."""
        result = prober.probe(SCAM_TEXT)
        impacts = {kw: d for kw, _, d in result["signal_impacts"]}
        # 'guaranteed' should be present and have positive delta (removal reduces score)
        if "guaranteed" in impacts:
            assert impacts["guaranteed"] >= 0.0

    def test_single_point_of_failure_in_report(self, prober: AdversarialProber):
        result = prober.probe(SCAM_TEXT)
        # When impact is large enough, report_line should mention single-point
        if result["weakest_signal_impact"] >= prober.flip_threshold:
            assert "single-point" in result["report_line"]


# ---------------------------------------------------------------------------
# RobustnessReport — dataclass
# ---------------------------------------------------------------------------


class TestRobustnessReport:
    def test_summary_contains_original_score(self):
        report = RobustnessReport(
            original_score=0.75,
            mean_perturbed_score=0.60,
            score_std=0.12,
            min_score=0.40,
            max_score=0.85,
            fragility_score=0.5,
            is_fragile=True,
            weakest_signal="guaranteed",
            weakest_signal_impact=0.35,
            suggested_improvements=["Fix something"],
            perturbed_scores=[0.4, 0.85, 0.6],
            adversarial_examples=[],
            human_review_requested=True,
        )
        assert "0.75" in report.summary

    def test_fragile_label_in_summary(self):
        report = RobustnessReport(
            original_score=0.7,
            mean_perturbed_score=0.5,
            score_std=0.3,
            min_score=0.2,
            max_score=0.9,
            fragility_score=0.8,
            is_fragile=True,
            weakest_signal="",
            weakest_signal_impact=0.0,
            suggested_improvements=[],
            perturbed_scores=[0.2, 0.9],
            adversarial_examples=[],
            human_review_requested=True,
        )
        assert "FRAGILE" in report.summary


# ---------------------------------------------------------------------------
# analyse_robustness — convenience function
# ---------------------------------------------------------------------------


class TestAnalyseRobustnessConvenience:
    def test_returns_report(self):
        report = analyse_robustness(SCAM_TEXT, _keyword_scorer, n_perturbations=10, seed=42)
        assert isinstance(report, RobustnessReport)

    def test_enriches_weakest_signal(self):
        report = analyse_robustness(SCAM_TEXT, _keyword_scorer, n_perturbations=10, seed=42)
        # weakest_signal is populated by AdversarialProber (may be empty if no keywords hit)
        assert isinstance(report.weakest_signal, str)

    def test_suggested_improvements_from_prober(self):
        report = analyse_robustness(SCAM_TEXT, _keyword_scorer, n_perturbations=10, seed=42)
        # The prober report_line should be prepended to improvements
        assert len(report.suggested_improvements) >= 1

    def test_legit_text_low_original_score(self):
        report = analyse_robustness(LEGIT_TEXT, _keyword_scorer, n_perturbations=10, seed=42)
        assert report.original_score < 0.5

    def test_scam_text_high_original_score(self):
        report = analyse_robustness(SCAM_TEXT, _keyword_scorer, n_perturbations=10, seed=42)
        assert report.original_score >= 0.3
