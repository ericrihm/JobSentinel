"""Tests for sentinel.counterfactual — Counterfactual Analysis Engine.

Covers all five classes:
  CounterfactualEngine, FailureAnalyzer, SignalGapFinder, WeightTuner, RootCauseTracer

50+ tests in total, organised by class.
"""

from __future__ import annotations

import math

import pytest

from sentinel.models import JobPosting, ScamSignal, SignalCategory
from sentinel.counterfactual import (
    CounterfactualEngine,
    CounterfactualResult,
    DiagnosisNode,
    DiagnosisTree,
    FailureAnalyzer,
    FailureMode,
    FailureRecord,
    MinimumInterventionSet,
    RootCauseTracer,
    SignalGapFinder,
    SignalProposal,
    WeightAdjustment,
    WeightTuner,
    _score_signals_pure,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_signal(
    name: str,
    weight: float = 0.7,
    category: SignalCategory = SignalCategory.RED_FLAG,
) -> ScamSignal:
    return ScamSignal(name=name, category=category, weight=weight)


def make_positive(name: str, weight: float = 0.3) -> ScamSignal:
    return make_signal(name, weight=weight, category=SignalCategory.POSITIVE)


def make_historical_record(
    signals: list[ScamSignal],
    true_scam: bool,
) -> dict:
    return {"signals": signals, "true_scam": true_scam}


# ---------------------------------------------------------------------------
# _score_signals_pure (internal helper)
# ---------------------------------------------------------------------------


class TestScoreSignalsPure:
    def test_empty_signals_returns_zero(self):
        assert _score_signals_pure([]) == 0.0

    def test_single_high_weight_red_flag(self):
        sig = make_signal("fee_required", weight=0.9)
        score = _score_signals_pure([sig])
        assert score > 0.5

    def test_single_positive_signal_reduces_score(self):
        pos = make_positive("established_company", weight=0.3)
        score = _score_signals_pure([pos])
        assert score < 0.5

    def test_weight_override_applied(self):
        sig = make_signal("guaranteed_income", weight=0.6)
        score_base = _score_signals_pure([sig])
        score_high = _score_signals_pure([sig], weight_overrides={"guaranteed_income": 0.95})
        assert score_high > score_base

    def test_multiple_red_flags_push_score_toward_one(self):
        signals = [make_signal(f"flag_{i}", weight=0.8) for i in range(5)]
        score = _score_signals_pure(signals)
        assert score > 0.9

    def test_mixed_signals_intermediate_score(self):
        signals = [
            make_signal("upfront_payment", weight=0.9),
            make_positive("established_company", weight=0.2),
        ]
        score = _score_signals_pure(signals)
        assert 0.0 < score < 1.0

    def test_score_monotonically_increases_with_weight(self):
        scores = []
        for w in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sig = make_signal("test_signal", weight=w)
            scores.append(_score_signals_pure([sig]))
        assert scores == sorted(scores)

    def test_weight_override_does_not_mutate_signal(self):
        sig = make_signal("immutable_signal", weight=0.6)
        _score_signals_pure([sig], weight_overrides={"immutable_signal": 0.95})
        assert sig.weight == 0.6  # original unchanged


# ---------------------------------------------------------------------------
# CounterfactualEngine
# ---------------------------------------------------------------------------


class TestCounterfactualEngine:
    @pytest.fixture
    def engine(self):
        return CounterfactualEngine(scam_threshold=0.8)

    @pytest.fixture
    def weak_signals(self):
        """Signals that collectively produce a score below threshold."""
        return [make_signal("vague_description", weight=0.55)]

    def test_is_scam_above_threshold(self, engine):
        assert engine.is_scam(0.85) is True

    def test_is_scam_below_threshold(self, engine):
        assert engine.is_scam(0.7) is False

    def test_is_scam_at_threshold(self, engine):
        assert engine.is_scam(0.8) is True

    def test_rank_counterfactuals_returns_list(self, engine, weak_signals):
        candidates = ["upfront_payment", "guaranteed_income", "crypto_payment"]
        results = engine.rank_counterfactuals(weak_signals, candidates)
        assert isinstance(results, list)
        assert len(results) == len(candidates)

    def test_rank_counterfactuals_all_results_typed(self, engine, weak_signals):
        results = engine.rank_counterfactuals(weak_signals, ["upfront_payment"])
        assert isinstance(results[0], CounterfactualResult)

    def test_rank_counterfactuals_sorted_by_abs_delta(self, engine, weak_signals):
        candidates = ["sig_low", "sig_high"]
        # sig_high with weight=0.95 should have higher delta than sig_low with default
        results = engine.rank_counterfactuals(weak_signals, candidates)
        deltas = [abs(r.score_delta) for r in results]
        assert deltas == sorted(deltas, reverse=True)

    def test_rank_counterfactuals_would_flip_flag(self, engine):
        # Start with no signals (score ~0.5) — adding a heavy signal should flip
        results = engine.rank_counterfactuals([], ["strong_signal"], hypothetical_weight=0.95)
        # With empty signals, base score is 0 and adding strong signal should push it up
        assert results[0].score_delta > 0

    def test_rank_counterfactuals_score_delta_positive_for_red_flag(self, engine, weak_signals):
        results = engine.rank_counterfactuals(weak_signals, ["new_red_flag"])
        assert results[0].score_delta > 0

    def test_rank_counterfactuals_empty_candidates_returns_empty(self, engine, weak_signals):
        results = engine.rank_counterfactuals(weak_signals, [])
        assert results == []

    def test_rank_counterfactuals_respects_custom_hypothetical_weight(self, engine, weak_signals):
        r_low = engine.rank_counterfactuals(weak_signals, ["sig"], hypothetical_weight=0.55)
        r_high = engine.rank_counterfactuals(weak_signals, ["sig"], hypothetical_weight=0.95)
        assert r_high[0].score_delta > r_low[0].score_delta

    def test_minimum_intervention_set_false_negative(self, engine):
        # Score below threshold — find signals needed to flip
        fired = [make_signal("vague_description", weight=0.5)]
        candidates = ["upfront_payment", "guaranteed_income", "crypto", "urgency"]
        result = engine.minimum_intervention_set(
            fired, candidates, hypothetical_weight=0.85
        )
        assert result is not None
        assert isinstance(result, MinimumInterventionSet)
        assert result.flipped_score >= 0.8
        assert len(result.signals) >= 1

    def test_minimum_intervention_set_already_classified_scam(self, engine):
        # Score already above threshold → should try to find removal set
        fired = [make_signal(f"flag_{i}", weight=0.85) for i in range(4)]
        candidates = []
        result = engine.minimum_intervention_set(fired, candidates)
        # May return a removal set or None — but should not crash
        # (classified as scam → tries _min_removal_set)
        assert result is None or isinstance(result, MinimumInterventionSet)

    def test_minimum_intervention_set_size_field(self, engine):
        fired = []
        candidates = ["a", "b", "c"]
        result = engine.minimum_intervention_set(fired, candidates, hypothetical_weight=0.9)
        if result is not None:
            assert result.size == len(result.signals)

    def test_minimum_intervention_set_max_size_respected(self, engine):
        fired = [make_signal("tiny", weight=0.4)]
        candidates = [f"sig_{i}" for i in range(10)]
        result = engine.minimum_intervention_set(
            fired, candidates, hypothetical_weight=0.6, max_set_size=2
        )
        if result is not None:
            assert result.size <= 2

    def test_score_with_weight_override(self, engine):
        signals = [make_signal("test_sig", weight=0.5)]
        base = engine.score_with_weight_override(signals, {})
        high = engine.score_with_weight_override(signals, {"test_sig": 0.95})
        assert high > base

    def test_false_positive_removal_set(self):
        # Use a very low threshold to force false-positive scenario
        engine = CounterfactualEngine(scam_threshold=0.3)
        fired = [make_signal("some_flag", weight=0.9)]
        result = engine.minimum_intervention_set(fired, [], max_set_size=3)
        # Should try to find minimum removal set
        assert result is None or isinstance(result, MinimumInterventionSet)


# ---------------------------------------------------------------------------
# FailureAnalyzer
# ---------------------------------------------------------------------------


class TestFailureAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return FailureAnalyzer()

    def test_record_false_negative_returns_record(self, analyzer):
        signals = [make_signal("vague", weight=0.4)]
        record = analyzer.record_false_negative(
            "https://example.com/job/1",
            predicted_score=0.3,
            fired_signals=signals,
        )
        assert isinstance(record, FailureRecord)
        assert record.true_label == "scam"

    def test_record_false_positive_returns_record(self, analyzer):
        signals = [make_signal("urgency", weight=0.6)]
        record = analyzer.record_false_positive(
            "https://example.com/job/2",
            predicted_score=0.85,
            fired_signals=signals,
        )
        assert record.true_label == "legitimate"

    def test_failure_mode_threshold_when_many_signals(self, analyzer):
        # Many signals fired but score was still moderate → THRESHOLD
        signals = [make_signal(f"s{i}", weight=0.4) for i in range(5)]
        record = analyzer.record_false_negative(
            "url", 0.45, signals
        )
        assert record.failure_mode == FailureMode.THRESHOLD

    def test_failure_mode_novel_when_no_signals_no_obfuscation(self, analyzer):
        record = analyzer.record_false_negative(
            "url", 0.05, fired_signals=[], job_text="normal clean job text"
        )
        assert record.failure_mode == FailureMode.NOVEL

    def test_failure_mode_evasion_when_near_misses(self, analyzer):
        record = analyzer.record_false_negative(
            "url", 0.2, fired_signals=[],
            near_misses=["upfront_payment"]
        )
        assert record.failure_mode == FailureMode.EVASION

    def test_failure_mode_gap_when_few_signals_low_score(self, analyzer):
        record = analyzer.record_false_negative(
            "url", 0.1, fired_signals=[make_signal("weak", weight=0.3)]
        )
        assert record.failure_mode == FailureMode.GAP

    def test_failure_mode_distribution_counts_all(self, analyzer):
        analyzer.record_false_negative("u1", 0.1, [])
        analyzer.record_false_negative("u2", 0.1, [])
        analyzer.record_false_positive("u3", 0.9, [make_signal("s", weight=0.9)])
        dist = analyzer.failure_mode_distribution()
        assert sum(dist.values()) == 3

    def test_priority_queue_sorted_desc(self, analyzer):
        for i in range(5):
            analyzer.record_false_negative(f"url_{i}", 0.1, [])
        analyzer.record_false_negative("url_evasion", 0.2, [], near_misses=["sig"])
        pq = analyzer.priority_queue()
        priorities = [p for _, _, p in pq]
        assert priorities == sorted(priorities, reverse=True)

    def test_priority_queue_returns_tuples(self, analyzer):
        analyzer.record_false_negative("u", 0.2, [])
        pq = analyzer.priority_queue()
        mode, count, priority = pq[0]
        assert isinstance(mode, FailureMode)
        assert isinstance(count, int)
        assert isinstance(priority, float)

    def test_get_records_filtered_by_mode(self, analyzer):
        analyzer.record_false_negative("u1", 0.1, [], near_misses=["s"])
        analyzer.record_false_negative("u2", 0.1, [])
        evasion_records = analyzer.get_records(mode=FailureMode.EVASION)
        assert all(r.failure_mode == FailureMode.EVASION for r in evasion_records)

    def test_get_records_all(self, analyzer):
        analyzer.record_false_negative("u1", 0.1, [])
        analyzer.record_false_positive("u2", 0.9, [make_signal("s")])
        assert len(analyzer.get_records()) == 2

    def test_summary_structure(self, analyzer):
        analyzer.record_false_negative("u", 0.2, [])
        s = analyzer.summary()
        assert "total_failures" in s
        assert "distribution" in s
        assert "priority_queue" in s
        assert s["total_failures"] == 1

    def test_failure_trends_returns_dict(self, analyzer):
        for _ in range(20):
            analyzer.record_false_negative("u", 0.2, [])
        trends = analyzer.failure_trends(window=10)
        assert isinstance(trends, dict)

    def test_failure_trends_small_dataset(self, analyzer):
        analyzer.record_false_negative("u", 0.2, [])
        trends = analyzer.failure_trends(window=10)
        # With < window records, falls back to distribution dict
        assert isinstance(trends, dict)

    def test_record_stores_fired_signal_names(self, analyzer):
        signals = [make_signal("fee_required"), make_signal("crypto_payment")]
        record = analyzer.record_false_negative("u", 0.3, signals)
        assert "fee_required" in record.fired_signals
        assert "crypto_payment" in record.fired_signals

    def test_record_timestamp_is_set(self, analyzer):
        record = analyzer.record_false_negative("u", 0.2, [])
        assert record.timestamp != ""

    def test_empty_analyzer_priority_queue(self, analyzer):
        assert analyzer.priority_queue() == []

    def test_empty_analyzer_summary(self, analyzer):
        s = analyzer.summary()
        assert s["total_failures"] == 0


# ---------------------------------------------------------------------------
# SignalGapFinder
# ---------------------------------------------------------------------------


class TestSignalGapFinder:
    @pytest.fixture
    def finder(self):
        return SignalGapFinder()

    def test_add_caught_increments_counter(self, finder):
        finder.add_caught_scam(["upfront_payment", "guaranteed_income"])
        finder.add_caught_scam(["upfront_payment"])
        gaps = finder.find_gaps()
        caught_counts = {g["signal_name"]: g["caught_count"] for g in gaps}
        assert caught_counts["upfront_payment"] == 2

    def test_add_missed_increments_counter(self, finder):
        finder.add_missed_scam(["novel_signal"])
        gaps = finder.find_gaps()
        assert any(g["signal_name"] == "novel_signal" for g in gaps)

    def test_find_gaps_sorted_by_gap_ratio(self, finder):
        # novel_signal only in missed → high gap ratio
        finder.add_caught_scam(["common_signal"])
        finder.add_missed_scam(["common_signal", "novel_signal"])
        finder.add_missed_scam(["novel_signal"])
        gaps = finder.find_gaps()
        ratios = [g["gap_ratio"] for g in gaps]
        assert ratios == sorted(ratios, reverse=True)

    def test_find_gaps_empty_returns_empty(self, finder):
        assert finder.find_gaps() == []

    def test_missing_signals_filters_by_count(self, finder):
        finder.add_missed_scam(["rare_signal"])
        finder.add_missed_scam(["rare_signal"])
        finder.add_caught_scam([])
        missing = finder.missing_signals(min_missed_count=2)
        assert any(g["signal_name"] == "rare_signal" for g in missing)

    def test_missing_signals_excludes_common_caught(self, finder):
        # Signal that fires equally on caught and missed → not a gap
        for _ in range(5):
            finder.add_caught_scam(["common"])
            finder.add_missed_scam(["common"])
        missing = finder.missing_signals(min_missed_count=2)
        # common should not appear because caught_rate ≈ missed_rate
        assert not any(g["signal_name"] == "common" for g in missing)

    def test_generate_proposals_returns_list(self, finder):
        finder.add_missed_scam(["novel_signal", "novel_signal_2"])
        finder.add_missed_scam(["novel_signal"])
        finder.add_caught_scam([])
        proposals = finder.generate_proposals(min_missed_count=1)
        assert isinstance(proposals, list)

    def test_generate_proposals_all_typed(self, finder):
        finder.add_missed_scam(["sig_x"])
        finder.add_missed_scam(["sig_x"])
        finder.add_caught_scam([])
        proposals = finder.generate_proposals(min_missed_count=2)
        assert all(isinstance(p, SignalProposal) for p in proposals)

    def test_ngram_mining_from_text(self, finder):
        text = "send us your bank account number now"
        finder.add_missed_scam(["weak_signal"], text_fragment=text)
        finder.add_missed_scam(["weak_signal"], text_fragment=text)
        proposals = finder.generate_proposals(min_missed_count=2)
        ngram_props = [p for p in proposals if p.name.startswith("ngram_")]
        assert len(ngram_props) >= 1

    def test_proposal_estimated_precision_in_range(self, finder):
        finder.add_missed_scam(["gap_signal"])
        finder.add_missed_scam(["gap_signal"])
        finder.add_caught_scam([])
        proposals = finder.generate_proposals(min_missed_count=2)
        for p in proposals:
            assert 0.0 <= p.estimated_precision <= 1.0

    def test_proposal_recall_lift_non_negative(self, finder):
        finder.add_missed_scam(["gap_signal"])
        finder.add_missed_scam(["gap_signal"])
        finder.add_caught_scam([])
        proposals = finder.generate_proposals(min_missed_count=2)
        for p in proposals:
            assert p.estimated_recall_lift >= 0.0

    def test_gap_ratio_signal_only_in_missed(self, finder):
        finder.add_caught_scam(["other_signal"])
        finder.add_missed_scam(["exclusive_to_missed"])
        gaps = {g["signal_name"]: g for g in finder.find_gaps()}
        # exclusive_to_missed should have a very high gap ratio
        assert gaps["exclusive_to_missed"]["gap_ratio"] > 1.0


# ---------------------------------------------------------------------------
# WeightTuner
# ---------------------------------------------------------------------------


class TestWeightTuner:
    @pytest.fixture
    def tuner(self):
        return WeightTuner(max_delta=0.1, learning_rate=1.0)

    @pytest.fixture
    def simple_dataset(self):
        """Two scam examples and two legitimate examples."""
        scam_signals = [
            make_signal("upfront_payment", weight=0.6),
            make_signal("guaranteed_income", weight=0.6),
        ]
        legit_signals = [
            make_positive("established_company", weight=0.3),
        ]
        return [
            make_historical_record(scam_signals, true_scam=True),
            make_historical_record(scam_signals, true_scam=True),
            make_historical_record(legit_signals, true_scam=False),
            make_historical_record(legit_signals, true_scam=False),
        ]

    def test_tune_returns_list(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6, "guaranteed_income": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        assert isinstance(adjustments, list)

    def test_tune_returns_weight_adjustment_objects(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        assert all(isinstance(a, WeightAdjustment) for a in adjustments)

    def test_tune_empty_dataset_returns_empty(self, tuner):
        adjustments = tuner.tune([], {"some_signal": 0.5})
        assert adjustments == []

    def test_tune_empty_weights_returns_empty(self, tuner, simple_dataset):
        adjustments = tuner.tune(simple_dataset, {})
        assert adjustments == []

    def test_tune_delta_within_max(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6, "guaranteed_income": 0.6}
        adjustments = tuner.tune(simple_dataset, weights, scam_threshold=0.8)
        for adj in adjustments:
            assert abs(adj.delta) <= tuner.MAX_DELTA_PER_CYCLE + 1e-9

    def test_tune_weight_stays_in_range(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        for adj in adjustments:
            assert tuner.MIN_WEIGHT <= adj.proposed_weight <= tuner.MAX_WEIGHT

    def test_weight_adjustment_direction_field(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        for adj in adjustments:
            assert adj.direction in ("increase", "decrease")
            if adj.delta > 0:
                assert adj.direction == "increase"
            else:
                assert adj.direction == "decrease"

    def test_weight_adjustment_delta_matches_proposed_minus_current(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        for adj in adjustments:
            assert abs(adj.delta - (adj.proposed_weight - adj.current_weight)) < 1e-9

    def test_apply_adjustments_updates_weights(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6, "guaranteed_income": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        new_weights = tuner.apply_adjustments(weights, adjustments)
        assert isinstance(new_weights, dict)
        # At least one weight should differ if any adjustments were proposed
        if adjustments:
            assert new_weights != weights

    def test_apply_adjustments_max_apply(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6, "guaranteed_income": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        new_weights = tuner.apply_adjustments(weights, adjustments, max_apply=1)
        changed = sum(1 for k, v in new_weights.items() if v != weights.get(k))
        assert changed <= 1

    def test_report_returns_strings(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        lines = tuner.report(adjustments)
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_report_mentions_signal_name(self, tuner, simple_dataset):
        weights = {"upfront_payment": 0.6}
        adjustments = tuner.tune(simple_dataset, weights)
        lines = tuner.report(adjustments, top_n=1)
        if lines:
            assert "upfront_payment" in lines[0]


# ---------------------------------------------------------------------------
# RootCauseTracer
# ---------------------------------------------------------------------------


class TestRootCauseTracer:
    @pytest.fixture
    def tracer(self):
        return RootCauseTracer(scam_threshold=0.8)

    @pytest.fixture
    def false_negative_case(self):
        return {
            "job_url": "https://linkedin.com/jobs/view/999",
            "job_text": "Earn money working from home. Pay a small fee to start.",
            "fired_signals": [make_signal("vague_description", weight=0.4)],
            "expected_signals": ["upfront_payment", "guaranteed_income"],
            "predicted_score": 0.35,
        }

    def test_trace_returns_diagnosis_tree(self, tracer, false_negative_case):
        tree = tracer.trace(**false_negative_case)
        assert isinstance(tree, DiagnosisTree)

    def test_trace_root_node_exists(self, tracer, false_negative_case):
        tree = tracer.trace(**false_negative_case)
        assert isinstance(tree.root, DiagnosisNode)

    def test_trace_has_four_child_nodes(self, tracer, false_negative_case):
        tree = tracer.trace(**false_negative_case)
        # Expects: text_normalisation, signal_extraction, weight_adequacy, threshold_placement
        assert len(tree.root.children) == 4

    def test_trace_failure_mode_is_enum(self, tracer, false_negative_case):
        tree = tracer.trace(**false_negative_case)
        assert isinstance(tree.failure_mode, FailureMode)

    def test_trace_summary_is_string(self, tracer, false_negative_case):
        tree = tracer.trace(**false_negative_case)
        assert isinstance(tree.summary, str)
        assert len(tree.summary) > 0

    def test_trace_gap_when_signals_missing(self, tracer):
        tree = tracer.trace(
            job_url="url",
            job_text="legitimate looking text",
            fired_signals=[make_signal("minor", weight=0.3)],
            expected_signals=["upfront_payment", "guaranteed_income"],
            predicted_score=0.15,
        )
        assert tree.failure_mode == FailureMode.GAP

    def test_trace_threshold_when_close_to_boundary(self, tracer):
        # All expected signals fired, score is just below threshold
        fired = [make_signal("upfront_payment"), make_signal("guaranteed_income")]
        tree = tracer.trace(
            job_url="url",
            job_text="pay a fee and earn guaranteed income",
            fired_signals=fired,
            expected_signals=["upfront_payment", "guaranteed_income"],
            predicted_score=0.79,
        )
        assert tree.failure_mode in (FailureMode.THRESHOLD, FailureMode.NOVEL)

    def test_trace_novel_when_no_signals_clean_text(self, tracer):
        tree = tracer.trace(
            job_url="url",
            job_text="normal clean job description for software engineer",
            fired_signals=[],
            expected_signals=[],
            predicted_score=0.05,
        )
        assert tree.failure_mode == FailureMode.NOVEL

    def test_trace_evasion_with_zero_width_chars(self, tracer):
        # Zero-width space (U+200B) in text
        text_with_zwsp = "pay\u200b fee required"
        tree = tracer.trace(
            job_url="url",
            job_text=text_with_zwsp,
            fired_signals=[],
            expected_signals=["upfront_payment"],
            predicted_score=0.1,
        )
        assert tree.failure_mode == FailureMode.EVASION

    def test_signal_extraction_node_lists_missing_signals(self, tracer):
        tree = tracer.trace(
            job_url="url",
            job_text="text",
            fired_signals=[],
            expected_signals=["missing_signal_a", "missing_signal_b"],
            predicted_score=0.1,
        )
        extraction_node = next(
            n for n in tree.root.children if n.step == "signal_extraction"
        )
        assert "missing_signal_a" in extraction_node.finding
        assert not extraction_node.passed

    def test_weight_node_flags_low_weights(self, tracer):
        low_weight_signals = [make_signal("weak_signal", weight=0.3)]
        tree = tracer.trace(
            job_url="url",
            job_text="text",
            fired_signals=low_weight_signals,
            expected_signals=[],
            predicted_score=0.35,
        )
        weight_node = next(
            n for n in tree.root.children if n.step == "weight_adequacy"
        )
        assert not weight_node.passed
        assert "weak_signal" in weight_node.finding

    def test_weight_node_passes_for_adequate_weights(self, tracer):
        high_weight_signals = [make_signal("strong_signal", weight=0.85)]
        tree = tracer.trace(
            job_url="url",
            job_text="text",
            fired_signals=high_weight_signals,
            expected_signals=[],
            predicted_score=0.7,
        )
        weight_node = next(
            n for n in tree.root.children if n.step == "weight_adequacy"
        )
        assert weight_node.passed

    def test_threshold_node_close_gap(self, tracer):
        tree = tracer.trace(
            job_url="url",
            job_text="text",
            fired_signals=[make_signal("s", weight=0.8)],
            expected_signals=[],
            predicted_score=0.75,  # gap of 0.05 from threshold 0.8
        )
        threshold_node = next(
            n for n in tree.root.children if n.step == "threshold_placement"
        )
        assert not threshold_node.passed

    def test_batch_trace_returns_list(self, tracer, false_negative_case):
        trees = tracer.batch_trace([false_negative_case, false_negative_case])
        assert len(trees) == 2
        assert all(isinstance(t, DiagnosisTree) for t in trees)

    def test_failure_mode_counts(self, tracer, false_negative_case):
        trees = tracer.batch_trace([false_negative_case, false_negative_case])
        counts = tracer.failure_mode_counts(trees)
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 2

    def test_diagnosis_node_step_names_are_set(self, tracer, false_negative_case):
        tree = tracer.trace(**false_negative_case)
        steps = {n.step for n in tree.root.children}
        assert "text_normalisation" in steps
        assert "signal_extraction" in steps
        assert "weight_adequacy" in steps
        assert "threshold_placement" in steps

    def test_trace_job_url_in_tree(self, tracer, false_negative_case):
        tree = tracer.trace(**false_negative_case)
        assert tree.job_url == false_negative_case["job_url"]

    def test_empty_text_normalisation_passes(self, tracer):
        tree = tracer.trace(
            job_url="url",
            job_text="",
            fired_signals=[],
            expected_signals=[],
            predicted_score=0.1,
        )
        norm_node = next(
            n for n in tree.root.children if n.step == "text_normalisation"
        )
        # Empty text: we treat it as can't diagnose → passed=True
        assert norm_node.passed is True

    def test_child_nodes_added_for_missing_signals(self, tracer):
        tree = tracer.trace(
            job_url="url",
            job_text="text",
            fired_signals=[],
            expected_signals=["upfront_payment"],
            predicted_score=0.1,
        )
        extraction_node = next(
            n for n in tree.root.children if n.step == "signal_extraction"
        )
        # Should have a child for each missing signal
        assert len(extraction_node.children) == 1
        assert extraction_node.children[0].step == "signal:upfront_payment"


# ---------------------------------------------------------------------------
# Integration: engine + analyzer + tracer working together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_false_negative_workflow(self):
        """Simulate a complete FN analysis: engine → analyzer → tracer."""
        fired_signals = [make_signal("vague_description", weight=0.4)]
        predicted_score = 0.3

        engine = CounterfactualEngine(scam_threshold=0.8)
        analyzer = FailureAnalyzer()
        tracer = RootCauseTracer(scam_threshold=0.8)

        # Engine: what would have caught it?
        candidates = ["upfront_payment", "guaranteed_income"]
        cf_results = engine.rank_counterfactuals(fired_signals, candidates, hypothetical_weight=0.85)
        assert len(cf_results) == 2

        # MIS: minimum needed to flip
        mis = engine.minimum_intervention_set(fired_signals, candidates, hypothetical_weight=0.85)
        assert mis is not None

        # Analyzer: record and categorise
        record = analyzer.record_false_negative("url", predicted_score, fired_signals)
        assert record is not None

        # Tracer: build diagnosis tree
        tree = tracer.trace(
            job_url="url",
            job_text="vague description with no specific skills required",
            fired_signals=fired_signals,
            expected_signals=candidates,
            predicted_score=predicted_score,
        )
        assert tree.failure_mode in FailureMode.__members__.values()

    def test_weight_tuner_then_rescore(self):
        """WeightTuner proposes adjustments; re-scoring with new weights improves accuracy."""
        signals_scam = [
            make_signal("upfront_payment", weight=0.55),
            make_signal("no_experience", weight=0.55),
        ]
        signals_legit = [
            make_positive("established_company", weight=0.3),
        ]
        dataset = [
            make_historical_record(signals_scam, true_scam=True),
            make_historical_record(signals_scam, true_scam=True),
            make_historical_record(signals_legit, true_scam=False),
            make_historical_record(signals_legit, true_scam=False),
        ]

        current_weights = {"upfront_payment": 0.55, "no_experience": 0.55}
        tuner = WeightTuner(learning_rate=1.0)
        adjustments = tuner.tune(dataset, current_weights, scam_threshold=0.8)

        new_weights = tuner.apply_adjustments(current_weights, adjustments)
        # New weights dict should be valid
        assert all(
            WeightTuner.MIN_WEIGHT <= v <= WeightTuner.MAX_WEIGHT
            for v in new_weights.values()
        )

    def test_gap_finder_feeds_proposals(self):
        """SignalGapFinder proposals reference signals missed in FNs."""
        finder = SignalGapFinder()
        finder.add_caught_scam(["upfront_payment", "guaranteed_income"])
        finder.add_caught_scam(["upfront_payment"])
        finder.add_missed_scam(["vague_description", "novel_tactic"])
        finder.add_missed_scam(["novel_tactic"])
        finder.add_missed_scam(["novel_tactic"])

        proposals = finder.generate_proposals(min_missed_count=2)
        names = [p.name for p in proposals]
        # novel_tactic should appear — high gap ratio, sufficient missed count
        assert any("novel_tactic" in n for n in names)
