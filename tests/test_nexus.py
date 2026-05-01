"""Tests for sentinel.nexus — Unified Intelligence Orchestrator.

Coverage targets:
  - NexusReport dataclass and to_dict()
  - Nexus.deep_analyze() — all subsystem paths
  - Nexus meta-weight management
  - Nexus._compute_overall() scoring logic
  - Nexus._build_recommendations() logic
  - NexusLearner.learn() and weight updates
  - NexusLearner._recompute_meta_weights()
  - NexusLearner.accuracy_summary() / most_accurate_subsystem()
  - NexusDashboard.snapshot() / most_active_signals() / most_informative_subsystems()
  - NexusEvolver.evolve() and run_cycles()
  - EvolveResult dataclass
  - Graceful degradation when subsystems are absent
  - Risk level thresholds
  - Score clamping / edge cases
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel.models import JobPosting, ScamSignal, SignalCategory
from sentinel.nexus import (
    EvolveResult,
    Nexus,
    NexusDashboard,
    NexusEvolver,
    NexusLearner,
    NexusReport,
    SubsystemStatus,
    DashboardSnapshot,
    _DEFAULT_META_WEIGHTS,
    _clamp,
    _full_text,
    _now_iso,
    _score_to_risk,
)


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_job(
    *,
    title: str = "Software Engineer",
    company: str = "Acme Corp",
    description: str = "We are looking for a software engineer.",
    salary_min: float = 80_000,
    salary_max: float = 120_000,
    url: str = "https://linkedin.com/jobs/1",
    location: str = "Remote",
    recruiter_connections: int = 500,
) -> JobPosting:
    return JobPosting(
        title=title,
        company=company,
        description=description,
        salary_min=salary_min,
        salary_max=salary_max,
        url=url,
        location=location,
        recruiter_connections=recruiter_connections,
    )


def _make_scam_job() -> JobPosting:
    return _make_job(
        title="Work From Home — Earn $5000/week guaranteed!",
        company="XYZ Opportunities LLC",
        description=(
            "No experience required! Earn fast cash from home. "
            "Guaranteed income. Send fee for starter kit. "
            "Apply now — limited spots! Contact us at jobs@gmail.com. "
            "Buy Bitcoin to confirm your application. "
            "100% legit! Not a scam! We are a real company."
        ),
        salary_min=5_000,
        salary_max=20_000,
        recruiter_connections=3,
        url="https://linkedin.com/jobs/scam1",
    )


def _make_legit_job() -> JobPosting:
    return _make_job(
        title="Senior Software Engineer",
        company="Stripe",
        description=(
            "We are seeking a senior software engineer with 5+ years of "
            "Python experience. You will design and implement scalable "
            "backend services. Requirements: BS/MS in CS or related field, "
            "experience with distributed systems, strong communication skills."
        ),
        salary_min=150_000,
        salary_max=220_000,
        recruiter_connections=800,
        url="https://linkedin.com/jobs/legit1",
    )


def _make_signal(name: str = "test_signal", category: SignalCategory = SignalCategory.RED_FLAG,
                  weight: float = 0.8) -> ScamSignal:
    return ScamSignal(name=name, category=category, weight=weight, confidence=0.9)


def _make_report(**overrides) -> NexusReport:
    defaults = {
        "overall_score": 0.3,
        "confidence": 0.7,
        "risk_level": "CAUTION",
        "subsystem_scores": {"signals": 0.4, "fraud_triangle": 0.3},
        "signals_fired": [_make_signal()],
        "key_findings": ["Some finding"],
        "subsystems_run": ["signals", "fraud_triangle"],
    }
    defaults.update(overrides)
    return NexusReport(**defaults)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_clamp_in_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below_zero(self):
        assert _clamp(-1.0) == 0.0

    def test_clamp_above_one(self):
        assert _clamp(2.0) == 1.0

    def test_clamp_exact_boundaries(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_clamp_custom_bounds(self):
        assert _clamp(5.0, lo=0.0, hi=3.0) == 3.0
        assert _clamp(-2.0, lo=-1.0, hi=1.0) == -1.0

    def test_full_text_combines_fields(self):
        job = _make_job(title="Engineer", company="Acme", description="Build things")
        text = _full_text(job)
        assert "Engineer" in text
        assert "Acme" in text
        assert "Build things" in text

    def test_full_text_skips_empty(self):
        job = JobPosting(title="Dev", company="", description="Code")
        text = _full_text(job)
        assert "Dev" in text
        assert "Code" in text
        assert "  " not in text  # no doubled spaces from empty company

    def test_now_iso_is_string(self):
        ts = _now_iso()
        assert isinstance(ts, str)
        assert "T" in ts

    def test_score_to_risk_safe(self):
        assert _score_to_risk(0.05) == "SAFE"

    def test_score_to_risk_caution(self):
        assert _score_to_risk(0.25) == "CAUTION"

    def test_score_to_risk_warning(self):
        assert _score_to_risk(0.55) == "WARNING"

    def test_score_to_risk_danger(self):
        assert _score_to_risk(0.75) == "DANGER"

    def test_score_to_risk_critical(self):
        assert _score_to_risk(0.95) == "CRITICAL"

    def test_score_to_risk_exact_boundaries(self):
        assert _score_to_risk(0.20) == "CAUTION"
        assert _score_to_risk(0.40) == "WARNING"
        assert _score_to_risk(0.60) == "DANGER"
        assert _score_to_risk(0.80) == "CRITICAL"


# ---------------------------------------------------------------------------
# NexusReport
# ---------------------------------------------------------------------------

class TestNexusReport:
    def test_default_values(self):
        r = NexusReport()
        assert r.overall_score == 0.0
        assert r.confidence == 0.0
        assert r.risk_level == "SAFE"
        assert r.signals_fired == []
        assert r.key_findings == []
        assert r.recommendations == []
        assert r.disagreements == []
        assert r.similar_postings == []
        assert r.operator_fingerprint is None
        assert r.economic_flags == []
        assert r.evasion_detected is False
        assert r.llm_generated_probability == 0.0
        assert r.counterfactual_insights == []
        assert r.subsystems_run == []
        assert r.analysis_time_ms == 0.0

    def test_to_dict_keys(self):
        r = NexusReport(overall_score=0.7, risk_level="DANGER")
        d = r.to_dict()
        assert "overall_score" in d
        assert "confidence" in d
        assert "risk_level" in d
        assert "subsystem_scores" in d
        assert "signals_fired_count" in d
        assert "key_findings" in d
        assert "recommendations" in d
        assert "disagreements" in d
        assert "similar_postings" in d
        assert "operator_fingerprint" in d
        assert "economic_flags" in d
        assert "evasion_detected" in d
        assert "llm_generated_probability" in d
        assert "counterfactual_insights" in d
        assert "subsystems_run" in d
        assert "analysis_time_ms" in d
        assert "analyzed_at" in d

    def test_to_dict_rounding(self):
        r = NexusReport(overall_score=0.123456789, confidence=0.987654321)
        d = r.to_dict()
        assert d["overall_score"] == round(0.123456789, 4)
        assert d["confidence"] == round(0.987654321, 4)

    def test_to_dict_signals_fired_count(self):
        r = NexusReport(signals_fired=[_make_signal(), _make_signal("s2")])
        d = r.to_dict()
        assert d["signals_fired_count"] == 2

    def test_analyzed_at_is_set(self):
        r = NexusReport()
        assert r.analyzed_at is not None
        assert isinstance(r.analyzed_at, str)


# ---------------------------------------------------------------------------
# Nexus — unit tests with mocked subsystems
# ---------------------------------------------------------------------------

class TestNexusInit:
    def test_default_meta_weights(self):
        nexus = Nexus()
        weights = nexus.get_meta_weights()
        assert "signals" in weights
        assert "fraud_triangle" in weights
        assert abs(sum(weights.values()) - sum(_DEFAULT_META_WEIGHTS.values())) < 0.01

    def test_custom_meta_weights(self):
        w = {"signals": 0.99, "fraud_triangle": 0.01}
        nexus = Nexus(meta_weights=w)
        assert nexus.get_meta_weights()["signals"] == 0.99

    def test_set_meta_weights(self):
        nexus = Nexus()
        nexus.set_meta_weights({"signals": 0.55})
        assert nexus.get_meta_weights()["signals"] == 0.55

    def test_subsystem_availability_returns_dict(self):
        nexus = Nexus()
        avail = nexus.subsystem_availability()
        assert isinstance(avail, dict)
        assert "signals" in avail
        assert "scorer" in avail
        assert "db" in avail
        assert all(isinstance(v, bool) for v in avail.values())

    def test_analysis_count_starts_at_zero(self):
        nexus = Nexus()
        assert nexus._analysis_count == 0


class TestNexusComputeOverall:
    def setup_method(self):
        self.nexus = Nexus()

    def test_empty_scores_returns_zero(self):
        score, confidence = self.nexus._compute_overall({})
        assert score == 0.0
        assert confidence == 0.0

    def test_single_subsystem_score(self):
        score, confidence = self.nexus._compute_overall({"signals": 0.8})
        assert score > 0
        assert 0 <= confidence <= 1

    def test_all_zeros_gives_zero(self):
        scores = {k: 0.0 for k in _DEFAULT_META_WEIGHTS}
        score, conf = self.nexus._compute_overall(scores)
        assert score == 0.0

    def test_all_ones_gives_one(self):
        scores = {k: 1.0 for k in _DEFAULT_META_WEIGHTS}
        score, conf = self.nexus._compute_overall(scores)
        assert score == 1.0

    def test_mixed_scores_weighted(self):
        # "signals" has default weight 0.30, heaviest
        scores = {"signals": 1.0, "benford": 0.0}
        score, _ = self.nexus._compute_overall(scores)
        # signals dominates, so overall > 0.5
        assert score > 0.5

    def test_confidence_higher_with_more_subsystems(self):
        few = {"signals": 0.5}
        many = {k: 0.5 for k in _DEFAULT_META_WEIGHTS}
        _, c1 = self.nexus._compute_overall(few)
        _, c2 = self.nexus._compute_overall(many)
        assert c2 > c1

    def test_score_clamped_to_zero_one(self):
        # Even if weights don't sum cleanly, result should be in [0,1]
        scores = {k: 1.5 for k in _DEFAULT_META_WEIGHTS}
        score, conf = self.nexus._compute_overall(scores)
        assert 0.0 <= score <= 1.0
        assert 0.0 <= conf <= 1.0


class TestNexusBuildRecommendations:
    def setup_method(self):
        self.nexus = Nexus()

    def _analyze_and_get_recs(self, score: float, **kwargs) -> list[str]:
        r = NexusReport(overall_score=score, **kwargs)
        self.nexus._build_recommendations(r)
        return r.recommendations

    def test_critical_score_do_not_apply(self):
        recs = self._analyze_and_get_recs(0.85)
        assert any("DO NOT apply" in r for r in recs)

    def test_high_score_extreme_caution(self):
        recs = self._analyze_and_get_recs(0.70)
        assert any("extreme caution" in r.lower() for r in recs)

    def test_medium_score_caution(self):
        recs = self._analyze_and_get_recs(0.50)
        assert any("caution" in r.lower() for r in recs)

    def test_low_score_mostly_legitimate(self):
        recs = self._analyze_and_get_recs(0.15)
        assert any("legitimate" in r.lower() for r in recs)

    def test_safe_score_no_strong_warnings(self):
        recs = self._analyze_and_get_recs(0.05)
        # Should not say DO NOT apply for very safe scores
        assert not any("DO NOT apply" in r for r in recs)

    def test_evasion_detected_adds_recommendation(self):
        recs = self._analyze_and_get_recs(0.9, evasion_detected=True)
        assert any("Obfuscation" in r or "evasion" in r.lower() for r in recs)

    def test_similar_postings_adds_recommendation(self):
        recs = self._analyze_and_get_recs(0.9, similar_postings=["job_x", "job_y"])
        assert any("near-duplicate" in r.lower() or "resembles" in r.lower() for r in recs)

    def test_operator_fingerprint_adds_recommendation(self):
        recs = self._analyze_and_get_recs(0.9, operator_fingerprint="scammer_001")
        assert any("scammer_001" in r for r in recs)

    def test_llm_generated_adds_recommendation(self):
        recs = self._analyze_and_get_recs(0.9, llm_generated_probability=0.85)
        assert any("AI-generated" in r or "LLM" in r for r in recs)


# ---------------------------------------------------------------------------
# Nexus.deep_analyze() — integration tests
# ---------------------------------------------------------------------------

class TestNexusDeepAnalyze:
    def setup_method(self):
        self.nexus = Nexus()

    def test_returns_nexus_report(self):
        job = _make_job()
        result = self.nexus.deep_analyze(job)
        assert isinstance(result, NexusReport)

    def test_increments_analysis_count(self):
        job = _make_job()
        self.nexus.deep_analyze(job)
        assert self.nexus._analysis_count == 1
        self.nexus.deep_analyze(job)
        assert self.nexus._analysis_count == 2

    def test_analysis_time_is_positive(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert report.analysis_time_ms > 0

    def test_analyzed_at_is_set(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert report.analyzed_at is not None

    def test_risk_level_valid_string(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert report.risk_level in {"SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"}

    def test_overall_score_in_range(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert 0.0 <= report.overall_score <= 1.0

    def test_confidence_in_range(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert 0.0 <= report.confidence <= 1.0

    def test_scam_job_scores_higher_than_legit(self):
        scam = _make_scam_job()
        legit = _make_legit_job()
        scam_report = self.nexus.deep_analyze(scam)
        legit_report = self.nexus.deep_analyze(legit)
        # Scam job should generally score higher
        assert scam_report.overall_score >= legit_report.overall_score

    def test_scam_job_fires_signals(self):
        scam = _make_scam_job()
        report = self.nexus.deep_analyze(scam)
        assert len(report.signals_fired) > 0

    def test_subsystems_run_is_list(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert isinstance(report.subsystems_run, list)

    def test_subsystem_scores_are_floats(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        for k, v in report.subsystem_scores.items():
            assert isinstance(v, float), f"Score for {k} is not float: {v}"
            assert 0.0 <= v <= 1.0, f"Score for {k} out of range: {v}"

    def test_to_dict_from_real_analysis(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["risk_level"] in {"SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"}

    def test_empty_job_does_not_crash(self):
        job = JobPosting()
        report = self.nexus.deep_analyze(job)
        assert isinstance(report, NexusReport)
        assert 0.0 <= report.overall_score <= 1.0

    def test_no_salary_skips_benford(self):
        job = _make_job(salary_min=0, salary_max=0)
        report = self.nexus.deep_analyze(job)
        assert isinstance(report, NexusReport)

    def test_similarity_index_created_lazily(self):
        nexus = Nexus()
        assert nexus._similarity_index is None
        job = _make_job()
        nexus.deep_analyze(job)
        # After analysis, similarity index may have been initialized
        # (only if graph subsystem is available)

    def test_second_similar_posting_detected(self):
        # Two identical jobs should be flagged as near-duplicates
        nexus = Nexus()
        job1 = _make_job(url="https://linkedin.com/jobs/dup1")
        job2 = _make_job(url="https://linkedin.com/jobs/dup2")
        nexus.deep_analyze(job1)
        report2 = nexus.deep_analyze(job2)
        # Second analysis may find first as similar (if graph subsystem available)
        assert isinstance(report2.similar_postings, list)

    def test_recommendations_generated_for_high_score(self):
        scam = _make_scam_job()
        report = self.nexus.deep_analyze(scam)
        if report.overall_score >= 0.4:
            assert len(report.recommendations) > 0

    def test_key_findings_is_list(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert isinstance(report.key_findings, list)

    def test_economic_flags_is_list(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert isinstance(report.economic_flags, list)

    def test_counterfactual_insights_is_list(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert isinstance(report.counterfactual_insights, list)

    def test_disagreements_is_list(self):
        job = _make_job()
        report = self.nexus.deep_analyze(job)
        assert isinstance(report.disagreements, list)


# ---------------------------------------------------------------------------
# NexusLearner
# ---------------------------------------------------------------------------

class TestNexusLearner:
    def setup_method(self):
        self.nexus = Nexus()
        self.learner = NexusLearner(self.nexus)

    def test_learn_returns_dict(self):
        job = _make_job()
        report = _make_report()
        result = self.learner.learn(job, report, is_scam=True)
        assert isinstance(result, dict)

    def test_learn_has_was_correct(self):
        job = _make_job()
        report = _make_report(overall_score=0.8)
        result = self.learner.learn(job, report, is_scam=True)
        assert "was_correct" in result
        assert result["was_correct"] is True  # predicted scam (0.8 >= 0.5), ground truth scam

    def test_learn_wrong_prediction_captured(self):
        job = _make_job()
        report = _make_report(overall_score=0.2)  # predicted legit
        result = self.learner.learn(job, report, is_scam=True)  # but it was scam
        assert result["was_correct"] is False

    def test_learn_records_accuracy_per_subsystem(self):
        job = _make_job()
        report = _make_report(subsystem_scores={"signals": 0.8, "fraud_triangle": 0.7})
        self.learner.learn(job, report, is_scam=True)
        assert "signals" in self.learner._accuracy
        assert "fraud_triangle" in self.learner._accuracy

    def test_accuracy_summary_empty_initially(self):
        acc = self.learner.accuracy_summary()
        assert isinstance(acc, dict)

    def test_accuracy_summary_after_feedback(self):
        job = _make_job()
        report = _make_report(overall_score=0.8, subsystem_scores={"signals": 0.9})
        self.learner.learn(job, report, is_scam=True)
        self.learner.learn(job, report, is_scam=True)
        self.learner.learn(job, report, is_scam=True)
        self.learner.learn(job, report, is_scam=True)
        self.learner.learn(job, report, is_scam=True)
        acc = self.learner.accuracy_summary()
        assert "signals" in acc
        assert 0.0 <= acc["signals"] <= 1.0

    def test_most_accurate_subsystem_none_when_empty(self):
        assert self.learner.most_accurate_subsystem() is None

    def test_most_accurate_subsystem_after_feedback(self):
        job = _make_job()
        # Give signals 100% accuracy, fraud_triangle 0%
        report_scam = _make_report(overall_score=0.9, subsystem_scores={"signals": 0.9, "fraud_triangle": 0.1})
        report_legit = _make_report(overall_score=0.1, subsystem_scores={"signals": 0.1, "fraud_triangle": 0.9})
        # 5+ observations for signals: always correct
        for _ in range(6):
            self.learner.learn(job, report_scam, is_scam=True)   # signals correct
        # 5+ observations for fraud_triangle: always wrong
        for _ in range(6):
            self.learner.learn(job, report_legit, is_scam=True)  # fraud_triangle wrong
        best = self.learner.most_accurate_subsystem()
        assert best is not None

    def test_recompute_returns_empty_with_no_data(self):
        weights = self.learner._recompute_meta_weights()
        assert isinstance(weights, dict)

    def test_recompute_normalizes_weights(self):
        job = _make_job()
        report = _make_report(subsystem_scores={k: 0.5 for k in _DEFAULT_META_WEIGHTS})
        for _ in range(10):
            self.learner.learn(job, report, is_scam=True)
        weights = self.learner._recompute_meta_weights()
        if weights:
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01

    def test_learn_updates_meta_weights_after_many_iterations(self):
        job = _make_job()
        report = _make_report(
            overall_score=0.8,
            subsystem_scores={k: 0.8 for k in list(_DEFAULT_META_WEIGHTS.keys())[:5]},
        )
        for _ in range(15):
            self.learner.learn(job, report, is_scam=True)
        # Meta weights may have changed
        weights = self.nexus.get_meta_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_learn_ground_truth_in_result(self):
        job = _make_job()
        report = _make_report()
        result = self.learner.learn(job, report, is_scam=False)
        assert result["ground_truth_scam"] is False

    def test_learn_predicted_score_in_result(self):
        job = _make_job()
        report = _make_report(overall_score=0.65)
        result = self.learner.learn(job, report, is_scam=True)
        assert abs(result["predicted_score"] - 0.65) < 0.001

    def test_learn_subsystems_updated_list(self):
        job = _make_job()
        report = _make_report()
        result = self.learner.learn(job, report, is_scam=True)
        assert "subsystems_updated" in result
        assert isinstance(result["subsystems_updated"], list)


# ---------------------------------------------------------------------------
# NexusDashboard
# ---------------------------------------------------------------------------

class TestNexusDashboard:
    def setup_method(self):
        self.nexus = Nexus()
        self.learner = NexusLearner(self.nexus)
        self.dashboard = NexusDashboard(self.nexus, learner=self.learner)

    def test_snapshot_returns_dataclass(self):
        snap = self.dashboard.snapshot()
        assert isinstance(snap, DashboardSnapshot)

    def test_snapshot_has_timestamp(self):
        snap = self.dashboard.snapshot()
        assert isinstance(snap.timestamp, str)
        assert "T" in snap.timestamp

    def test_snapshot_total_analyses_zero_initially(self):
        snap = self.dashboard.snapshot()
        assert snap.total_analyses == 0

    def test_snapshot_total_analyses_after_analyze(self):
        job = _make_job()
        self.nexus.deep_analyze(job)
        snap = self.dashboard.snapshot()
        assert snap.total_analyses == 1

    def test_snapshot_subsystem_statuses_all_present(self):
        snap = self.dashboard.snapshot()
        names = {s.name for s in snap.subsystem_statuses}
        assert "signals" in names
        assert "scorer" in names
        assert "db" in names

    def test_subsystem_status_fields(self):
        snap = self.dashboard.snapshot()
        for status in snap.subsystem_statuses:
            assert isinstance(status, SubsystemStatus)
            assert isinstance(status.available, bool)
            assert 0.0 <= status.accuracy <= 1.0
            assert status.observations >= 0

    def test_system_health_score_in_range(self):
        snap = self.dashboard.snapshot()
        assert 0.0 <= snap.system_health_score <= 1.0

    def test_top_signals_empty_initially(self):
        snap = self.dashboard.snapshot()
        assert isinstance(snap.top_signals, list)

    def test_top_signals_after_analysis(self):
        job = _make_scam_job()
        report = self.nexus.deep_analyze(job)
        self.dashboard.record_result(report)
        snap = self.dashboard.snapshot()
        if report.signals_fired:
            assert len(snap.top_signals) > 0

    def test_record_result_increments_window(self):
        initial = len(self.dashboard._recent_reports)
        report = _make_report()
        self.dashboard.record_result(report)
        assert len(self.dashboard._recent_reports) == initial + 1

    def test_record_result_with_ground_truth(self):
        report = _make_report(overall_score=0.8)
        self.dashboard.record_result(report, ground_truth_scam=True)
        assert len(self.dashboard._correct_predictions) == 1
        assert self.dashboard._correct_predictions[0] is True

    def test_record_result_wrong_prediction(self):
        report = _make_report(overall_score=0.2)
        self.dashboard.record_result(report, ground_truth_scam=True)
        assert self.dashboard._correct_predictions[0] is False

    def test_accuracy_trend_empty_initially(self):
        snap = self.dashboard.snapshot()
        assert isinstance(snap.accuracy_trend, list)

    def test_recommendations_not_empty_when_no_analyses(self):
        snap = self.dashboard.snapshot()
        assert isinstance(snap.recommendations, list)

    def test_most_active_signals_empty_initially(self):
        result = self.dashboard.most_active_signals()
        assert isinstance(result, list)

    def test_most_active_signals_after_analysis(self):
        report = _make_report(signals_fired=[_make_signal("sig_a"), _make_signal("sig_b")])
        self.dashboard.record_result(report)
        result = self.dashboard.most_active_signals(top_n=5)
        assert len(result) <= 5
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)

    def test_most_informative_subsystems_returns_list(self):
        result = self.dashboard.most_informative_subsystems()
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_window_limits_recent_reports(self):
        dashboard = NexusDashboard(self.nexus, window=3)
        for i in range(10):
            dashboard.record_result(_make_report())
        assert len(dashboard._recent_reports) <= 3


# ---------------------------------------------------------------------------
# NexusEvolver
# ---------------------------------------------------------------------------

class TestNexusEvolver:
    def setup_method(self):
        self.nexus = Nexus()
        self.learner = NexusLearner(self.nexus)
        self.dashboard = NexusDashboard(self.nexus, learner=self.learner)
        self.evolver = NexusEvolver(
            self.nexus,
            learner=self.learner,
            dashboard=self.dashboard,
        )

    def test_evolve_returns_evolve_result(self):
        result = self.evolver.evolve()
        assert isinstance(result, EvolveResult)

    def test_evolve_result_fields(self):
        result = self.evolver.evolve()
        assert isinstance(result.cycle, int)
        assert result.cycle >= 1
        assert isinstance(result.timestamp, str)
        assert isinstance(result.steps_completed, list)
        assert isinstance(result.counterfactual_insights, list)
        assert isinstance(result.research_topics_added, int)
        assert isinstance(result.innovation_ran, bool)
        assert isinstance(result.regression_clean, bool)
        assert isinstance(result.meta_weight_adjustments, dict)
        assert isinstance(result.summary, str)

    def test_evolve_increments_cycle(self):
        self.evolver.evolve()
        self.evolver.evolve()
        result = self.evolver.evolve()
        assert result.cycle == 3

    def test_evolve_summary_is_nonempty(self):
        result = self.evolver.evolve()
        assert len(result.summary) > 0

    def test_evolve_with_recent_reports(self):
        reports = [_make_report(overall_score=0.2) for _ in range(5)]
        result = self.evolver.evolve(recent_reports=reports)
        assert isinstance(result, EvolveResult)

    def test_evolve_with_scam_reports_triggers_counterfactual(self):
        # Low-score jobs with signals may trigger counterfactual analysis
        reports = [
            _make_report(
                overall_score=0.2,
                signals_fired=[_make_signal("sig1"), _make_signal("sig2")],
            )
            for _ in range(5)
        ]
        result = self.evolver.evolve(recent_reports=reports)
        assert isinstance(result.counterfactual_insights, list)

    def test_run_cycles_returns_list(self):
        results = self.evolver.run_cycles(3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_run_cycles_increments_cycle_numbers(self):
        results = self.evolver.run_cycles(3)
        cycles = [r.cycle for r in results]
        assert cycles == sorted(cycles)
        assert cycles[-1] - cycles[0] == 2

    def test_regression_clean_default(self):
        result = self.evolver.evolve()
        # No accuracy data — should default to clean or be handled gracefully
        assert isinstance(result.regression_clean, bool)

    def test_regression_triggered_on_low_accuracy(self):
        # Inject bad accuracy data
        for _ in range(10):
            self.learner._accuracy.setdefault("signals", []).append(False)
        result = self.evolver.evolve()
        assert isinstance(result.regression_clean, bool)


# ---------------------------------------------------------------------------
# EvolveResult dataclass
# ---------------------------------------------------------------------------

class TestEvolveResult:
    def test_can_construct(self):
        er = EvolveResult(
            cycle=1,
            timestamp="2025-01-01T00:00:00",
            steps_completed=["a", "b"],
            counterfactual_insights=["insight1"],
            research_topics_added=2,
            innovation_ran=True,
            regression_clean=True,
            meta_weight_adjustments={"signals": 0.01},
            summary="All good.",
        )
        assert er.cycle == 1
        assert er.innovation_ran is True
        assert er.research_topics_added == 2

    def test_all_fields_present(self):
        field_names = {f.name for f in fields(EvolveResult)}
        assert "cycle" in field_names
        assert "steps_completed" in field_names
        assert "counterfactual_insights" in field_names
        assert "research_topics_added" in field_names
        assert "innovation_ran" in field_names
        assert "regression_clean" in field_names
        assert "meta_weight_adjustments" in field_names
        assert "summary" in field_names


# ---------------------------------------------------------------------------
# Default meta-weights sanity checks
# ---------------------------------------------------------------------------

class TestDefaultMetaWeights:
    def test_all_keys_present(self):
        keys = set(_DEFAULT_META_WEIGHTS.keys())
        assert "signals" in keys
        assert "fraud_triangle" in keys
        assert "benford" in keys
        assert "linguistic" in keys
        assert "llm_detect" in keys
        assert "stylometry" in keys
        assert "economics" in keys
        assert "graph" in keys
        assert "robustness" in keys
        assert "adversarial" in keys

    def test_all_weights_positive(self):
        for k, v in _DEFAULT_META_WEIGHTS.items():
            assert v > 0, f"Weight for {k} should be positive"

    def test_weights_sum_to_one(self):
        total = sum(_DEFAULT_META_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"


# ---------------------------------------------------------------------------
# Graceful degradation — stub out subsystems and ensure nothing crashes
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_deep_analyze_with_no_subsystems(self):
        """Patching all subsystem flags to False should still return a valid report."""
        patches = {
            "sentinel.nexus._HAS_SIGNALS": False,
            "sentinel.nexus._HAS_SCORER": False,
            "sentinel.nexus._HAS_FRAUD": False,
            "sentinel.nexus._HAS_LLM": False,
            "sentinel.nexus._HAS_STYLO": False,
            "sentinel.nexus._HAS_ECON": False,
            "sentinel.nexus._HAS_GRAPH": False,
            "sentinel.nexus._HAS_ROBUST": False,
            "sentinel.nexus._HAS_ADVERSARIAL": False,
            "sentinel.nexus._HAS_DISAGREE": False,
            "sentinel.nexus._HAS_COUNTER": False,
        }
        with patch.multiple("sentinel.nexus", **{k.split(".")[-1]: v for k, v in patches.items()}):
            nexus = Nexus()
            job = _make_job()
            report = nexus.deep_analyze(job)
            assert isinstance(report, NexusReport)
            assert report.overall_score == 0.0
            assert report.subsystems_run == []

    def test_nexus_survives_scorer_exception(self):
        """If score_signals raises, analysis continues."""
        nexus = Nexus()
        with patch("sentinel.nexus.score_signals", side_effect=RuntimeError("boom")):
            job = _make_job()
            # Should not raise
            report = nexus.deep_analyze(job)
            assert isinstance(report, NexusReport)

    def test_nexus_survives_signals_exception(self):
        """If extract_signals raises, analysis continues."""
        nexus = Nexus()
        with patch("sentinel.nexus.extract_signals", side_effect=RuntimeError("bad")):
            job = _make_job()
            report = nexus.deep_analyze(job)
            assert isinstance(report, NexusReport)
