"""Tests for the Shadow Scoring / A/B weight testing system."""

import json

import pytest

from sentinel.db import SentinelDB
from sentinel.models import ScamSignal, SignalCategory
from sentinel.shadow import (
    DualScoreResult,
    EvaluationResult,
    ShadowScorer,
    _dicts_to_signals,
    _load_primary_weights,
    _score_signals_with_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def shadow_db(tmp_path) -> SentinelDB:
    """Fresh SentinelDB for shadow tests."""
    db = SentinelDB(path=str(tmp_path / "shadow_test.db"))
    yield db
    db.close()


@pytest.fixture
def shadow(shadow_db) -> ShadowScorer:
    return ShadowScorer(shadow_db)


def _red_flag(name: str, weight: float = 0.7) -> ScamSignal:
    return ScamSignal(name=name, category=SignalCategory.RED_FLAG, weight=weight)


def _positive(name: str, weight: float = 0.3) -> ScamSignal:
    return ScamSignal(name=name, category=SignalCategory.POSITIVE, weight=weight)


def _seed_pattern(db: SentinelDB, pattern_id: str, alpha: float, beta: float, obs: int = 15) -> None:
    """Insert a pattern into the DB with the given alpha/beta."""
    db.save_pattern({
        "pattern_id": pattern_id,
        "name": pattern_id,
        "alpha": alpha,
        "beta": beta,
        "observations": obs,
        "status": "active",
    })


def _seed_job_with_report(
    db: SentinelDB,
    url: str,
    signals: list[dict],
    is_scam: bool,
    scam_score: float = 0.5,
) -> None:
    """Insert a job and a corresponding user report."""
    db.save_job({
        "url": url,
        "title": "Test Job",
        "company": "Test Corp",
        "signals_json": json.dumps(signals),
        "scam_score": scam_score,
        "risk_level": "suspicious",
        "signal_count": len(signals),
    })
    db.save_report({
        "url": url,
        "is_scam": is_scam,
        "reason": "test",
        "our_prediction": scam_score,
        "was_correct": (scam_score >= 0.5) == is_scam,
    })


# ---------------------------------------------------------------------------
# Test: _score_signals_with_weights produces different scores with different weights
# ---------------------------------------------------------------------------

class TestScoreSignalsWithWeights:
    def test_empty_signals(self):
        score, conf = _score_signals_with_weights([], {})
        assert score == 0.0
        assert conf == 0.0

    def test_red_flag_increases_score(self):
        signals = [_red_flag("upfront_payment", weight=0.8)]
        score, _ = _score_signals_with_weights(signals, {})
        assert score > 0.5

    def test_positive_decreases_score(self):
        signals = [_positive("established_company", weight=0.3)]
        score, _ = _score_signals_with_weights(signals, {})
        assert score < 0.5

    def test_different_weights_produce_different_scores(self):
        signals = [_red_flag("sig_a", weight=0.5)]
        score_low, _ = _score_signals_with_weights(signals, {"sig_a": 0.3})
        score_high, _ = _score_signals_with_weights(signals, {"sig_a": 0.9})
        assert score_high > score_low


# ---------------------------------------------------------------------------
# Test: DualScoreResult and ShadowScorer.dual_score
# ---------------------------------------------------------------------------

class TestDualScore:
    def test_dual_score_without_active_run(self, shadow):
        """When no shadow run is active, shadow mirrors primary."""
        signals = [_red_flag("test_sig", weight=0.7)]
        result = shadow.dual_score(signals)
        assert isinstance(result, DualScoreResult)
        assert result.primary_score == result.shadow_score
        assert result.primary_confidence == result.shadow_confidence

    def test_dual_score_with_active_run(self, shadow):
        """Shadow and primary produce different scores with different weights."""
        shadow.propose_weights({"test_sig": 0.9})
        signals = [_red_flag("test_sig", weight=0.5)]
        result = shadow.dual_score(signals)
        # Shadow uses weight 0.9, primary uses default 0.5 (no DB patterns)
        assert result.shadow_score != result.primary_score
        assert result.shadow_score > result.primary_score

    def test_dual_score_signal_count(self, shadow):
        signals = [_red_flag("a"), _red_flag("b"), _positive("c")]
        result = shadow.dual_score(signals)
        assert result.signal_count == 3


# ---------------------------------------------------------------------------
# Test: propose_weights / active / lifecycle
# ---------------------------------------------------------------------------

class TestProposeWeights:
    def test_propose_starts_shadow_run(self, shadow):
        assert not shadow.active
        run_id = shadow.propose_weights({"sig_a": 0.8})
        assert shadow.active
        assert run_id is not None

    def test_propose_creates_db_record(self, shadow, shadow_db):
        shadow.propose_weights({"sig_a": 0.8})
        active = shadow_db.get_active_shadow_run()
        assert active is not None
        assert active["candidate_weights"]["sig_a"] == 0.8

    def test_propose_rejects_previous_active_run(self, shadow, shadow_db):
        first_id = shadow.propose_weights({"sig_a": 0.5})
        second_id = shadow.propose_weights({"sig_b": 0.9})
        assert second_id != first_id
        # First run should be rejected
        history = shadow_db.get_shadow_history()
        statuses = {h["id"]: h["status"] for h in history}
        assert statuses[first_id] == "rejected"
        assert statuses[second_id] == "active"


# ---------------------------------------------------------------------------
# Test: should_promote criteria
# ---------------------------------------------------------------------------

class TestShouldPromote:
    def test_returns_false_without_evaluation(self, shadow):
        shadow.propose_weights({"sig": 0.8})
        assert not shadow.should_promote()

    def test_returns_false_with_too_few_jobs(self, shadow):
        shadow.propose_weights({"sig": 0.8})
        shadow._evaluation = EvaluationResult(
            baseline_precision=0.5,
            shadow_precision=0.7,
            jobs_evaluated=10,  # below default min_jobs=30
            improvement=0.2,
        )
        assert not shadow.should_promote()

    def test_returns_false_when_shadow_is_worse(self, shadow):
        shadow.propose_weights({"sig": 0.8})
        shadow._evaluation = EvaluationResult(
            baseline_precision=0.7,
            shadow_precision=0.6,
            jobs_evaluated=50,
            improvement=-0.1,
        )
        assert not shadow.should_promote()

    def test_returns_false_when_improvement_too_small(self, shadow):
        shadow.propose_weights({"sig": 0.8})
        shadow._evaluation = EvaluationResult(
            baseline_precision=0.7,
            shadow_precision=0.71,
            jobs_evaluated=50,
            improvement=0.01,  # below default min_improvement=0.02
        )
        assert not shadow.should_promote()

    def test_returns_true_when_criteria_met(self, shadow):
        shadow.propose_weights({"sig": 0.8})
        shadow._evaluation = EvaluationResult(
            baseline_precision=0.7,
            shadow_precision=0.75,
            jobs_evaluated=50,
            improvement=0.05,
        )
        assert shadow.should_promote()

    def test_custom_thresholds(self, shadow):
        shadow.propose_weights({"sig": 0.8})
        shadow._evaluation = EvaluationResult(
            baseline_precision=0.7,
            shadow_precision=0.72,
            jobs_evaluated=15,
            improvement=0.02,
        )
        # Default min_jobs=30 would reject, but custom min_jobs=10 accepts
        assert shadow.should_promote(min_improvement=0.01, min_jobs=10)
        assert not shadow.should_promote(min_improvement=0.05, min_jobs=10)


# ---------------------------------------------------------------------------
# Test: promote
# ---------------------------------------------------------------------------

class TestPromote:
    def test_promote_updates_db_patterns(self, shadow, shadow_db):
        _seed_pattern(shadow_db, "sig_a", alpha=8.0, beta=2.0)
        run_id = shadow.propose_weights({"sig_a": 0.6})
        shadow._evaluation = EvaluationResult(
            baseline_precision=0.7, shadow_precision=0.8,
            jobs_evaluated=50, improvement=0.1,
        )
        result = shadow.promote()
        assert result["promoted"] is True
        assert "sig_a" in result["patterns_updated"]

        # Verify the pattern alpha/beta changed
        patterns = shadow_db.get_patterns(status="active")
        sig_a = [p for p in patterns if p["pattern_id"] == "sig_a"][0]
        total = sig_a["alpha"] + sig_a["beta"]
        actual_weight = sig_a["alpha"] / total
        assert abs(actual_weight - 0.6) < 0.01

    def test_promote_marks_run_promoted(self, shadow, shadow_db):
        run_id = shadow.propose_weights({"sig_a": 0.6})
        shadow._evaluation = EvaluationResult(
            baseline_precision=0.7, shadow_precision=0.8,
            jobs_evaluated=50, improvement=0.1,
        )
        shadow.promote()
        history = shadow_db.get_shadow_history()
        run = [h for h in history if h["id"] == run_id][0]
        assert run["status"] == "promoted"
        assert run["promoted"] == 1

    def test_promote_resets_state(self, shadow):
        shadow.propose_weights({"sig_a": 0.6})
        shadow._evaluation = EvaluationResult(
            baseline_precision=0.7, shadow_precision=0.8,
            jobs_evaluated=50, improvement=0.1,
        )
        shadow.promote()
        assert not shadow.active

    def test_promote_without_active_run(self, shadow):
        result = shadow.promote()
        assert result["promoted"] is False


# ---------------------------------------------------------------------------
# Test: reject
# ---------------------------------------------------------------------------

class TestReject:
    def test_reject_marks_run_rejected(self, shadow, shadow_db):
        run_id = shadow.propose_weights({"sig_a": 0.6})
        result = shadow.reject()
        assert result["rejected"] is True
        assert result["run_id"] == run_id
        history = shadow_db.get_shadow_history()
        run = [h for h in history if h["id"] == run_id][0]
        assert run["status"] == "rejected"

    def test_reject_resets_state(self, shadow):
        shadow.propose_weights({"sig_a": 0.6})
        shadow.reject()
        assert not shadow.active


# ---------------------------------------------------------------------------
# Test: evaluate with real DB data
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_empty_db(self, shadow):
        shadow.propose_weights({"sig_a": 0.8})
        result = shadow.evaluate()
        assert result.jobs_evaluated == 0

    def test_evaluate_with_reports(self, shadow, shadow_db):
        """Seed jobs and reports, then evaluate."""
        # Create signals that the shadow weights will change
        signals = [
            {"name": "upfront_payment", "category": "red_flag", "weight": 0.7},
            {"name": "guaranteed_income", "category": "red_flag", "weight": 0.6},
        ]
        for i in range(5):
            _seed_job_with_report(
                shadow_db,
                url=f"https://example.com/scam-{i}",
                signals=signals,
                is_scam=True,
                scam_score=0.8,
            )
        for i in range(5):
            _seed_job_with_report(
                shadow_db,
                url=f"https://example.com/legit-{i}",
                signals=[{"name": "established_company", "category": "positive", "weight": 0.3}],
                is_scam=False,
                scam_score=0.2,
            )

        shadow.propose_weights({"upfront_payment": 0.9, "guaranteed_income": 0.85})
        result = shadow.evaluate(n_jobs=20)
        assert result.jobs_evaluated == 10
        assert result.baseline_precision >= 0.0
        assert result.shadow_precision >= 0.0

    def test_evaluate_persists_to_db(self, shadow, shadow_db):
        signals = [{"name": "sig_a", "category": "red_flag", "weight": 0.7}]
        for i in range(3):
            _seed_job_with_report(shadow_db, f"https://example.com/{i}", signals, True)
        run_id = shadow.propose_weights({"sig_a": 0.9})
        shadow.evaluate(n_jobs=10)
        run = shadow_db.get_active_shadow_run()
        assert run is not None
        assert run["jobs_evaluated"] == 3


# ---------------------------------------------------------------------------
# Test: DB methods
# ---------------------------------------------------------------------------

class TestDBShadowMethods:
    def test_insert_and_get_active(self, shadow_db):
        run_id = shadow_db.insert_shadow_run({"sig_a": 0.5})
        active = shadow_db.get_active_shadow_run()
        assert active is not None
        assert active["id"] == run_id
        assert active["candidate_weights"]["sig_a"] == 0.5
        assert active["status"] == "active"

    def test_no_active_when_empty(self, shadow_db):
        assert shadow_db.get_active_shadow_run() is None

    def test_update_shadow_run(self, shadow_db):
        run_id = shadow_db.insert_shadow_run({"sig": 0.5})
        shadow_db.update_shadow_run(run_id, {
            "baseline_precision": 0.7,
            "shadow_precision": 0.8,
            "jobs_evaluated": 42,
        })
        active = shadow_db.get_active_shadow_run()
        assert active["baseline_precision"] == 0.7
        assert active["shadow_precision"] == 0.8
        assert active["jobs_evaluated"] == 42

    def test_promote_shadow_run(self, shadow_db):
        run_id = shadow_db.insert_shadow_run({"sig": 0.5})
        shadow_db.promote_shadow_run(run_id)
        active = shadow_db.get_active_shadow_run()
        assert active is None  # no longer active
        history = shadow_db.get_shadow_history()
        assert history[0]["status"] == "promoted"
        assert history[0]["promoted"] == 1

    def test_reject_shadow_run(self, shadow_db):
        run_id = shadow_db.insert_shadow_run({"sig": 0.5})
        shadow_db.reject_shadow_run(run_id)
        active = shadow_db.get_active_shadow_run()
        assert active is None
        history = shadow_db.get_shadow_history()
        assert history[0]["status"] == "rejected"

    def test_shadow_history_ordering(self, shadow_db):
        id1 = shadow_db.insert_shadow_run({"a": 0.1})
        id2 = shadow_db.insert_shadow_run({"b": 0.2})
        id3 = shadow_db.insert_shadow_run({"c": 0.3})
        history = shadow_db.get_shadow_history()
        assert [h["id"] for h in history] == [id3, id2, id1]

    def test_shadow_history_limit(self, shadow_db):
        for i in range(10):
            shadow_db.insert_shadow_run({f"sig_{i}": 0.5})
        history = shadow_db.get_shadow_history(limit=3)
        assert len(history) == 3


# ---------------------------------------------------------------------------
# Test: _dicts_to_signals
# ---------------------------------------------------------------------------

class TestDictsToSignals:
    def test_converts_valid_dicts(self):
        dicts = [
            {"name": "sig_a", "category": "red_flag", "weight": 0.8},
            {"name": "sig_b", "category": "positive", "weight": 0.3},
        ]
        signals = _dicts_to_signals(dicts)
        assert len(signals) == 2
        assert signals[0].name == "sig_a"
        assert signals[0].category == SignalCategory.RED_FLAG
        assert signals[1].category == SignalCategory.POSITIVE

    def test_skips_invalid_entries(self):
        dicts = [
            {"name": "good", "category": "warning"},
            "not_a_dict",
            {"no_name": True},
            {},
        ]
        signals = _dicts_to_signals(dicts)
        assert len(signals) == 1
        assert signals[0].name == "good"

    def test_handles_unknown_category(self):
        dicts = [{"name": "test", "category": "nonexistent"}]
        signals = _dicts_to_signals(dicts)
        assert signals[0].category == SignalCategory.WARNING


# ---------------------------------------------------------------------------
# Test: _load_primary_weights
# ---------------------------------------------------------------------------

class TestLoadPrimaryWeights:
    def test_empty_db(self, shadow_db):
        weights = _load_primary_weights(shadow_db)
        assert weights == {}

    def test_loads_active_patterns(self, shadow_db):
        _seed_pattern(shadow_db, "sig_a", alpha=8.0, beta=2.0, obs=15)
        weights = _load_primary_weights(shadow_db)
        assert "sig_a" in weights
        assert abs(weights["sig_a"] - 0.8) < 0.01

    def test_skips_low_observation_patterns(self, shadow_db):
        _seed_pattern(shadow_db, "sig_a", alpha=8.0, beta=2.0, obs=5)
        weights = _load_primary_weights(shadow_db)
        assert "sig_a" not in weights


# ---------------------------------------------------------------------------
# Test: get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_status_no_active_run(self, shadow):
        status = shadow.get_status()
        assert status["active"] is False
        assert status["run_id"] is None

    def test_status_with_active_run(self, shadow):
        shadow.propose_weights({"sig_a": 0.8, "sig_b": 0.7})
        status = shadow.get_status()
        assert status["active"] is True
        assert status["candidate_weight_count"] == 2


# ---------------------------------------------------------------------------
# Test: ShadowScorer resumes active run from DB
# ---------------------------------------------------------------------------

class TestResumeFromDB:
    def test_resumes_active_shadow_run(self, shadow_db):
        shadow_db.insert_shadow_run({"sig_a": 0.75})
        scorer = ShadowScorer(shadow_db)
        assert scorer.active
        assert scorer._candidate_weights["sig_a"] == 0.75


# ---------------------------------------------------------------------------
# Test: flywheel integration
# ---------------------------------------------------------------------------

class TestFlywheelIntegration:
    def test_flywheel_has_shadow_attribute(self, shadow_db):
        from sentinel.flywheel import DetectionFlywheel
        fw = DetectionFlywheel(db=shadow_db)
        assert hasattr(fw, "shadow")
        assert isinstance(fw.shadow, ShadowScorer)

    def test_run_cycle_includes_shadow_evaluation(self, shadow_db):
        from sentinel.flywheel import DetectionFlywheel
        fw = DetectionFlywheel(db=shadow_db)
        metrics = fw.run_cycle()
        assert "shadow_evaluation" in metrics
        # No active shadow run, so it should report inactive
        assert metrics["shadow_evaluation"]["active"] is False

    def test_evolve_proposes_weights_on_promotion(self, shadow_db):
        from sentinel.flywheel import DetectionFlywheel

        # Create a candidate pattern that will be promoted
        shadow_db.save_pattern({
            "pattern_id": "candidate_sig",
            "name": "candidate_sig",
            "alpha": 9.0,
            "beta": 1.0,
            "observations": 15,
            "true_positives": 12,
            "false_positives": 3,
            "status": "candidate",
        })

        fw = DetectionFlywheel(db=shadow_db)
        evolution = fw.evolve_patterns()
        assert "candidate_sig" in evolution["promoted"]
        # After promotion the shadow should have been proposed weights
        assert fw.shadow.active

    def test_run_cycle_with_shadow_evaluation(self, shadow_db):
        """Full cycle with shadow evaluation that does not promote (too few jobs)."""
        from sentinel.flywheel import DetectionFlywheel

        fw = DetectionFlywheel(db=shadow_db)
        # Manually propose weights
        fw.shadow.propose_weights({"test_sig": 0.8})
        metrics = fw.run_cycle()
        shadow_eval = metrics["shadow_evaluation"]
        assert shadow_eval["active"] is True
        assert shadow_eval["jobs_evaluated"] == 0
        # Not enough jobs, should not promote or reject (need >= 30)
        assert shadow_eval["promoted"] is False
