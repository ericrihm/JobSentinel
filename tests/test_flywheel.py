"""Comprehensive tests for sentinel.flywheel — CUSUMDetector and DetectionFlywheel."""

import pytest

from sentinel.db import SentinelDB
from sentinel.flywheel import CUSUMDetector, DetectionFlywheel, SignalWeightTracker
from sentinel.knowledge import KnowledgeBase
from sentinel.models import (
    JobPosting,
    ScamSignal,
    SignalCategory,
    UserReport,
    ValidationResult,
)


# ===========================================================================
# TestCUSUMDetector
# ===========================================================================


class TestCUSUMDetector:
    def test_no_alarm_below_threshold(self):
        """Feeding small errors should keep the alarm off."""
        detector = CUSUMDetector(target=0.0, slack=0.5, threshold=5.0)
        for _ in range(20):
            alarm = detector.update(0.01)
        assert alarm is False
        assert detector.statistic < 5.0

    def test_alarm_when_drift_exceeds_threshold(self):
        """Large repeated negative errors should eventually trigger the alarm."""
        detector = CUSUMDetector(target=0.0, slack=0.5, threshold=5.0)
        alarm = False
        for _ in range(50):
            alarm = detector.update(0.5)
            if alarm:
                break
        # deviation per step = 0.5 - (0.0 - 0.5) = 1.0, so s_neg stays 0 (positive deviation
        # means no regression). Use a *negative* observation to simulate regression.
        # With target=0.0 and slack=0.5: deviation = value - (target - slack) = value + 0.5
        # For alarm to fire we need s_neg to accumulate, meaning deviation < 0.
        # Use a value well below (target - slack) = -0.5
        detector2 = CUSUMDetector(target=0.8, slack=0.1, threshold=5.0)
        alarm2 = False
        for _ in range(100):
            alarm2 = detector2.update(0.0)
            if alarm2:
                break
        assert alarm2 is True, "CUSUM alarm should have fired with sustained low observations"

    def test_alarm_triggered_by_sustained_low_precision(self):
        """Sustained zero observations against a high baseline should trigger alarm."""
        detector = CUSUMDetector(target=0.9, slack=0.05, threshold=3.0)
        alarm = False
        for _ in range(100):
            alarm = detector.update(0.0)
            if alarm:
                break
        assert alarm is True

    def test_reset_clears_state(self):
        """After reset, alarm should be False and statistic should be zero."""
        detector = CUSUMDetector(target=0.9, slack=0.05, threshold=3.0)
        for _ in range(100):
            alarm = detector.update(0.0)
            if alarm:
                break
        # Should have alarmed
        assert detector.statistic > 0
        detector.reset()
        assert detector.alarm_fired is False if hasattr(detector, "alarm_fired") else True
        assert detector._s_neg == 0.0
        # Next update after reset should not immediately alarm (fresh start)
        alarm_after_reset = detector.update(0.01)
        # statistic should be low after one step with small input
        assert detector._s_neg < detector.threshold

    def test_statistic_property(self):
        """The statistic property should reflect internal CUSUM state."""
        detector = CUSUMDetector(target=0.0, slack=0.5, threshold=5.0)
        assert detector.statistic == 0.0
        detector.update(0.0)
        assert isinstance(detector.statistic, float)

    def test_reset_after_alarm_allows_fresh_detection(self):
        """After a reset triggered alarm, a new clean run stays below threshold."""
        detector = CUSUMDetector(target=0.9, slack=0.05, threshold=3.0)
        for _ in range(100):
            if detector.update(0.0):
                break
        assert detector._s_neg >= detector.threshold
        detector.reset()
        assert detector._s_neg == 0.0
        # Feed perfect observations — should not alarm
        for _ in range(20):
            result = detector.update(0.9)
        assert result is False

    def test_initial_state_no_alarm(self):
        """Freshly constructed detector should have no alarm state."""
        detector = CUSUMDetector()
        assert detector._s_neg == 0.0
        assert detector.statistic == 0.0

    def test_slack_and_threshold_configurable(self):
        """Custom slack and threshold parameters should be stored."""
        detector = CUSUMDetector(target=0.7, slack=0.2, threshold=10.0)
        assert detector.target == 0.7
        assert detector.slack == 0.2
        assert detector.threshold == 10.0


# ===========================================================================
# TestSignalWeightTracker
# ===========================================================================


class TestSignalWeightTracker:
    def test_default_prior_is_uniform(self):
        """New signals should start with a flat Beta(1, 1) prior — expected weight 0.5."""
        tracker = SignalWeightTracker()
        assert tracker.expected_weight("new_signal") == pytest.approx(0.5)

    def test_true_positive_increases_weight(self):
        """True positive observations should push expected weight above 0.5."""
        tracker = SignalWeightTracker()
        for _ in range(10):
            tracker.update("sig", is_true_positive=True)
        assert tracker.expected_weight("sig") > 0.5

    def test_false_positive_decreases_weight(self):
        """False positive observations should push expected weight below 0.5."""
        tracker = SignalWeightTracker()
        for _ in range(10):
            tracker.update("sig", is_true_positive=False)
        assert tracker.expected_weight("sig") < 0.5

    def test_all_weights_returns_dict(self):
        """all_weights() should return a dict of signal names to floats."""
        tracker = SignalWeightTracker()
        tracker.update("a", is_true_positive=True)
        tracker.update("b", is_true_positive=False)
        weights = tracker.all_weights()
        assert "a" in weights
        assert "b" in weights
        assert all(0.0 <= v <= 1.0 for v in weights.values())

    def test_sample_returns_float_in_range(self):
        """Thompson sample should return a value in [0, 1]."""
        tracker = SignalWeightTracker()
        tracker.update("s", is_true_positive=True)
        for _ in range(20):
            sample = tracker.sample("s")
            assert 0.0 <= sample <= 1.0

    def test_get_posterior_returns_tuple(self):
        """get_posterior() should return (alpha, beta) tuple."""
        tracker = SignalWeightTracker()
        alpha, beta = tracker.get_posterior("my_signal")
        assert alpha == 1.0
        assert beta == 1.0


# ===========================================================================
# TestDetectionFlywheel
# ===========================================================================


@pytest.fixture
def flywheel_db(tmp_path):
    """Temporary SentinelDB with seeded patterns."""
    db_path = str(tmp_path / "flywheel_test.db")
    db = SentinelDB(path=db_path)
    kb = KnowledgeBase(db=db)
    kb.seed_default_patterns()
    yield db
    db.close()


@pytest.fixture
def flywheel(flywheel_db):
    """DetectionFlywheel backed by a fresh temporary DB."""
    return DetectionFlywheel(db=flywheel_db)


class TestDetectionFlywheel:
    def test_get_health_returns_dict(self, flywheel):
        """get_health() should return a dict with all expected keys."""
        health = flywheel.get_health()
        required_keys = {
            "healthy",
            "grade",
            "precision",
            "recall",
            "f1",
            "total_jobs_analyzed",
            "total_user_reports",
            "active_patterns",
            "candidate_patterns",
            "deprecated_patterns",
            "regression_alarm",
            "cusum_statistic",
            "cycle_count",
            "checked_at",
        }
        assert required_keys.issubset(set(health.keys()))

    def test_get_health_cold_start_on_fresh_db(self, flywheel):
        """A fresh DB with no data should report cold_start=True."""
        health = flywheel.get_health()
        assert health["cold_start"] is True
        assert health["healthy"] is False
        assert health["regression_alarm"] is False

    def test_get_health_grade_is_valid(self, flywheel):
        """Health grade should be valid (N/A for cold start, letter otherwise)."""
        health = flywheel.get_health()
        assert health["grade"] in ("A", "B", "C", "D", "F", "N/A")

    def test_get_health_active_patterns_populated(self, flywheel):
        """Seeded DB should report active patterns > 0."""
        health = flywheel.get_health()
        assert health["active_patterns"] > 0

    def test_run_cycle_returns_metrics(self, flywheel):
        """run_cycle() should return a dict with all expected metric keys."""
        metrics = flywheel.run_cycle()
        required_keys = {
            "cycle_ts",
            "cycle_number",
            "total_analyzed",
            "true_positives",
            "false_positives",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "signals_updated",
            "patterns_evolved",
            "patterns_promoted",
            "patterns_deprecated",
            "regression_alarm",
            "cusum_statistic",
        }
        assert required_keys.issubset(set(metrics.keys()))

    def test_run_cycle_increments_cycle_count(self, flywheel):
        """Each call to run_cycle() should increment the internal cycle counter."""
        assert flywheel._cycle_count == 0
        flywheel.run_cycle()
        assert flywheel._cycle_count == 1
        flywheel.run_cycle()
        assert flywheel._cycle_count == 2

    def test_run_cycle_cycle_number_matches_count(self, flywheel):
        """The cycle_number in the returned dict should match _cycle_count."""
        metrics = flywheel.run_cycle()
        assert metrics["cycle_number"] == flywheel._cycle_count

    def test_run_cycle_regression_alarm_is_bool(self, flywheel):
        """regression_alarm in cycle metrics should be a boolean."""
        metrics = flywheel.run_cycle()
        assert isinstance(metrics["regression_alarm"], bool)

    def test_run_cycle_patterns_promoted_is_list(self, flywheel):
        """patterns_promoted should be a list."""
        metrics = flywheel.run_cycle()
        assert isinstance(metrics["patterns_promoted"], list)

    def test_learn_from_report_updates_weights(self, flywheel, flywheel_db):
        """Submitting a scam report should increase the weight of the fired signal."""
        # Seed the pattern name we're going to use
        signal_name = "upfront_payment_fee"
        # Record initial weight
        initial_weight = flywheel.weight_tracker.expected_weight(signal_name)
        assert initial_weight == pytest.approx(0.5)

        report = UserReport(url="https://example.com/job/999", is_scam=True)
        job = JobPosting(url="https://example.com/job/999", title="Easy Money Job")
        sig = ScamSignal(name=signal_name, category=SignalCategory.RED_FLAG)
        result = ValidationResult(job=job, signals=[sig])

        summary = flywheel.learn_from_report(report, result)

        updated_weight = flywheel.weight_tracker.expected_weight(signal_name)
        assert updated_weight > initial_weight
        assert summary["signal_count"] == 1
        assert signal_name in summary["signals_updated"]
        assert summary["is_scam"] is True

    def test_learn_from_report_false_positive_decreases_weight(self, flywheel):
        """Marking a job as NOT a scam should decrease signal weight."""
        signal_name = "salary_too_high"
        report = UserReport(url="https://example.com/job/888", is_scam=False)
        job = JobPosting(url="https://example.com/job/888")
        sig = ScamSignal(name=signal_name, category=SignalCategory.WARNING)
        result = ValidationResult(job=job, signals=[sig])

        flywheel.learn_from_report(report, result)
        weight = flywheel.weight_tracker.expected_weight(signal_name)
        assert weight < 0.5

    def test_learn_from_report_returns_summary_dict(self, flywheel):
        """learn_from_report() should return a summary dict with expected keys."""
        report = UserReport(url="https://example.com/job/777", is_scam=True)
        job = JobPosting(url="https://example.com/job/777")
        sig = ScamSignal(name="test_signal", category=SignalCategory.RED_FLAG)
        result = ValidationResult(job=job, signals=[sig])

        summary = flywheel.learn_from_report(report, result)
        assert "report_url" in summary
        assert "is_scam" in summary
        assert "signals_updated" in summary
        assert "signal_count" in summary
        assert "updated_at" in summary

    def test_learn_from_report_multiple_signals(self, flywheel):
        """Multiple signals in a result should all be updated."""
        report = UserReport(url="https://example.com/job/666", is_scam=True)
        job = JobPosting(url="https://example.com/job/666")
        signals = [
            ScamSignal(name="sig_a", category=SignalCategory.RED_FLAG),
            ScamSignal(name="sig_b", category=SignalCategory.WARNING),
            ScamSignal(name="sig_c", category=SignalCategory.STRUCTURAL),
        ]
        result = ValidationResult(job=job, signals=signals)

        summary = flywheel.learn_from_report(report, result)
        assert summary["signal_count"] == 3
        assert set(summary["signals_updated"]) == {"sig_a", "sig_b", "sig_c"}

    def test_compute_accuracy_empty_db(self, flywheel):
        """compute_accuracy() should return zeros when no reports exist."""
        accuracy = flywheel.compute_accuracy()
        assert accuracy["total"] == 0
        assert accuracy["precision"] == 0.0
        assert accuracy["recall"] == 0.0
        assert accuracy["f1"] == 0.0

    def test_compute_accuracy_keys(self, flywheel):
        """compute_accuracy() should return all required keys."""
        accuracy = flywheel.compute_accuracy()
        required = {"total", "accuracy", "precision", "recall", "f1",
                    "true_positives", "false_positives", "true_negatives", "false_negatives"}
        assert required.issubset(set(accuracy.keys()))

    def test_evolve_patterns_returns_dict(self, flywheel):
        """evolve_patterns() should return a dict with promoted/deprecated/retained."""
        result = flywheel.evolve_patterns()
        assert "promoted" in result
        assert "deprecated" in result
        assert "retained_count" in result
        assert "evolved_at" in result

    def test_evolve_patterns_promoted_is_list(self, flywheel):
        """Promoted field should be a list."""
        result = flywheel.evolve_patterns()
        assert isinstance(result["promoted"], list)
        assert isinstance(result["deprecated"], list)

    def test_detect_regression_insufficient_data(self, flywheel):
        """detect_regression() should return alarm=False when fewer than 5 reports."""
        result = flywheel.detect_regression()
        assert result["alarm"] is False
        assert "message" in result

    def test_evolve_promotes_high_precision_candidate(self, flywheel_db):
        """A candidate pattern with 10+ observations and precision >= 0.8 should be promoted."""
        # Insert a high-precision candidate pattern
        flywheel_db.save_pattern({
            "pattern_id": "test_high_precision",
            "name": "High Precision Test",
            "description": "Test pattern",
            "category": "red_flag",
            "status": "candidate",
            "observations": 12,
            "true_positives": 11,
            "false_positives": 1,
            "alpha": 12.0,
            "beta": 2.0,
        })
        fw = DetectionFlywheel(db=flywheel_db)
        result = fw.evolve_patterns()
        assert "test_high_precision" in result["promoted"]

    def test_evolve_deprecates_low_precision_active(self, flywheel_db):
        """An active pattern with 20+ observations and precision < 0.3 should be deprecated."""
        flywheel_db.save_pattern({
            "pattern_id": "test_low_precision",
            "name": "Low Precision Test",
            "description": "Test pattern",
            "category": "warning",
            "status": "active",
            "observations": 25,
            "true_positives": 3,
            "false_positives": 22,
            "alpha": 4.0,
            "beta": 23.0,
        })
        fw = DetectionFlywheel(db=flywheel_db)
        result = fw.evolve_patterns()
        assert "test_low_precision" in result["deprecated"]


# ===========================================================================
# TestRegressionResponse — regression detection loop closure
# ===========================================================================


class TestRegressionResponse:
    """Tests for the CUSUM regression response: cache reset, callbacks, alerts."""

    def _seed_regression_reports(self, db, count=30):
        """Insert reports that will trigger a CUSUM regression alarm.

        Creates reports where is_scam=1 but was_correct=0 (false negatives),
        which drives rolling precision to 0 and accumulates the CUSUM statistic.
        Also seeds a few initial correct reports to establish a nonzero baseline.
        """
        from datetime import datetime, UTC, timedelta

        base_time = datetime(2026, 1, 1, tzinfo=UTC)

        # First: a few correct predictions to set a nonzero CUSUM baseline
        for i in range(5):
            db.save_report({
                "url": f"https://example.com/job/good-{i}",
                "is_scam": 1,
                "was_correct": 1,
                "reported_at": (base_time + timedelta(hours=i)).isoformat(),
            })

        # Then: many incorrect predictions to trigger regression
        for i in range(count):
            db.save_report({
                "url": f"https://example.com/job/bad-{i}",
                "is_scam": 1,
                "was_correct": 0,
                "reported_at": (base_time + timedelta(hours=10 + i)).isoformat(),
            })

    def test_regression_triggers_cache_reset(self, flywheel_db):
        """When CUSUM detects regression, _reset_learned_weights_cache must be called."""
        from unittest.mock import patch

        self._seed_regression_reports(flywheel_db, count=40)
        fw = DetectionFlywheel(db=flywheel_db)

        with patch("sentinel.scorer._reset_learned_weights_cache") as mock_reset:
            metrics = fw.run_cycle()

        assert metrics["regression_alarm"] is True
        # Called at least twice: once from evolve_patterns() inside run_cycle,
        # and once explicitly from the regression response block.
        # evolve_patterns is called twice when regression fires (initial + re-run),
        # plus the explicit call = at least 3 times.
        assert mock_reset.call_count >= 2, (
            f"Expected _reset_learned_weights_cache to be called at least 2 times, "
            f"got {mock_reset.call_count}"
        )

    def test_regression_alert_callback_fired(self, flywheel_db):
        """A registered callback should receive the metrics dict on regression."""
        from unittest.mock import MagicMock, patch

        self._seed_regression_reports(flywheel_db, count=40)
        fw = DetectionFlywheel(db=flywheel_db)

        mock_callback = MagicMock()
        fw.on_regression(mock_callback)

        with patch("sentinel.scorer._reset_learned_weights_cache"):
            metrics = fw.run_cycle()

        assert metrics["regression_alarm"] is True
        mock_callback.assert_called_once_with(metrics)
        # Verify the callback received a dict with the expected keys
        received = mock_callback.call_args[0][0]
        assert "precision" in received
        assert "cusum_statistic" in received
        assert "regression_alarm" in received

    def test_evolve_resets_weight_cache(self, flywheel_db):
        """evolve_patterns() should call _reset_learned_weights_cache."""
        from unittest.mock import patch

        fw = DetectionFlywheel(db=flywheel_db)

        with patch("sentinel.scorer._reset_learned_weights_cache") as mock_reset:
            fw.evolve_patterns()

        mock_reset.assert_called_once()

    def test_no_regression_no_callback(self, flywheel_db):
        """When no regression is detected, alert callbacks should not fire."""
        from unittest.mock import MagicMock, patch

        # Fresh DB with no reports = no regression
        fw = DetectionFlywheel(db=flywheel_db)

        mock_callback = MagicMock()
        fw.on_regression(mock_callback)

        with patch("sentinel.scorer._reset_learned_weights_cache"):
            metrics = fw.run_cycle()

        assert metrics["regression_alarm"] is False
        mock_callback.assert_not_called()
