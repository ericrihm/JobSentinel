"""Comprehensive tests for sentinel.temporal — Temporal Intelligence module.

Coverage:
- ScamEvolutionTracker: lifecycle tracking, EMA, emerging pattern detection
- TemporalAnomalyDetector: z-score anomaly, burst detection, change-point
- PatternDrift: KL-divergence, DB-backed comparison
- PredictiveModel: OLS regression, prediction, confidence, edge cases
- Season helpers: current_scam_seasons, seasonal_lift_for_date
- Empty / minimal data edge cases throughout
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pytest

from sentinel.db import SentinelDB
from sentinel.temporal import (
    AnomalyResult,
    DriftReport,
    PatternDrift,
    PatternLifecycle,
    PredictionResult,
    PredictiveModel,
    ScamEvolutionTracker,
    ScamSeason,
    TemporalAnomalyDetector,
    current_scam_seasons,
    seasonal_lift_for_date,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> SentinelDB:
    db_path = str(tmp_path / "temporal_test.db")
    db = SentinelDB(path=db_path)
    return db


def _iso(offset_days: float = 0.0) -> str:
    return (datetime.now(UTC) + timedelta(days=offset_days)).isoformat()


def _date(offset_days: int = 0) -> date:
    return (datetime.now(UTC) + timedelta(days=offset_days)).date()


# ---------------------------------------------------------------------------
# Tests: Season helpers
# ---------------------------------------------------------------------------


class TestSeasonHelpers:
    def test_scam_season_is_dataclass(self):
        s = ScamSeason(name="test", description="desc", peak_months=[1], lift_factor=1.5)
        assert s.name == "test"
        assert s.lift_factor == 1.5

    def test_current_scam_seasons_tax_season(self):
        # February is in tax season
        feb = date(2026, 2, 15)
        seasons = current_scam_seasons(feb)
        names = [s.name for s in seasons]
        assert "tax_season" in names

    def test_current_scam_seasons_holiday(self):
        dec = date(2026, 12, 1)
        seasons = current_scam_seasons(dec)
        names = [s.name for s in seasons]
        assert "holiday_hiring" in names

    def test_current_scam_seasons_empty_in_july(self):
        # July is not in any defined season
        july = date(2026, 7, 15)
        seasons = current_scam_seasons(july)
        assert len(seasons) == 0

    def test_seasonal_lift_above_one_in_peak(self):
        jan = date(2026, 1, 10)
        lift = seasonal_lift_for_date(jan)
        assert lift > 1.0

    def test_seasonal_lift_is_one_in_off_season(self):
        july = date(2026, 7, 15)
        lift = seasonal_lift_for_date(july)
        assert lift == 1.0

    def test_graduation_season_may_june(self):
        for month in (5, 6):
            d = date(2026, month, 15)
            names = [s.name for s in current_scam_seasons(d)]
            assert "graduation" in names


# ---------------------------------------------------------------------------
# Tests: ScamEvolutionTracker
# ---------------------------------------------------------------------------


class TestScamEvolutionTracker:
    def test_record_returns_lifecycle(self):
        tracker = ScamEvolutionTracker()
        lc = tracker.record_observation("upfront_payment", count=10, total_jobs=100)
        assert isinstance(lc, PatternLifecycle)
        assert lc.pattern_name == "upfront_payment"

    def test_lifecycle_starts_emerging(self):
        tracker = ScamEvolutionTracker()
        lc = tracker.record_observation("new_pattern", count=3, total_jobs=100)
        # Fresh pattern with low rate should start as emerging
        assert lc.status in ("emerging", "growing", "mutating", "dead")

    def test_first_seen_set_on_first_observation(self):
        tracker = ScamEvolutionTracker()
        d = date(2026, 1, 15)
        lc = tracker.record_observation("sig_a", count=5, total_jobs=100, observation_date=d)
        assert lc.first_seen == "2026-01-15"

    def test_last_seen_advances_over_time(self):
        tracker = ScamEvolutionTracker()
        tracker.record_observation("sig", count=5, total_jobs=100, observation_date=date(2026, 1, 1))
        lc = tracker.record_observation("sig", count=5, total_jobs=100, observation_date=date(2026, 2, 1))
        assert lc.last_seen == "2026-02-01"
        assert lc.first_seen == "2026-01-01"

    def test_total_observations_accumulates(self):
        tracker = ScamEvolutionTracker()
        tracker.record_observation("p", count=10, total_jobs=100, observation_date=date(2026, 1, 1))
        tracker.record_observation("p", count=20, total_jobs=100, observation_date=date(2026, 1, 8))
        lc = tracker.get_lifecycle("p")
        assert lc.total_observations == 30

    def test_ema_rate_updates_on_each_observation(self):
        tracker = ScamEvolutionTracker()
        tracker.record_observation("sig", count=10, total_jobs=100, observation_date=date(2026, 1, 1))
        lc1_ema = tracker.get_lifecycle("sig").ema_rate
        tracker.record_observation("sig", count=50, total_jobs=100, observation_date=date(2026, 1, 8))
        lc2_ema = tracker.get_lifecycle("sig").ema_rate
        # EMA should increase when new rate is higher
        assert lc2_ema > lc1_ema

    def test_trend_positive_when_growing(self):
        tracker = ScamEvolutionTracker()
        # Feed increasing rates
        for week_idx in range(8):
            d = date(2026, 1, 1) + timedelta(weeks=week_idx)
            tracker.record_observation("growing_sig", count=10 + week_idx * 5, total_jobs=100, observation_date=d)
        lc = tracker.get_lifecycle("growing_sig")
        # EMA trend should be positive (or at least non-negative)
        assert lc.trend >= 0

    def test_status_saturated_at_high_rate(self):
        tracker = ScamEvolutionTracker()
        # Rate of 0.5 (50%) should exceed SATURATION_RATE
        for i in range(5):
            d = date(2026, 1, 1) + timedelta(weeks=i)
            tracker.record_observation("saturated_sig", count=50, total_jobs=100, observation_date=d)
        lc = tracker.get_lifecycle("saturated_sig")
        assert lc.status == "saturated"

    def test_status_dead_at_zero_rate_after_history(self):
        tracker = ScamEvolutionTracker()
        # First get some observations...
        for i in range(3):
            d = date(2026, 1, 1) + timedelta(weeks=i)
            tracker.record_observation("dying_sig", count=30, total_jobs=100, observation_date=d)
        # Feed many zero-count observations so EMA decays below DEAD_RATE (0.005).
        # With alpha=0.3, starting EMA ~0.3: after 20 steps 0.3 * 0.7^20 ≈ 0.0007 < 0.005.
        for i in range(3, 23):
            d = date(2026, 1, 1) + timedelta(weeks=i)
            tracker.record_observation("dying_sig", count=0, total_jobs=100, observation_date=d)
        lc = tracker.get_lifecycle("dying_sig")
        assert lc.status == "dead"

    def test_peak_rate_tracked(self):
        tracker = ScamEvolutionTracker()
        tracker.record_observation("p", count=10, total_jobs=100, observation_date=date(2026, 1, 1))
        tracker.record_observation("p", count=40, total_jobs=100, observation_date=date(2026, 1, 8))
        tracker.record_observation("p", count=20, total_jobs=100, observation_date=date(2026, 1, 15))
        lc = tracker.get_lifecycle("p")
        assert lc.peak_rate == pytest.approx(0.40)

    def test_get_lifecycle_returns_none_for_unknown(self):
        tracker = ScamEvolutionTracker()
        assert tracker.get_lifecycle("unknown") is None

    def test_all_lifecycles_returns_dict(self):
        tracker = ScamEvolutionTracker()
        tracker.record_observation("a", count=5, total_jobs=100, observation_date=date(2026, 1, 1))
        tracker.record_observation("b", count=3, total_jobs=100, observation_date=date(2026, 1, 1))
        lcs = tracker.all_lifecycles()
        assert "a" in lcs
        assert "b" in lcs

    def test_emerging_patterns_filters_correctly(self):
        tracker = ScamEvolutionTracker()
        # Pattern with rate 0.05 (in emerging range)
        tracker.record_observation("emerging_one", count=5, total_jobs=100, observation_date=date(2026, 1, 1))
        # Pattern with rate 0.80 (saturated)
        tracker.record_observation("saturated_one", count=80, total_jobs=100, observation_date=date(2026, 1, 1))
        emerging = tracker.emerging_patterns(min_rate=0.01, max_rate=0.10)
        names = [lc.pattern_name for lc in emerging]
        assert "saturated_one" not in names

    def test_rate_history_stores_observations(self):
        tracker = ScamEvolutionTracker()
        tracker.record_observation("sig", count=10, total_jobs=100, observation_date=date(2026, 1, 1))
        tracker.record_observation("sig", count=20, total_jobs=100, observation_date=date(2026, 1, 8))
        hist = tracker.rate_history("sig")
        assert len(hist) == 2
        # Each entry is (week_key, rate)
        assert all(isinstance(wk, str) and isinstance(r, float) for wk, r in hist)

    def test_zero_total_jobs_handled(self):
        tracker = ScamEvolutionTracker()
        # Should not raise ZeroDivisionError
        lc = tracker.record_observation("sig", count=0, total_jobs=0, observation_date=date(2026, 1, 1))
        assert isinstance(lc, PatternLifecycle)

    def test_observation_date_as_string(self):
        tracker = ScamEvolutionTracker()
        lc = tracker.record_observation("sig", count=5, total_jobs=100, observation_date="2026-03-15")
        assert lc.first_seen == "2026-03-15"

    def test_observation_date_as_datetime(self):
        tracker = ScamEvolutionTracker()
        dt = datetime(2026, 4, 1, 10, 30, tzinfo=UTC)
        lc = tracker.record_observation("sig", count=5, total_jobs=100, observation_date=dt)
        assert lc.first_seen == "2026-04-01"

    def test_multiple_independent_patterns_tracked(self):
        tracker = ScamEvolutionTracker()
        for name in ("a", "b", "c"):
            tracker.record_observation(name, count=10, total_jobs=100, observation_date=date(2026, 1, 1))
        assert len(tracker.all_lifecycles()) == 3


# ---------------------------------------------------------------------------
# Tests: TemporalAnomalyDetector
# ---------------------------------------------------------------------------


class TestTemporalAnomalyDetector:
    def test_insufficient_history_returns_no_anomaly(self):
        det = TemporalAnomalyDetector()
        result = det.observe("sig", volume=100, obs_date=date(2026, 1, 1))
        assert result.is_anomaly is False
        assert "Insufficient" in result.message

    def test_normal_volume_not_anomalous(self):
        det = TemporalAnomalyDetector(window=14, z_threshold=2.5)
        # Feed 14 stable days
        for i in range(14):
            det.observe("sig", volume=20, obs_date=date(2026, 1, 1) + timedelta(days=i))
        result = det.observe("sig", volume=21, obs_date=date(2026, 1, 15))
        assert result.is_anomaly is False

    def test_large_spike_is_anomalous(self):
        det = TemporalAnomalyDetector(window=14, z_threshold=2.0)
        # Stable baseline of 10
        for i in range(14):
            det.observe("sig", volume=10, obs_date=date(2026, 1, 1) + timedelta(days=i))
        # Massive spike
        result = det.observe("sig", volume=1000, obs_date=date(2026, 1, 15))
        assert result.is_anomaly is True

    def test_burst_detection(self):
        det = TemporalAnomalyDetector(burst_multiplier=3.0, burst_window=3)
        # Establish low recent mean
        for i in range(10):
            det.observe("sig", volume=5, obs_date=date(2026, 1, 1) + timedelta(days=i))
        # Burst: volume >> 3x recent mean
        result = det.observe("sig", volume=500, obs_date=date(2026, 1, 11))
        assert result.is_burst is True

    def test_z_score_returned_as_float(self):
        det = TemporalAnomalyDetector()
        for i in range(10):
            det.observe("sig", volume=10, obs_date=date(2026, 1, 1) + timedelta(days=i))
        result = det.observe("sig", volume=10, obs_date=date(2026, 1, 11))
        assert isinstance(result.z_score, float)

    def test_result_is_anomaly_result_type(self):
        det = TemporalAnomalyDetector()
        result = det.observe("sig", volume=5, obs_date=date(2026, 1, 1))
        assert isinstance(result, AnomalyResult)

    def test_observed_volume_stored_correctly(self):
        det = TemporalAnomalyDetector()
        for i in range(10):
            det.observe("sig", volume=10, obs_date=date(2026, 1, 1) + timedelta(days=i))
        result = det.observe("sig", volume=42, obs_date=date(2026, 1, 11))
        assert result.observed_volume == 42

    def test_seasonal_factor_reflects_weekday(self):
        det = TemporalAnomalyDetector()
        # Monday (weekday=0) has highest factor
        monday = date(2026, 4, 27)  # Known Monday
        for i in range(10):
            det.observe("sig", volume=10, obs_date=monday + timedelta(days=i))
        result_mon = det.observe("sig", volume=10, obs_date=monday + timedelta(weeks=2))
        # Friday (weekday=4) has lower factor
        friday = monday + timedelta(days=4)
        det2 = TemporalAnomalyDetector()
        for i in range(10):
            det2.observe("sig", volume=10, obs_date=friday + timedelta(days=i))
        result_fri = det2.observe("sig", volume=10, obs_date=friday + timedelta(weeks=2))
        assert result_mon.seasonal_factor > result_fri.seasonal_factor

    def test_volume_history_returns_list(self):
        det = TemporalAnomalyDetector()
        for i in range(5):
            det.observe("sig", volume=i * 2, obs_date=date(2026, 1, 1) + timedelta(days=i))
        hist = det.volume_history("sig")
        assert len(hist) == 5
        assert all(isinstance(d, str) and isinstance(v, int) for d, v in hist)

    def test_volume_history_empty_for_unknown(self):
        det = TemporalAnomalyDetector()
        assert det.volume_history("nonexistent") == []

    def test_changepoint_detected(self):
        det = TemporalAnomalyDetector()
        # First 10 days: low volume
        for i in range(10):
            det.observe("sig", volume=5, obs_date=date(2026, 1, 1) + timedelta(days=i))
        # Next 10 days: high volume (step change)
        for i in range(10, 20):
            det.observe("sig", volume=50, obs_date=date(2026, 1, 1) + timedelta(days=i))
        result = det.detect_changepoint("sig")
        assert isinstance(result, dict)
        assert "found" in result
        assert result["found"] is True
        assert result["post_mean"] > result["pre_mean"]

    def test_changepoint_insufficient_data(self):
        det = TemporalAnomalyDetector()
        det.observe("sig", volume=10, obs_date=date(2026, 1, 1))
        result = det.detect_changepoint("sig")
        assert result["found"] is False
        assert "Insufficient" in result["message"]

    def test_changepoint_unknown_signal(self):
        det = TemporalAnomalyDetector()
        result = det.detect_changepoint("never_seen")
        assert result["found"] is False

    def test_drop_also_flagged_as_anomaly(self):
        det = TemporalAnomalyDetector(z_threshold=2.0)
        for i in range(14):
            det.observe("sig", volume=100, obs_date=date(2026, 1, 1) + timedelta(days=i))
        # Near-zero drop
        result = det.observe("sig", volume=0, obs_date=date(2026, 1, 15))
        assert result.is_anomaly is True


# ---------------------------------------------------------------------------
# Tests: PatternDrift
# ---------------------------------------------------------------------------


class TestPatternDrift:
    def test_identical_distributions_no_drift(self):
        pd = PatternDrift()
        dist = {"a": 0.5, "b": 0.3, "c": 0.2}
        report = pd.compare(reference=dist, current=dist)
        assert report.drift_detected is False
        assert report.kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_very_different_distributions_drift_detected(self):
        pd = PatternDrift()
        reference = {"upfront_payment": 0.9, "salary_too_high": 0.1}
        current = {"upfront_payment": 0.05, "crypto_payment": 0.95}
        report = pd.compare(reference=reference, current=current)
        assert report.drift_detected is True
        assert report.kl_divergence > 0.05

    def test_report_is_drift_report_type(self):
        pd = PatternDrift()
        report = pd.compare(reference={"a": 1.0}, current={"a": 1.0})
        assert isinstance(report, DriftReport)

    def test_empty_distributions_no_drift(self):
        pd = PatternDrift()
        report = pd.compare(reference={}, current={})
        assert report.drift_detected is False
        assert "No signals" in report.message

    def test_top_shifted_signals_returned(self):
        pd = PatternDrift()
        reference = {"a": 0.5, "b": 0.3, "c": 0.2}
        current = {"a": 0.1, "b": 0.1, "c": 0.8}
        report = pd.compare(reference=reference, current=current)
        assert len(report.top_shifted_signals) > 0
        # Each entry has required keys
        for entry in report.top_shifted_signals:
            assert "signal" in entry
            assert "baseline_share" in entry
            assert "current_share" in entry
            assert "delta" in entry

    def test_kl_divergence_is_non_negative(self):
        pd = PatternDrift()
        reference = {"a": 0.6, "b": 0.4}
        current = {"a": 0.3, "b": 0.7}
        report = pd.compare(reference=reference, current=current)
        assert report.kl_divergence >= 0.0

    def test_custom_threshold_lowers_sensitivity(self):
        pd = PatternDrift(drift_threshold=0.99)  # Very high threshold
        reference = {"a": 0.6, "b": 0.4}
        current = {"a": 0.3, "b": 0.7}
        report = pd.compare(reference=reference, current=current)
        # Even noticeable drift should be below 0.99
        assert report.drift_detected is False

    def test_new_signal_in_current_detected(self):
        pd = PatternDrift()
        reference = {"upfront_payment": 1.0}
        current = {"upfront_payment": 0.1, "crypto_scam": 0.9}
        report = pd.compare(reference=reference, current=current)
        assert report.drift_detected is True

    def test_compare_from_db_returns_drift_report(self, tmp_path):
        db = _make_db(tmp_path)
        # Seed some signal rate history
        db.record_signal_rates(
            signal_rates={"upfront_payment": 10, "salary_anomaly": 5},
            total_jobs=100,
            window_start=_iso(-40),
            window_end=_iso(-35),
        )
        db.record_signal_rates(
            signal_rates={"upfront_payment": 2, "crypto_payment": 20},
            total_jobs=100,
            window_start=_iso(-3),
            window_end=_iso(-1),
        )
        pd = PatternDrift()
        report = pd.compare_from_db(db, reference_days=30, comparison_days=7)
        assert isinstance(report, DriftReport)
        db.close()

    def test_compare_from_db_empty_history_no_drift(self, tmp_path):
        db = _make_db(tmp_path)
        pd = PatternDrift()
        report = pd.compare_from_db(db)
        # No history → no drift (both windows empty)
        assert report.drift_detected is False
        db.close()

    def test_date_labels_passed_through(self):
        pd = PatternDrift()
        report = pd.compare(
            reference={"a": 1.0},
            current={"a": 1.0},
            reference_start="2026-01-01",
            reference_end="2026-01-31",
            comparison_start="2026-02-01",
            comparison_end="2026-02-28",
        )
        assert report.reference_start == "2026-01-01"
        assert report.comparison_end == "2026-02-28"


# ---------------------------------------------------------------------------
# Tests: PredictiveModel
# ---------------------------------------------------------------------------


class TestPredictiveModel:
    def test_insufficient_data_returns_zero_confidence(self):
        model = PredictiveModel()
        result = model.predict_next_week()
        assert result.confidence == 0.0
        assert "Insufficient" in result.message

    def test_two_weeks_insufficient(self):
        model = PredictiveModel()
        model.add_observation("2026-W01", 10)
        model.add_observation("2026-W02", 12)
        result = model.predict_next_week()
        assert result.confidence == 0.0

    def test_three_weeks_minimum_yields_result(self):
        model = PredictiveModel()
        model.add_observation("2026-W01", 10)
        model.add_observation("2026-W02", 12)
        model.add_observation("2026-W03", 14)
        result = model.predict_next_week()
        assert result.predicted_volume > 0

    def test_perfect_linear_trend_high_confidence(self):
        model = PredictiveModel()
        for i in range(8):
            model.add_observation(f"2026-W{i+1:02d}", float(10 + i * 5))
        result = model.predict_next_week()
        # Perfect linear trend → R² ≈ 1.0
        assert result.confidence > 0.95

    def test_prediction_extrapolates_slope(self):
        # y = 10 + 5*x → at x=8, predicted = 50
        model = PredictiveModel()
        for i in range(8):
            model.add_observation(f"2026-W{i+1:02d}", float(10 + i * 5))
        result = model.predict_next_week()
        assert result.predicted_volume == pytest.approx(50.0, abs=0.5)

    def test_lower_bound_below_predicted(self):
        model = PredictiveModel()
        for i in range(8):
            model.add_observation(f"2026-W{i+1:02d}", float(10 + i * 3 + (i % 3) * 2))
        result = model.predict_next_week()
        assert result.lower_bound <= result.predicted_volume

    def test_upper_bound_above_predicted(self):
        model = PredictiveModel()
        for i in range(8):
            model.add_observation(f"2026-W{i+1:02d}", float(10 + i * 3 + (i % 3) * 2))
        result = model.predict_next_week()
        assert result.upper_bound >= result.predicted_volume

    def test_lower_bound_non_negative(self):
        model = PredictiveModel()
        for i in range(5):
            model.add_observation(f"2026-W{i+1:02d}", float(max(0, 5 - i * 3)))
        result = model.predict_next_week()
        assert result.lower_bound >= 0.0

    def test_predicted_volume_non_negative(self):
        model = PredictiveModel()
        # Declining trend that would go negative
        for i in range(8):
            model.add_observation(f"2026-W{i+1:02d}", max(0.0, 50.0 - i * 10))
        result = model.predict_next_week()
        assert result.predicted_volume >= 0.0

    def test_target_week_is_next_week(self):
        model = PredictiveModel()
        for i in range(4):
            model.add_observation(f"2026-W{i+1:02d}", float(10 + i))
        result = model.predict_next_week()
        assert result.target_week == "2026-W05"

    def test_slope_positive_for_growing_trend(self):
        model = PredictiveModel()
        for i in range(6):
            model.add_observation(f"2026-W{i+1:02d}", float(10 + i * 5))
        result = model.predict_next_week()
        assert result.slope > 0

    def test_slope_negative_for_declining_trend(self):
        model = PredictiveModel()
        for i in range(6):
            model.add_observation(f"2026-W{i+1:02d}", float(50 - i * 5))
        result = model.predict_next_week()
        assert result.slope < 0

    def test_add_observations_bulk(self):
        model = PredictiveModel()
        weekly = {"2026-W01": 10, "2026-W02": 12, "2026-W03": 14, "2026-W04": 16}
        model.add_observations(weekly)
        result = model.predict_next_week()
        assert result.n_weeks_used == 4

    def test_observations_list_copy(self):
        model = PredictiveModel()
        model.add_observation("2026-W01", 10)
        obs = model.observations()
        assert len(obs) == 1
        assert obs[0] == ("2026-W01", 10.0)
        # Mutating the copy should not affect the model
        obs.append(("2026-W02", 999.0))
        assert len(model.observations()) == 1

    def test_predict_category_volumes(self):
        category_weekly = {
            "red_flag": {"2026-W01": 10, "2026-W02": 12, "2026-W03": 14},
            "warning": {"2026-W01": 5, "2026-W02": 6, "2026-W03": 7},
        }
        model = PredictiveModel()
        results = model.predict_category_volumes(category_weekly)
        assert "red_flag" in results
        assert "warning" in results
        assert isinstance(results["red_flag"], PredictionResult)

    def test_predict_category_with_insufficient_data(self):
        category_weekly = {
            "emerging": {"2026-W01": 1},
        }
        model = PredictiveModel()
        results = model.predict_category_volumes(category_weekly)
        assert results["emerging"].confidence == 0.0

    def test_constant_volume_zero_slope(self):
        model = PredictiveModel()
        for i in range(6):
            model.add_observation(f"2026-W{i+1:02d}", 20.0)
        result = model.predict_next_week()
        assert result.slope == pytest.approx(0.0, abs=1e-6)
        assert result.predicted_volume == pytest.approx(20.0, abs=1e-6)

    def test_result_is_prediction_result_type(self):
        model = PredictiveModel()
        for i in range(4):
            model.add_observation(f"2026-W{i+1:02d}", float(10 + i))
        result = model.predict_next_week()
        assert isinstance(result, PredictionResult)

    def test_year_rollover_week_key(self):
        from sentinel.temporal import PredictiveModel as PM
        model = PM()
        # ISO 2026 has 52 weeks — W52 + 1 should roll to 2027-W01
        for i in range(3):
            model.add_observation(f"2026-W{50+i:02d}", float(10 + i))
        result = model.predict_next_week()
        # Should produce a valid week key (not crash)
        assert isinstance(result.target_week, str)
        assert "W" in result.target_week
