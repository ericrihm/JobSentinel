"""Tests for input data drift detection, signal rate tracking, and trend/health CLI."""

from __future__ import annotations

import math
import os
import tempfile
from datetime import UTC, datetime, timedelta

import pytest
from click.testing import CliRunner

from sentinel.cli import main
from sentinel.db import SentinelDB
from sentinel.flywheel import DetectionFlywheel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(offset_days: float = 0.0) -> str:
    """Return ISO timestamp *offset_days* days from now (negative = in the past)."""
    return (datetime.now(UTC) + timedelta(days=offset_days)).isoformat()


def _seed_history(
    db: SentinelDB,
    signal_rates: dict[str, int],
    total_jobs: int,
    days_ago_start: float,
    days_ago_end: float,
) -> None:
    """Insert a synthetic signal_rate_history entry."""
    db.record_signal_rates(
        signal_rates=signal_rates,
        total_jobs=total_jobs,
        window_start=_iso(-days_ago_start),
        window_end=_iso(-days_ago_end),
    )


# ---------------------------------------------------------------------------
# Part 1: signal_rate_history DB methods
# ---------------------------------------------------------------------------


class TestSignalRateHistory:
    def test_record_and_retrieve_single_entry(self, temp_db):
        """Inserting a signal rate entry should be retrievable."""
        temp_db.record_signal_rates(
            signal_rates={"upfront_payment": 5, "urgency_language": 3},
            total_jobs=20,
            window_start=_iso(-2),
            window_end=_iso(-1),
        )
        rows = temp_db.get_signal_rate_history()
        assert len(rows) == 2
        names = {r["signal_name"] for r in rows}
        assert "upfront_payment" in names
        assert "urgency_language" in names

    def test_fire_counts_stored_correctly(self, temp_db):
        """Fire counts and total_jobs should match what was inserted."""
        temp_db.record_signal_rates(
            signal_rates={"crypto_payment": 7},
            total_jobs=100,
            window_start=_iso(-3),
            window_end=_iso(-2),
        )
        rows = temp_db.get_signal_rate_history(signal_name="crypto_payment")
        assert len(rows) == 1
        assert rows[0]["fire_count"] == 7
        assert rows[0]["total_jobs"] == 100

    def test_filter_by_signal_name(self, temp_db):
        """get_signal_rate_history with signal_name should filter correctly."""
        temp_db.record_signal_rates(
            signal_rates={"a": 1, "b": 2, "c": 3},
            total_jobs=50,
            window_start=_iso(-5),
            window_end=_iso(-4),
        )
        rows_a = temp_db.get_signal_rate_history(signal_name="a")
        rows_b = temp_db.get_signal_rate_history(signal_name="b")
        assert len(rows_a) == 1
        assert rows_a[0]["fire_count"] == 1
        assert len(rows_b) == 1
        assert rows_b[0]["fire_count"] == 2

    def test_multiple_windows_accumulated(self, temp_db):
        """Multiple inserts for the same signal should accumulate as separate rows."""
        for i in range(3):
            temp_db.record_signal_rates(
                signal_rates={"upfront_payment": 4},
                total_jobs=10,
                window_start=_iso(-(i + 2)),
                window_end=_iso(-(i + 1)),
            )
        rows = temp_db.get_signal_rate_history(signal_name="upfront_payment")
        assert len(rows) == 3
        assert all(r["fire_count"] == 4 for r in rows)

    def test_empty_history_returns_empty_list(self, temp_db):
        """No entries in DB should return empty list."""
        rows = temp_db.get_signal_rate_history()
        assert rows == []

    def test_limit_parameter_respected(self, temp_db):
        """The limit parameter should cap returned rows."""
        for i in range(10):
            temp_db.record_signal_rates(
                signal_rates={"x": i},
                total_jobs=10,
                window_start=_iso(-(i + 2)),
                window_end=_iso(-(i + 1)),
            )
        rows = temp_db.get_signal_rate_history(limit=5)
        assert len(rows) == 5


# ---------------------------------------------------------------------------
# Part 2: detect_input_drift
# ---------------------------------------------------------------------------


class TestDetectInputDrift:
    def test_no_data_returns_no_alarm(self, temp_db):
        """Empty history should return alarm=False with a descriptive message."""
        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        assert result["alarm"] is False
        assert result["drift_score"] == 0.0
        assert "No signal rate history" in result["message"]

    def test_only_recent_no_baseline(self, temp_db):
        """Data only in recent window → cannot compute drift → alarm=False."""
        _seed_history(temp_db, {"upfront_payment": 5}, 20, 1, 0)
        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        assert result["alarm"] is False
        assert "baseline" in result["message"].lower()

    def test_only_baseline_no_recent(self, temp_db):
        """Data only in old baseline, nothing recent → alarm=False."""
        _seed_history(temp_db, {"upfront_payment": 5}, 20, 30, 25)
        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        assert result["alarm"] is False

    def test_no_drift_identical_distributions(self, temp_db):
        """Identical recent and baseline distributions should produce near-zero drift."""
        for _ in range(5):
            _seed_history(temp_db, {"upfront_payment": 10, "urgency_language": 5}, 100, 20, 15)
        _seed_history(temp_db, {"upfront_payment": 10, "urgency_language": 5}, 100, 3, 1)

        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        assert result["alarm"] is False
        assert result["drift_score"] < 0.10

    def test_significant_drift_triggers_alarm(self, temp_db):
        """A large distribution shift should trigger the drift alarm."""
        # Baseline: upfront_payment fires a lot, urgency_language rarely
        for _ in range(5):
            _seed_history(
                temp_db,
                {"upfront_payment": 50, "urgency_language": 2},
                100, 30, 25,
            )
        # Recent: completely different — urgency_language dominates
        _seed_history(
            temp_db,
            {"upfront_payment": 2, "urgency_language": 50},
            100, 3, 1,
        )

        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        assert result["alarm"] is True
        assert result["drift_score"] > 0.10

    def test_changed_signals_sorted_by_abs_delta(self, temp_db):
        """changed_signals should be sorted by absolute delta descending."""
        for _ in range(3):
            _seed_history(
                temp_db,
                {"signal_a": 10, "signal_b": 1, "signal_c": 5},
                100, 20, 15,
            )
        # Recent: signal_a drops sharply, signal_b unchanged, signal_c rises
        _seed_history(
            temp_db,
            {"signal_a": 1, "signal_b": 1, "signal_c": 20},
            100, 2, 1,
        )

        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        deltas = [abs(c["delta"]) for c in result["changed_signals"]]
        assert deltas == sorted(deltas, reverse=True), "changed_signals not sorted by |delta|"

    def test_drift_result_contains_required_keys(self, temp_db):
        """Result dict must contain all documented keys."""
        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        required = {
            "drift_score", "alarm", "changed_signals", "recent_window_start",
            "baseline_jobs", "recent_jobs", "chi2_statistic", "message",
        }
        assert required.issubset(set(result.keys()))

    def test_kl_divergence_zero_for_identical_single_signal(self, temp_db):
        """KL divergence of identical distributions should be near zero."""
        for _ in range(4):
            _seed_history(temp_db, {"signal_x": 10}, 100, 20, 15)
        _seed_history(temp_db, {"signal_x": 10}, 100, 3, 1)

        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        assert result["drift_score"] < 0.001

    def test_chi2_statistic_non_negative(self, temp_db):
        """Chi-squared statistic must always be >= 0."""
        for _ in range(3):
            _seed_history(temp_db, {"s1": 5, "s2": 3}, 50, 15, 10)
        _seed_history(temp_db, {"s1": 8, "s2": 1}, 50, 2, 1)

        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        assert result["chi2_statistic"] >= 0.0

    def test_record_signal_rates_helper(self, temp_db):
        """DetectionFlywheel.record_signal_rates should persist to DB."""
        fw = DetectionFlywheel(db=temp_db)
        fw.record_signal_rates(
            signal_counts={"upfront_payment": 3, "crypto_payment": 1},
            total_jobs=15,
            window_start=_iso(-2),
            window_end=_iso(-1),
        )
        rows = temp_db.get_signal_rate_history()
        assert len(rows) == 2
        names = {r["signal_name"] for r in rows}
        assert "upfront_payment" in names

    def test_single_signal_edge_case(self, temp_db):
        """Drift detection with only one signal in both windows should not crash."""
        for _ in range(3):
            _seed_history(temp_db, {"lone_signal": 8}, 50, 20, 15)
        _seed_history(temp_db, {"lone_signal": 2}, 50, 2, 1)

        fw = DetectionFlywheel(db=temp_db)
        result = fw.detect_input_drift(window_days=7)
        # Should not raise; result should be valid
        assert isinstance(result["drift_score"], float)
        assert isinstance(result["alarm"], bool)

    def test_drift_score_increases_with_larger_shift(self):
        """Larger distribution shifts should produce higher drift scores.

        Uses multi-signal distributions so the JSD has meaningful shape to compare.
        """
        with tempfile.TemporaryDirectory() as td:
            db_small = SentinelDB(path=os.path.join(td, "small.db"))
            db_large = SentinelDB(path=os.path.join(td, "large.db"))

            # Both baselines: signal_a fires often, signal_b rarely
            baseline = {"signal_a": 80, "signal_b": 20}
            for _ in range(4):
                _seed_history(db_small, baseline, 100, 25, 20)
                _seed_history(db_large, baseline, 100, 25, 20)

            # Small shift: barely changed
            _seed_history(db_small, {"signal_a": 75, "signal_b": 25}, 100, 2, 1)

            # Large shift: complete reversal
            _seed_history(db_large, {"signal_a": 20, "signal_b": 80}, 100, 2, 1)

            fw_small = DetectionFlywheel(db=db_small)
            fw_large = DetectionFlywheel(db=db_large)
            small_score = fw_small.detect_input_drift(window_days=7)["drift_score"]
            large_score = fw_large.detect_input_drift(window_days=7)["drift_score"]

            db_small.close()
            db_large.close()

        assert large_score > small_score, (
            f"Larger shift should produce higher JSD score: {large_score} vs {small_score}"
        )


# ---------------------------------------------------------------------------
# Part 3: trends CLI command
# ---------------------------------------------------------------------------


class _NonClosingDB:
    """Wraps a SentinelDB and makes close() a no-op so tests don't close shared fixtures."""

    def __init__(self, db: SentinelDB) -> None:
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def close(self):
        pass  # intentional no-op

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def _patch_db(monkeypatch, temp_db):
    """Patch SentinelDB at the db module level with a non-closing wrapper."""
    import sentinel.db as db_mod

    wrapper = _NonClosingDB(temp_db)
    monkeypatch.setattr(db_mod, "SentinelDB", lambda *a, **kw: wrapper)

    original_init = DetectionFlywheel.__init__

    def _patched_fw_init(self, db=None):
        original_init(self, db=temp_db)

    monkeypatch.setattr(DetectionFlywheel, "__init__", _patched_fw_init)


class TestTrendsCLI:
    def test_trends_no_data_runs_without_error(self, temp_db, monkeypatch):
        """sentinel trends should exit 0 and display headers even with no data."""
        _patch_db(monkeypatch, temp_db)

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--days", "30"])
        assert result.exit_code == 0, result.output
        assert "Sentinel Trends" in result.output

    def test_trends_json_output(self, temp_db, monkeypatch):
        """sentinel --json-output trends should produce valid JSON."""
        _patch_db(monkeypatch, temp_db)

        runner = CliRunner()
        result = runner.invoke(main, ["--json-output", "trends", "--days", "7"])
        assert result.exit_code == 0, result.output
        import json
        data = json.loads(result.output)
        assert "metrics_history" in data
        assert "drift" in data

    def test_trends_with_flywheel_data(self, temp_db, monkeypatch):
        """sentinel trends should show precision/recall rows when data exists."""
        # Seed flywheel metrics
        for i in range(5):
            temp_db.save_flywheel_metrics({
                "cycle_ts": _iso(-(5 - i)),
                "precision": 0.80 + i * 0.01,
                "recall": 0.70,
                "f1": 0.75,
                "accuracy": 0.78,
                "total_analyzed": 100,
                "true_positives": 80,
                "false_positives": 20,
                "signals_updated": 10,
                "patterns_evolved": 1,
                "cycle_number": i,
                "regression_alarm": False,
                "cusum_statistic": 0.0,
            })

        _patch_db(monkeypatch, temp_db)

        runner = CliRunner()
        result = runner.invoke(main, ["trends", "--days", "30"])
        assert result.exit_code == 0, result.output
        assert "Precision" in result.output
        assert "Recall" in result.output


# ---------------------------------------------------------------------------
# Part 4: health CLI command
# ---------------------------------------------------------------------------


class TestHealthCLI:
    def test_health_exits_cleanly(self, temp_db, monkeypatch):
        """sentinel health should always exit 0."""
        _patch_db(monkeypatch, temp_db)

        runner = CliRunner()
        result = runner.invoke(main, ["health"])
        assert result.exit_code == 0, result.output

    def test_health_dashboard_contains_key_sections(self, temp_db, monkeypatch):
        """Health dashboard should contain all major section headings."""
        _patch_db(monkeypatch, temp_db)

        runner = CliRunner()
        result = runner.invoke(main, ["health"])
        assert result.exit_code == 0
        output = result.output
        assert "Sentinel Health Dashboard" in output
        assert "Model Metrics" in output
        assert "Regression" in output
        assert "Drift" in output
        assert "Patterns" in output
        assert "Shadow Tests" in output

    def test_health_json_output(self, temp_db, monkeypatch):
        """sentinel --json-output health should produce valid JSON with required keys."""
        _patch_db(monkeypatch, temp_db)

        runner = CliRunner()
        result = runner.invoke(main, ["--json-output", "health"])
        assert result.exit_code == 0, result.output

        import json
        data = json.loads(result.output)
        assert "health" in data
        assert "drift" in data
        assert "source_stats" in data

    def test_health_shows_no_active_shadow(self, temp_db, monkeypatch):
        """With no active shadow run, the dashboard should say so."""
        _patch_db(monkeypatch, temp_db)

        runner = CliRunner()
        result = runner.invoke(main, ["health"])
        assert result.exit_code == 0
        assert "No active shadow run" in result.output

    def test_health_shows_active_shadow(self, temp_db, monkeypatch):
        """When a shadow run is active, the dashboard should show its stats."""
        temp_db.insert_shadow_run({"upfront_payment": 0.9})
        temp_db.update_shadow_run(1, {
            "baseline_precision": 0.80,
            "shadow_precision": 0.85,
            "jobs_evaluated": 50,
        })

        _patch_db(monkeypatch, temp_db)

        runner = CliRunner()
        result = runner.invoke(main, ["health"])
        assert result.exit_code == 0
        assert "Active shadow run" in result.output


# ---------------------------------------------------------------------------
# Part 5: sparkline helper
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_sparkline_returns_correct_width(self):
        from sentinel.cli import _sparkline
        assert len(_sparkline([1, 2, 3], width=8)) == 8

    def test_sparkline_empty_returns_spaces(self):
        from sentinel.cli import _sparkline
        result = _sparkline([], width=5)
        assert len(result) == 5
        assert result == " " * 5

    def test_sparkline_constant_uses_lowest_block(self):
        from sentinel.cli import _sparkline
        BLOCKS = " ▁▂▃▄▅▆▇█"
        result = _sparkline([5.0, 5.0, 5.0], width=3)
        # All equal → span=0 → all map to index 0 (lowest block)
        assert all(c == BLOCKS[0] for c in result)
