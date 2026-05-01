"""Tests for flywheel_metrics schema columns and date parsing helpers."""

import tempfile
import os
import pytest

from sentinel.db import SentinelDB
from sentinel.scanner import _days_since_posted, _parse_relative_date


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_db() -> SentinelDB:
    """Create a SentinelDB backed by a temporary file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # let SentinelDB create it fresh
    return SentinelDB(path=path)


# ---------------------------------------------------------------------------
# Fix 1 — flywheel_metrics schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "id", "cycle_ts", "total_analyzed", "true_positives", "false_positives",
    "precision", "recall", "signals_updated", "patterns_evolved",
    "f1", "accuracy", "cycle_number", "regression_alarm",
    "cusum_statistic", "patterns_promoted", "patterns_deprecated",
    "calibration_ece", "thresholds_adjusted", "shadow_evaluation_json",
}


def test_flywheel_metrics_has_all_columns():
    """CREATE TABLE must include all new columns."""
    with _fresh_db() as db:
        rows = db.conn.execute(
            "PRAGMA table_info(flywheel_metrics)"
        ).fetchall()
        cols = {row["name"] for row in rows}
    assert EXPECTED_COLUMNS == cols, f"Missing columns: {EXPECTED_COLUMNS - cols}"


def test_save_flywheel_metrics_round_trip():
    """save_flywheel_metrics stores all fields; they can be read back."""
    with _fresh_db() as db:
        metrics = {
            "cycle_ts": "2026-04-30T00:00:00+00:00",
            "total_analyzed": 100,
            "true_positives": 80,
            "false_positives": 5,
            "precision": 0.941,
            "recall": 0.888,
            "signals_updated": 12,
            "patterns_evolved": 2,
            "f1": 0.914,
            "accuracy": 0.900,
            "cycle_number": 7,
            "regression_alarm": False,
            "cusum_statistic": 1.23,
            "patterns_promoted": 3,
            "patterns_deprecated": 1,
        }
        db.save_flywheel_metrics(metrics)

        row = db.conn.execute(
            "SELECT * FROM flywheel_metrics ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        row_dict = dict(row)

        assert row_dict["total_analyzed"] == 100
        assert row_dict["cycle_number"] == 7
        assert abs(row_dict["f1"] - 0.914) < 1e-6
        assert abs(row_dict["accuracy"] - 0.900) < 1e-6
        assert row_dict["regression_alarm"] == 0
        assert abs(row_dict["cusum_statistic"] - 1.23) < 1e-6
        assert row_dict["patterns_promoted"] == 3
        assert row_dict["patterns_deprecated"] == 1


def test_save_flywheel_metrics_list_promoted_deprecated():
    """patterns_promoted / patterns_deprecated may be lists; should be stored as counts."""
    with _fresh_db() as db:
        metrics = {
            "patterns_promoted": ["pat_a", "pat_b"],
            "patterns_deprecated": ["pat_c"],
        }
        db.save_flywheel_metrics(metrics)

        row = db.conn.execute(
            "SELECT patterns_promoted, patterns_deprecated FROM flywheel_metrics LIMIT 1"
        ).fetchone()
        assert row["patterns_promoted"] == 2
        assert row["patterns_deprecated"] == 1


def test_migration_idempotent():
    """Opening the same DB file twice must not raise (migration is re-entrant)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    try:
        db1 = SentinelDB(path=path)
        db1.close()
        db2 = SentinelDB(path=path)  # second open triggers migration again
        db2.close()
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# Fix 2 — date parsing
# ---------------------------------------------------------------------------

def test_days_since_iso_basic():
    """Plain ISO date string returns a non-negative integer."""
    result = _days_since_posted("2026-04-15")
    assert isinstance(result, int)
    assert result >= 0


def test_days_since_iso_timezone():
    """ISO 8601 with timezone offset is handled correctly."""
    result = _days_since_posted("2026-04-15T10:30:00+00:00")
    assert isinstance(result, int)
    assert result >= 0


def test_days_since_relative_days():
    assert _days_since_posted("3 days ago") == 3


def test_days_since_relative_weeks():
    assert _days_since_posted("2 weeks ago") == 14


def test_days_since_relative_months():
    assert _days_since_posted("1 month ago") == 30


def test_days_since_yesterday():
    assert _days_since_posted("yesterday") == 1


def test_days_since_just_now():
    assert _days_since_posted("just now") == 0


def test_days_since_invalid():
    assert _days_since_posted("invalid") is None


def test_days_since_empty():
    assert _days_since_posted("") is None


def test_parse_relative_date_today():
    assert _parse_relative_date("today") == 0


def test_parse_relative_date_hours_ago():
    """Hours-ago is treated as 0 days (same day)."""
    assert _parse_relative_date("5 hours ago") == 0


def test_parse_relative_date_no_match():
    assert _parse_relative_date("no match here") is None
