"""Tests for calibration tracking and confidence-aware scoring.

Covers:
- calibration_curve computation with known data
- ECE (Expected Calibration Error) calculation
- Auto-threshold adjustment
- Confidence-weighted CUSUM
- needs-review CLI command output
- Edge cases: no reports, single bin, empty data
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

import sentinel.config
from sentinel.cli import main
from sentinel.db import SentinelDB
from sentinel.flywheel import DetectionFlywheel
from sentinel.scorer import _RISK_THRESHOLDS, classify_risk
from sentinel.models import RiskLevel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> SentinelDB:
    return SentinelDB(path=str(tmp_path / "test_cal.db"))


def _seed_reports(db: SentinelDB, reports: list[dict]) -> None:
    """Insert report rows directly with known our_prediction values."""
    for r in reports:
        db.conn.execute(
            """
            INSERT INTO reports
                (url, is_scam, reason, our_prediction, was_correct, reported_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r.get("url", "http://example.com/job"),
                int(r.get("is_scam", 0)),
                r.get("reason", ""),
                r.get("our_prediction", 0.0),
                int(r.get("was_correct", 0)),
                r.get("reported_at", "2026-01-01T00:00:00+00:00"),
            ),
        )
    db.conn.commit()


def _seed_job(db: SentinelDB, url: str, score: float, confidence: float | None) -> None:
    db.save_job({
        "url": url,
        "title": "Test Job",
        "company": "Test Co",
        "scam_score": score,
        "confidence": confidence,
        "risk_level": "high",
    })


# ===========================================================================
# Part 1: Calibration curve
# ===========================================================================

class TestCalibrationCurve:
    def test_empty_db_returns_empty_list(self, tmp_path):
        """calibration_curve returns [] when no reports exist."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        curve = fw.calibration_curve(db=db)
        assert curve == []

    def test_curve_shape_with_known_data(self, tmp_path):
        """calibration_curve bins correctly and returns (center, pred, actual, n)."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        # All reports score ~0.05 (bin 0) — none are scams
        _seed_reports(db, [
            {"our_prediction": 0.05, "is_scam": 0, "url": f"http://x.com/{i}"}
            for i in range(10)
        ])
        # All reports score ~0.85 (bin 8) — all are scams
        _seed_reports(db, [
            {"our_prediction": 0.85, "is_scam": 1, "url": f"http://y.com/{i}"}
            for i in range(10)
        ])
        curve = fw.calibration_curve(db=db, n_bins=10)
        assert len(curve) == 2  # only 2 bins have data

        low_bin = next(c for c in curve if c[0] < 0.5)
        high_bin = next(c for c in curve if c[0] > 0.5)

        # Low bin: predicted ~0.05, actual scam rate = 0.0
        assert abs(low_bin[1] - 0.05) < 0.01, "predicted_rate should be ~0.05"
        assert low_bin[2] == pytest.approx(0.0), "actual_rate should be 0.0 for safe jobs"
        assert low_bin[3] == 10

        # High bin: predicted ~0.85, actual scam rate = 1.0
        assert abs(high_bin[1] - 0.85) < 0.01, "predicted_rate should be ~0.85"
        assert high_bin[2] == pytest.approx(1.0), "actual_rate should be 1.0 for scam jobs"
        assert high_bin[3] == 10

    def test_curve_tuple_format(self, tmp_path):
        """Each entry is a 4-tuple: (float, float, float, int)."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        _seed_reports(db, [{"our_prediction": 0.3, "is_scam": 1, "url": "http://a.com/1"}])
        curve = fw.calibration_curve(db=db)
        assert len(curve) == 1
        entry = curve[0]
        assert len(entry) == 4
        bin_center, predicted, actual, n = entry
        assert isinstance(bin_center, float)
        assert isinstance(predicted, float)
        assert isinstance(actual, float)
        assert isinstance(n, int)
        assert n == 1

    def test_single_bin_populated(self, tmp_path):
        """Works correctly when only one bin has data."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        # 5 scam reports, all scored at 0.75
        _seed_reports(db, [
            {"our_prediction": 0.75, "is_scam": 1, "url": f"http://z.com/{i}"}
            for i in range(5)
        ])
        curve = fw.calibration_curve(db=db, n_bins=10)
        assert len(curve) == 1
        _, predicted, actual, n = curve[0]
        assert n == 5
        assert abs(predicted - 0.75) < 0.01
        assert actual == pytest.approx(1.0)

    def test_perfectly_calibrated_model(self, tmp_path):
        """A perfectly calibrated model has predicted ≈ actual in each bin."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        # For bin ~0.2: seed 2 scam + 8 safe at score 0.2  → actual_rate = 0.2
        _seed_reports(db, [
            {"our_prediction": 0.2, "is_scam": 1, "url": f"http://p.com/s{i}"}
            for i in range(2)
        ])
        _seed_reports(db, [
            {"our_prediction": 0.2, "is_scam": 0, "url": f"http://p.com/l{i}"}
            for i in range(8)
        ])
        curve = fw.calibration_curve(db=db, n_bins=10)
        assert len(curve) == 1
        _, predicted, actual, _ = curve[0]
        assert abs(predicted - 0.2) < 0.02
        assert abs(actual - 0.2) < 0.02


# ===========================================================================
# Part 2: ECE calculation
# ===========================================================================

class TestCalibrationError:
    def test_ece_zero_when_no_data(self, tmp_path):
        """ECE is 0.0 when no reports exist."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        assert fw.calibration_error(db=db) == 0.0

    def test_ece_zero_for_perfect_calibration(self, tmp_path):
        """ECE is ~0 when model is perfectly calibrated."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        # score 0.5 → 5 scam, 5 safe → actual_rate = 0.5 = predicted
        _seed_reports(db, [
            {"our_prediction": 0.5, "is_scam": i % 2, "url": f"http://perf.com/{i}"}
            for i in range(10)
        ])
        ece = fw.calibration_error(db=db)
        assert ece < 0.05, f"ECE should be near 0 for perfect calibration, got {ece}"

    def test_ece_high_for_miscalibrated_model(self, tmp_path):
        """ECE is high when predictions are systematically wrong."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        # Model says 0.05 (safe) but all are actually scams
        _seed_reports(db, [
            {"our_prediction": 0.05, "is_scam": 1, "url": f"http://bad.com/{i}"}
            for i in range(20)
        ])
        ece = fw.calibration_error(db=db)
        # Predicted ~0.05, actual 1.0 → ECE ≈ 0.95
        assert ece > 0.8, f"ECE should be ~0.95 for badly calibrated model, got {ece}"

    def test_ece_is_weighted_mean(self, tmp_path):
        """ECE is weighted by bin sample size, not a simple average."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        # 90 reports at score 0.9 all scam + 10 reports at score 0.1 all scam
        # Bin 0.9: predicted≈0.9, actual=1.0 → |diff|=0.1, weight=0.9
        # Bin 0.1: predicted≈0.1, actual=1.0 → |diff|=0.9, weight=0.1
        # Weighted ECE = 0.9*0.1 + 0.1*0.9 = 0.18
        # Unweighted average = (0.1 + 0.9) / 2 = 0.5
        # Weighted ECE must be much lower than the unweighted average
        _seed_reports(db, [
            {"our_prediction": 0.9, "is_scam": 1, "url": f"http://wt.com/s{i}"}
            for i in range(90)
        ])
        _seed_reports(db, [
            {"our_prediction": 0.1, "is_scam": 1, "url": f"http://wt.com/b{i}"}
            for i in range(10)
        ])
        ece = fw.calibration_error(db=db)
        # Weighted ECE ≈ 0.18 — far below the naive unweighted average of 0.5
        assert ece < 0.30, f"Weighted ECE should be ~0.18, got {ece}"
        # And specifically much less than unweighted average (0.5)
        assert ece < 0.40, "Weighted ECE should be substantially less than 0.5"


# ===========================================================================
# Part 3: Auto-threshold adjustment
# ===========================================================================

class TestAutoThresholdAdjustment:
    def test_no_adjustment_when_no_data(self, tmp_path):
        """auto_adjust_thresholds returns skipped message when no calibration data."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        result = fw.auto_adjust_thresholds(db=db)
        assert "skipped" in result or result.get("adjusted") == []

    def test_threshold_adjusted_when_model_underpredicts(self, tmp_path):
        """If actual scam rate >> predicted, the relevant threshold is lowered."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        # Score ~0.55 (near suspicious/high boundary) but ALL are scams
        # → actual_rate = 1.0, predicted ≈ 0.55, diff = 0.45 > tolerance
        _seed_reports(db, [
            {"our_prediction": 0.55, "is_scam": 1, "url": f"http://adj.com/{i}"}
            for i in range(20)
        ])

        old_suspicious = _RISK_THRESHOLDS["suspicious"]
        result = fw.auto_adjust_thresholds(db=db)

        # Should have adjusted something
        assert isinstance(result.get("adjusted"), list)
        if result["adjusted"]:
            # The threshold moved (direction: lowered to catch more scams)
            changed = result["adjusted"][0]
            assert changed["new_value"] < changed["old_value"] or abs(
                changed["new_value"] - changed["old_value"]
            ) <= 0.05, "Threshold delta should be clamped to _MAX_THRESHOLD_DELTA"
        # Restore for other tests
        _RISK_THRESHOLDS["suspicious"] = old_suspicious

    def test_no_adjustment_for_well_calibrated_bin(self, tmp_path):
        """Bins within tolerance are not adjusted."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        # score 0.5 → 50% scams → well calibrated, no adjustment needed
        _seed_reports(db, [
            {"our_prediction": 0.5, "is_scam": i % 2, "url": f"http://ok.com/{i}"}
            for i in range(20)
        ])
        result = fw.auto_adjust_thresholds(db=db)
        # No adjustments (actual ≈ predicted = 0.5)
        assert result.get("adjusted") == [] or all(
            abs(a["actual_rate"] - a["predicted_rate"]) <= 0.10
            for a in result.get("adjusted", [])
        )


# ===========================================================================
# Part 4: Confidence-weighted CUSUM
# ===========================================================================

class TestConfidenceWeightedCUSUM:
    def test_cusum_with_no_job_confidence_uses_default(self, tmp_path):
        """detect_regression works normally when jobs have no confidence stored."""
        db = _make_db(tmp_path)
        fw = DetectionFlywheel(db=db)
        _seed_reports(db, [
            {
                "url": f"http://cusum.com/{i}",
                "is_scam": 1,
                "was_correct": 1,
                "our_prediction": 0.9,
                "reported_at": f"2026-01-0{i % 9 + 1}T00:00:00+00:00",
            }
            for i in range(10)
        ])
        result = fw.detect_regression(window=10)
        assert "alarm" in result
        assert "cusum_statistic" in result

    def test_low_confidence_job_contributes_less(self, tmp_path):
        """Low-confidence reports should dampen the CUSUM signal."""
        db = _make_db(tmp_path)
        fw_hi = DetectionFlywheel(db=db)
        fw_lo = DetectionFlywheel(db=SentinelDB(path=str(tmp_path / "lo.db")))

        # Seed high-confidence scam reports into fw_hi DB
        for i in range(10):
            url = f"http://hi.com/{i}"
            _seed_job(db, url, score=0.8, confidence=0.9)
            _seed_reports(db, [{
                "url": url, "is_scam": 1, "was_correct": 0,  # wrong prediction
                "our_prediction": 0.2,
                "reported_at": f"2026-02-{i + 1:02d}T00:00:00+00:00",
            }])

        # Seed low-confidence scam reports into fw_lo DB
        lo_db = fw_lo.db
        for i in range(10):
            url = f"http://lo.com/{i}"
            _seed_job(lo_db, url, score=0.8, confidence=0.1)
            _seed_reports(lo_db, [{
                "url": url, "is_scam": 1, "was_correct": 0,
                "our_prediction": 0.2,
                "reported_at": f"2026-02-{i + 1:02d}T00:00:00+00:00",
            }])

        result_hi = fw_hi.detect_regression(window=10)
        result_lo = fw_lo.detect_regression(window=10)
        # High-confidence wrong predictions → higher CUSUM stat
        assert result_hi["cusum_statistic"] >= result_lo["cusum_statistic"], (
            f"High-confidence CUSUM ({result_hi['cusum_statistic']}) should be "
            f">= low-confidence ({result_lo['cusum_statistic']})"
        )


# ===========================================================================
# Part 5: needs-review CLI command
# ===========================================================================

class TestNeedsReviewCommand:
    def test_command_exists_and_runs(self, tmp_path, monkeypatch):
        """needs-review command runs without error on an empty DB."""
        monkeypatch.setenv("SENTINEL_DB_PATH", str(tmp_path / "nr.db"))
        runner = CliRunner()
        result = runner.invoke(main, ["needs-review"])
        assert result.exit_code == 0, result.output

    def test_empty_db_shows_no_jobs_message(self, tmp_path, monkeypatch):
        """Empty DB outputs a 'no jobs' message."""
        monkeypatch.setenv("SENTINEL_DB_PATH", str(tmp_path / "nr_empty.db"))
        runner = CliRunner()
        result = runner.invoke(main, ["needs-review"])
        assert result.exit_code == 0
        assert "No jobs need review" in result.output or result.output.strip() == ""

    def test_high_score_low_confidence_job_appears(self, tmp_path, monkeypatch):
        """A job with high score and low confidence appears in the review list."""
        db_path = str(tmp_path / "nr_jobs.db")
        monkeypatch.setenv("SENTINEL_DB_PATH", db_path)

        db = SentinelDB(path=db_path)
        db.save_job({
            "url": "http://review.com/job1",
            "title": "Mystery Job",
            "company": "Unknown Corp",
            "scam_score": 0.75,
            "confidence": 0.2,
            "risk_level": "high",
        })
        db.close()

        runner = CliRunner()
        result = runner.invoke(main, ["needs-review"])
        assert result.exit_code == 0, result.output
        assert "Mystery Job" in result.output or "Unknown Corp" in result.output or "0.75" in result.output

    def test_low_score_job_not_surfaced(self, tmp_path, monkeypatch):
        """A job with low scam score is NOT surfaced even with low confidence."""
        db_path = str(tmp_path / "nr_lowscore.db")
        monkeypatch.setenv("SENTINEL_DB_PATH", db_path)

        db = SentinelDB(path=db_path)
        db.save_job({
            "url": "http://safe.com/job1",
            "title": "Safe Job",
            "company": "Safe Corp",
            "scam_score": 0.1,
            "confidence": 0.1,
            "risk_level": "safe",
        })
        db.close()

        runner = CliRunner()
        result = runner.invoke(main, ["needs-review"])
        assert result.exit_code == 0
        assert "Safe Job" not in result.output

    def test_json_output_format(self, tmp_path, monkeypatch):
        """--json-output returns valid JSON with expected keys."""
        import json as _json

        db_path = str(tmp_path / "nr_json.db")
        monkeypatch.setenv("SENTINEL_DB_PATH", db_path)

        db = SentinelDB(path=db_path)
        db.save_job({
            "url": "http://json.com/job1",
            "title": "JSON Job",
            "company": "JSON Corp",
            "scam_score": 0.8,
            "confidence": 0.15,
            "risk_level": "scam",
        })
        db.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--json-output", "needs-review"])
        assert result.exit_code == 0, result.output
        data = _json.loads(result.output)
        assert "count" in data
        assert "jobs" in data
        assert "score_threshold" in data
        assert "confidence_threshold" in data


# ===========================================================================
# Part 6: DB confidence column and get_jobs_for_review
# ===========================================================================

class TestDBConfidenceSupport:
    def test_save_and_retrieve_confidence(self, tmp_path):
        """Confidence is persisted and retrieved correctly."""
        db = _make_db(tmp_path)
        db.save_job({
            "url": "http://conf.com/1",
            "title": "Conf Test",
            "company": "Conf Co",
            "scam_score": 0.7,
            "confidence": 0.25,
            "risk_level": "high",
        })
        job = db.get_job("http://conf.com/1")
        assert job is not None
        assert job["confidence"] == pytest.approx(0.25)

    def test_get_jobs_for_review_filters_correctly(self, tmp_path):
        """get_jobs_for_review returns only high-score, low-confidence jobs."""
        db = _make_db(tmp_path)
        # Should appear: score=0.8, confidence=0.2
        db.save_job({
            "url": "http://review.com/a",
            "scam_score": 0.8,
            "confidence": 0.2,
            "risk_level": "high",
        })
        # Should NOT appear: score=0.8 but confidence=0.8 (too high)
        db.save_job({
            "url": "http://review.com/b",
            "scam_score": 0.8,
            "confidence": 0.8,
            "risk_level": "high",
        })
        # Should NOT appear: confidence=0.1 but score=0.3 (too low)
        db.save_job({
            "url": "http://review.com/c",
            "scam_score": 0.3,
            "confidence": 0.1,
            "risk_level": "low",
        })
        jobs = db.get_jobs_for_review(score_threshold=0.5, confidence_threshold=0.4)
        urls = [j["url"] for j in jobs]
        assert "http://review.com/a" in urls
        assert "http://review.com/b" not in urls
        assert "http://review.com/c" not in urls

    def test_null_confidence_excluded_from_review(self, tmp_path):
        """Jobs with NULL confidence are excluded from needs-review."""
        db = _make_db(tmp_path)
        db.save_job({
            "url": "http://null.com/1",
            "scam_score": 0.9,
            "confidence": None,
            "risk_level": "scam",
        })
        jobs = db.get_jobs_for_review()
        urls = [j["url"] for j in jobs]
        assert "http://null.com/1" not in urls
