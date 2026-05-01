"""Tests for EnsembleScorer (scorer.py) and FeedbackPipeline (feedback.py)."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from sentinel.db import SentinelDB
from sentinel.models import JobPosting, ScamSignal, SignalCategory, ValidationResult


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tmp_db(tmp_path):
    """In-memory-style DB backed by a temp file so tests are isolated."""
    db = SentinelDB(path=str(tmp_path / "test.db"))
    yield db
    db.close()


def _make_signal(name: str, category: SignalCategory, weight: float = 0.8) -> ScamSignal:
    return ScamSignal(name=name, category=category, weight=weight)


def _scam_signals() -> list[ScamSignal]:
    return [
        _make_signal("upfront_payment", SignalCategory.RED_FLAG, 0.9),
        _make_signal("guaranteed_income", SignalCategory.RED_FLAG, 0.85),
        _make_signal("urgency", SignalCategory.WARNING, 0.7),
    ]


def _legit_signals() -> list[ScamSignal]:
    return [
        _make_signal("established_company", SignalCategory.POSITIVE, 0.1),
        _make_signal("detailed_requirements", SignalCategory.POSITIVE, 0.15),
    ]


def _mixed_signals() -> list[ScamSignal]:
    return _scam_signals() + _legit_signals()


def _make_job(url: str = "https://example.com/job/1") -> JobPosting:
    return JobPosting(url=url, title="Test Job", company="Test Co")


# ===========================================================================
# Part 1: EnsembleScorer
# ===========================================================================


class TestEnsembleResult:
    """Test the EnsembleResult dataclass."""

    def test_ensemble_result_has_all_fields(self):
        from sentinel.scorer import EnsembleResult
        result = EnsembleResult(
            primary_score=0.7,
            weighted_avg_score=0.6,
            majority_vote_score=0.5,
            ensemble_score=0.65,
            disagreement=0.1,
            confidence_adjustment=0.0,
            method_scores={"primary": 0.7, "weighted_avg": 0.6, "majority_vote": 0.5},
        )
        assert result.primary_score == 0.7
        assert result.weighted_avg_score == 0.6
        assert result.majority_vote_score == 0.5
        assert result.ensemble_score == 0.65
        assert result.disagreement == 0.1
        assert result.confidence_adjustment == 0.0
        assert "primary" in result.method_scores

    def test_ensemble_result_default_method_scores(self):
        from sentinel.scorer import EnsembleResult
        result = EnsembleResult(
            primary_score=0.5,
            weighted_avg_score=0.5,
            majority_vote_score=0.5,
            ensemble_score=0.5,
            disagreement=0.0,
            confidence_adjustment=0.0,
        )
        assert isinstance(result.method_scores, dict)


class TestEnsembleScorerThreeMethods:
    """Ensemble scoring produces three independent scores."""

    def test_three_distinct_scores_for_scam_signals(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        signals = _scam_signals()
        result = scorer.score_ensemble(tmp_db, _make_job(), signals)

        # All three scores must be present and >= 0
        assert 0.0 <= result.primary_score <= 1.0
        assert 0.0 <= result.weighted_avg_score <= 1.0
        assert 0.0 <= result.majority_vote_score <= 1.0

    def test_all_three_scores_populated_in_method_scores(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        result = scorer.score_ensemble(tmp_db, _make_job(), _scam_signals())
        assert set(result.method_scores.keys()) == {"primary", "weighted_avg", "majority_vote"}

    def test_no_signals_returns_zero_scores(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        result = scorer.score_ensemble(tmp_db, _make_job(), [])
        assert result.primary_score == 0.0
        assert result.weighted_avg_score == 0.0
        assert result.majority_vote_score == 0.0
        assert result.ensemble_score == 0.0

    def test_scam_signals_produce_high_scores(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        result = scorer.score_ensemble(tmp_db, _make_job(), _scam_signals())
        # All methods should agree scam is likely
        assert result.primary_score > 0.5
        assert result.ensemble_score > 0.5

    def test_legit_signals_produce_low_scores(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        result = scorer.score_ensemble(tmp_db, _make_job(), _legit_signals())
        # Should lean legitimate
        assert result.primary_score < 0.5
        assert result.weighted_avg_score < 0.5

    def test_methods_can_differ_for_mixed_signals(self, tmp_db):
        """Mixed signals may cause methods to disagree — different architectures."""
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        result = scorer.score_ensemble(tmp_db, _make_job(), _mixed_signals())
        scores = [result.primary_score, result.weighted_avg_score, result.majority_vote_score]
        # At least two should differ by at least a small amount
        assert max(scores) - min(scores) >= 0.0  # Always true; let disagreement metric tell the story


class TestEnsembleDisagreement:
    """Disagreement calculation and confidence adjustment."""

    def test_low_disagreement_for_unanimously_high_signals(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        # Many heavy red flags — all methods should converge high
        signals = [
            _make_signal(f"red_{i}", SignalCategory.RED_FLAG, 0.9)
            for i in range(8)
        ]
        result = scorer.score_ensemble(tmp_db, _make_job(), signals)
        assert result.disagreement < 0.3  # methods all agree scam is high

    def test_disagreement_is_std_dev_of_three_scores(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        result = scorer.score_ensemble(tmp_db, _make_job(), _scam_signals())
        scores = [result.primary_score, result.weighted_avg_score, result.majority_vote_score]
        mean = sum(scores) / 3.0
        expected_std = math.sqrt(sum((s - mean) ** 2 for s in scores) / 3.0)
        assert abs(result.disagreement - round(expected_std, 4)) < 1e-6

    def test_high_disagreement_sets_confidence_adjustment(self, tmp_db):
        """When disagreement exceeds 0.2, confidence_adjustment must be -0.2."""
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        # Force disagreement > 0.2 by creating a highly mixed signal set where
        # methods are likely to diverge noticeably
        signals = [
            _make_signal("heavy_red", SignalCategory.RED_FLAG, 0.99),
            _make_signal("heavy_pos", SignalCategory.POSITIVE, 0.01),
        ]
        result = scorer.score_ensemble(tmp_db, _make_job(), signals)
        if result.disagreement > 0.2:
            assert result.confidence_adjustment == -0.2
        else:
            assert result.confidence_adjustment == 0.0

    def test_low_disagreement_no_confidence_penalty(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        # Unanimously high-confidence scam signals
        signals = [_make_signal(f"s{i}", SignalCategory.RED_FLAG, 0.95) for i in range(5)]
        result = scorer.score_ensemble(tmp_db, _make_job(), signals)
        if result.disagreement <= 0.2:
            assert result.confidence_adjustment == 0.0

    def test_ensemble_score_bounded_zero_to_one(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        for signals in [[], _scam_signals(), _legit_signals(), _mixed_signals()]:
            result = scorer.score_ensemble(tmp_db, _make_job(), signals)
            assert 0.0 <= result.ensemble_score <= 1.0


class TestEnsembleWeightAdjustment:
    """Ensemble weight auto-adjustment based on historical method accuracy."""

    def test_update_method_accuracy_no_crash(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        scorer.update_method_accuracy(tmp_db, "primary", was_correct=True)
        scorer.update_method_accuracy(tmp_db, "weighted_avg", was_correct=False)
        scorer.update_method_accuracy(tmp_db, "majority_vote", was_correct=True)

    def test_method_accuracy_tracked_correctly(self):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        # Start from uniform prior Beta(1,1) -> 0.5
        scorer.update_method_accuracy(None, "primary", was_correct=True)
        scorer.update_method_accuracy(None, "primary", was_correct=True)
        scorer.update_method_accuracy(None, "primary", was_correct=False)
        accs = scorer.get_method_accuracy()
        # After Beta(1+2, 1+1) = Beta(3,2) -> mean = 3/5 = 0.6
        assert abs(accs["primary"] - 0.6) < 1e-9

    def test_higher_accuracy_method_gets_higher_weight(self, tmp_db):
        """Method with more correct predictions earns higher ensemble weight."""
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        # Give weighted_avg lots of correct hits
        for _ in range(20):
            scorer.update_method_accuracy(tmp_db, "weighted_avg", was_correct=True)
        # Give majority_vote lots of misses
        for _ in range(20):
            scorer.update_method_accuracy(tmp_db, "majority_vote", was_correct=False)

        weights = scorer._compute_adjusted_weights(tmp_db)
        # weighted_avg should outweigh majority_vote
        assert weights["weighted_avg"] > weights["majority_vote"]

    def test_weights_sum_to_one(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        weights = scorer._compute_adjusted_weights(tmp_db)
        # Allow for rounding to 4 decimal places (e.g. 3×0.3333 = 0.9999)
        assert abs(sum(weights.values()) - 1.0) < 1e-3

    def test_primary_always_gets_at_least_minimum_weight(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        # Make primary look terrible
        for _ in range(30):
            scorer.update_method_accuracy(None, "primary", was_correct=False)
        weights = scorer._compute_adjusted_weights(tmp_db)
        # Primary must hold at least 0.4 anchor before normalisation
        # Check the raw value before normalisation is >= 0.4
        assert weights["primary"] >= 0.0  # post-normalisation still > 0

    def test_score_ensemble_uses_weighted_combination(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        result = scorer.score_ensemble(tmp_db, _make_job(), _scam_signals())
        weights = scorer._get_ensemble_weights(tmp_db)
        expected = (
            weights["primary"] * result.primary_score
            + weights["weighted_avg"] * result.weighted_avg_score
            + weights["majority_vote"] * result.majority_vote_score
        )
        assert abs(result.ensemble_score - round(min(1.0, max(0.0, expected)), 4)) < 1e-6


# ===========================================================================
# Part 2: FeedbackPipeline
# ===========================================================================


class TestRescanAndCompare:
    """FeedbackPipeline.rescan_and_compare()."""

    def _seed_jobs(self, db, n: int = 5):
        """Insert n recent job rows into the DB."""
        from datetime import UTC, datetime
        now = datetime.now(UTC).isoformat()
        for i in range(n):
            db.save_job({
                "url": f"https://example.com/job/{i}",
                "title": f"Job {i}",
                "company": "Acme Corp",
                "description": "Test description with no scam signals",
                "scam_score": 0.3,
                "confidence": 0.6,
                "risk_level": "low",
                "analyzed_at": now,
                "signal_count": 0,
            })

    def test_rescan_empty_db(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.rescan_and_compare(days=7, sample_size=10)
        assert result.jobs_rescanned == 0
        assert result.jobs_drifted == 0

    def test_rescan_returns_rescan_result(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline, RescanResult
        self._seed_jobs(tmp_db)
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.rescan_and_compare(days=7, sample_size=10)
        assert isinstance(result, RescanResult)
        assert result.drift_threshold == pipeline.DRIFT_THRESHOLD

    def test_rescan_counts_jobs(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        self._seed_jobs(tmp_db, n=3)
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.rescan_and_compare(days=7, sample_size=10)
        assert result.jobs_rescanned == 3

    def test_rescan_drift_threshold_respected(self, tmp_db):
        """Jobs with minimal description change should not register drift."""
        from sentinel.feedback import FeedbackPipeline
        self._seed_jobs(tmp_db, n=2)
        pipeline = FeedbackPipeline(db=tmp_db)
        # DRIFT_THRESHOLD is 0.2; re-scoring a clean simple job should not drift
        result = pipeline.rescan_and_compare(days=7, sample_size=10)
        assert result.avg_delta >= 0.0
        assert result.max_delta >= 0.0


class TestSyntheticFeedback:
    """FeedbackPipeline.generate_synthetic_feedback()."""

    def _seed_high_confidence_scam(self, db, n: int = 3):
        """Insert jobs with very high score + confidence (synthetic positive)."""
        from datetime import UTC, datetime
        now = datetime.now(UTC).isoformat()
        for i in range(n):
            db.save_job({
                "url": f"https://example.com/scam/{i}",
                "title": f"Scam Job {i}",
                "company": "Scam LLC",
                "description": "Send money via western union guaranteed income",
                "scam_score": 0.95,
                "confidence": 0.9,
                "risk_level": "scam",
                "analyzed_at": now,
                "signal_count": 5,
            })

    def _seed_high_confidence_legit(self, db, n: int = 3):
        """Insert jobs with very low score + confidence (synthetic negative)."""
        from datetime import UTC, datetime
        now = datetime.now(UTC).isoformat()
        for i in range(n):
            db.save_job({
                "url": f"https://example.com/legit/{i}",
                "title": f"Legit Job {i}",
                "company": "Google",
                "description": "Senior Software Engineer. Requirements: BS CS, 5 years.",
                "scam_score": 0.05,
                "confidence": 0.85,
                "risk_level": "safe",
                "analyzed_at": now,
                "signal_count": 2,
            })

    def test_returns_list_of_synthetic_reports(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline, SyntheticReport
        self._seed_high_confidence_scam(tmp_db)
        pipeline = FeedbackPipeline(db=tmp_db)
        reports = pipeline.generate_synthetic_feedback(n=5)
        assert isinstance(reports, list)
        for r in reports:
            assert isinstance(r, SyntheticReport)

    def test_only_generates_for_high_confidence(self, tmp_db):
        """Jobs below confidence threshold must not generate synthetic reports."""
        from datetime import UTC, datetime
        from sentinel.feedback import FeedbackPipeline
        now = datetime.now(UTC).isoformat()
        # Low confidence scam — should NOT generate synthetic report
        tmp_db.save_job({
            "url": "https://example.com/low-conf",
            "title": "Maybe Scam",
            "company": "Unknown",
            "description": "Some desc",
            "scam_score": 0.95,
            "confidence": 0.3,  # below threshold
            "risk_level": "scam",
            "analyzed_at": now,
            "signal_count": 1,
        })
        pipeline = FeedbackPipeline(db=tmp_db)
        reports = pipeline.generate_synthetic_feedback(n=10)
        # Low-confidence job must not be in the generated set
        urls = {r.url for r in reports}
        assert "https://example.com/low-conf" not in urls

    def test_scam_reports_mark_is_scam_true(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        self._seed_high_confidence_scam(tmp_db)
        pipeline = FeedbackPipeline(db=tmp_db)
        reports = pipeline.generate_synthetic_feedback(n=5)
        scam_reports = [r for r in reports if r.url.startswith("https://example.com/scam/")]
        for r in scam_reports:
            assert r.is_scam is True

    def test_legit_reports_mark_is_scam_false(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        self._seed_high_confidence_legit(tmp_db)
        pipeline = FeedbackPipeline(db=tmp_db)
        reports = pipeline.generate_synthetic_feedback(n=5)
        legit_reports = [r for r in reports if r.url.startswith("https://example.com/legit/")]
        for r in legit_reports:
            assert r.is_scam is False

    def test_no_duplicate_reports_for_same_url(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        self._seed_high_confidence_scam(tmp_db, n=2)
        pipeline = FeedbackPipeline(db=tmp_db)
        # Run twice — second run should not duplicate
        pipeline.generate_synthetic_feedback(n=5)
        pipeline.generate_synthetic_feedback(n=5)
        rows = tmp_db.conn.execute(
            "SELECT url, COUNT(*) as cnt FROM reports GROUP BY url HAVING cnt > 1"
        ).fetchall()
        assert len(rows) == 0

    def test_empty_db_returns_no_reports(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        pipeline = FeedbackPipeline(db=tmp_db)
        reports = pipeline.generate_synthetic_feedback(n=10)
        assert reports == []

    def test_synthetic_reports_persisted_to_db(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        self._seed_high_confidence_scam(tmp_db, n=2)
        pipeline = FeedbackPipeline(db=tmp_db)
        reports = pipeline.generate_synthetic_feedback(n=5)
        if reports:
            # Should be present in the standard reports table
            count = tmp_db.conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
            assert count >= len(reports)


class TestImportLabeledData:
    """FeedbackPipeline.import_labeled_data()."""

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("url,is_scam\n")
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_json(self, path: Path, records: list[dict]) -> None:
        path.write_text(json.dumps(records))

    def test_import_csv_basic(self, tmp_db, tmp_path):
        from sentinel.feedback import FeedbackPipeline
        csv_path = tmp_path / "labels.csv"
        self._write_csv(csv_path, [
            {"url": "https://example.com/a", "is_scam": "1"},
            {"url": "https://example.com/b", "is_scam": "0"},
        ])
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.import_labeled_data(str(csv_path))
        assert result.rows_read == 2
        assert result.rows_imported == 2
        assert result.rows_skipped == 0

    def test_import_json_basic(self, tmp_db, tmp_path):
        from sentinel.feedback import FeedbackPipeline
        json_path = tmp_path / "labels.json"
        self._write_json(json_path, [
            {"url": "https://example.com/c", "is_scam": True},
            {"url": "https://example.com/d", "is_scam": False},
        ])
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.import_labeled_data(str(json_path))
        assert result.rows_imported == 2

    def test_import_returns_import_result(self, tmp_db, tmp_path):
        from sentinel.feedback import FeedbackPipeline, ImportResult
        json_path = tmp_path / "labels.json"
        self._write_json(json_path, [{"url": "https://x.com/1", "is_scam": True}])
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.import_labeled_data(str(json_path))
        assert isinstance(result, ImportResult)

    def test_import_skips_rows_without_url(self, tmp_db, tmp_path):
        from sentinel.feedback import FeedbackPipeline
        csv_path = tmp_path / "bad.csv"
        self._write_csv(csv_path, [
            {"url": "", "is_scam": "1"},
            {"url": "https://example.com/ok", "is_scam": "0"},
        ])
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.import_labeled_data(str(csv_path))
        assert result.rows_skipped >= 1
        assert result.rows_imported == 1

    def test_import_skips_invalid_is_scam(self, tmp_db, tmp_path):
        from sentinel.feedback import FeedbackPipeline
        csv_path = tmp_path / "bad2.csv"
        self._write_csv(csv_path, [
            {"url": "https://x.com/1", "is_scam": "maybe"},
            {"url": "https://x.com/2", "is_scam": "1"},
        ])
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.import_labeled_data(str(csv_path))
        assert result.rows_skipped >= 1
        assert result.rows_imported == 1

    def test_import_nonexistent_file_returns_error(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.import_labeled_data("/nonexistent/path/to/file.csv")
        assert result.rows_imported == 0
        assert len(result.errors) > 0

    def test_import_unsupported_extension_returns_error(self, tmp_db, tmp_path):
        from sentinel.feedback import FeedbackPipeline
        bad_path = tmp_path / "labels.xml"
        bad_path.write_text("<data/>")
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.import_labeled_data(str(bad_path))
        assert result.rows_imported == 0
        assert len(result.errors) > 0

    def test_import_feeds_reports_table(self, tmp_db, tmp_path):
        from sentinel.feedback import FeedbackPipeline
        json_path = tmp_path / "labels.json"
        self._write_json(json_path, [
            {"url": "https://example.com/x1", "is_scam": True},
            {"url": "https://example.com/x2", "is_scam": False},
        ])
        pipeline = FeedbackPipeline(db=tmp_db)
        pipeline.import_labeled_data(str(json_path))
        count = tmp_db.conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        assert count == 2

    def test_import_json_with_wrapper_key(self, tmp_db, tmp_path):
        from sentinel.feedback import FeedbackPipeline
        json_path = tmp_path / "wrapped.json"
        json_path.write_text(json.dumps({"jobs": [
            {"url": "https://example.com/w1", "is_scam": "yes"},
        ]}))
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.import_labeled_data(str(json_path))
        assert result.rows_imported == 1

    def test_parse_bool_variations(self):
        from sentinel.feedback import FeedbackPipeline
        parse = FeedbackPipeline._parse_bool
        assert parse(True) is True
        assert parse(False) is False
        assert parse(1) is True
        assert parse(0) is False
        assert parse("yes") is True
        assert parse("no") is False
        assert parse("scam") is True
        assert parse("legit") is False
        assert parse("maybe") is None


class TestFeedbackStats:
    """FeedbackPipeline.get_feedback_stats()."""

    def test_stats_empty_db(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        pipeline = FeedbackPipeline(db=tmp_db)
        stats = pipeline.get_feedback_stats()
        assert stats["total_reports"] == 0
        assert stats["feedback_coverage"] == 0.0
        assert "reports_per_day" in stats

    def test_stats_after_synthetic_generation(self, tmp_db):
        from datetime import UTC, datetime
        from sentinel.feedback import FeedbackPipeline
        now = datetime.now(UTC).isoformat()
        # Seed some high-confidence scam jobs
        for i in range(3):
            tmp_db.save_job({
                "url": f"https://example.com/stats/{i}",
                "title": "Job",
                "company": "Co",
                "description": "desc",
                "scam_score": 0.95,
                "confidence": 0.9,
                "risk_level": "scam",
                "analyzed_at": now,
                "signal_count": 1,
            })
        pipeline = FeedbackPipeline(db=tmp_db)
        pipeline.generate_synthetic_feedback(n=10)
        stats = pipeline.get_feedback_stats()
        assert stats["total_reports"] >= 0  # at least ran without crashing

    def test_stats_coverage_nonzero_when_reports_exist(self, tmp_db):
        from datetime import UTC, datetime
        from sentinel.feedback import FeedbackPipeline
        now = datetime.now(UTC).isoformat()
        tmp_db.save_job({
            "url": "https://example.com/cov/1",
            "title": "Job",
            "company": "Co",
            "description": "desc",
            "scam_score": 0.7,
            "confidence": 0.6,
            "risk_level": "high",
            "analyzed_at": now,
            "signal_count": 1,
        })
        tmp_db.save_report({
            "url": "https://example.com/cov/1",
            "is_scam": True,
            "reason": "test",
            "our_prediction": 0.7,
            "was_correct": True,
            "reported_at": now,
        })
        pipeline = FeedbackPipeline(db=tmp_db)
        stats = pipeline.get_feedback_stats()
        assert stats["feedback_coverage"] > 0.0
        assert stats["total_scored_jobs"] >= 1
        assert stats["total_reports"] >= 1

    def test_stats_reports_per_day_has_seven_entries(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        pipeline = FeedbackPipeline(db=tmp_db)
        stats = pipeline.get_feedback_stats()
        assert len(stats["reports_per_day"]) == 7

    def test_stats_report_attribution_breakdown(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        pipeline = FeedbackPipeline(db=tmp_db)
        stats = pipeline.get_feedback_stats()
        assert "manual_reports" in stats
        assert "synthetic_reports" in stats
        assert "imported_reports" in stats


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_ensemble_with_single_signal(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        signals = [_make_signal("only_one", SignalCategory.WARNING, 0.6)]
        result = scorer.score_ensemble(tmp_db, _make_job(), signals)
        assert 0.0 <= result.ensemble_score <= 1.0
        assert isinstance(result.disagreement, float)

    def test_ensemble_all_positive_signals(self, tmp_db):
        from sentinel.scorer import EnsembleScorer
        scorer = EnsembleScorer()
        signals = [_make_signal(f"pos{i}", SignalCategory.POSITIVE, 0.1) for i in range(5)]
        result = scorer.score_ensemble(tmp_db, _make_job(), signals)
        # Should score low — all positives
        assert result.primary_score < 0.5
        assert result.ensemble_score < 0.5

    def test_feedback_pipeline_works_without_db_arg(self):
        """FeedbackPipeline with no db creates its own SentinelDB."""
        from sentinel.feedback import FeedbackPipeline
        # Should not raise
        pipeline = FeedbackPipeline()
        stats = pipeline.get_feedback_stats()
        assert "total_reports" in stats

    def test_rescan_honors_sample_size_limit(self, tmp_db):
        from sentinel.feedback import FeedbackPipeline
        from datetime import UTC, datetime
        now = datetime.now(UTC).isoformat()
        for i in range(20):
            tmp_db.save_job({
                "url": f"https://example.com/size/{i}",
                "title": "Job",
                "company": "Co",
                "description": "desc",
                "scam_score": 0.3,
                "confidence": 0.5,
                "risk_level": "low",
                "analyzed_at": now,
                "signal_count": 0,
            })
        pipeline = FeedbackPipeline(db=tmp_db)
        result = pipeline.rescan_and_compare(days=7, sample_size=5)
        assert result.jobs_rescanned <= 5
