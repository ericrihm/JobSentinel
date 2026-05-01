"""Tests for source quality tracking and warm-start optimisations.

Coverage:
  1.  source_stats table – table exists after DB init
  2.  upsert_source_stats – insert creates a row
  3.  upsert_source_stats – update increments counts cumulatively
  4.  get_source_stats – returns all rows
  5.  get_source_stats – empty DB returns []
  6.  get_best_sources – ranked by yield rate
  7.  get_best_sources – sources with 0 ingested jobs are excluded
  8.  get_best_sources – respects n limit
  9.  ingest pipeline updates source_stats after scoring
  10. InnovationEngine._evaluate_source_quality – no stats returns failure
  11. InnovationEngine._evaluate_source_quality – returns success with stats
  12. InnovationEngine._evaluate_source_quality – identifies worst source
  13. InnovationEngine STRATEGIES includes 'source_quality' arm
  14. Warm-start daemon query prioritisation
  15. Warm-start daemon skips circuit-broken sources
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from sentinel.db import SentinelDB
from sentinel.innovation import InnovationEngine, ImprovementResult
from sentinel.models import JobPosting, RiskLevel, ScamSignal, SignalCategory, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> SentinelDB:
    return SentinelDB(path=str(tmp_path / "test.db"))


def _make_job(url: str, title: str = "Test Job", source: str = "remoteok") -> JobPosting:
    return JobPosting(
        url=url,
        title=title,
        company="TestCo",
        location="Remote",
        description="A test job posting.",
        source=source,
    )


def _make_result(job: JobPosting, score: float = 0.2) -> ValidationResult:
    if score >= 0.8:
        risk = RiskLevel.SCAM
    elif score >= 0.6:
        risk = RiskLevel.HIGH
    elif score >= 0.4:
        risk = RiskLevel.SUSPICIOUS
    else:
        risk = RiskLevel.LOW
    return ValidationResult(
        job=job,
        scam_score=score,
        confidence=0.8,
        risk_level=risk,
        signals=[ScamSignal(name="test_signal", category=SignalCategory.WARNING, detail="test")],
    )


# ---------------------------------------------------------------------------
# 1. source_stats table exists after DB init
# ---------------------------------------------------------------------------

class TestSourceStatsSchema:
    def test_table_exists(self, tmp_path):
        db = _make_db(tmp_path)
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='source_stats'"
        ).fetchall()
        assert len(rows) == 1, "source_stats table should exist after DB init"
        db.close()

    def test_table_has_expected_columns(self, tmp_path):
        db = _make_db(tmp_path)
        info = db.conn.execute("PRAGMA table_info(source_stats)").fetchall()
        cols = {row["name"] for row in info}
        assert cols >= {"source", "jobs_ingested", "scams_detected", "avg_score", "last_updated"}
        db.close()


# ---------------------------------------------------------------------------
# 2. upsert_source_stats – insert creates a row
# ---------------------------------------------------------------------------

class TestUpsertSourceStats:
    def test_insert_creates_row(self, tmp_path):
        db = _make_db(tmp_path)
        db.upsert_source_stats("remoteok", jobs_ingested=10, scams_detected=3, avg_score=0.45)
        rows = db.get_source_stats()
        assert len(rows) == 1
        row = rows[0]
        assert row["source"] == "remoteok"
        assert row["jobs_ingested"] == 10
        assert row["scams_detected"] == 3
        assert abs(row["avg_score"] - 0.45) < 0.001
        assert row["last_updated"] is not None
        db.close()

    # 3. upsert increments counts cumulatively
    def test_update_accumulates_counts(self, tmp_path):
        db = _make_db(tmp_path)
        db.upsert_source_stats("adzuna", jobs_ingested=5, scams_detected=1, avg_score=0.3)
        db.upsert_source_stats("adzuna", jobs_ingested=8, scams_detected=2, avg_score=0.5)
        rows = db.get_source_stats()
        assert len(rows) == 1
        row = rows[0]
        assert row["jobs_ingested"] == 13, "jobs_ingested should accumulate"
        assert row["scams_detected"] == 3, "scams_detected should accumulate"
        # avg_score should be the latest value (replaced, not averaged)
        assert abs(row["avg_score"] - 0.5) < 0.001
        db.close()

    def test_multiple_sources_stored_separately(self, tmp_path):
        db = _make_db(tmp_path)
        db.upsert_source_stats("remoteok", jobs_ingested=10, scams_detected=2, avg_score=0.3)
        db.upsert_source_stats("themuse", jobs_ingested=20, scams_detected=1, avg_score=0.2)
        rows = db.get_source_stats()
        sources = {r["source"] for r in rows}
        assert "remoteok" in sources
        assert "themuse" in sources
        db.close()


# ---------------------------------------------------------------------------
# 4 & 5. get_source_stats
# ---------------------------------------------------------------------------

class TestGetSourceStats:
    def test_returns_all_rows(self, tmp_path):
        db = _make_db(tmp_path)
        for src in ("remoteok", "adzuna", "remotive"):
            db.upsert_source_stats(src, jobs_ingested=10, scams_detected=1, avg_score=0.2)
        rows = db.get_source_stats()
        assert len(rows) == 3
        db.close()

    def test_empty_db_returns_empty_list(self, tmp_path):
        db = _make_db(tmp_path)
        rows = db.get_source_stats()
        assert rows == []
        db.close()


# ---------------------------------------------------------------------------
# 6–8. get_best_sources
# ---------------------------------------------------------------------------

class TestGetBestSources:
    def test_ranked_by_yield_rate(self, tmp_path):
        db = _make_db(tmp_path)
        # remoteok: 5/10 = 50%  ← best
        # adzuna:   1/20 = 5%
        # themuse:  2/10 = 20%
        db.upsert_source_stats("remoteok", jobs_ingested=10, scams_detected=5, avg_score=0.7)
        db.upsert_source_stats("adzuna",   jobs_ingested=20, scams_detected=1, avg_score=0.2)
        db.upsert_source_stats("themuse",  jobs_ingested=10, scams_detected=2, avg_score=0.3)
        best = db.get_best_sources(n=3)
        assert best[0] == "remoteok", "Highest yield source should be first"
        assert best[-1] == "adzuna", "Lowest yield source should be last"
        db.close()

    def test_excludes_zero_ingested(self, tmp_path):
        db = _make_db(tmp_path)
        db.upsert_source_stats("remoteok", jobs_ingested=10, scams_detected=2, avg_score=0.3)
        # Manually insert a row with 0 jobs_ingested to test exclusion
        db.conn.execute(
            "INSERT INTO source_stats (source, jobs_ingested, scams_detected, avg_score, last_updated)"
            " VALUES ('ghost', 0, 0, 0.0, '2026-01-01')"
        )
        db.conn.commit()
        best = db.get_best_sources(n=10)
        assert "ghost" not in best
        db.close()

    def test_respects_n_limit(self, tmp_path):
        db = _make_db(tmp_path)
        for i in range(5):
            db.upsert_source_stats(f"source_{i}", jobs_ingested=10, scams_detected=i, avg_score=0.1 * i)
        best = db.get_best_sources(n=2)
        assert len(best) == 2
        db.close()


# ---------------------------------------------------------------------------
# 9. Ingest pipeline updates source_stats after scoring
# ---------------------------------------------------------------------------

class TestIngestUpdatesSourceStats:
    def test_source_stats_updated_after_pipeline_run(self, tmp_path):
        db_path = str(tmp_path / "ingest.db")

        from sentinel.ingest import IngestionPipeline

        # Two fake jobs from 'remoteok'
        jobs = [
            _make_job("https://example.com/job/1", source="remoteok"),
            _make_job("https://example.com/job/2", source="remoteok"),
        ]

        # Patch source fetchers and analyzer so no network/AI is needed
        fake_fetchers = {"remoteok": lambda query, location="", limit=25: jobs}

        with (
            patch.object(IngestionPipeline, "_get_source_fetchers", return_value=fake_fetchers),
            patch("sentinel.analyzer.analyze_job", side_effect=[
                _make_result(jobs[0], score=0.7),  # high-risk → scam
                _make_result(jobs[1], score=0.2),  # low-risk
            ]),
        ):
            pipeline = IngestionPipeline(db_path=db_path)
            pipeline.run(query="test")

        db = SentinelDB(path=db_path)
        rows = db.get_source_stats()
        db.close()

        assert len(rows) >= 1
        remoteok_row = next((r for r in rows if r["source"] == "remoteok"), None)
        assert remoteok_row is not None, "remoteok should have a stats row"
        assert remoteok_row["jobs_ingested"] == 2
        assert remoteok_row["scams_detected"] == 1


# ---------------------------------------------------------------------------
# 10–12. InnovationEngine._evaluate_source_quality
# ---------------------------------------------------------------------------

class TestSourceQualityStrategy:
    def test_no_stats_returns_failure(self, tmp_path):
        db = _make_db(tmp_path)
        engine = InnovationEngine(db=db)
        result = engine._evaluate_source_quality(baseline=0.8)
        assert result.strategy == "source_quality"
        assert result.success is False
        assert "No source stats" in result.detail
        db.close()

    def test_returns_success_with_stats(self, tmp_path):
        db = _make_db(tmp_path)
        db.upsert_source_stats("remoteok", jobs_ingested=20, scams_detected=10, avg_score=0.6)
        db.upsert_source_stats("adzuna", jobs_ingested=20, scams_detected=1, avg_score=0.2)
        engine = InnovationEngine(db=db)
        result = engine._evaluate_source_quality(baseline=0.8)
        assert result.strategy == "source_quality"
        assert result.success is True
        db.close()

    def test_identifies_worst_source(self, tmp_path):
        db = _make_db(tmp_path)
        # adzuna has the lowest scam yield (1/100 = 1%)
        db.upsert_source_stats("remoteok",  jobs_ingested=10, scams_detected=5,  avg_score=0.6)
        db.upsert_source_stats("adzuna",    jobs_ingested=100, scams_detected=1, avg_score=0.1)
        db.upsert_source_stats("remotive",  jobs_ingested=10,  scams_detected=3, avg_score=0.4)
        engine = InnovationEngine(db=db)
        # Run multiple times to smooth out Thompson Sampling randomness
        worst_mentioned = 0
        for _ in range(20):
            result = engine._evaluate_source_quality(baseline=0.8)
            if "adzuna" in result.detail:
                worst_mentioned += 1
        # adzuna should appear in the majority of runs as the worst source
        assert worst_mentioned >= 10, (
            f"Expected adzuna to be identified as worst in most runs, got {worst_mentioned}/20"
        )
        db.close()


# ---------------------------------------------------------------------------
# 13. InnovationEngine STRATEGIES includes 'source_quality' arm
# ---------------------------------------------------------------------------

class TestSourceQualityArmRegistration:
    def test_source_quality_arm_in_strategies(self, tmp_path):
        db = _make_db(tmp_path)
        engine = InnovationEngine(db=db)
        names = [arm.name for arm in engine.STRATEGIES]
        assert "source_quality" in names, "source_quality arm must be registered in STRATEGIES"
        db.close()

    def test_source_quality_dispatches_to_method(self, tmp_path):
        db = _make_db(tmp_path)
        engine = InnovationEngine(db=db)
        # Calling _execute_strategy with the source_quality arm should not fall through to _noop
        sq_arm = next(a for a in engine.STRATEGIES if a.name == "source_quality")
        result = engine._execute_strategy(sq_arm, baseline_precision=0.8)
        assert result.strategy == "source_quality", (
            "Dispatch should route to _evaluate_source_quality, not _noop"
        )
        db.close()


# ---------------------------------------------------------------------------
# 14. Warm-start daemon query prioritisation
# ---------------------------------------------------------------------------

class TestDaemonWarmStart:
    def test_prioritized_queries_uses_best_sources(self, tmp_path):
        from sentinel.daemon import SentinelDaemon

        db = _make_db(tmp_path)
        # Populate source_stats so remotive has the highest scam yield
        db.upsert_source_stats("remotive",  jobs_ingested=10, scams_detected=9, avg_score=0.9)
        db.upsert_source_stats("remoteok",  jobs_ingested=10, scams_detected=1, avg_score=0.2)

        daemon = SentinelDaemon(
            queries=["remoteok", "remotive", "themuse"],
            db_path=str(tmp_path / "daemon.db"),
        )
        # Manually seed source_stats in the daemon's DB
        daemon_db = SentinelDB(path=str(tmp_path / "daemon.db"))
        daemon_db.upsert_source_stats("remotive", jobs_ingested=10, scams_detected=9, avg_score=0.9)
        daemon_db.upsert_source_stats("remoteok", jobs_ingested=10, scams_detected=1, avg_score=0.2)

        prioritized = daemon._prioritized_queries(daemon_db)
        daemon_db.close()

        # remotive (highest yield) should come before remoteok (lower yield)
        assert prioritized.index("remotive") < prioritized.index("remoteok"), (
            "Higher-yield source should sort before lower-yield source"
        )

    def test_prioritized_queries_falls_back_on_empty_stats(self, tmp_path):
        from sentinel.daemon import SentinelDaemon

        db = _make_db(tmp_path)  # no source stats
        queries = ["a", "b", "c"]
        daemon = SentinelDaemon(queries=queries)
        result = daemon._prioritized_queries(db)
        assert sorted(result) == sorted(queries), "All queries should be returned"
        db.close()

    # 15. Warm-start daemon skips circuit-broken sources
    def test_active_source_names_excludes_broken(self, tmp_path):
        from sentinel.daemon import SentinelDaemon

        mock_throttler = MagicMock()
        mock_throttler.get_stats.return_value = {
            "remoteok.com": {"circuit_broken": True, "total_requests": 10, "total_errors": 5, "consecutive_errors": 5},
            "api.adzuna.com": {"circuit_broken": False, "total_requests": 5, "total_errors": 0, "consecutive_errors": 0},
        }

        daemon = SentinelDaemon()
        with patch("sentinel.daemon.SentinelDaemon._active_source_names",
                   wraps=daemon._active_source_names):
            with patch("sentinel.sources.get_throttler", return_value=mock_throttler):
                broken = daemon._active_source_names()

        assert "remoteok" in broken, "Circuit-broken remoteok should be in the broken set"
        assert "adzuna" not in broken, "Non-broken adzuna should not be in the broken set"
