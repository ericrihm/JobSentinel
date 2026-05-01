"""Tests for the ingestion pipeline."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from sentinel.db import SentinelDB
from sentinel.ingest import IngestionPipeline, IngestionRun
from sentinel.models import JobPosting, RiskLevel, ScamSignal, SignalCategory, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(url: str, title: str = "Test Job", company: str = "TestCo") -> JobPosting:
    return JobPosting(
        url=url,
        title=title,
        company=company,
        location="Remote",
        description="A legitimate software engineering role with clear requirements.",
        source="test",
    )


def _make_result(job: JobPosting, score: float = 0.2) -> ValidationResult:
    risk = RiskLevel.LOW
    if score >= 0.8:
        risk = RiskLevel.SCAM
    elif score >= 0.6:
        risk = RiskLevel.HIGH
    elif score >= 0.4:
        risk = RiskLevel.SUSPICIOUS
    return ValidationResult(
        job=job,
        scam_score=score,
        confidence=0.8,
        risk_level=risk,
        signals=[
            ScamSignal(name="test_signal", category=SignalCategory.WARNING, detail="test"),
        ],
    )


def _fake_source_fetcher(jobs: list[JobPosting]):
    """Return a callable that mimics a source fetcher."""
    def fetcher(query: str, location: str = "", limit: int = 25):
        return jobs[:limit]
    return fetcher


def _failing_source_fetcher(query: str, location: str = "", limit: int = 25):
    """Source fetcher that always raises."""
    raise ConnectionError("Network unavailable")


# ---------------------------------------------------------------------------
# 1. Pipeline initializes and seeds patterns on empty DB
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_initializes_with_seeded_patterns(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)
        patterns = pipeline.db.get_patterns(status="active")
        assert len(patterns) > 0, "Default patterns should be seeded on init"
        pipeline.db.close()

    def test_does_not_reseed_on_second_init(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        p1 = IngestionPipeline(db_path=db_path)
        count_1 = len(p1.db.get_patterns(status="active"))
        p1.db.close()

        p2 = IngestionPipeline(db_path=db_path)
        count_2 = len(p2.db.get_patterns(status="active"))
        p2.db.close()

        assert count_1 == count_2, "Patterns should not be duplicated on second init"


# ---------------------------------------------------------------------------
# 2. Pipeline deduplicates jobs already in DB
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_skips_existing_jobs(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        # Pre-insert a job
        pipeline.db.save_job({
            "url": "https://linkedin.com/jobs/view/111",
            "title": "Existing Job",
            "company": "OldCo",
        })

        jobs = [
            _make_job("https://linkedin.com/jobs/view/111", "Existing Job"),
            _make_job("https://linkedin.com/jobs/view/222", "New Job"),
        ]

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze:
            mock_fetchers.return_value = {"test": _fake_source_fetcher(jobs)}
            mock_analyze.side_effect = lambda job, use_ai=False: _make_result(job)

            result = pipeline.run(query="engineer", sources=["test"])

        assert result.jobs_fetched == 2
        assert result.jobs_new == 1  # only the new one
        assert result.jobs_scored == 1
        pipeline.db.close()

    def test_jobs_without_url_are_never_deduped(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        jobs = [_make_job("", "No URL Job")]

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze:
            mock_fetchers.return_value = {"test": _fake_source_fetcher(jobs)}
            mock_analyze.side_effect = lambda job, use_ai=False: _make_result(job)

            result = pipeline.run(query="any")

        assert result.jobs_new == 1
        pipeline.db.close()


# ---------------------------------------------------------------------------
# 3. Pipeline handles source fetch failures gracefully
# ---------------------------------------------------------------------------

class TestSourceFailures:
    def test_source_error_captured_in_errors(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers:
            mock_fetchers.return_value = {"broken": _failing_source_fetcher}
            result = pipeline.run(query="engineer")

        assert result.jobs_fetched == 0
        assert len(result.errors) > 0
        assert "Network unavailable" in result.errors[0]
        pipeline.db.close()

    def test_one_source_failure_doesnt_block_others(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)
        good_jobs = [_make_job("https://example.com/jobs/1")]

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze:
            mock_fetchers.return_value = {
                "broken": _failing_source_fetcher,
                "good": _fake_source_fetcher(good_jobs),
            }
            mock_analyze.side_effect = lambda job, use_ai=False: _make_result(job)

            result = pipeline.run(query="engineer", throttle_seconds=0)

        assert result.jobs_fetched == 1
        assert result.jobs_scored == 1
        assert len(result.errors) == 1  # only the broken source
        pipeline.db.close()

    def test_scoring_error_captured_gracefully(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)
        jobs = [_make_job("https://example.com/jobs/score-fail")]

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze:
            mock_fetchers.return_value = {"test": _fake_source_fetcher(jobs)}
            mock_analyze.side_effect = RuntimeError("Analyzer crashed")

            result = pipeline.run(query="engineer")

        assert result.jobs_fetched == 1
        assert result.jobs_scored == 0
        assert any("Scoring failed" in e for e in result.errors)
        pipeline.db.close()


# ---------------------------------------------------------------------------
# 4. Pipeline records ingestion run to DB
# ---------------------------------------------------------------------------

class TestIngestionRunPersistence:
    def test_run_saved_to_ingestion_runs_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)
        jobs = [_make_job("https://example.com/jobs/persist-1")]

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze:
            mock_fetchers.return_value = {"test": _fake_source_fetcher(jobs)}
            mock_analyze.side_effect = lambda job, use_ai=False: _make_result(job)

            result = pipeline.run(query="data scientist", location="NYC")

        history = pipeline.db.get_ingestion_history(limit=10)
        assert len(history) == 1
        record = history[0]
        assert record["run_id"] == result.run_id
        assert record["query"] == "data scientist"
        assert record["location"] == "NYC"
        assert record["jobs_fetched"] == 1
        assert record["jobs_new"] == 1
        assert record["jobs_scored"] == 1
        pipeline.db.close()

    def test_ingestion_run_has_valid_timestamps(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers:
            mock_fetchers.return_value = {}
            result = pipeline.run(query="noop")

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.started_at <= result.completed_at
        pipeline.db.close()


# ---------------------------------------------------------------------------
# 5. auto_ingest runs multiple queries
# ---------------------------------------------------------------------------

class TestAutoIngest:
    def test_auto_ingest_runs_all_queries(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        def make_jobs(query, **kwargs):
            return [_make_job(f"https://example.com/jobs/{query}/1", title=f"Job for {query}")]

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze:
            mock_fetchers.return_value = {"test": make_jobs}
            mock_analyze.side_effect = lambda job, use_ai=False: _make_result(job)

            runs = pipeline.auto_ingest(
                queries=["python dev", "data engineer", "ml engineer"],
                location="Remote",
                run_flywheel=False,
            )

        assert len(runs) == 3
        assert all(isinstance(r, IngestionRun) for r in runs)
        assert runs[0].query == "python dev"
        assert runs[1].query == "data engineer"
        assert runs[2].query == "ml engineer"
        pipeline.db.close()

    def test_auto_ingest_continues_on_query_failure(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        call_count = 0

        def flaky_fetcher(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First query explodes")
            return [_make_job(f"https://example.com/jobs/{call_count}")]

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze:
            mock_fetchers.return_value = {"test": flaky_fetcher}
            mock_analyze.side_effect = lambda job, use_ai=False: _make_result(job)

            runs = pipeline.auto_ingest(
                queries=["q1", "q2"],
                run_flywheel=False,
            )

        # First query errors (caught in run()), second succeeds
        # Both should produce IngestionRun objects
        assert len(runs) == 2
        pipeline.db.close()


# ---------------------------------------------------------------------------
# 6. Flywheel cycle runs after ingestion
# ---------------------------------------------------------------------------

class TestFlywheelIntegration:
    def test_flywheel_cycle_runs_after_auto_ingest(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze, \
             patch("sentinel.flywheel.DetectionFlywheel") as mock_fw_cls:
            mock_fetchers.return_value = {}
            mock_analyze.return_value = _make_result(_make_job("x"))
            mock_fw_instance = MagicMock()
            mock_fw_instance.run_cycle.return_value = {"cycle_number": 1}
            mock_fw_cls.return_value = mock_fw_instance

            pipeline.auto_ingest(queries=["test"], run_flywheel=True)

        mock_fw_instance.run_cycle.assert_called_once()

    def test_flywheel_cycle_skipped_when_disabled(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.flywheel.DetectionFlywheel") as mock_fw_cls:
            mock_fetchers.return_value = {}
            mock_fw_instance = MagicMock()
            mock_fw_cls.return_value = mock_fw_instance

            pipeline.auto_ingest(queries=["test"], run_flywheel=False)

        mock_fw_instance.run_cycle.assert_not_called()

    def test_run_flywheel_cycle_returns_metrics(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        with patch("sentinel.flywheel.DetectionFlywheel") as mock_fw_cls:
            mock_fw_instance = MagicMock()
            mock_fw_instance.run_cycle.return_value = {
                "cycle_number": 42,
                "precision": 0.95,
            }
            mock_fw_cls.return_value = mock_fw_instance

            metrics = pipeline.run_flywheel_cycle()

        assert metrics["cycle_number"] == 42
        assert metrics["precision"] == 0.95
        pipeline.db.close()


# ---------------------------------------------------------------------------
# 7. High risk counting
# ---------------------------------------------------------------------------

class TestHighRiskCounting:
    def test_high_risk_jobs_counted(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        jobs = [
            _make_job("https://example.com/jobs/safe", "Safe Job"),
            _make_job("https://example.com/jobs/risky", "Risky Job"),
            _make_job("https://example.com/jobs/scam", "Scam Job"),
        ]

        scores = [0.1, 0.7, 0.9]

        with patch.object(pipeline, "_get_source_fetchers") as mock_fetchers, \
             patch("sentinel.analyzer.analyze_job") as mock_analyze:
            mock_fetchers.return_value = {"test": _fake_source_fetcher(jobs)}

            call_idx = 0
            def score_by_index(job, use_ai=False):
                nonlocal call_idx
                result = _make_result(job, scores[call_idx])
                call_idx += 1
                return result

            mock_analyze.side_effect = score_by_index
            result = pipeline.run(query="engineer")

        assert result.high_risk_count == 2  # 0.7 and 0.9 both >= 0.6
        pipeline.db.close()


# ---------------------------------------------------------------------------
# 8. IngestionRun dataclass
# ---------------------------------------------------------------------------

class TestIngestionRunDataclass:
    def test_to_dict_round_trip(self):
        run = IngestionRun(
            run_id="abc-123",
            started_at="2026-04-30T00:00:00+00:00",
            completed_at="2026-04-30T00:01:00+00:00",
            sources_queried=["linkedin", "indeed"],
            query="python developer",
            location="NYC",
            jobs_fetched=50,
            jobs_new=30,
            jobs_scored=30,
            high_risk_count=5,
            errors=["Source 'indeed' timed out"],
        )
        d = run.to_dict()
        assert d["run_id"] == "abc-123"
        assert d["query"] == "python developer"
        assert d["jobs_fetched"] == 50
        assert d["high_risk_count"] == 5
        assert len(d["errors"]) == 1


# ---------------------------------------------------------------------------
# 9. Source fetcher fallback (no sources module)
# ---------------------------------------------------------------------------

class TestSourceFetcherFallback:
    def test_falls_back_to_linkedin_when_no_sources_module(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        errors: list[str] = []
        # Simulate sentinel.sources being unavailable by making its import raise
        import sentinel as _sentinel_pkg
        original = getattr(_sentinel_pkg, "sources", None)
        if hasattr(_sentinel_pkg, "sources"):
            delattr(_sentinel_pkg, "sources")
        try:
            with patch.dict(sys.modules, {"sentinel.sources": None}):
                fetchers = pipeline._get_source_fetchers(None, errors)
        finally:
            if original is not None:
                _sentinel_pkg.sources = original

        assert "linkedin" in fetchers
        pipeline.db.close()

    def test_unavailable_requested_source_recorded_as_error(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        pipeline = IngestionPipeline(db_path=db_path)

        # Mock a sources module with limited availability
        mock_sources = types.ModuleType("sentinel.sources")
        mock_sources.AVAILABLE_SOURCES = {"linkedin": lambda **kw: []}

        errors: list[str] = []
        with patch.dict(sys.modules, {"sentinel.sources": mock_sources}):
            fetchers = pipeline._get_source_fetchers(["linkedin", "nonexistent"], errors)

        assert "linkedin" in fetchers
        assert "nonexistent" not in fetchers
        assert any("nonexistent" in e for e in errors)
        pipeline.db.close()


# ---------------------------------------------------------------------------
# 10. DB methods for ingestion runs
# ---------------------------------------------------------------------------

class TestDBIngestionMethods:
    def test_get_job_by_url_returns_none_for_missing(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        assert db.get_job_by_url("https://nonexistent.com/job/1") is None
        db.close()

    def test_get_job_by_url_returns_existing(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        db.save_job({"url": "https://example.com/job/1", "title": "Test"})
        result = db.get_job_by_url("https://example.com/job/1")
        assert result is not None
        assert result["title"] == "Test"
        db.close()

    def test_save_and_get_ingestion_history(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        db.save_ingestion_run({
            "run_id": "run-001",
            "started_at": "2026-04-30T00:00:00+00:00",
            "completed_at": "2026-04-30T00:01:00+00:00",
            "sources_queried": ["linkedin"],
            "query": "engineer",
            "location": "Remote",
            "jobs_fetched": 10,
            "jobs_new": 8,
            "jobs_scored": 8,
            "high_risk_count": 2,
            "errors": [],
        })
        history = db.get_ingestion_history(limit=10)
        assert len(history) == 1
        assert history[0]["run_id"] == "run-001"
        assert history[0]["jobs_fetched"] == 10
        # sources and errors should be deserialized to lists
        assert isinstance(history[0]["sources"], list)
        assert isinstance(history[0]["errors"], list)
        db.close()

    def test_ingestion_history_ordering(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        for i in range(3):
            db.save_ingestion_run({
                "run_id": f"run-{i:03d}",
                "started_at": f"2026-04-{28 + i}T00:00:00+00:00",
                "query": f"query-{i}",
            })
        history = db.get_ingestion_history(limit=2)
        assert len(history) == 2
        # Most recent first
        assert history[0]["run_id"] == "run-002"
        assert history[1]["run_id"] == "run-001"
        db.close()
