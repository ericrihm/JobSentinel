"""Tests for sentinel/db.py — SentinelDB persistence layer."""

import json
import pytest
from sentinel.db import SentinelDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """In-memory-like SentinelDB backed by a temp file; closed after test."""
    db_path = str(tmp_path / "sentinel_test.db")
    d = SentinelDB(path=db_path)
    yield d
    d.close()


def _job(url="https://example.com/job/1", **kwargs) -> dict:
    """Build a minimal valid job dict."""
    base = {
        "url": url,
        "title": "Software Engineer",
        "company": "Acme Corp",
        "location": "Remote",
        "description": "Write code every day.",
        "salary_min": 80_000.0,
        "salary_max": 120_000.0,
        "scam_score": 0.1,
        "confidence": 0.9,
        "risk_level": "low",
        "signal_count": 0,
        "signals": [],
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Schema / initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_tables_created(self, db):
        """All core tables must exist after __init__."""
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for expected in (
            "jobs", "patterns", "reports", "companies",
            "ingestion_runs", "flywheel_metrics", "salary_benchmarks",
            "scam_entities", "shadow_runs", "signal_rate_history",
        ):
            assert expected in tables, f"missing table: {expected}"

    def test_salary_benchmarks_seeded(self, db):
        """Salary benchmarks are pre-populated on first open."""
        count = db.conn.execute("SELECT COUNT(*) FROM salary_benchmarks").fetchone()[0]
        assert count > 0

    def test_scam_entities_seeded(self, db):
        """Scam entity seeds are pre-populated on first open."""
        count = db.conn.execute("SELECT COUNT(*) FROM scam_entities").fetchone()[0]
        assert count > 0

    def test_wal_mode(self, db):
        """WAL journal mode should be active."""
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_context_manager(self, tmp_path):
        """SentinelDB works as a context manager and closes cleanly."""
        db_path = str(tmp_path / "cm_test.db")
        with SentinelDB(path=db_path) as d:
            d.save_job(_job())
        # After close, further calls raise sqlite3.ProgrammingError
        import sqlite3
        with pytest.raises(Exception):
            d.conn.execute("SELECT 1")


# ---------------------------------------------------------------------------
# Jobs — save / get
# ---------------------------------------------------------------------------

class TestSaveGetJob:
    def test_save_and_retrieve(self, db):
        job = _job()
        db.save_job(job)
        result = db.get_job(job["url"])
        assert result is not None
        assert result["title"] == "Software Engineer"
        assert result["company"] == "Acme Corp"

    def test_signals_roundtrip_list(self, db):
        """signals stored as a list are deserialised back to a list."""
        job = _job(signals=[{"name": "upfront_payment", "weight": 0.9}])
        db.save_job(job)
        result = db.get_job(job["url"])
        assert isinstance(result["signals"], list)
        assert result["signals"][0]["name"] == "upfront_payment"

    def test_signals_roundtrip_json_string(self, db):
        """signals_json stored as a JSON string are also deserialised."""
        job = _job()
        job.pop("signals")
        job["signals_json"] = json.dumps([{"name": "too_good_to_be_true"}])
        db.save_job(job)
        result = db.get_job(job["url"])
        assert result["signals"][0]["name"] == "too_good_to_be_true"

    def test_upsert_updates_existing(self, db):
        """Saving the same URL twice updates the existing row."""
        db.save_job(_job(scam_score=0.1))
        db.save_job(_job(scam_score=0.95, title="Updated Title"))
        result = db.get_job("https://example.com/job/1")
        assert result["scam_score"] == pytest.approx(0.95)
        assert result["title"] == "Updated Title"
        count = db.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert count == 1

    def test_get_missing_job_returns_none(self, db):
        assert db.get_job("https://nonexistent.example.com") is None

    def test_get_job_by_url(self, db):
        db.save_job(_job())
        row = db.get_job_by_url("https://example.com/job/1")
        assert row is not None
        assert row["company"] == "Acme Corp"

    def test_get_job_by_url_missing(self, db):
        assert db.get_job_by_url("https://missing.example.com") is None

    def test_save_job_minimal_fields(self, db):
        """Saving a job with only url should not raise."""
        db.save_job({"url": "https://minimal.example.com"})
        result = db.get_job("https://minimal.example.com")
        assert result is not None
        assert result["scam_score"] == pytest.approx(0.0)

    def test_confidence_none_stored(self, db):
        """confidence=None is persisted and returned as None."""
        job = _job()
        job["confidence"] = None
        db.save_job(job)
        result = db.get_job(job["url"])
        assert result["confidence"] is None


# ---------------------------------------------------------------------------
# Jobs — search
# ---------------------------------------------------------------------------

class TestSearchJobs:
    def test_fts_search_by_title(self, db):
        db.save_job(_job(url="https://example.com/1", title="Python Developer"))
        db.save_job(_job(url="https://example.com/2", title="Java Developer"))
        results = db.search_jobs("Python")
        assert len(results) == 1
        assert results[0]["title"] == "Python Developer"

    def test_fts_no_results(self, db):
        db.save_job(_job())
        results = db.search_jobs("xyzzy_nonexistent_keyword")
        assert results == []

    def test_fts_special_chars_dont_raise(self, db):
        """FTS query escaping should prevent crashes on special characters."""
        db.save_job(_job())
        # These characters could break FTS5 if not escaped
        for q in ['"quoted"', "hyphen-word", "colon:val", "star*"]:
            results = db.search_jobs(q)
            assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Jobs for review
# ---------------------------------------------------------------------------

class TestGetJobsForReview:
    def test_returns_high_score_low_confidence(self, db):
        db.save_job(_job(url="u1", scam_score=0.8, confidence=0.3))
        db.save_job(_job(url="u2", scam_score=0.3, confidence=0.3))
        results = db.get_jobs_for_review()
        urls = [r["url"] for r in results]
        assert "u1" in urls
        assert "u2" not in urls

    def test_excludes_null_confidence(self, db):
        job = _job(url="u_null", scam_score=0.9)
        job["confidence"] = None
        db.save_job(job)
        results = db.get_jobs_for_review()
        assert all(r["url"] != "u_null" for r in results)

    def test_ordered_by_scam_score_desc(self, db):
        db.save_job(_job(url="u_low",  scam_score=0.6, confidence=0.2))
        db.save_job(_job(url="u_high", scam_score=0.9, confidence=0.2))
        results = db.get_jobs_for_review()
        assert results[0]["url"] == "u_high"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class TestReports:
    def test_save_and_retrieve_report(self, db):
        db.save_job(_job(url="https://r.example.com"))
        db.save_report({
            "url": "https://r.example.com",
            "is_scam": True,
            "reason": "Upfront fee demanded",
            "our_prediction": 0.8,
            "was_correct": True,
        })
        reports = db.get_reports()
        assert len(reports) == 1
        assert reports[0]["is_scam"] == 1

    def test_report_marks_job_as_reported(self, db):
        db.save_job(_job(url="https://r2.example.com"))
        db.save_report({"url": "https://r2.example.com", "is_scam": True})
        job = db.get_job("https://r2.example.com")
        assert job["user_reported"] == 1
        assert job["user_verdict"] == "scam"

    def test_report_legitimate_verdict(self, db):
        db.save_job(_job(url="https://legit.example.com"))
        db.save_report({"url": "https://legit.example.com", "is_scam": False})
        job = db.get_job("https://legit.example.com")
        assert job["user_verdict"] == "legitimate"


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

class TestPatterns:
    def _pattern(self, pid="p1", **kwargs) -> dict:
        base = {
            "pattern_id": pid,
            "name": "Test Pattern",
            "description": "Detects test scams",
            "category": "red_flag",
            "regex": r"guaranteed income",
            "keywords": ["guaranteed", "income"],
            "alpha": 1.0,
            "beta": 1.0,
            "status": "active",
        }
        base.update(kwargs)
        return base

    def test_save_and_get_pattern(self, db):
        db.save_pattern(self._pattern())
        patterns = db.get_patterns()
        assert any(p["pattern_id"] == "p1" for p in patterns)

    def test_get_patterns_filters_by_status(self, db):
        db.save_pattern(self._pattern(pid="active1", status="active"))
        db.save_pattern(self._pattern(pid="dep1", status="deprecated"))
        active = db.get_patterns(status="active")
        deprecated = db.get_patterns(status="deprecated")
        assert any(p["pattern_id"] == "active1" for p in active)
        assert not any(p["pattern_id"] == "dep1" for p in active)
        assert any(p["pattern_id"] == "dep1" for p in deprecated)

    def test_update_pattern_stats_true_positive(self, db):
        db.save_pattern(self._pattern(alpha=1.0, beta=1.0))
        db.update_pattern_stats("p1", is_true_positive=True)
        row = db.conn.execute(
            "SELECT alpha, beta, observations, true_positives FROM patterns WHERE pattern_id = 'p1'"
        ).fetchone()
        assert row["alpha"] == pytest.approx(2.0)
        assert row["beta"] == pytest.approx(1.0)
        assert row["observations"] == 1
        assert row["true_positives"] == 1

    def test_update_pattern_stats_false_positive(self, db):
        db.save_pattern(self._pattern(alpha=1.0, beta=1.0))
        db.update_pattern_stats("p1", is_true_positive=False)
        row = db.conn.execute(
            "SELECT alpha, beta, false_positives FROM patterns WHERE pattern_id = 'p1'"
        ).fetchone()
        assert row["alpha"] == pytest.approx(1.0)
        assert row["beta"] == pytest.approx(2.0)
        assert row["false_positives"] == 1

    def test_update_missing_pattern_is_noop(self, db):
        """update_pattern_stats on unknown pattern_id should not raise."""
        db.update_pattern_stats("nonexistent_pattern", is_true_positive=True)

    def test_keywords_roundtrip(self, db):
        db.save_pattern(self._pattern(keywords=["fast", "money", "easy"]))
        patterns = db.get_patterns()
        p = next(p for p in patterns if p["pattern_id"] == "p1")
        assert "fast" in p["keywords"]


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

class TestCompanies:
    def test_save_and_get_company(self, db):
        db.save_company({"name": "Google", "domain": "google.com", "is_verified": True})
        result = db.get_company("Google")
        assert result is not None
        assert result["domain"] == "google.com"
        assert result["is_verified"] == 1

    def test_get_company_case_insensitive(self, db):
        db.save_company({"name": "Acme Corp"})
        assert db.get_company("acme corp") is not None
        assert db.get_company("ACME CORP") is not None

    def test_get_company_missing(self, db):
        assert db.get_company("Nonexistent Co") is None

    def test_save_company_upsert(self, db):
        db.save_company({"name": "Acme", "domain": "acme.com"})
        db.save_company({"name": "Acme", "domain": "acme.io"})
        result = db.get_company("Acme")
        assert result["domain"] == "acme.io"
        count = db.conn.execute("SELECT COUNT(*) FROM companies WHERE name='Acme'").fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# Scam entities
# ---------------------------------------------------------------------------

class TestScamEntities:
    def test_is_known_scam_entity_by_name(self, db):
        assert db.is_known_scam_entity(name="Global Solutions LLC")

    def test_is_known_scam_entity_by_domain(self, db):
        assert db.is_known_scam_entity(domain="amazon-jobs.net")

    def test_is_not_known_scam(self, db):
        assert not db.is_known_scam_entity(name="Google", domain="google.com")

    def test_add_custom_scam_entity(self, db):
        db.add_scam_entity(name="Fake Corp", entity_type="fake_company", source="test")
        assert db.is_known_scam_entity(name="Fake Corp")

    def test_get_scam_entities_filtered(self, db):
        fakes = db.get_scam_entities(entity_type="fake_company")
        assert all(e["type"] == "fake_company" for e in fakes)

    def test_get_scam_entities_unfiltered(self, db):
        all_entities = db.get_scam_entities()
        types = {e["type"] for e in all_entities}
        assert len(types) > 1  # seed has multiple types


# ---------------------------------------------------------------------------
# Source stats
# ---------------------------------------------------------------------------

class TestSourceStats:
    def test_upsert_and_retrieve(self, db):
        db.upsert_source_stats("linkedin", jobs_ingested=10, scams_detected=2, avg_score=0.3)
        stats = db.get_source_stats()
        linkedin = next(s for s in stats if s["source"] == "linkedin")
        assert linkedin["jobs_ingested"] == 10
        assert linkedin["scams_detected"] == 2

    def test_upsert_accumulates_counts(self, db):
        db.upsert_source_stats("indeed", jobs_ingested=5, scams_detected=1)
        db.upsert_source_stats("indeed", jobs_ingested=3, scams_detected=2)
        stats = db.get_source_stats()
        indeed = next(s for s in stats if s["source"] == "indeed")
        assert indeed["jobs_ingested"] == 8
        assert indeed["scams_detected"] == 3

    def test_get_best_sources(self, db):
        db.upsert_source_stats("high_yield", jobs_ingested=10, scams_detected=8)
        db.upsert_source_stats("low_yield",  jobs_ingested=10, scams_detected=1)
        best = db.get_best_sources(n=1)
        assert best == ["high_yield"]


# ---------------------------------------------------------------------------
# Salary benchmarks
# ---------------------------------------------------------------------------

class TestSalaryBenchmarks:
    def test_get_known_benchmark(self, db):
        result = db.get_salary_benchmark("software_engineer", "senior")
        assert result is not None
        assert result["p50"] > result["p25"]

    def test_get_benchmark_case_insensitive(self, db):
        result = db.get_salary_benchmark("Software_Engineer", "Senior")
        assert result is not None

    def test_get_unknown_benchmark(self, db):
        result = db.get_salary_benchmark("astrologer", "cosmic")
        assert result is None

    def test_get_all_salary_benchmarks(self, db):
        all_benchmarks = db.get_all_salary_benchmarks()
        assert len(all_benchmarks) > 10


# ---------------------------------------------------------------------------
# Signal rate history
# ---------------------------------------------------------------------------

class TestSignalRateHistory:
    def test_record_and_retrieve(self, db):
        db.record_signal_rates(
            {"upfront_fee": 5, "guaranteed_income": 3},
            total_jobs=100,
            window_start="2026-04-01T00:00:00",
            window_end="2026-04-08T00:00:00",
        )
        history = db.get_signal_rate_history()
        assert len(history) == 2

    def test_filter_by_signal_name(self, db):
        db.record_signal_rates(
            {"sig_a": 2, "sig_b": 4},
            total_jobs=50,
            window_start="2026-04-01T00:00:00",
            window_end="2026-04-08T00:00:00",
        )
        results = db.get_signal_rate_history(signal_name="sig_a")
        assert all(r["signal_name"] == "sig_a" for r in results)
        assert len(results) == 1

    def test_empty_signal_rates(self, db):
        """Empty dict should store nothing and not raise."""
        db.record_signal_rates({}, total_jobs=10,
                               window_start="2026-01-01T00:00:00",
                               window_end="2026-01-02T00:00:00")
        assert db.get_signal_rate_history() == []


# ---------------------------------------------------------------------------
# Shadow runs
# ---------------------------------------------------------------------------

class TestShadowRuns:
    def test_insert_and_get_active(self, db):
        run_id = db.insert_shadow_run({"signal_a": 1.2, "signal_b": 0.8})
        assert isinstance(run_id, int)
        active = db.get_active_shadow_run()
        assert active is not None
        assert active["candidate_weights"] == {"signal_a": 1.2, "signal_b": 0.8}

    def test_promote_shadow_run(self, db):
        run_id = db.insert_shadow_run({"w": 1.0})
        db.promote_shadow_run(run_id)
        active = db.get_active_shadow_run()
        assert active is None  # promoted, not active

    def test_reject_shadow_run(self, db):
        run_id = db.insert_shadow_run({"w": 1.0})
        db.reject_shadow_run(run_id)
        active = db.get_active_shadow_run()
        assert active is None

    def test_update_shadow_run(self, db):
        run_id = db.insert_shadow_run({"w": 1.0})
        db.update_shadow_run(run_id, {"jobs_evaluated": 50, "shadow_precision": 0.85})
        row = db.conn.execute(
            "SELECT jobs_evaluated, shadow_precision FROM shadow_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["jobs_evaluated"] == 50
        assert row["shadow_precision"] == pytest.approx(0.85)

    def test_shadow_history(self, db):
        db.insert_shadow_run({"a": 1})
        db.insert_shadow_run({"b": 2})
        history = db.get_shadow_history()
        assert len(history) >= 2


# ---------------------------------------------------------------------------
# Near misses
# ---------------------------------------------------------------------------

class TestNearMisses:
    def test_insert_and_retrieve(self, db):
        db.insert_near_miss("guaranteed_income", "garanteed income", "https://j.example.com")
        results = db.get_near_misses()
        assert len(results) == 1
        assert results[0]["partial_match"] == "garanteed income"

    def test_filter_by_signal_name(self, db):
        db.insert_near_miss("sig_x", "match1")
        db.insert_near_miss("sig_y", "match2")
        results = db.get_near_misses(signal_name="sig_x")
        assert all(r["signal_name"] == "sig_x" for r in results)


# ---------------------------------------------------------------------------
# Stats aggregate
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_empty_db_stats(self, db):
        stats = db.get_stats()
        assert stats["total_jobs_analyzed"] == 0
        assert stats["scam_jobs_detected"] == 0
        assert stats["avg_scam_score"] == 0.0

    def test_stats_counts_jobs(self, db):
        db.save_job(_job(url="j1", scam_score=0.7))
        db.save_job(_job(url="j2", scam_score=0.2))
        stats = db.get_stats()
        assert stats["total_jobs_analyzed"] == 2
        assert stats["scam_jobs_detected"] == 1

    def test_prediction_accuracy(self, db):
        db.save_job(_job(url="acc_j"))
        db.save_report({"url": "acc_j", "is_scam": True, "was_correct": True})
        stats = db.get_stats()
        assert stats["prediction_accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Ingestion runs
# ---------------------------------------------------------------------------

class TestIngestionRuns:
    def test_save_and_retrieve(self, db):
        run = {
            "run_id": "run_001",
            "sources": ["linkedin", "indeed"],
            "query": "engineer",
            "location": "Remote",
            "jobs_fetched": 50,
            "jobs_new": 10,
            "jobs_scored": 10,
            "high_risk_count": 2,
        }
        db.save_ingestion_run(run)
        history = db.get_ingestion_history()
        assert len(history) == 1
        assert history[0]["run_id"] == "run_001"
        assert isinstance(history[0]["sources"], list)

    def test_upsert_updates_run(self, db):
        db.save_ingestion_run({"run_id": "r1", "jobs_fetched": 10})
        db.save_ingestion_run({"run_id": "r1", "jobs_fetched": 20})
        history = db.get_ingestion_history()
        assert len(history) == 1
        assert history[0]["jobs_fetched"] == 20


# ---------------------------------------------------------------------------
# Cascade events
# ---------------------------------------------------------------------------

class TestCascadeEvents:
    def test_insert_and_retrieve(self, db):
        eid = db.insert_cascade_event("flywheel", "pattern_update", json.dumps({"patterns_affected": 3}))
        assert isinstance(eid, int)
        history = db.get_cascade_history()
        assert len(history) == 1
        assert history[0]["trigger"] == "flywheel"
        assert isinstance(history[0]["impact"], dict)

    def test_cascade_history_newest_first(self, db):
        db.insert_cascade_event("t1", "ct1")
        db.insert_cascade_event("t2", "ct2")
        history = db.get_cascade_history()
        assert history[0]["trigger"] == "t2"


# ---------------------------------------------------------------------------
# Cortex state
# ---------------------------------------------------------------------------

class TestCortexState:
    def test_save_and_get_latest(self, db):
        db.save_cortex_state(
            cycle_number=1,
            state_json="{}",
            learning_velocity=0.5,
            health_grade="B",
            strategic_mode="LEARN",
        )
        state = db.get_latest_cortex_state()
        assert state is not None
        assert state["cycle_number"] == 1
        assert state["health_grade"] == "B"

    def test_get_latest_returns_most_recent(self, db):
        db.save_cortex_state(1, "{}", 0.1, "C", "OBSERVE")
        db.save_cortex_state(2, "{}", 0.9, "A", "EXPLOIT")
        latest = db.get_latest_cortex_state()
        assert latest["cycle_number"] == 2

    def test_get_latest_empty(self, db):
        assert db.get_latest_cortex_state() is None


# ---------------------------------------------------------------------------
# Flywheel metrics
# ---------------------------------------------------------------------------

class TestFlywheelMetrics:
    def test_save_flywheel_metrics(self, db):
        db.save_flywheel_metrics({
            "total_analyzed": 100,
            "true_positives": 80,
            "false_positives": 5,
            "precision": 0.94,
            "recall": 0.88,
            "f1": 0.91,
            "accuracy": 0.93,
            "cycle_number": 1,
        })
        history = db.get_flywheel_metrics_history(days=1)
        assert len(history) == 1
        assert history[0]["precision"] == pytest.approx(0.94)

    def test_promoted_list_coerced_to_count(self, db):
        """patterns_promoted as a list should be stored as its length."""
        db.save_flywheel_metrics({"patterns_promoted": ["p1", "p2", "p3"]})
        history = db.get_flywheel_metrics_history(days=1)
        assert history[0]["patterns_promoted"] == 3
