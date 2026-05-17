"""Focused stress tests for high-contention paths."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier

from sentinel.db import SentinelDB


def _stress_job(index: int) -> dict:
    return {
        "url": f"https://stress.example.com/job/{index}",
        "title": f"Stress Engineer {index}",
        "company": "Load Test Corp",
        "location": "Remote",
        "description": "Build reliable systems under concurrent database load.",
        "salary_min": 100_000.0,
        "salary_max": 140_000.0,
        "scam_score": 0.05,
        "confidence": 0.95,
        "risk_level": "low",
        "signal_count": 0,
        "signals": [],
    }


def _save_and_verify_job(db_path: str, start: Barrier, index: int) -> str:
    start.wait()
    db = SentinelDB(path=db_path)
    try:
        job = _stress_job(index)
        db.save_job(job)
        saved = db.get_job(job["url"])
        assert saved is not None
        assert saved["title"] == job["title"]
        return job["url"]
    finally:
        db.close()
        try:
            db.conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            pass
        else:
            raise AssertionError("SentinelDB connection remained usable after close")


def test_db_handles_100_concurrent_job_writes_without_crashes_or_connection_leaks(tmp_path):
    db_path = str(tmp_path / "stress.db")
    operations = 100
    start = Barrier(operations)

    with ThreadPoolExecutor(max_workers=operations) as pool:
        futures = [
            pool.submit(_save_and_verify_job, db_path, start, index)
            for index in range(operations)
        ]
        urls = [future.result(timeout=30) for future in as_completed(futures)]

    assert len(urls) == operations
    assert len(set(urls)) == operations

    db = SentinelDB(path=db_path)
    try:
        count = db.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert count == operations
    finally:
        db.close()
