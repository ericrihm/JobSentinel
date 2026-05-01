"""Ingestion pipeline — fetches jobs from sources, scores, and persists them."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sentinel.db import SentinelDB
from sentinel.knowledge import KnowledgeBase
from sentinel.models import JobPosting

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class IngestionRun:
    """Summary of a single ingestion pipeline execution."""

    run_id: str
    started_at: str
    completed_at: str | None
    sources_queried: list[str]
    query: str
    location: str
    jobs_fetched: int
    jobs_new: int
    jobs_scored: int
    high_risk_count: int
    errors: list[str]

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "sources_queried": self.sources_queried,
            "query": self.query,
            "location": self.location,
            "jobs_fetched": self.jobs_fetched,
            "jobs_new": self.jobs_new,
            "jobs_scored": self.jobs_scored,
            "high_risk_count": self.high_risk_count,
            "errors": self.errors,
        }


class IngestionPipeline:
    """Orchestrator that pulls jobs from sources, scores them, and persists results."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path:
            self.db = SentinelDB(path=db_path)
        else:
            self.db = SentinelDB()

        # Seed default patterns if the DB has none
        kb = KnowledgeBase(db=self.db)
        kb.seed_default_patterns()

    def run(
        self,
        query: str,
        location: str = "",
        sources: list[str] | None = None,
        limit_per_source: int = 25,
        use_ai: bool = False,
        throttle_seconds: float = 1.0,
    ) -> IngestionRun:
        """Full pipeline: fetch -> deduplicate -> score -> persist -> return summary."""
        run_id = str(uuid.uuid4())
        started_at = _now_iso()
        errors: list[str] = []
        all_jobs: list[JobPosting] = []
        sources_queried: list[str] = []

        # 1. Import sources module (lazy to avoid circular imports)
        source_fetchers = self._get_source_fetchers(sources, errors)
        if source_fetchers:
            sources_queried = list(source_fetchers.keys())

        # 2. Fetch jobs from each source with throttle delay between sources
        for i, (source_name, fetch_fn) in enumerate(source_fetchers.items()):
            if i > 0:
                time.sleep(throttle_seconds)
            try:
                logger.info(
                    "Fetching from source '%s': query=%r location=%r limit=%d",
                    source_name, query, location, limit_per_source,
                )
                jobs = fetch_fn(query=query, location=location, limit=limit_per_source)
                if jobs:
                    all_jobs.extend(jobs)
                    logger.info("Source '%s' returned %d jobs", source_name, len(jobs))
                else:
                    logger.info("Source '%s' returned 0 jobs", source_name)
            except Exception as exc:
                msg = f"Source '{source_name}' failed: {exc}"
                logger.warning(msg)
                errors.append(msg)

        jobs_fetched = len(all_jobs)

        # 3. Deduplicate against DB
        new_jobs: list[JobPosting] = []
        for job in all_jobs:
            if not job.url:
                new_jobs.append(job)
                continue
            try:
                existing = self.db.get_job_by_url(job.url)
                if existing is None:
                    new_jobs.append(job)
                else:
                    logger.debug("Skipping duplicate: %s", job.url)
            except Exception as exc:
                msg = f"Dedup check failed for {job.url}: {exc}"
                logger.warning(msg)
                errors.append(msg)
                new_jobs.append(job)

        jobs_new = len(new_jobs)
        logger.info("Dedup: %d fetched, %d new", jobs_fetched, jobs_new)

        # 4. Score each new job via analyzer
        jobs_scored = 0
        high_risk_count = 0
        for job in new_jobs:
            try:
                from sentinel.analyzer import analyze_job

                result = analyze_job(job, use_ai=use_ai)

                # 5. Persist scored job to DB
                job_data = {
                    "url": job.url,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "description": job.description,
                    "salary_min": job.salary_min,
                    "salary_max": job.salary_max,
                    "scam_score": result.scam_score,
                    "risk_level": result.risk_level.value,
                    "analyzed_at": _now_iso(),
                    "signal_count": len(result.signals),
                    "signals_json": [
                        {"name": s.name, "category": s.category.value, "detail": s.detail}
                        for s in result.signals
                    ],
                    "user_reported": 0,
                    "user_verdict": "",
                }
                self.db.save_job(job_data)
                jobs_scored += 1

                if result.scam_score >= 0.6:
                    high_risk_count += 1

            except Exception as exc:
                msg = f"Scoring failed for '{job.title or job.url}': {exc}"
                logger.warning(msg)
                errors.append(msg)

        completed_at = _now_iso()

        # 6. Build and persist ingestion run record
        ingestion_run = IngestionRun(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            sources_queried=sources_queried,
            query=query,
            location=location,
            jobs_fetched=jobs_fetched,
            jobs_new=jobs_new,
            jobs_scored=jobs_scored,
            high_risk_count=high_risk_count,
            errors=errors,
        )

        try:
            self.db.save_ingestion_run({
                "run_id": run_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "sources_queried": sources_queried,
                "query": query,
                "location": location,
                "jobs_fetched": jobs_fetched,
                "jobs_new": jobs_new,
                "jobs_scored": jobs_scored,
                "high_risk_count": high_risk_count,
                "errors": errors,
            })
        except Exception as exc:
            logger.error("Failed to save ingestion run record: %s", exc)

        logger.info(
            "Ingestion complete: run_id=%s fetched=%d new=%d scored=%d high_risk=%d errors=%d",
            run_id, jobs_fetched, jobs_new, jobs_scored, high_risk_count, len(errors),
        )
        return ingestion_run

    def run_flywheel_cycle(self) -> dict:
        """Run the full flywheel after ingestion: LEARN -> EVOLVE."""
        from sentinel.flywheel import DetectionFlywheel

        flywheel = DetectionFlywheel(db=self.db)
        return flywheel.run_cycle()

    def auto_ingest(
        self,
        queries: list[str],
        location: str = "",
        run_flywheel: bool = True,
    ) -> list[IngestionRun]:
        """Run ingestion for multiple queries, then optionally run the flywheel."""
        results: list[IngestionRun] = []
        for query in queries:
            try:
                run = self.run(query=query, location=location)
                results.append(run)
            except Exception as exc:
                logger.error("auto_ingest failed for query %r: %s", query, exc)

        if run_flywheel:
            try:
                cycle = self.run_flywheel_cycle()
                logger.info("Flywheel cycle complete after auto_ingest: %s", cycle)
            except Exception as exc:
                logger.error("Flywheel cycle failed: %s", exc)

        return results

    def _get_source_fetchers(
        self,
        requested: list[str] | None,
        errors: list[str],
    ) -> dict[str, object]:
        """Build a dict of {source_name: fetch_function}.

        Attempts to import sentinel.sources. If that module is not available,
        falls back to the built-in LinkedIn scanner as the sole source.
        """
        fetchers: dict[str, object] = {}

        # Try to import the sources module
        try:
            from sentinel import sources as sources_module  # type: ignore[attr-defined]

            available = getattr(sources_module, "AVAILABLE_SOURCES", {})
            if requested:
                for name in requested:
                    if name in available:
                        fetchers[name] = available[name]
                    else:
                        msg = f"Requested source '{name}' not available"
                        logger.warning(msg)
                        errors.append(msg)
            else:
                fetchers = dict(available)
        except ImportError:
            logger.debug("sentinel.sources not available; falling back to linkedin scanner")
            # Fall back to the built-in scanner
            if requested is None or "linkedin" in requested:
                fetchers["linkedin"] = self._fetch_linkedin

        # If specific sources were requested but none resolved, try linkedin fallback
        if not fetchers and requested and "linkedin" in requested:
            fetchers["linkedin"] = self._fetch_linkedin

        return fetchers

    @staticmethod
    def _fetch_linkedin(
        query: str, location: str = "", limit: int = 25,
    ) -> list[JobPosting]:
        """Built-in LinkedIn source using scanner.scrape_search_results."""
        from sentinel.scanner import scrape_search_results

        return scrape_search_results(query=query, location=location, limit=limit)
