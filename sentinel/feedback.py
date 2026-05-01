"""Active learning feedback pipeline for JobSentinel.

FeedbackPipeline closes the learning loop by:
1. Rescanning previously-scored jobs and detecting score drift.
2. Generating synthetic feedback for extreme high-confidence predictions.
3. Importing labeled datasets (CSV / JSON) to bootstrap training.
4. Reporting feedback coverage and volume trends.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RescanResult:
    """Output of FeedbackPipeline.rescan_and_compare()."""
    jobs_rescanned: int
    jobs_drifted: int
    drift_threshold: float
    drifted_urls: list[str] = field(default_factory=list)
    avg_delta: float = 0.0
    max_delta: float = 0.0
    rescanned_at: str = field(default_factory=_now_iso)


@dataclass
class SyntheticReport:
    """A single auto-generated feedback report."""
    url: str
    is_scam: bool
    our_prediction: float
    confidence: float
    reason: str
    generated_at: str = field(default_factory=_now_iso)


@dataclass
class ImportResult:
    """Output of FeedbackPipeline.import_labeled_data()."""
    filepath: str
    rows_read: int
    rows_imported: int
    rows_skipped: int
    errors: list[str] = field(default_factory=list)
    imported_at: str = field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class FeedbackPipeline:
    """Active learning feedback pipeline.

    Bootstraps and sustains the learning loop without requiring manual CLI
    reports for every prediction.
    """

    # Minimum score delta considered significant drift
    DRIFT_THRESHOLD: float = 0.2

    # Thresholds for synthetic feedback generation
    SYNTHETIC_HIGH_SCORE: float = 0.9   # score above this → positive (scam) report
    SYNTHETIC_LOW_SCORE: float = 0.1    # score below this → negative (legit) report
    SYNTHETIC_MIN_CONFIDENCE: float = 0.8  # only generate when this confident

    def __init__(self, db=None) -> None:
        if db is None:
            from sentinel.db import SentinelDB
            db = SentinelDB()
        self.db = db

    # ------------------------------------------------------------------
    # Rescan and compare
    # ------------------------------------------------------------------

    def rescan_and_compare(
        self,
        days: int = 7,
        sample_size: int = 50,
    ) -> RescanResult:
        """Re-analyse previously-scored jobs and detect score drift.

        Pulls up to *sample_size* jobs scored in the last *days* days,
        re-scores them with the current signal weights, and identifies jobs
        where the score changed by more than DRIFT_THRESHOLD.

        Args:
            days:        Look-back window for selecting jobs to rescan.
            sample_size: Maximum number of jobs to rescan per call.

        Returns:
            RescanResult summarising how many jobs drifted and by how much.
        """
        from datetime import timedelta

        from sentinel.analyzer import analyze_job
        from sentinel.models import JobPosting

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        rows = self.db.conn.execute(
            """
            SELECT url, title, company, location, description,
                   salary_min, salary_max, scam_score, confidence
            FROM jobs
            WHERE analyzed_at >= ?
              AND scam_score IS NOT NULL
            ORDER BY analyzed_at DESC
            LIMIT ?
            """,
            (cutoff, sample_size),
        ).fetchall()

        drifted: list[str] = []
        deltas: list[float] = []

        for row in rows:
            url = row["url"] or ""
            old_score = float(row["scam_score"] or 0.0)

            job = JobPosting(
                url=url,
                title=row["title"] or "",
                company=row["company"] or "",
                location=row["location"] or "",
                description=row["description"] or "",
                salary_min=float(row["salary_min"] or 0.0),
                salary_max=float(row["salary_max"] or 0.0),
            )

            try:
                result = analyze_job(job, use_ai=False)
                new_score = result.scam_score
            except Exception:
                logger.debug("Rescan failed for %s", url, exc_info=True)
                continue

            delta = abs(new_score - old_score)
            deltas.append(delta)

            if delta > self.DRIFT_THRESHOLD:
                drifted.append(url)
                # Update the stored score to reflect current weights
                try:
                    self.db.save_job({
                        "url": url,
                        "title": row["title"] or "",
                        "company": row["company"] or "",
                        "location": row["location"] or "",
                        "description": row["description"] or "",
                        "salary_min": float(row["salary_min"] or 0.0),
                        "salary_max": float(row["salary_max"] or 0.0),
                        "scam_score": new_score,
                        "confidence": result.confidence,
                        "risk_level": result.risk_level.value,
                        "analyzed_at": _now_iso(),
                        "signal_count": len(result.signals),
                        "signals_json": json.dumps([
                            {"name": s.name, "category": s.category.value, "weight": s.weight}
                            for s in result.signals
                        ]),
                    })
                except Exception:
                    logger.debug("Could not persist rescored job %s", url, exc_info=True)

        jobs_rescanned = len(rows)
        avg_delta = round(sum(deltas) / len(deltas), 4) if deltas else 0.0
        max_delta = round(max(deltas), 4) if deltas else 0.0

        return RescanResult(
            jobs_rescanned=jobs_rescanned,
            jobs_drifted=len(drifted),
            drift_threshold=self.DRIFT_THRESHOLD,
            drifted_urls=drifted,
            avg_delta=avg_delta,
            max_delta=max_delta,
        )

    # ------------------------------------------------------------------
    # Synthetic feedback
    # ------------------------------------------------------------------

    def generate_synthetic_feedback(
        self,
        n: int = 20,
    ) -> list[SyntheticReport]:
        """Auto-generate feedback for extreme high-confidence predictions.

        For jobs with score > SYNTHETIC_HIGH_SCORE AND confidence > SYNTHETIC_MIN_CONFIDENCE,
        generate a positive (scam) report.

        For jobs with score < SYNTHETIC_LOW_SCORE AND confidence > SYNTHETIC_MIN_CONFIDENCE,
        generate a negative (legitimate) report.

        Only generates reports for jobs that do not already have a user-supplied
        report, to avoid overwriting human labels with synthetic ones.

        Args:
            n: Maximum number of synthetic reports to generate.

        Returns:
            List of SyntheticReport objects (also persisted to the DB).
        """
        reports: list[SyntheticReport] = []

        # Jobs we already have feedback for (human or synthetic)
        existing_urls = self._get_reported_urls()

        # High-confidence scam candidates
        high_rows = self.db.conn.execute(
            """
            SELECT url, scam_score, confidence
            FROM jobs
            WHERE scam_score > ?
              AND confidence > ?
              AND url IS NOT NULL
              AND url != ''
            ORDER BY scam_score DESC
            LIMIT ?
            """,
            (self.SYNTHETIC_HIGH_SCORE, self.SYNTHETIC_MIN_CONFIDENCE, n),
        ).fetchall()

        for row in high_rows:
            if len(reports) >= n:
                break
            url = row["url"]
            if url in existing_urls:
                continue
            score = float(row["scam_score"])
            conf = float(row["confidence"])
            report = SyntheticReport(
                url=url,
                is_scam=True,
                our_prediction=score,
                confidence=conf,
                reason=f"synthetic: score={score:.2f} > {self.SYNTHETIC_HIGH_SCORE}, conf={conf:.2f}",
            )
            self._persist_synthetic_report(report)
            existing_urls.add(url)
            reports.append(report)

        # High-confidence legitimate candidates
        low_rows = self.db.conn.execute(
            """
            SELECT url, scam_score, confidence
            FROM jobs
            WHERE scam_score < ?
              AND confidence > ?
              AND url IS NOT NULL
              AND url != ''
            ORDER BY scam_score ASC
            LIMIT ?
            """,
            (self.SYNTHETIC_LOW_SCORE, self.SYNTHETIC_MIN_CONFIDENCE, n),
        ).fetchall()

        for row in low_rows:
            if len(reports) >= n:
                break
            url = row["url"]
            if url in existing_urls:
                continue
            score = float(row["scam_score"])
            conf = float(row["confidence"])
            report = SyntheticReport(
                url=url,
                is_scam=False,
                our_prediction=score,
                confidence=conf,
                reason=f"synthetic: score={score:.2f} < {self.SYNTHETIC_LOW_SCORE}, conf={conf:.2f}",
            )
            self._persist_synthetic_report(report)
            existing_urls.add(url)
            reports.append(report)

        logger.info(
            "Generated %d synthetic feedback reports (%d scam, %d legit)",
            len(reports),
            sum(1 for r in reports if r.is_scam),
            sum(1 for r in reports if not r.is_scam),
        )
        return reports

    # ------------------------------------------------------------------
    # Labeled data import
    # ------------------------------------------------------------------

    def import_labeled_data(self, filepath: str) -> ImportResult:
        """Bulk import labeled job data from CSV or JSON.

        Expected columns / keys: ``url`` (str), ``is_scam`` (bool/int/str).
        Optional: ``reason`` (str), ``our_prediction`` (float).

        Rows are scored (heuristic only), persisted to the jobs table, and
        feedback reports are written to the feedback_reports table (and the
        standard reports table so the flywheel sees them).

        Args:
            filepath: Path to a CSV or JSON file.

        Returns:
            ImportResult with import statistics.
        """
        from sentinel.analyzer import analyze_job
        from sentinel.models import JobPosting

        path = Path(filepath)
        rows_read = 0
        rows_imported = 0
        rows_skipped = 0
        errors: list[str] = []

        try:
            records = self._load_labeled_file(path)
        except Exception as exc:
            return ImportResult(
                filepath=filepath,
                rows_read=0,
                rows_imported=0,
                rows_skipped=0,
                errors=[f"File load error: {exc}"],
            )

        existing_urls = self._get_reported_urls()

        for i, rec in enumerate(records):
            rows_read += 1
            try:
                url = str(rec.get("url", "")).strip()
                if not url:
                    rows_skipped += 1
                    continue

                is_scam_raw = rec.get("is_scam", rec.get("label", ""))
                is_scam = self._parse_bool(is_scam_raw)
                if is_scam is None:
                    errors.append(f"Row {i + 1}: unrecognised is_scam value {is_scam_raw!r}")
                    rows_skipped += 1
                    continue

                reason = str(rec.get("reason", "imported")).strip()
                our_prediction_raw = rec.get("our_prediction", rec.get("score", None))

                # Score the job if we don't have a pre-computed score
                if our_prediction_raw is not None:
                    our_prediction = float(our_prediction_raw)
                else:
                    try:
                        job = JobPosting(
                            url=url,
                            title=str(rec.get("title", "")),
                            company=str(rec.get("company", "")),
                            description=str(rec.get("description", "")),
                        )
                        result = analyze_job(job, use_ai=False)
                        our_prediction = result.scam_score
                        # Persist the job itself
                        self.db.save_job({
                            "url": url,
                            "title": rec.get("title", ""),
                            "company": rec.get("company", ""),
                            "description": rec.get("description", ""),
                            "scam_score": our_prediction,
                            "confidence": result.confidence,
                            "risk_level": result.risk_level.value,
                            "analyzed_at": _now_iso(),
                            "signal_count": len(result.signals),
                        })
                    except Exception as exc2:
                        our_prediction = 0.5
                        logger.debug("Could not score imported job %s: %s", url, exc2)

                was_correct = our_prediction >= 0.5 if is_scam else our_prediction < 0.5

                # Write to standard reports table (flywheel reads from here)
                self.db.save_report({
                    "url": url,
                    "is_scam": is_scam,
                    "reason": reason,
                    "our_prediction": our_prediction,
                    "was_correct": was_correct,
                    "reported_at": _now_iso(),
                })

                # Write to feedback_reports for attribution tracking
                self._save_feedback_report(
                    url=url,
                    is_scam=is_scam,
                    our_prediction=our_prediction,
                    source="imported",
                    reason=reason,
                    was_correct=was_correct,
                )

                existing_urls.add(url)
                rows_imported += 1

            except Exception as exc:
                errors.append(f"Row {i + 1}: {exc}")
                rows_skipped += 1

        logger.info(
            "Import complete: %d read, %d imported, %d skipped from %s",
            rows_read, rows_imported, rows_skipped, filepath,
        )
        return ImportResult(
            filepath=filepath,
            rows_read=rows_read,
            rows_imported=rows_imported,
            rows_skipped=rows_skipped,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Feedback stats
    # ------------------------------------------------------------------

    def get_feedback_stats(self, db=None) -> dict[str, Any]:
        """Return feedback pipeline statistics.

        Returns:
            dict with:
            - total_reports: total feedback records across all sources
            - manual_reports: count of human-submitted reports
            - synthetic_reports: count of auto-generated reports
            - imported_reports: count of bulk-imported labeled records
            - reports_per_day: list of daily report counts (last 7 days)
            - feedback_coverage: fraction of scored jobs that have feedback
            - total_scored_jobs: total jobs in the jobs table
        """
        _db = db or self.db

        # Counts from feedback_reports (fine-grained attribution)
        try:
            manual_count = _db.conn.execute(
                "SELECT COUNT(*) FROM feedback_reports WHERE source = 'manual'"
            ).fetchone()[0]
            synthetic_count = _db.conn.execute(
                "SELECT COUNT(*) FROM feedback_reports WHERE source = 'synthetic'"
            ).fetchone()[0]
            imported_count = _db.conn.execute(
                "SELECT COUNT(*) FROM feedback_reports WHERE source = 'imported'"
            ).fetchone()[0]
            total_feedback = _db.conn.execute(
                "SELECT COUNT(*) FROM feedback_reports"
            ).fetchone()[0]
        except Exception:
            # Table may not exist yet on very old DBs
            manual_count = _db.conn.execute(
                "SELECT COUNT(*) FROM reports"
            ).fetchone()[0]
            synthetic_count = 0
            imported_count = 0
            total_feedback = manual_count

        # Fallback: if feedback_reports is empty, use reports table
        if total_feedback == 0:
            total_feedback = _db.conn.execute(
                "SELECT COUNT(*) FROM reports"
            ).fetchone()[0]
            manual_count = total_feedback

        # Daily trend (last 7 days)
        reports_per_day = self._reports_per_day(_db, days=7)

        # Coverage
        total_scored = _db.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE scam_score IS NOT NULL"
        ).fetchone()[0]

        reported_urls = _db.conn.execute(
            "SELECT COUNT(DISTINCT url) FROM reports"
        ).fetchone()[0]

        coverage = round(reported_urls / total_scored, 4) if total_scored > 0 else 0.0

        return {
            "total_reports": total_feedback,
            "manual_reports": manual_count,
            "synthetic_reports": synthetic_count,
            "imported_reports": imported_count,
            "reports_per_day": reports_per_day,
            "feedback_coverage": coverage,
            "total_scored_jobs": total_scored,
            "reported_jobs": reported_urls,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reported_urls(self) -> set[str]:
        """Return the set of URLs that already have any feedback record."""
        rows = self.db.conn.execute("SELECT DISTINCT url FROM reports").fetchall()
        urls = {row[0] for row in rows if row[0]}
        try:
            extra = self.db.conn.execute(
                "SELECT DISTINCT url FROM feedback_reports"
            ).fetchall()
            urls |= {row[0] for row in extra if row[0]}
        except Exception:
            pass
        return urls

    def _persist_synthetic_report(self, report: SyntheticReport) -> None:
        """Persist a synthetic report to both reports and feedback_reports tables."""
        was_correct = report.our_prediction >= 0.5 if report.is_scam else report.our_prediction < 0.5
        self.db.save_report({
            "url": report.url,
            "is_scam": report.is_scam,
            "reason": report.reason,
            "our_prediction": report.our_prediction,
            "was_correct": was_correct,
            "reported_at": report.generated_at,
        })
        self._save_feedback_report(
            url=report.url,
            is_scam=report.is_scam,
            our_prediction=report.our_prediction,
            source="synthetic",
            reason=report.reason,
            was_correct=was_correct,
        )

    def _save_feedback_report(
        self,
        url: str,
        is_scam: bool,
        our_prediction: float,
        source: str,
        reason: str,
        was_correct: bool,
    ) -> None:
        """Insert a row into feedback_reports (attribution-aware table)."""
        try:
            self.db.conn.execute(
                """
                INSERT INTO feedback_reports
                    (url, is_scam, our_prediction, source, reason, was_correct, reported_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, int(is_scam), our_prediction, source, reason, int(was_correct), _now_iso()),
            )
            self.db.conn.commit()
        except Exception:
            logger.debug("Could not write to feedback_reports", exc_info=True)

    @staticmethod
    def _load_labeled_file(path: Path) -> list[dict]:
        """Load CSV or JSON from *path* and return a list of record dicts."""
        suffix = path.suffix.lower()
        if suffix == ".json":
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Support {"jobs": [...]} or {"data": [...]} wrappers
                for key in ("jobs", "data", "records", "items"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # Single record
                return [data]
            raise ValueError(f"Unexpected JSON structure in {path}")
        elif suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                return list(reader)
        else:
            raise ValueError(f"Unsupported file format: {suffix!r}. Use .csv or .json")

    @staticmethod
    def _parse_bool(value) -> bool | None:
        """Parse a flexible is_scam representation to bool or None on failure."""
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "true", "yes", "scam", "fraud"):
                return True
            if v in ("0", "false", "no", "legit", "legitimate", "safe"):
                return False
        return None

    def _reports_per_day(self, db, days: int = 7) -> list[int]:
        """Return daily report counts for the last *days* days (oldest first)."""
        from datetime import timedelta

        counts = []
        now = datetime.now(UTC)
        for offset in range(days - 1, -1, -1):
            day_start = (now - timedelta(days=offset + 1)).isoformat()
            day_end = (now - timedelta(days=offset)).isoformat()
            count = db.conn.execute(
                "SELECT COUNT(*) FROM reports WHERE reported_at >= ? AND reported_at < ?",
                (day_start, day_end),
            ).fetchone()[0]
            counts.append(count)
        return counts
