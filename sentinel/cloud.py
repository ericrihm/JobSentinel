"""Cloud-ready execution module for Sentinel.

Wraps daemon phases for stateless serverless execution (GitHub Actions,
AWS Lambda, Cloudflare Workers, etc.) with support for cloud databases
(Turso/libsql) and webhook-based result reporting.

All dependencies are Python stdlib + optional ``libsql-experimental``.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# CloudConfig — environment-driven configuration
# ---------------------------------------------------------------------------

@dataclass
class CloudConfig:
    """Configuration sourced from environment variables.

    Environment variables:
        SENTINEL_DB_URL       — Turso/D1/local SQLite path
        SENTINEL_API_KEY      — API key for Claude AI analysis (optional)
        SENTINEL_SCAN_QUERIES — comma-separated job search queries
        SENTINEL_SCAN_INTERVAL — hours between scans (default: 1)
        SENTINEL_REPORT_WEBHOOK — webhook URL for scan results
    """

    db_url: str = ""
    api_key: str = ""
    scan_queries: list[str] = field(default_factory=list)
    scan_interval_hours: float = 1.0
    report_webhook: str = ""

    @classmethod
    def from_env(cls) -> CloudConfig:
        """Build config from environment variables."""
        queries_raw = os.environ.get("SENTINEL_SCAN_QUERIES", "")
        queries = [q.strip() for q in queries_raw.split(",") if q.strip()] if queries_raw else []

        interval_raw = os.environ.get("SENTINEL_SCAN_INTERVAL", "1")
        try:
            interval = float(interval_raw)
        except (ValueError, TypeError):
            interval = 1.0

        return cls(
            db_url=os.environ.get("SENTINEL_DB_URL", ""),
            api_key=os.environ.get("SENTINEL_API_KEY", ""),
            scan_queries=queries,
            scan_interval_hours=interval,
            report_webhook=os.environ.get("SENTINEL_REPORT_WEBHOOK", ""),
        )

    def validate(self) -> list[str]:
        """Return a list of configuration issues (empty = valid)."""
        issues: list[str] = []
        if not self.db_url:
            issues.append("SENTINEL_DB_URL not set (will use local SQLite default)")
        if not self.scan_queries:
            issues.append("SENTINEL_SCAN_QUERIES not set (will use default queries)")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_url": self.db_url if self.db_url else "(local default)",
            "api_key": "***" if self.api_key else "(not set)",
            "scan_queries": self.scan_queries,
            "scan_interval_hours": self.scan_interval_hours,
            "report_webhook": self.report_webhook if self.report_webhook else "(not set)",
        }


# ---------------------------------------------------------------------------
# TursoAdapter — libsql-compatible database adapter
# ---------------------------------------------------------------------------

class TursoAdapter:
    """SQLite-compatible adapter for Turso (libsql) cloud databases.

    Provides the same interface as SentinelDB.  When ``libsql-experimental``
    is available and a ``libsql://`` URL is configured, uses the Turso cloud
    database.  Otherwise, falls back gracefully to local SQLite via SentinelDB.
    """

    def __init__(self, db_url: str = "") -> None:
        self._db_url = db_url
        self._is_turso = False
        self._db: Any = None  # SentinelDB or libsql connection wrapper
        self._init_connection()

    def _init_connection(self) -> None:
        """Initialise the database connection."""
        if self._db_url and self._db_url.startswith("libsql://"):
            try:
                self._init_turso()
                return
            except Exception as exc:
                logger.warning(
                    "Turso connection failed (%s), falling back to local SQLite", exc
                )

        # Fallback: use SentinelDB with local SQLite
        self._init_local(self._db_url)

    def _init_turso(self) -> None:
        """Connect via libsql-experimental."""
        import libsql_experimental as libsql  # type: ignore[import-untyped]

        auth_token = os.environ.get("SENTINEL_TURSO_TOKEN", "")
        self._conn = libsql.connect(
            self._db_url,
            auth_token=auth_token,
        )
        self._is_turso = True

        # Apply schema
        from sentinel.db import SCHEMA
        self._conn.executescript(SCHEMA)
        self._conn.commit()

        # Wrap in a SentinelDB-like interface
        self._db = _TursoDBWrapper(self._conn)
        logger.info("Connected to Turso: %s", self._db_url)

    def _init_local(self, path: str = "") -> None:
        """Fall back to local SentinelDB."""
        from sentinel.db import SentinelDB

        if path and not path.startswith("libsql://"):
            self._db = SentinelDB(path=path)
        else:
            self._db = SentinelDB()
        self._is_turso = False

    @property
    def is_turso(self) -> bool:
        return self._is_turso

    @property
    def db(self) -> Any:
        """Return the underlying database object (SentinelDB or wrapper)."""
        return self._db

    def check_connectivity(self) -> dict[str, Any]:
        """Test database connectivity and return status."""
        try:
            if self._is_turso:
                self._conn.execute("SELECT 1")
            else:
                self._db.conn.execute("SELECT 1")
            stats = self._db.get_stats() if hasattr(self._db, "get_stats") else {}
            return {
                "connected": True,
                "backend": "turso" if self._is_turso else "sqlite",
                "url": self._db_url or "(local default)",
                "stats": stats,
            }
        except Exception as exc:
            return {
                "connected": False,
                "backend": "turso" if self._is_turso else "sqlite",
                "url": self._db_url or "(local default)",
                "error": str(exc),
            }

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._db, "close"):
            self._db.close()


class _TursoDBWrapper:
    """Thin wrapper around a libsql connection to provide SentinelDB-compatible methods.

    Delegates to the underlying connection for raw SQL while providing
    the higher-level methods that CloudRunner and other Sentinel components
    expect (save_job, get_stats, etc.).
    """

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.path = "(turso)"

    def save_job(self, job_data: dict) -> None:
        signals = job_data.get("signals_json", job_data.get("signals", []))
        if not isinstance(signals, str):
            signals = json.dumps(signals)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO jobs
                (url, title, company, location, description,
                 salary_min, salary_max, scam_score, confidence, risk_level,
                 analyzed_at, signal_count, signals_json,
                 user_reported, user_verdict)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_data.get("url", ""),
                job_data.get("title", ""),
                job_data.get("company", ""),
                job_data.get("location", ""),
                job_data.get("description", ""),
                job_data.get("salary_min", 0),
                job_data.get("salary_max", 0),
                job_data.get("scam_score", 0),
                job_data.get("confidence"),
                job_data.get("risk_level", ""),
                job_data.get("analyzed_at", ""),
                job_data.get("signal_count", 0),
                signals,
                job_data.get("user_reported", 0),
                job_data.get("user_verdict", ""),
            ),
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        total_jobs = self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        scam_jobs = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE scam_score >= 0.6"
        ).fetchone()[0]
        total_reports = self.conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        return {
            "total_jobs_analyzed": total_jobs,
            "scam_jobs_detected": scam_jobs,
            "total_user_reports": total_reports,
        }

    def get_reports(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM reports ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows] if rows else []

    def get_patterns(self, status: str = "active") -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM patterns WHERE status = ?", (status,)
        ).fetchall()
        return [dict(r) for r in rows] if rows else []

    def save_pattern(self, pattern_data: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO patterns
               (pattern_id, name, description, category, regex,
                keywords_json, alpha, beta, observations,
                true_positives, false_positives, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pattern_data.get("pattern_id", ""),
                pattern_data.get("name", ""),
                pattern_data.get("description", ""),
                pattern_data.get("category", ""),
                pattern_data.get("regex", ""),
                pattern_data.get("keywords_json", "[]"),
                pattern_data.get("alpha", 1.0),
                pattern_data.get("beta", 1.0),
                pattern_data.get("observations", 0),
                pattern_data.get("true_positives", 0),
                pattern_data.get("false_positives", 0),
                pattern_data.get("status", "active"),
                pattern_data.get("created_at", _now_iso()),
                _now_iso(),
            ),
        )
        self.conn.commit()

    def get_source_stats(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM source_stats").fetchall()
        return [dict(r) for r in rows] if rows else []

    def get_best_sources(self, n: int = 5) -> list[str]:
        rows = self.conn.execute(
            """SELECT source FROM source_stats
               WHERE jobs_ingested > 0
               ORDER BY CAST(scams_detected AS REAL) / MAX(jobs_ingested, 1) DESC
               LIMIT ?""",
            (n,),
        ).fetchall()
        return [r[0] for r in rows] if rows else []

    def close(self) -> None:
        pass  # libsql connections don't need explicit close


# ---------------------------------------------------------------------------
# WebhookReporter — sends scan results to external services
# ---------------------------------------------------------------------------

class WebhookReporter:
    """Sends scan results to webhooks (Slack, Discord, generic HTTP POST)."""

    def __init__(self, webhook_url: str = "", timeout: float = 10.0) -> None:
        self._url = webhook_url
        self._timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._url)

    def detect_type(self) -> str:
        """Detect webhook type from URL pattern."""
        if not self._url:
            return "none"
        if "hooks.slack.com" in self._url:
            return "slack"
        if "discord.com/api/webhooks" in self._url or "discordapp.com/api/webhooks" in self._url:
            return "discord"
        return "generic"

    def format_slack(self, results: dict) -> dict:
        """Format results as a Slack incoming webhook payload."""
        summary = results.get("summary", {})
        jobs_new = summary.get("jobs_new", 0)
        high_risk = summary.get("high_risk_count", 0)
        errors = summary.get("errors", [])
        duration = summary.get("duration_seconds", 0)

        status_emoji = ":warning:" if high_risk > 0 else ":white_check_mark:"
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} Sentinel Scan Complete",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*New Jobs:* {jobs_new}"},
                    {"type": "mrkdwn", "text": f"*High Risk:* {high_risk}"},
                    {"type": "mrkdwn", "text": f"*Duration:* {duration:.1f}s"},
                    {"type": "mrkdwn", "text": f"*Errors:* {len(errors)}"},
                ],
            },
        ]

        if errors:
            error_text = "\n".join(f"- {e}" for e in errors[:5])
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Errors:*\n{error_text}"},
            })

        # Add phase details if present
        phases = results.get("phases", {})
        if phases:
            phase_parts = []
            for phase_name, phase_data in phases.items():
                status = phase_data.get("status", "unknown")
                icon = ":white_check_mark:" if status == "success" else ":x:"
                phase_parts.append(f"{icon} {phase_name}")
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": " | ".join(phase_parts)},
            })

        return {"blocks": blocks}

    def format_discord(self, results: dict) -> dict:
        """Format results as a Discord webhook payload."""
        summary = results.get("summary", {})
        jobs_new = summary.get("jobs_new", 0)
        high_risk = summary.get("high_risk_count", 0)
        errors = summary.get("errors", [])
        duration = summary.get("duration_seconds", 0)

        color = 0xFF0000 if high_risk > 0 else 0x00FF00

        fields = [
            {"name": "New Jobs", "value": str(jobs_new), "inline": True},
            {"name": "High Risk", "value": str(high_risk), "inline": True},
            {"name": "Duration", "value": f"{duration:.1f}s", "inline": True},
        ]

        if errors:
            error_text = "\n".join(f"- {e}" for e in errors[:5])
            fields.append({"name": "Errors", "value": error_text, "inline": False})

        embed = {
            "title": "Sentinel Scan Complete",
            "color": color,
            "fields": fields,
            "timestamp": _now_iso(),
        }

        return {"embeds": [embed]}

    def format_generic(self, results: dict) -> dict:
        """Format results as a generic JSON payload."""
        return {
            "event": "sentinel_scan_complete",
            "timestamp": _now_iso(),
            "results": results,
        }

    def format_results(self, results: dict) -> dict:
        """Auto-detect webhook type and format accordingly."""
        wh_type = self.detect_type()
        if wh_type == "slack":
            return self.format_slack(results)
        elif wh_type == "discord":
            return self.format_discord(results)
        else:
            return self.format_generic(results)

    def send(self, results: dict) -> dict[str, Any]:
        """Send formatted results to the configured webhook.

        Returns a dict with status info. Uses only stdlib urllib.
        """
        if not self._url:
            return {"sent": False, "reason": "no webhook URL configured"}

        payload = self.format_results(results)
        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                status_code = resp.status
                response_body = resp.read().decode("utf-8", errors="replace")
            return {
                "sent": True,
                "status_code": status_code,
                "webhook_type": self.detect_type(),
                "response": response_body[:500],
            }
        except urllib.error.HTTPError as exc:
            return {
                "sent": False,
                "status_code": exc.code,
                "webhook_type": self.detect_type(),
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "sent": False,
                "webhook_type": self.detect_type(),
                "error": str(exc),
            }


# ---------------------------------------------------------------------------
# CloudRunner — serverless execution wrapper
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    """Result of a single cloud phase execution."""

    phase: str
    status: str  # "success" | "error" | "skipped"
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    data: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CloudRunner:
    """Wraps daemon phases for stateless serverless execution.

    Each method is idempotent and safe to retry.  Accepts a DB URL
    (local SQLite or cloud Turso/D1) and returns structured JSON
    results suitable for GitHub Actions summaries or API responses.
    """

    def __init__(
        self,
        db_url: str = "",
        config: CloudConfig | None = None,
    ) -> None:
        self._config = config or CloudConfig.from_env()
        self._db_url = db_url or self._config.db_url
        self._adapter: TursoAdapter | None = None
        self._webhook: WebhookReporter | None = None

    def _get_adapter(self) -> TursoAdapter:
        """Lazy-init the database adapter."""
        if self._adapter is None:
            self._adapter = TursoAdapter(db_url=self._db_url)
        return self._adapter

    def _get_db(self) -> Any:
        """Return the underlying DB object."""
        return self._get_adapter().db

    def _get_db_path(self) -> str | None:
        """Return a file path for SentinelDB, or None for Turso."""
        adapter = self._get_adapter()
        if adapter.is_turso:
            return None
        db = adapter.db
        return getattr(db, "path", None)

    def _get_webhook(self) -> WebhookReporter:
        """Lazy-init the webhook reporter."""
        if self._webhook is None:
            url = self._config.report_webhook
            self._webhook = WebhookReporter(webhook_url=url)
        return self._webhook

    def _queries(self) -> list[str]:
        """Return scan queries from config or defaults."""
        if self._config.scan_queries:
            return self._config.scan_queries
        return [
            "software engineer",
            "data analyst",
            "remote work from home",
        ]

    # ------------------------------------------------------------------
    # Phase: INGEST
    # ------------------------------------------------------------------

    def run_ingest(
        self,
        queries: list[str] | None = None,
        location: str = "",
    ) -> PhaseResult:
        """Run the ingestion phase: fetch jobs from sources, score, persist.

        Idempotent: duplicate jobs are automatically skipped.
        """
        started = _now_iso()
        start_time = datetime.now(UTC)
        use_queries = queries or self._queries()

        try:
            from sentinel.ingest import IngestionPipeline

            db_path = self._get_db_path()
            pipeline = IngestionPipeline(db_path=db_path)

            runs = pipeline.auto_ingest(
                queries=use_queries,
                location=location,
                run_flywheel=False,
            )

            total_fetched = sum(r.jobs_fetched for r in runs)
            total_new = sum(r.jobs_new for r in runs)
            total_high_risk = sum(r.high_risk_count for r in runs)
            all_errors: list[str] = []
            for r in runs:
                all_errors.extend(r.errors)

            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()

            return PhaseResult(
                phase="ingest",
                status="success",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                data={
                    "queries": use_queries,
                    "location": location,
                    "jobs_fetched": total_fetched,
                    "jobs_new": total_new,
                    "high_risk_count": total_high_risk,
                    "runs": len(runs),
                    "errors": all_errors,
                },
            )
        except Exception as exc:
            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.error("Cloud ingest failed: %s", exc)
            return PhaseResult(
                phase="ingest",
                status="error",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Phase: SCORE (flywheel cycle)
    # ------------------------------------------------------------------

    def run_score(self) -> PhaseResult:
        """Run the scoring/flywheel phase: learn, evolve, detect regression.

        Idempotent: safe to run multiple times (metrics accumulate).
        """
        started = _now_iso()
        start_time = datetime.now(UTC)

        try:
            from sentinel.db import SentinelDB
            from sentinel.flywheel import DetectionFlywheel

            db_path = self._get_db_path()
            db = SentinelDB(path=db_path) if db_path else SentinelDB()
            flywheel = DetectionFlywheel(db=db)
            metrics = flywheel.run_cycle()

            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()

            return PhaseResult(
                phase="score",
                status="success",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                data={
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "regression_alarm": metrics.get("regression_alarm", False),
                    "cusum_statistic": metrics.get("cusum_statistic", 0.0),
                    "patterns_evolved": metrics.get("patterns_evolved", 0),
                    "calibration_ece": metrics.get("calibration_ece", 0.0),
                },
            )
        except Exception as exc:
            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.error("Cloud score failed: %s", exc)
            return PhaseResult(
                phase="score",
                status="error",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Phase: EVOLVE (pattern lifecycle)
    # ------------------------------------------------------------------

    def run_evolve(self) -> PhaseResult:
        """Run the evolution phase: promote/deprecate patterns, check regression.

        Idempotent: pattern lifecycle transitions are based on observation counts.
        """
        started = _now_iso()
        start_time = datetime.now(UTC)

        try:
            from sentinel.db import SentinelDB
            from sentinel.flywheel import DetectionFlywheel

            db_path = self._get_db_path()
            db = SentinelDB(path=db_path) if db_path else SentinelDB()
            flywheel = DetectionFlywheel(db=db)

            evolution = flywheel.evolve_patterns()
            regression = flywheel.detect_regression()

            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()

            return PhaseResult(
                phase="evolve",
                status="success",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                data={
                    "promoted": evolution.get("promoted", []),
                    "deprecated": evolution.get("deprecated", []),
                    "retained_count": evolution.get("retained_count", 0),
                    "cascade_risk": evolution.get("cascade_risk", "SAFE"),
                    "regression_alarm": regression.get("alarm", False),
                    "cusum_statistic": regression.get("cusum_statistic", 0.0),
                    "rolling_precision": regression.get("rolling_precision"),
                },
            )
        except Exception as exc:
            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.error("Cloud evolve failed: %s", exc)
            return PhaseResult(
                phase="evolve",
                status="error",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Phase: INNOVATE (Thompson Sampling improvement)
    # ------------------------------------------------------------------

    def run_innovate(self, max_strategies: int = 3) -> PhaseResult:
        """Run the innovation phase: Thompson Sampling strategy exploration.

        Idempotent: strategy selection is stochastic but safe to retry.
        """
        started = _now_iso()
        start_time = datetime.now(UTC)

        try:
            from sentinel.db import SentinelDB
            from sentinel.innovation import InnovationEngine

            db_path = self._get_db_path()
            db = SentinelDB(path=db_path) if db_path else SentinelDB()
            engine = InnovationEngine(db=db)
            results = engine.run_cycle(max_strategies=max_strategies)

            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()

            return PhaseResult(
                phase="innovate",
                status="success",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                data={
                    "strategies_run": len(results),
                    "successful": sum(1 for r in results if r.precision_delta > 0),
                    "strategies": [
                        {
                            "name": r.strategy,
                            "success": r.success,
                            "detail": r.detail,
                            "precision_delta": r.precision_delta,
                            "new_patterns": r.new_patterns,
                        }
                        for r in results
                    ],
                    "total_precision_delta": sum(r.precision_delta for r in results),
                },
            )
        except Exception as exc:
            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.error("Cloud innovate failed: %s", exc)
            return PhaseResult(
                phase="innovate",
                status="error",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Phase: RESEARCH
    # ------------------------------------------------------------------

    def run_research(self, budget: int = 2) -> PhaseResult:
        """Run the research phase: discover new fraud detection knowledge.

        Idempotent: research topics are tracked to avoid redundant work.
        """
        started = _now_iso()
        start_time = datetime.now(UTC)

        try:
            from sentinel.db import SentinelDB
            from sentinel.research import ResearchEngine

            db_path = self._get_db_path()
            db = SentinelDB(path=db_path) if db_path else SentinelDB()
            research = ResearchEngine(db=db, research_budget=budget)
            results = research.run_cycle()

            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()

            return PhaseResult(
                phase="research",
                status="success",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                data={
                    "topics_researched": len(results),
                    "patterns_found": sum(
                        len(r.extracted_patterns) for r in results
                    ),
                    "topics": [
                        {
                            "topic": r.topic,
                            "patterns": len(r.extracted_patterns),
                        }
                        for r in results
                    ],
                },
            )
        except Exception as exc:
            completed = _now_iso()
            duration = (datetime.now(UTC) - start_time).total_seconds()
            logger.error("Cloud research failed: %s", exc)
            return PhaseResult(
                phase="research",
                status="error",
                started_at=started,
                completed_at=completed,
                duration_seconds=round(duration, 2),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------

    def run_full_cycle(
        self,
        queries: list[str] | None = None,
        location: str = "",
        skip_research: bool = False,
    ) -> dict[str, Any]:
        """Run a complete cloud scan cycle (all phases).

        Returns a structured dict suitable for JSON serialisation.
        """
        cycle_started = _now_iso()
        cycle_start_time = datetime.now(UTC)
        phases: dict[str, dict] = {}
        all_errors: list[str] = []

        # Phase 1: Ingest
        ingest_result = self.run_ingest(queries=queries, location=location)
        phases["ingest"] = ingest_result.to_dict()
        if ingest_result.error:
            all_errors.append(f"ingest: {ingest_result.error}")

        # Phase 2: Score
        score_result = self.run_score()
        phases["score"] = score_result.to_dict()
        if score_result.error:
            all_errors.append(f"score: {score_result.error}")

        # Phase 3: Evolve
        evolve_result = self.run_evolve()
        phases["evolve"] = evolve_result.to_dict()
        if evolve_result.error:
            all_errors.append(f"evolve: {evolve_result.error}")

        # Phase 4: Innovate
        innovate_result = self.run_innovate()
        phases["innovate"] = innovate_result.to_dict()
        if innovate_result.error:
            all_errors.append(f"innovate: {innovate_result.error}")

        # Phase 5: Research (optional)
        if not skip_research:
            research_result = self.run_research()
            phases["research"] = research_result.to_dict()
            if research_result.error:
                all_errors.append(f"research: {research_result.error}")

        cycle_completed = _now_iso()
        cycle_duration = (datetime.now(UTC) - cycle_start_time).total_seconds()

        # Build summary
        ingest_data = ingest_result.data
        summary = {
            "started_at": cycle_started,
            "completed_at": cycle_completed,
            "duration_seconds": round(cycle_duration, 2),
            "jobs_fetched": ingest_data.get("jobs_fetched", 0),
            "jobs_new": ingest_data.get("jobs_new", 0),
            "high_risk_count": ingest_data.get("high_risk_count", 0),
            "regression_alarm": score_result.data.get("regression_alarm", False),
            "errors": all_errors,
        }

        result = {
            "summary": summary,
            "phases": phases,
        }

        # Send webhook notification if configured
        webhook = self._get_webhook()
        if webhook.configured:
            try:
                webhook_result = webhook.send(result)
                result["webhook"] = webhook_result
            except Exception as exc:
                result["webhook"] = {"sent": False, "error": str(exc)}

        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return cloud configuration and connectivity status."""
        adapter = self._get_adapter()
        connectivity = adapter.check_connectivity()
        webhook = self._get_webhook()

        return {
            "config": self._config.to_dict(),
            "config_issues": self._config.validate(),
            "database": connectivity,
            "webhook": {
                "configured": webhook.configured,
                "type": webhook.detect_type(),
                "url": self._config.report_webhook if self._config.report_webhook else "(not set)",
            },
            "checked_at": _now_iso(),
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._adapter is not None:
            self._adapter.close()


# ---------------------------------------------------------------------------
# CLI command registration helpers
# ---------------------------------------------------------------------------

def register_cloud_commands(cli_group: Any) -> None:
    """Register cloud-* CLI commands on the given click group.

    Called from sentinel.cli to add:
      sentinel cloud-scan
      sentinel cloud-evolve
      sentinel cloud-status
    """
    import click

    @cli_group.command("cloud-scan")
    @click.option("--queries", "-q", multiple=True, default=(),
                  help="Search queries (overrides SENTINEL_SCAN_QUERIES)")
    @click.option("--location", "-l", default="", help="Location filter")
    @click.option("--skip-research", is_flag=True, help="Skip the research phase")
    @click.option("--db-url", default="", help="Database URL (overrides SENTINEL_DB_URL)")
    @click.pass_context
    def cloud_scan(ctx: Any, queries: tuple, location: str, skip_research: bool, db_url: str) -> None:
        """Run a single cloud-compatible scan cycle (all phases)."""
        config = CloudConfig.from_env()
        if queries:
            config.scan_queries = list(queries)
        if db_url:
            config.db_url = db_url

        runner = CloudRunner(db_url=config.db_url, config=config)
        try:
            result = runner.run_full_cycle(
                queries=list(queries) if queries else None,
                location=location,
                skip_research=skip_research,
            )

            if ctx.obj.get("json"):
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                summary = result.get("summary", {})
                click.echo("")
                click.echo(click.style("  Cloud Scan Complete", bold=True))
                click.echo("  " + "-" * 50)
                click.echo(f"  Jobs fetched:  {summary.get('jobs_fetched', 0)}")
                click.echo(f"  Jobs new:      {summary.get('jobs_new', 0)}")
                click.echo(f"  High risk:     {summary.get('high_risk_count', 0)}")
                click.echo(f"  Duration:      {summary.get('duration_seconds', 0):.1f}s")
                errors = summary.get("errors", [])
                if errors:
                    click.echo(f"  Errors:        {len(errors)}")
                    for e in errors[:5]:
                        click.echo(f"    - {e}")

                # Phase statuses
                phases = result.get("phases", {})
                click.echo("")
                for phase_name, phase_data in phases.items():
                    status = phase_data.get("status", "unknown")
                    icon = click.style("OK", fg="green") if status == "success" else click.style("FAIL", fg="red")
                    click.echo(f"  {phase_name:<12} [{icon}]  {phase_data.get('duration_seconds', 0):.1f}s")
                click.echo("")
        finally:
            runner.close()

    @cli_group.command("cloud-evolve")
    @click.option("--db-url", default="", help="Database URL (overrides SENTINEL_DB_URL)")
    @click.option("--max-strategies", default=3, help="Max innovation strategies per cycle")
    @click.pass_context
    def cloud_evolve(ctx: Any, db_url: str, max_strategies: int) -> None:
        """Run a single cloud-compatible evolution cycle (evolve + innovate)."""
        config = CloudConfig.from_env()
        if db_url:
            config.db_url = db_url

        runner = CloudRunner(db_url=config.db_url, config=config)
        try:
            evolve_result = runner.run_evolve()
            innovate_result = runner.run_innovate(max_strategies=max_strategies)

            result = {
                "evolve": evolve_result.to_dict(),
                "innovate": innovate_result.to_dict(),
            }

            if ctx.obj.get("json"):
                click.echo(json.dumps(result, indent=2, default=str))
            else:
                click.echo("")
                click.echo(click.style("  Cloud Evolution Complete", bold=True))
                click.echo("  " + "-" * 50)

                # Evolve details
                ed = evolve_result.data
                click.echo(f"  Promoted:      {len(ed.get('promoted', []))}")
                click.echo(f"  Deprecated:    {len(ed.get('deprecated', []))}")
                click.echo(f"  Retained:      {ed.get('retained_count', 0)}")
                click.echo(f"  Cascade risk:  {ed.get('cascade_risk', 'SAFE')}")
                alarm = ed.get("regression_alarm", False)
                alarm_style = click.style("YES", fg="red") if alarm else click.style("no", fg="green")
                click.echo(f"  Regression:    {alarm_style}")

                # Innovation details
                click.echo("")
                idata = innovate_result.data
                click.echo(f"  Strategies run: {idata.get('strategies_run', 0)}")
                click.echo(f"  Successful:     {idata.get('successful', 0)}")
                for s in idata.get("strategies", []):
                    icon = click.style("+", fg="green") if s.get("success") else click.style("-", fg="red")
                    click.echo(f"    {icon} {s['name']}: {s.get('detail', '')[:60]}")
                click.echo("")
        finally:
            runner.close()

    @cli_group.command("cloud-status")
    @click.pass_context
    def cloud_status(ctx: Any) -> None:
        """Show cloud configuration, database connectivity, and webhook status."""
        config = CloudConfig.from_env()
        runner = CloudRunner(config=config)

        try:
            status = runner.get_status()

            if ctx.obj.get("json"):
                click.echo(json.dumps(status, indent=2, default=str))
            else:
                click.echo("")
                click.echo(click.style("  Cloud Status", bold=True))
                click.echo("  " + "=" * 50)

                # Config
                cfg = status.get("config", {})
                click.echo(click.style("  Configuration:", bold=True))
                click.echo(f"    DB URL:         {cfg.get('db_url', '?')}")
                click.echo(f"    API Key:        {cfg.get('api_key', '?')}")
                click.echo(f"    Scan Queries:   {cfg.get('scan_queries', [])}")
                click.echo(f"    Interval:       {cfg.get('scan_interval_hours', '?')}h")
                click.echo(f"    Webhook:        {cfg.get('report_webhook', '?')}")

                issues = status.get("config_issues", [])
                if issues:
                    click.echo("")
                    click.echo(click.style("  Warnings:", fg="yellow"))
                    for issue in issues:
                        click.echo(f"    - {issue}")

                # Database
                db = status.get("database", {})
                connected = db.get("connected", False)
                conn_style = click.style("Connected", fg="green") if connected else click.style("DISCONNECTED", fg="red")
                click.echo("")
                click.echo(click.style("  Database:", bold=True))
                click.echo(f"    Status:     {conn_style}")
                click.echo(f"    Backend:    {db.get('backend', '?')}")
                if db.get("stats"):
                    stats = db["stats"]
                    click.echo(f"    Jobs:       {stats.get('total_jobs_analyzed', 0)}")
                    click.echo(f"    Reports:    {stats.get('total_user_reports', 0)}")

                # Webhook
                wh = status.get("webhook", {})
                click.echo("")
                click.echo(click.style("  Webhook:", bold=True))
                click.echo(f"    Configured: {'Yes' if wh.get('configured') else 'No'}")
                click.echo(f"    Type:       {wh.get('type', 'none')}")
                click.echo("")
        finally:
            runner.close()
