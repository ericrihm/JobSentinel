"""SQLite persistence layer for Sentinel — WAL mode + FTS5 full-text search."""

import contextlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".sentinel", "sentinel.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,
    title TEXT,
    company TEXT,
    location TEXT,
    description TEXT,
    salary_min REAL,
    salary_max REAL,
    scam_score REAL,
    confidence REAL,
    risk_level TEXT,
    analyzed_at TEXT,
    signal_count INTEGER DEFAULT 0,
    signals_json TEXT DEFAULT '[]',
    user_reported INTEGER DEFAULT 0,
    user_verdict TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,
    regex TEXT,
    keywords_json TEXT DEFAULT '[]',
    alpha REAL DEFAULT 1.0,
    beta REAL DEFAULT 1.0,
    observations INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active',
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT,
    is_scam INTEGER,
    reason TEXT,
    our_prediction REAL,
    was_correct INTEGER,
    reported_at TEXT
);

CREATE TABLE IF NOT EXISTS companies (
    name TEXT PRIMARY KEY,
    domain TEXT,
    employee_count INTEGER,
    is_verified INTEGER DEFAULT 0,
    linkedin_url TEXT,
    glassdoor_rating REAL,
    whois_age_days INTEGER DEFAULT 0,
    last_checked TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS jobs_fts USING fts5(
    title, company, description, content='jobs', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS jobs_ai AFTER INSERT ON jobs BEGIN
    INSERT INTO jobs_fts(rowid, title, company, description) VALUES (new.id, new.title, new.company, new.description);
END;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    sources TEXT,
    query TEXT,
    location TEXT,
    jobs_fetched INTEGER DEFAULT 0,
    jobs_new INTEGER DEFAULT 0,
    jobs_scored INTEGER DEFAULT 0,
    high_risk_count INTEGER DEFAULT 0,
    errors TEXT
);

CREATE TABLE IF NOT EXISTS flywheel_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_ts TEXT,
    total_analyzed INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    precision REAL DEFAULT 0.0,
    recall REAL DEFAULT 0.0,
    signals_updated INTEGER DEFAULT 0,
    patterns_evolved INTEGER DEFAULT 0,
    f1 REAL DEFAULT 0.0,
    accuracy REAL DEFAULT 0.0,
    cycle_number INTEGER DEFAULT 0,
    regression_alarm INTEGER DEFAULT 0,
    cusum_statistic REAL DEFAULT 0.0,
    patterns_promoted INTEGER DEFAULT 0,
    patterns_deprecated INTEGER DEFAULT 0,
    calibration_ece REAL DEFAULT 0.0,
    thresholds_adjusted INTEGER DEFAULT 0,
    shadow_evaluation_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS source_stats (
    source TEXT PRIMARY KEY,
    jobs_ingested INT DEFAULT 0,
    scams_detected INT DEFAULT 0,
    avg_score REAL DEFAULT 0.0,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS signal_rate_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    fire_count INTEGER DEFAULT 0,
    total_jobs INTEGER DEFAULT 0,
    recorded_at TEXT
);

CREATE TABLE IF NOT EXISTS shadow_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_weights TEXT NOT NULL DEFAULT '{}',
    status TEXT DEFAULT 'active',
    baseline_precision REAL DEFAULT 0.0,
    shadow_precision REAL DEFAULT 0.0,
    jobs_evaluated INTEGER DEFAULT 0,
    promoted INTEGER DEFAULT 0,
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS salary_benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    level TEXT NOT NULL,
    p25 INTEGER NOT NULL,
    p50 INTEGER NOT NULL,
    p75 INTEGER NOT NULL,
    p90 INTEGER NOT NULL,
    UNIQUE(category, level)
);

CREATE TABLE IF NOT EXISTS scam_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    domain TEXT,
    type TEXT,
    source TEXT,
    added_at TEXT
);

CREATE TABLE IF NOT EXISTS posting_velocity (
    company_name TEXT PRIMARY KEY,
    postings_24h INTEGER DEFAULT 0,
    postings_7d INTEGER DEFAULT 0,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS description_hashes (
    hash TEXT NOT NULL,
    company_name TEXT NOT NULL,
    job_url TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    PRIMARY KEY (hash, company_name)
);

CREATE TABLE IF NOT EXISTS near_misses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name TEXT NOT NULL,
    partial_match TEXT NOT NULL,
    job_url TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signal_decay_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name TEXT NOT NULL,
    window_start TEXT NOT NULL,
    fire_rate REAL NOT NULL DEFAULT 0.0,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    prompt TEXT,
    response_summary TEXT,
    patterns_extracted INTEGER DEFAULT 0,
    patterns_adopted INTEGER DEFAULT 0,
    precision_delta REAL DEFAULT 0.0,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS research_topics (
    topic TEXT PRIMARY KEY,
    priority REAL DEFAULT 0.5,
    last_researched TEXT,
    total_patterns_found INTEGER DEFAULT 0,
    avg_precision_impact REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS ensemble_method_accuracy (
    method_name TEXT PRIMARY KEY,
    alpha REAL DEFAULT 1.0,
    beta REAL DEFAULT 1.0,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS feedback_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    is_scam INTEGER NOT NULL,
    our_prediction REAL,
    source TEXT DEFAULT 'manual',
    reason TEXT DEFAULT '',
    was_correct INTEGER,
    reported_at TEXT
);

CREATE TABLE IF NOT EXISTS cascade_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger TEXT NOT NULL,
    change_type TEXT NOT NULL DEFAULT '',
    impact_json TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS flywheel_mesh_edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'data',
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (source, target)
);

CREATE TABLE IF NOT EXISTS cortex_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_number INTEGER NOT NULL,
    state_json TEXT NOT NULL DEFAULT '{}',
    learning_velocity REAL DEFAULT 0.0,
    health_grade TEXT DEFAULT 'C',
    strategic_mode TEXT DEFAULT 'OBSERVE',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cortex_investigations (
    id TEXT PRIMARY KEY,
    trigger TEXT NOT NULL,
    hypothesis TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    actions_json TEXT NOT NULL DEFAULT '[]',
    opened_at TEXT NOT NULL,
    resolved_at TEXT,
    resolution TEXT
);

CREATE TABLE IF NOT EXISTS cortex_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    priority REAL DEFAULT 0.5,
    acted_on INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


class SentinelDB:
    def __init__(self, path: str = "") -> None:
        if path:
            self.path = path
        else:
            from sentinel.config import get_config
            self.path = get_config().db_path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

        # Migrate existing DBs
        for col_sql in [
            "ALTER TABLE flywheel_metrics ADD COLUMN f1 REAL DEFAULT 0.0",
            "ALTER TABLE flywheel_metrics ADD COLUMN accuracy REAL DEFAULT 0.0",
            "ALTER TABLE flywheel_metrics ADD COLUMN cycle_number INTEGER DEFAULT 0",
            "ALTER TABLE flywheel_metrics ADD COLUMN regression_alarm INTEGER DEFAULT 0",
            "ALTER TABLE flywheel_metrics ADD COLUMN cusum_statistic REAL DEFAULT 0.0",
            "ALTER TABLE flywheel_metrics ADD COLUMN patterns_promoted INTEGER DEFAULT 0",
            "ALTER TABLE flywheel_metrics ADD COLUMN patterns_deprecated INTEGER DEFAULT 0",
            # Calibration / confidence additions
            "ALTER TABLE jobs ADD COLUMN confidence REAL",
            # Flywheel metrics: calibration + shadow + thresholds
            "ALTER TABLE flywheel_metrics ADD COLUMN calibration_ece REAL DEFAULT 0.0",
            "ALTER TABLE flywheel_metrics ADD COLUMN thresholds_adjusted INTEGER DEFAULT 0",
            "ALTER TABLE flywheel_metrics ADD COLUMN shadow_evaluation_json TEXT DEFAULT '{}'",
        ]:
            with contextlib.suppress(sqlite3.OperationalError):
                self.conn.execute(col_sql)
        self.conn.commit()

        # Seed reference tables (no-ops if already populated)
        self.seed_salary_benchmarks()
        self.seed_scam_entities()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def save_job(self, job_data: dict) -> None:
        """Insert or replace a job record."""
        signals = job_data.get("signals_json", job_data.get("signals", []))
        if not isinstance(signals, str):
            signals = json.dumps(signals)

        self.conn.execute(
            """
            INSERT INTO jobs
                (url, title, company, location, description,
                 salary_min, salary_max, scam_score, confidence, risk_level,
                 analyzed_at, signal_count, signals_json,
                 user_reported, user_verdict)
            VALUES
                (:url, :title, :company, :location, :description,
                 :salary_min, :salary_max, :scam_score, :confidence, :risk_level,
                 :analyzed_at, :signal_count, :signals_json,
                 :user_reported, :user_verdict)
            ON CONFLICT(url) DO UPDATE SET
                title         = excluded.title,
                company       = excluded.company,
                location      = excluded.location,
                description   = excluded.description,
                salary_min    = excluded.salary_min,
                salary_max    = excluded.salary_max,
                scam_score    = excluded.scam_score,
                confidence    = excluded.confidence,
                risk_level    = excluded.risk_level,
                analyzed_at   = excluded.analyzed_at,
                signal_count  = excluded.signal_count,
                signals_json  = excluded.signals_json,
                user_reported = excluded.user_reported,
                user_verdict  = excluded.user_verdict
            """,
            {
                "url": job_data.get("url", ""),
                "title": job_data.get("title", ""),
                "company": job_data.get("company", ""),
                "location": job_data.get("location", ""),
                "description": job_data.get("description", ""),
                "salary_min": job_data.get("salary_min", 0.0),
                "salary_max": job_data.get("salary_max", 0.0),
                "scam_score": job_data.get("scam_score", 0.0),
                "confidence": job_data.get("confidence"),
                "risk_level": job_data.get("risk_level", ""),
                "analyzed_at": job_data.get("analyzed_at", _now_iso()),
                "signal_count": job_data.get("signal_count", 0),
                "signals_json": signals,
                "user_reported": int(job_data.get("user_reported", False)),
                "user_verdict": job_data.get("user_verdict", ""),
            },
        )
        self.conn.commit()

    def get_job(self, url: str) -> dict | None:
        """Fetch a single job by URL. Returns None if not found."""
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE url = ?", (url,)
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        # Deserialise signals_json for convenience
        try:
            result["signals"] = json.loads(result.get("signals_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            result["signals"] = []
        return result

    def search_jobs(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over title, company, and description via FTS5."""
        # Wrap as an FTS5 phrase so special characters (-, ", :, *) are
        # treated as literals rather than operators. Doubled quotes inside
        # the phrase escape an embedded quote per FTS5 syntax.
        safe_query = '"' + query.replace('"', '""') + '"'
        rows = self.conn.execute(
            """
            SELECT j.*
            FROM jobs j
            JOIN jobs_fts f ON j.id = f.rowid
            WHERE jobs_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (safe_query, limit),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            try:
                d["signals"] = json.loads(d.get("signals_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                d["signals"] = []
            results.append(d)
        return results

    def get_job_by_url(self, url: str) -> dict | None:
        """Check if a job with this URL already exists. Returns the row dict or None."""
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE url = ?", (url,)
        ).fetchone()
        return _row_to_dict(row)

    def get_jobs_for_review(
        self,
        score_threshold: float = 0.5,
        confidence_threshold: float = 0.4,
        limit: int = 100,
    ) -> list[dict]:
        """Return jobs with high scam score but low model confidence — need human review.

        Criteria: scam_score > score_threshold AND confidence < confidence_threshold
        AND confidence IS NOT NULL.
        Results are ordered by scam_score descending (highest risk first).
        """
        rows = self.conn.execute(
            """
            SELECT * FROM jobs
            WHERE scam_score > ?
              AND confidence IS NOT NULL
              AND confidence < ?
            ORDER BY scam_score DESC
            LIMIT ?
            """,
            (score_threshold, confidence_threshold, limit),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            try:
                d["signals"] = json.loads(d.get("signals_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                d["signals"] = []
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Ingestion runs
    # ------------------------------------------------------------------

    def save_ingestion_run(self, run: dict) -> None:
        """Persist an IngestionRun record."""
        sources = run.get("sources", run.get("sources_queried", []))
        if not isinstance(sources, str):
            sources = json.dumps(sources)
        errors = run.get("errors", [])
        if not isinstance(errors, str):
            errors = json.dumps(errors)

        self.conn.execute(
            """
            INSERT INTO ingestion_runs
                (run_id, started_at, completed_at, sources, query, location,
                 jobs_fetched, jobs_new, jobs_scored, high_risk_count, errors)
            VALUES
                (:run_id, :started_at, :completed_at, :sources, :query, :location,
                 :jobs_fetched, :jobs_new, :jobs_scored, :high_risk_count, :errors)
            ON CONFLICT(run_id) DO UPDATE SET
                completed_at   = excluded.completed_at,
                jobs_fetched   = excluded.jobs_fetched,
                jobs_new       = excluded.jobs_new,
                jobs_scored    = excluded.jobs_scored,
                high_risk_count= excluded.high_risk_count,
                errors         = excluded.errors
            """,
            {
                "run_id": run.get("run_id", ""),
                "started_at": run.get("started_at", _now_iso()),
                "completed_at": run.get("completed_at"),
                "sources": sources,
                "query": run.get("query", ""),
                "location": run.get("location", ""),
                "jobs_fetched": run.get("jobs_fetched", 0),
                "jobs_new": run.get("jobs_new", 0),
                "jobs_scored": run.get("jobs_scored", 0),
                "high_risk_count": run.get("high_risk_count", 0),
                "errors": errors,
            },
        )
        self.conn.commit()

    def get_ingestion_history(self, limit: int = 20) -> list[dict]:
        """Return the most recent ingestion runs, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            for field in ("sources", "errors"):
                raw = d.get(field, "[]")
                try:
                    d[field] = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def save_report(self, report_data: dict) -> None:
        """Persist a user-submitted scam/legitimate report."""
        self.conn.execute(
            """
            INSERT INTO reports (url, is_scam, reason, our_prediction, was_correct, reported_at)
            VALUES (:url, :is_scam, :reason, :our_prediction, :was_correct, :reported_at)
            """,
            {
                "url": report_data.get("url", ""),
                "is_scam": int(report_data.get("is_scam", False)),
                "reason": report_data.get("reason", ""),
                "our_prediction": report_data.get("our_prediction", 0.0),
                "was_correct": int(report_data.get("was_correct", False)),
                "reported_at": report_data.get("reported_at", _now_iso()),
            },
        )
        # Mark the job as user-reported and store the verdict
        verdict = "scam" if report_data.get("is_scam") else "legitimate"
        self.conn.execute(
            "UPDATE jobs SET user_reported = 1, user_verdict = ? WHERE url = ?",
            (verdict, report_data.get("url", "")),
        )
        self.conn.commit()

    def get_reports(self, limit: int = 50) -> list[dict]:
        """Return the most recent user reports, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM reports ORDER BY reported_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def save_pattern(self, pattern_data: dict) -> None:
        """Insert or replace a scam detection pattern."""
        keywords = pattern_data.get("keywords_json", pattern_data.get("keywords", []))
        if not isinstance(keywords, str):
            keywords = json.dumps(keywords)

        now = _now_iso()
        self.conn.execute(
            """
            INSERT INTO patterns
                (pattern_id, name, description, category, regex,
                 keywords_json, alpha, beta, observations,
                 true_positives, false_positives, status, created_at, updated_at)
            VALUES
                (:pattern_id, :name, :description, :category, :regex,
                 :keywords_json, :alpha, :beta, :observations,
                 :true_positives, :false_positives, :status, :created_at, :updated_at)
            ON CONFLICT(pattern_id) DO UPDATE SET
                name           = excluded.name,
                description    = excluded.description,
                category       = excluded.category,
                regex          = excluded.regex,
                keywords_json  = excluded.keywords_json,
                alpha          = excluded.alpha,
                beta           = excluded.beta,
                observations   = excluded.observations,
                true_positives = excluded.true_positives,
                false_positives= excluded.false_positives,
                status         = excluded.status,
                updated_at     = excluded.updated_at
            """,
            {
                "pattern_id": pattern_data["pattern_id"],
                "name": pattern_data.get("name", ""),
                "description": pattern_data.get("description", ""),
                "category": pattern_data.get("category", "red_flag"),
                "regex": pattern_data.get("regex", ""),
                "keywords_json": keywords,
                "alpha": pattern_data.get("alpha", 1.0),
                "beta": pattern_data.get("beta", 1.0),
                "observations": pattern_data.get("observations", 0),
                "true_positives": pattern_data.get("true_positives", 0),
                "false_positives": pattern_data.get("false_positives", 0),
                "status": pattern_data.get("status", "active"),
                "created_at": pattern_data.get("created_at", now),
                "updated_at": pattern_data.get("updated_at", now),
            },
        )
        self.conn.commit()

    def get_patterns(self, status: str = "active") -> list[dict]:
        """Return all patterns with the given status."""
        rows = self.conn.execute(
            "SELECT * FROM patterns WHERE status = ? ORDER BY alpha DESC",
            (status,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            try:
                d["keywords"] = json.loads(d.get("keywords_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                d["keywords"] = []
            results.append(d)
        return results

    def update_pattern_stats(self, pattern_id: str, is_true_positive: bool) -> None:
        """Increment observations and TP/FP counters; update Bayesian alpha/beta."""
        row = self.conn.execute(
            "SELECT alpha, beta, observations, true_positives, false_positives FROM patterns WHERE pattern_id = ?",
            (pattern_id,),
        ).fetchone()
        if row is None:
            return

        alpha = row["alpha"]
        beta = row["beta"]
        observations = row["observations"] + 1
        true_positives = row["true_positives"] + (1 if is_true_positive else 0)
        false_positives = row["false_positives"] + (0 if is_true_positive else 1)

        # Thompson Sampling / Beta-Binomial update
        new_alpha = alpha + (1.0 if is_true_positive else 0.0)
        new_beta = beta + (0.0 if is_true_positive else 1.0)

        self.conn.execute(
            """
            UPDATE patterns
            SET alpha = ?, beta = ?, observations = ?,
                true_positives = ?, false_positives = ?, updated_at = ?
            WHERE pattern_id = ?
            """,
            (new_alpha, new_beta, observations, true_positives, false_positives, _now_iso(), pattern_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Companies
    # ------------------------------------------------------------------

    def save_company(self, company_data: dict) -> None:
        """Insert or replace a company record."""
        self.conn.execute(
            """
            INSERT INTO companies
                (name, domain, employee_count, is_verified,
                 linkedin_url, glassdoor_rating, whois_age_days, last_checked)
            VALUES
                (:name, :domain, :employee_count, :is_verified,
                 :linkedin_url, :glassdoor_rating, :whois_age_days, :last_checked)
            ON CONFLICT(name) DO UPDATE SET
                domain          = excluded.domain,
                employee_count  = excluded.employee_count,
                is_verified     = excluded.is_verified,
                linkedin_url    = excluded.linkedin_url,
                glassdoor_rating= excluded.glassdoor_rating,
                whois_age_days  = excluded.whois_age_days,
                last_checked    = excluded.last_checked
            """,
            {
                "name": company_data["name"],
                "domain": company_data.get("domain", ""),
                "employee_count": company_data.get("employee_count", 0),
                "is_verified": int(company_data.get("is_verified", False)),
                "linkedin_url": company_data.get("linkedin_url", ""),
                "glassdoor_rating": company_data.get("glassdoor_rating", 0.0),
                "whois_age_days": company_data.get("whois_age_days", 0),
                "last_checked": company_data.get("last_checked", _now_iso()),
            },
        )
        self.conn.commit()

    def get_company(self, name: str) -> dict | None:
        """Fetch a company by exact name (case-insensitive)."""
        row = self.conn.execute(
            "SELECT * FROM companies WHERE lower(name) = lower(?)", (name,)
        ).fetchone()
        return _row_to_dict(row)

    # ------------------------------------------------------------------
    # Flywheel metrics
    # ------------------------------------------------------------------

    def save_flywheel_metrics(self, metrics: dict) -> None:
        """Append a flywheel cycle snapshot."""
        # patterns_promoted / patterns_deprecated may be lists (evolution result)
        promoted = metrics.get("patterns_promoted", 0)
        deprecated = metrics.get("patterns_deprecated", 0)
        if isinstance(promoted, list):
            promoted = len(promoted)
        if isinstance(deprecated, list):
            deprecated = len(deprecated)

        # Serialize shadow_evaluation if present
        shadow_eval = metrics.get("shadow_evaluation", {})
        if not isinstance(shadow_eval, str):
            import json as _json
            shadow_eval = _json.dumps(shadow_eval)

        self.conn.execute(
            """
            INSERT INTO flywheel_metrics
                (cycle_ts, total_analyzed, true_positives, false_positives,
                 precision, recall, signals_updated, patterns_evolved,
                 f1, accuracy, cycle_number, regression_alarm,
                 cusum_statistic, patterns_promoted, patterns_deprecated,
                 calibration_ece, thresholds_adjusted, shadow_evaluation_json)
            VALUES
                (:cycle_ts, :total_analyzed, :true_positives, :false_positives,
                 :precision, :recall, :signals_updated, :patterns_evolved,
                 :f1, :accuracy, :cycle_number, :regression_alarm,
                 :cusum_statistic, :patterns_promoted, :patterns_deprecated,
                 :calibration_ece, :thresholds_adjusted, :shadow_evaluation_json)
            """,
            {
                "cycle_ts": metrics.get("cycle_ts", _now_iso()),
                "total_analyzed": metrics.get("total_analyzed", 0),
                "true_positives": metrics.get("true_positives", 0),
                "false_positives": metrics.get("false_positives", 0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "signals_updated": metrics.get("signals_updated", 0),
                "patterns_evolved": metrics.get("patterns_evolved", 0),
                "f1": metrics.get("f1", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "cycle_number": metrics.get("cycle_number", 0),
                "regression_alarm": int(bool(metrics.get("regression_alarm", False))),
                "cusum_statistic": metrics.get("cusum_statistic", 0.0),
                "patterns_promoted": promoted,
                "patterns_deprecated": deprecated,
                "calibration_ece": metrics.get("calibration_ece", 0.0),
                "thresholds_adjusted": metrics.get("thresholds_adjusted", 0),
                "shadow_evaluation_json": shadow_eval,
            },
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a summary dict used by `sentinel stats` and the API."""
        total_jobs = self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        scam_jobs = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE scam_score >= 0.6"
        ).fetchone()[0]
        total_reports = self.conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        scam_reports = self.conn.execute(
            "SELECT COUNT(*) FROM reports WHERE is_scam = 1"
        ).fetchone()[0]
        correct_reports = self.conn.execute(
            "SELECT COUNT(*) FROM reports WHERE was_correct = 1"
        ).fetchone()[0]
        total_patterns = self.conn.execute(
            "SELECT COUNT(*) FROM patterns WHERE status = 'active'"
        ).fetchone()[0]
        total_companies = self.conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        verified_companies = self.conn.execute(
            "SELECT COUNT(*) FROM companies WHERE is_verified = 1"
        ).fetchone()[0]
        avg_score_row = self.conn.execute(
            "SELECT AVG(scam_score) FROM jobs WHERE scam_score IS NOT NULL"
        ).fetchone()
        avg_score = round(avg_score_row[0] or 0.0, 3)

        accuracy = 0.0
        if total_reports > 0:
            accuracy = round(correct_reports / total_reports, 3)

        last_cycle_row = self.conn.execute(
            "SELECT cycle_ts, precision, recall FROM flywheel_metrics ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_cycle = dict(last_cycle_row) if last_cycle_row else {}

        return {
            "total_jobs_analyzed": total_jobs,
            "scam_jobs_detected": scam_jobs,
            "total_user_reports": total_reports,
            "scam_reports": scam_reports,
            "prediction_accuracy": accuracy,
            "active_patterns": total_patterns,
            "total_companies": total_companies,
            "verified_companies": verified_companies,
            "avg_scam_score": avg_score,
            "last_flywheel_cycle": last_cycle,
        }

    # ------------------------------------------------------------------
    # Source quality stats
    # ------------------------------------------------------------------

    def upsert_source_stats(
        self,
        source: str,
        jobs_ingested: int = 0,
        scams_detected: int = 0,
        avg_score: float = 0.0,
    ) -> None:
        """Insert or update source stats, accumulating counts and refreshing avg_score."""
        self.conn.execute(
            """
            INSERT INTO source_stats (source, jobs_ingested, scams_detected, avg_score, last_updated)
            VALUES (:source, :jobs_ingested, :scams_detected, :avg_score, :last_updated)
            ON CONFLICT(source) DO UPDATE SET
                jobs_ingested  = source_stats.jobs_ingested + excluded.jobs_ingested,
                scams_detected = source_stats.scams_detected + excluded.scams_detected,
                avg_score      = excluded.avg_score,
                last_updated   = excluded.last_updated
            """,
            {
                "source": source,
                "jobs_ingested": jobs_ingested,
                "scams_detected": scams_detected,
                "avg_score": avg_score,
                "last_updated": _now_iso(),
            },
        )
        self.conn.commit()

    def get_source_stats(self) -> list[dict]:
        """Return all source stats rows, ordered by jobs_ingested descending."""
        rows = self.conn.execute(
            "SELECT * FROM source_stats ORDER BY jobs_ingested DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best_sources(self, n: int = 5) -> list[str]:
        """Return the top-n source names ranked by scam yield rate (scams_detected / jobs_ingested).

        Sources with zero jobs_ingested are excluded.
        """
        rows = self.conn.execute(
            """
            SELECT source,
                   CAST(scams_detected AS REAL) / jobs_ingested AS yield_rate
            FROM source_stats
            WHERE jobs_ingested > 0
            ORDER BY yield_rate DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        return [row["source"] for row in rows]

    # ------------------------------------------------------------------
    # Signal rate history (input drift detection)
    # ------------------------------------------------------------------

    def record_signal_rates(
        self,
        signal_rates: dict[str, int],
        total_jobs: int,
        window_start: str,
        window_end: str,
    ) -> None:
        """Insert one row per signal into signal_rate_history for the given window.

        *signal_rates* maps signal_name -> fire_count.
        """
        now = _now_iso()
        for signal_name, fire_count in signal_rates.items():
            self.conn.execute(
                """
                INSERT INTO signal_rate_history
                    (signal_name, window_start, window_end, fire_count, total_jobs, recorded_at)
                VALUES
                    (?, ?, ?, ?, ?, ?)
                """,
                (signal_name, window_start, window_end, int(fire_count), int(total_jobs), now),
            )
        self.conn.commit()

    def get_signal_rate_history(
        self,
        signal_name: str | None = None,
        since: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Return signal_rate_history rows, optionally filtered by signal_name and/or since timestamp."""
        params: list = []
        conditions: list[str] = []

        if signal_name is not None:
            conditions.append("signal_name = ?")
            params.append(signal_name)
        if since is not None:
            conditions.append("window_end >= ?")
            params.append(since)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        rows = self.conn.execute(
            f"SELECT * FROM signal_rate_history {where} ORDER BY window_end DESC LIMIT ?",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_flywheel_metrics_history(self, days: int = 30, limit: int = 200) -> list[dict]:
        """Return flywheel_metrics rows from the last *days* days, newest first."""
        from datetime import UTC, datetime, timedelta

        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            """
            SELECT * FROM flywheel_metrics
            WHERE cycle_ts >= ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (since, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Shadow runs (A/B weight testing)
    # ------------------------------------------------------------------

    def insert_shadow_run(self, candidate_weights: dict) -> int:
        """Insert a new shadow run record and return its id."""
        cursor = self.conn.execute(
            """
            INSERT INTO shadow_runs
                (candidate_weights, status, started_at)
            VALUES
                (?, 'active', ?)
            """,
            (json.dumps(candidate_weights), _now_iso()),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_active_shadow_run(self) -> dict | None:
        """Return the most recent active shadow run, or None if none exists."""
        row = self.conn.execute(
            "SELECT * FROM shadow_runs WHERE status = 'active' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["candidate_weights"] = json.loads(d.get("candidate_weights") or "{}")
        except (json.JSONDecodeError, TypeError):
            d["candidate_weights"] = {}
        return d

    def update_shadow_run(self, run_id: int, updates: dict) -> None:
        """Update fields on a shadow run record."""
        allowed = {
            "baseline_precision", "shadow_precision", "jobs_evaluated",
            "status", "promoted", "completed_at",
        }
        sets = []
        params = []
        for key, val in updates.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                params.append(val)
        if not sets:
            return
        params.append(run_id)
        self.conn.execute(
            f"UPDATE shadow_runs SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self.conn.commit()

    def promote_shadow_run(self, run_id: int) -> None:
        """Mark a shadow run as promoted."""
        self.conn.execute(
            "UPDATE shadow_runs SET status = 'promoted', promoted = 1, completed_at = ? WHERE id = ?",
            (_now_iso(), run_id),
        )
        self.conn.commit()

    def reject_shadow_run(self, run_id: int) -> None:
        """Mark a shadow run as rejected."""
        self.conn.execute(
            "UPDATE shadow_runs SET status = 'rejected', completed_at = ? WHERE id = ?",
            (_now_iso(), run_id),
        )
        self.conn.commit()

    def get_shadow_history(self, limit: int = 20) -> list[dict]:
        """Return shadow run history, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM shadow_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            try:
                d["candidate_weights"] = json.loads(d.get("candidate_weights") or "{}")
            except (json.JSONDecodeError, TypeError):
                d["candidate_weights"] = {}
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Salary benchmarks
    # ------------------------------------------------------------------

    _SALARY_SEED: list[tuple[str, str, int, int, int, int]] = [
        # (category, level, p25, p50, p75, p90)
        # Software Engineering
        ("software_engineer", "entry", 75_000, 90_000, 110_000, 130_000),
        ("software_engineer", "mid",   105_000, 130_000, 160_000, 195_000),
        ("software_engineer", "senior", 150_000, 180_000, 220_000, 270_000),
        # Data / Analytics
        ("data_analyst", "entry",  55_000,  68_000,  85_000, 100_000),
        ("data_analyst", "mid",    75_000,  95_000, 115_000, 140_000),
        ("data_analyst", "senior", 105_000, 130_000, 155_000, 185_000),
        # Data Science / ML
        ("data_scientist", "entry",  80_000,  98_000, 120_000, 145_000),
        ("data_scientist", "mid",   110_000, 140_000, 170_000, 210_000),
        ("data_scientist", "senior", 150_000, 185_000, 225_000, 275_000),
        # Customer Service
        ("customer_service", "entry",  30_000, 38_000,  48_000,  58_000),
        ("customer_service", "mid",    38_000, 48_000,  60_000,  72_000),
        ("customer_service", "senior", 50_000, 62_000,  78_000,  95_000),
        # Administrative Assistant
        ("admin_assistant", "entry",  32_000, 40_000,  50_000,  60_000),
        ("admin_assistant", "mid",    42_000, 52_000,  64_000,  76_000),
        ("admin_assistant", "senior", 55_000, 68_000,  82_000,  98_000),
        # Marketing
        ("marketing", "entry",  40_000,  50_000,  63_000,  78_000),
        ("marketing", "mid",    60_000,  75_000,  95_000, 115_000),
        ("marketing", "senior", 90_000, 115_000, 145_000, 175_000),
        # Sales
        ("sales", "entry",  35_000,  45_000,  58_000,  75_000),
        ("sales", "mid",    55_000,  72_000,  92_000, 120_000),
        ("sales", "senior", 80_000, 105_000, 140_000, 180_000),
        # Nursing / Healthcare
        ("nursing", "entry",  55_000, 65_000,  78_000,  92_000),
        ("nursing", "mid",    68_000, 82_000,  98_000, 115_000),
        ("nursing", "senior", 85_000, 102_000, 122_000, 145_000),
        # Accounting / Finance
        ("accounting", "entry",  45_000, 55_000,  68_000,  82_000),
        ("accounting", "mid",    65_000, 82_000, 100_000, 120_000),
        ("accounting", "senior", 90_000, 115_000, 142_000, 170_000),
        # Project Manager
        ("project_manager", "entry",  50_000,  63_000,  78_000,  95_000),
        ("project_manager", "mid",    75_000,  95_000, 118_000, 140_000),
        ("project_manager", "senior", 100_000, 128_000, 158_000, 190_000),
        # Human Resources
        ("human_resources", "entry",  40_000, 50_000,  62_000,  75_000),
        ("human_resources", "mid",    58_000, 72_000,  90_000, 108_000),
        ("human_resources", "senior", 80_000, 100_000, 125_000, 150_000),
        # Graphic Designer
        ("graphic_designer", "entry",  38_000, 47_000,  58_000,  72_000),
        ("graphic_designer", "mid",    52_000, 65_000,  80_000,  98_000),
        ("graphic_designer", "senior", 72_000, 90_000, 112_000, 135_000),
        # Teacher / Education
        ("teacher", "entry",  35_000, 43_000,  52_000,  62_000),
        ("teacher", "mid",    45_000, 56_000,  68_000,  80_000),
        ("teacher", "senior", 58_000, 72_000,  88_000, 105_000),
        # Warehouse / Logistics
        ("warehouse", "entry",  28_000, 35_000,  44_000,  54_000),
        ("warehouse", "mid",    38_000, 47_000,  58_000,  70_000),
        ("warehouse", "senior", 50_000, 62_000,  76_000,  92_000),
        # Legal
        ("legal", "entry",  55_000,  70_000,  90_000, 115_000),
        ("legal", "mid",    90_000, 120_000, 155_000, 200_000),
        ("legal", "senior", 140_000, 185_000, 240_000, 310_000),
    ]

    _SCAM_ENTITY_SEED: list[tuple[str, str, str, str]] = [
        # (name, domain, type, source)
        # Generic fake company names
        ("Global Solutions LLC", "", "fake_company", "sentinel_seed"),
        ("Apex Digital Services", "", "fake_company", "sentinel_seed"),
        ("Premier Staffing Group", "", "fake_company", "sentinel_seed"),
        ("Elite Workforce Solutions", "", "fake_company", "sentinel_seed"),
        ("National Career Opportunities", "", "fake_company", "sentinel_seed"),
        ("United Business Services", "", "fake_company", "sentinel_seed"),
        ("Dynamic Growth Partners", "", "fake_company", "sentinel_seed"),
        ("Synergy Global Enterprises", "", "fake_company", "sentinel_seed"),
        ("ProStaff Unlimited", "", "fake_company", "sentinel_seed"),
        ("Horizon Consulting Group", "", "fake_company", "sentinel_seed"),
        ("Infinity Talent Solutions", "", "fake_company", "sentinel_seed"),
        ("Alpha Workforce International", "", "fake_company", "sentinel_seed"),
        ("Nexus Employment Services", "", "fake_company", "sentinel_seed"),
        ("Summit Career Associates", "", "fake_company", "sentinel_seed"),
        ("Pinnacle Staffing Solutions", "", "fake_company", "sentinel_seed"),
        # Known typosquat / scam domains
        ("", "amazon-jobs.net", "typosquat_domain", "sentinel_seed"),
        ("", "amazon-hiring.com", "typosquat_domain", "sentinel_seed"),
        ("", "google-careers.net", "typosquat_domain", "sentinel_seed"),
        ("", "meta-jobs.net", "typosquat_domain", "sentinel_seed"),
        ("", "microsoft-hiring.net", "typosquat_domain", "sentinel_seed"),
        ("", "apple-jobs.net", "typosquat_domain", "sentinel_seed"),
        ("", "netflix-careers.net", "typosquat_domain", "sentinel_seed"),
        ("", "linkedin-jobs.net", "typosquat_domain", "sentinel_seed"),
        ("", "indeed-jobs.net", "typosquat_domain", "sentinel_seed"),
        ("", "glassdoor-jobs.net", "typosquat_domain", "sentinel_seed"),
        # Common fake recruiter/company patterns
        ("HR Department Team", "", "fake_recruiter", "sentinel_seed"),
        ("Recruitment Team USA", "", "fake_recruiter", "sentinel_seed"),
        ("Jobs4You Staffing", "", "fake_recruiter", "sentinel_seed"),
        ("QuickHire Solutions", "", "fake_recruiter", "sentinel_seed"),
        ("InstantHire Corp", "", "fake_recruiter", "sentinel_seed"),
    ]

    def seed_salary_benchmarks(self) -> None:
        """Seed the salary_benchmarks table with market-rate data if empty."""
        count = self.conn.execute("SELECT COUNT(*) FROM salary_benchmarks").fetchone()[0]
        if count > 0:
            return
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO salary_benchmarks (category, level, p25, p50, p75, p90)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            self._SALARY_SEED,
        )
        self.conn.commit()

    def get_salary_benchmark(self, category: str, level: str) -> dict | None:
        """Return salary benchmark for the given category and level, or None."""
        row = self.conn.execute(
            "SELECT * FROM salary_benchmarks WHERE category = ? AND level = ?",
            (category.lower(), level.lower()),
        ).fetchone()
        return _row_to_dict(row)

    def get_all_salary_benchmarks(self) -> list[dict]:
        """Return all salary benchmark rows."""
        rows = self.conn.execute(
            "SELECT * FROM salary_benchmarks ORDER BY category, level"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Scam entities
    # ------------------------------------------------------------------

    def seed_scam_entities(self) -> None:
        """Seed the scam_entities table with known patterns if empty."""
        count = self.conn.execute("SELECT COUNT(*) FROM scam_entities").fetchone()[0]
        if count > 0:
            return
        now = _now_iso()
        self.conn.executemany(
            """
            INSERT INTO scam_entities (name, domain, type, source, added_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(n, d, t, s, now) for n, d, t, s in self._SCAM_ENTITY_SEED],
        )
        self.conn.commit()

    def add_scam_entity(
        self,
        name: str = "",
        domain: str = "",
        entity_type: str = "fake_company",
        source: str = "manual",
    ) -> None:
        """Add a new scam entity to the database."""
        self.conn.execute(
            """
            INSERT INTO scam_entities (name, domain, type, source, added_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name.strip(), domain.strip().lower(), entity_type, source, _now_iso()),
        )
        self.conn.commit()

    def is_known_scam_entity(self, name: str = "", domain: str = "") -> bool:
        """Return True if the name or domain matches a known scam entity (exact, case-insensitive)."""
        if name:
            row = self.conn.execute(
                "SELECT 1 FROM scam_entities WHERE lower(name) = lower(?) AND name != ''",
                (name.strip(),),
            ).fetchone()
            if row:
                return True
        if domain:
            row = self.conn.execute(
                "SELECT 1 FROM scam_entities WHERE lower(domain) = lower(?) AND domain != ''",
                (domain.strip().lower(),),
            ).fetchone()
            if row:
                return True
        return False

    def get_scam_entities(self, entity_type: str | None = None) -> list[dict]:
        """Return all scam entities, optionally filtered by type."""
        if entity_type:
            rows = self.conn.execute(
                "SELECT * FROM scam_entities WHERE type = ? ORDER BY added_at DESC",
                (entity_type,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM scam_entities ORDER BY added_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Posting velocity
    # ------------------------------------------------------------------

    def upsert_posting_velocity(
        self,
        company_name: str,
        postings_24h: int,
        postings_7d: int,
    ) -> None:
        """Insert or replace posting velocity stats for a company."""
        self.conn.execute(
            """
            INSERT INTO posting_velocity (company_name, postings_24h, postings_7d, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(company_name) DO UPDATE SET
                postings_24h = excluded.postings_24h,
                postings_7d  = excluded.postings_7d,
                last_updated = excluded.last_updated
            """,
            (company_name, postings_24h, postings_7d, _now_iso()),
        )
        self.conn.commit()

    def get_posting_velocity(self, company_name: str) -> dict | None:
        """Return posting velocity row for company_name, or None if not tracked."""
        row = self.conn.execute(
            "SELECT * FROM posting_velocity WHERE lower(company_name) = lower(?)",
            (company_name,),
        ).fetchone()
        return _row_to_dict(row)

    # ------------------------------------------------------------------
    # Description hash deduplication
    # ------------------------------------------------------------------

    def record_description_hash(
        self,
        hash_val: str,
        company_name: str,
        job_url: str,
    ) -> None:
        """Record a description hash for cross-posting dedup.

        Uses INSERT OR IGNORE so re-recording the same (hash, company) is a no-op.
        """
        self.conn.execute(
            """
            INSERT OR IGNORE INTO description_hashes (hash, company_name, job_url, first_seen)
            VALUES (?, ?, ?, ?)
            """,
            (hash_val, company_name, job_url, _now_iso()),
        )
        self.conn.commit()

    def get_duplicate_description(
        self,
        hash_val: str,
        exclude_company: str = "",
    ) -> list[dict]:
        """Return all description_hashes rows matching *hash_val* under a different company.

        If *exclude_company* is empty, all matching rows are returned.
        """
        if exclude_company:
            rows = self.conn.execute(
                """
                SELECT * FROM description_hashes
                WHERE hash = ? AND lower(company_name) != lower(?)
                """,
                (hash_val, exclude_company),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM description_hashes WHERE hash = ?",
                (hash_val,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Near-misses (adversarial evasion tracking)
    # ------------------------------------------------------------------

    def insert_near_miss(
        self, signal_name: str, partial_match: str, job_url: str = ""
    ) -> None:
        """Record a near-miss evasion observation."""
        self.conn.execute(
            """
            INSERT INTO near_misses (signal_name, partial_match, job_url, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (signal_name, partial_match, job_url, _now_iso()),
        )
        self.conn.commit()

    def get_near_misses(
        self,
        signal_name: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """Return near-miss rows, optionally filtered by signal_name."""
        if signal_name is not None:
            rows = self.conn.execute(
                "SELECT * FROM near_misses WHERE signal_name = ? ORDER BY timestamp DESC LIMIT ?",
                (signal_name, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM near_misses ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Signal decay history (evasion rate monitoring)
    # ------------------------------------------------------------------

    def insert_signal_rate(
        self, signal_name: str, window_start: str, fire_rate: float
    ) -> None:
        """Record a per-signal firing rate snapshot for a time window."""
        self.conn.execute(
            """
            INSERT INTO signal_decay_history (signal_name, window_start, fire_rate, recorded_at)
            VALUES (?, ?, ?, ?)
            """,
            (signal_name, window_start, fire_rate, _now_iso()),
        )
        self.conn.commit()

    def get_signal_decay(
        self, signal_name: str | None = None, limit: int = 200
    ) -> list[dict]:
        """Return signal_decay_history rows, optionally filtered by signal_name."""
        if signal_name is not None:
            rows = self.conn.execute(
                """
                SELECT * FROM signal_decay_history
                WHERE signal_name = ?
                ORDER BY window_start DESC LIMIT ?
                """,
                (signal_name, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM signal_decay_history ORDER BY window_start DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Research history
    # ------------------------------------------------------------------

    def insert_research_run(self, run: dict) -> None:
        """Persist a research run record."""
        self.conn.execute(
            """
            INSERT INTO research_history
                (topic, prompt, response_summary, patterns_extracted,
                 patterns_adopted, precision_delta, timestamp)
            VALUES
                (:topic, :prompt, :response_summary, :patterns_extracted,
                 :patterns_adopted, :precision_delta, :timestamp)
            """,
            {
                "topic": run.get("topic", ""),
                "prompt": run.get("prompt", ""),
                "response_summary": run.get("response_summary", ""),
                "patterns_extracted": run.get("patterns_extracted", 0),
                "patterns_adopted": run.get("patterns_adopted", 0),
                "precision_delta": run.get("precision_delta", 0.0),
                "timestamp": run.get("timestamp", _now_iso()),
            },
        )
        self.conn.commit()

    def get_research_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent research runs, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM research_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_topic_priority(
        self,
        topic: str,
        priority: float,
        patterns_found: int = 0,
        precision_impact: float = 0.0,
    ) -> None:
        """Insert or update a research topic's priority and stats."""
        now = _now_iso()
        self.conn.execute(
            """
            INSERT INTO research_topics
                (topic, priority, last_researched, total_patterns_found, avg_precision_impact)
            VALUES
                (:topic, :priority, :last_researched, :patterns_found, :precision_impact)
            ON CONFLICT(topic) DO UPDATE SET
                priority             = excluded.priority,
                last_researched      = excluded.last_researched,
                total_patterns_found = research_topics.total_patterns_found + excluded.total_patterns_found,
                avg_precision_impact = CASE
                    WHEN research_topics.total_patterns_found > 0
                    THEN (research_topics.avg_precision_impact * research_topics.total_patterns_found
                          + excluded.avg_precision_impact)
                         / (research_topics.total_patterns_found + 1)
                    ELSE excluded.avg_precision_impact
                END
            """,
            {
                "topic": topic,
                "priority": priority,
                "last_researched": now,
                "patterns_found": patterns_found,
                "precision_impact": precision_impact,
            },
        )
        self.conn.commit()

    def get_top_research_topics(self, n: int = 10) -> list[dict]:
        """Return the top-n research topics by priority, highest first."""
        rows = self.conn.execute(
            "SELECT * FROM research_topics ORDER BY priority DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cascade events
    # ------------------------------------------------------------------

    def insert_cascade_event(
        self,
        trigger: str,
        change_type: str,
        impact_json: str = "{}",
    ) -> int:
        """Insert a cascade event record and return its id."""
        cursor = self.conn.execute(
            """
            INSERT INTO cascade_events (trigger, change_type, impact_json, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (trigger, change_type, impact_json, _now_iso()),
        )
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_cascade_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent cascade events, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM cascade_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            try:
                d["impact"] = json.loads(d.get("impact_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                d["impact"] = {}
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Flywheel mesh edges
    # ------------------------------------------------------------------

    def upsert_mesh_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "data",
        weight: float = 1.0,
    ) -> None:
        """Insert or replace a mesh edge record."""
        self.conn.execute(
            """
            INSERT INTO flywheel_mesh_edges (source, target, edge_type, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source, target) DO UPDATE SET
                edge_type = excluded.edge_type,
                weight    = excluded.weight
            """,
            (source, target, edge_type, weight),
        )
        self.conn.commit()

    def get_mesh_topology(self) -> list[dict]:
        """Return all mesh edges as a list of dicts."""
        rows = self.conn.execute(
            "SELECT * FROM flywheel_mesh_edges ORDER BY source, target"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Jobs sampling (for cascade preview)
    # ------------------------------------------------------------------

    def get_recent_jobs_for_sampling(self, limit: int = 100) -> list[dict]:
        """Return recent scored jobs for cascade impact preview."""
        rows = self.conn.execute(
            """
            SELECT url, scam_score, risk_level, signals_json, confidence
            FROM jobs
            WHERE scam_score IS NOT NULL
            ORDER BY analyzed_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cortex state
    # ------------------------------------------------------------------

    def save_cortex_state(
        self,
        cycle_number: int,
        state_json: str,
        learning_velocity: float,
        health_grade: str,
        strategic_mode: str,
    ) -> None:
        """Persist a cortex state snapshot."""
        self.conn.execute(
            """
            INSERT INTO cortex_state
                (cycle_number, state_json, learning_velocity, health_grade, strategic_mode, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cycle_number, state_json, learning_velocity, health_grade, strategic_mode, _now_iso()),
        )
        self.conn.commit()

    def get_latest_cortex_state(self) -> dict | None:
        """Return the most recent cortex state row, or None."""
        row = self.conn.execute(
            "SELECT * FROM cortex_state ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return _row_to_dict(row)

    def get_cortex_state_history(self, limit: int = 50) -> list[dict]:
        """Return cortex state history, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM cortex_state ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cortex investigations
    # ------------------------------------------------------------------

    def insert_cortex_investigation(
        self,
        id: str,
        trigger: str,
        hypothesis: str,
    ) -> None:
        """Insert a new cortex investigation."""
        self.conn.execute(
            """
            INSERT OR IGNORE INTO cortex_investigations
                (id, trigger, hypothesis, status, actions_json, opened_at)
            VALUES (?, ?, ?, 'open', '[]', ?)
            """,
            (id, trigger, hypothesis, _now_iso()),
        )
        self.conn.commit()

    def update_cortex_investigation(self, inv_id: str, updates: dict) -> None:
        """Update fields on a cortex investigation."""
        allowed = {"status", "resolution", "resolved_at", "actions_json"}
        sets = []
        params = []
        for key, val in updates.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                params.append(val)
        if not sets:
            return
        params.append(inv_id)
        self.conn.execute(
            f"UPDATE cortex_investigations SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        self.conn.commit()

    def get_cortex_investigations(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Return cortex investigations, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM cortex_investigations WHERE status = ? ORDER BY opened_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM cortex_investigations ORDER BY opened_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Cortex signals
    # ------------------------------------------------------------------

    def insert_cortex_signal(
        self,
        source: str,
        target: str,
        signal_type: str,
        payload: dict,
        priority: float = 0.5,
    ) -> None:
        """Insert a cross-system signal record."""
        self.conn.execute(
            """
            INSERT INTO cortex_signals
                (source, target, signal_type, payload_json, priority, acted_on, timestamp)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (source, target, signal_type, json.dumps(payload), priority, _now_iso()),
        )
        self.conn.commit()

    def get_recent_cortex_signals(self, limit: int = 20) -> list[dict]:
        """Return recent cortex signals, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM cortex_signals ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            try:
                d["payload"] = json.loads(d.get("payload_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                d["payload"] = {}
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()

    def __enter__(self) -> "SentinelDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
