"""SQLite persistence layer for Sentinel — WAL mode + FTS5 full-text search."""

import json
import os
import sqlite3
from datetime import datetime, timezone
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
    patterns_deprecated INTEGER DEFAULT 0
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        ]:
            try:
                self.conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass  # column already exists
        self.conn.commit()

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
                 salary_min, salary_max, scam_score, risk_level,
                 analyzed_at, signal_count, signals_json,
                 user_reported, user_verdict)
            VALUES
                (:url, :title, :company, :location, :description,
                 :salary_min, :salary_max, :scam_score, :risk_level,
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
        # Escape special FTS characters to avoid query errors on raw input
        safe_query = query.replace('"', '""')
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

        self.conn.execute(
            """
            INSERT INTO flywheel_metrics
                (cycle_ts, total_analyzed, true_positives, false_positives,
                 precision, recall, signals_updated, patterns_evolved,
                 f1, accuracy, cycle_number, regression_alarm,
                 cusum_statistic, patterns_promoted, patterns_deprecated)
            VALUES
                (:cycle_ts, :total_analyzed, :true_positives, :false_positives,
                 :precision, :recall, :signals_updated, :patterns_evolved,
                 :f1, :accuracy, :cycle_number, :regression_alarm,
                 :cusum_statistic, :patterns_promoted, :patterns_deprecated)
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
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()

    def __enter__(self) -> "SentinelDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
