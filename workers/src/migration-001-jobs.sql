-- migration-001-jobs.sql — Add jobs table for aggregated job listings
--
-- Apply with:
--   wrangler d1 execute jobsentinel-db --file=src/migration-001-jobs.sql
--
-- For remote (deployed) database:
--   wrangler d1 execute jobsentinel-db --remote --file=src/migration-001-jobs.sql

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id     TEXT    NOT NULL DEFAULT '',    -- source-specific ID for dedup
    url             TEXT    NOT NULL,
    title           TEXT    NOT NULL DEFAULT '',
    company         TEXT    NOT NULL DEFAULT '',
    location        TEXT    NOT NULL DEFAULT '',
    description     TEXT    NOT NULL DEFAULT '',
    salary_min      REAL    NOT NULL DEFAULT 0.0,
    salary_max      REAL    NOT NULL DEFAULT 0.0,
    salary_currency TEXT    NOT NULL DEFAULT 'USD',
    employment_type TEXT    NOT NULL DEFAULT '',
    experience_level TEXT   NOT NULL DEFAULT '',
    is_remote       INTEGER NOT NULL DEFAULT 0,
    source          TEXT    NOT NULL DEFAULT '',     -- greenhouse, lever, adzuna, etc.
    source_company  TEXT    NOT NULL DEFAULT '',     -- the company slug on the ATS
    scam_score      REAL    NOT NULL DEFAULT 0.0,
    risk_level      TEXT    NOT NULL DEFAULT 'safe',
    signal_count    INTEGER NOT NULL DEFAULT 0,
    posted_at       TEXT    NOT NULL DEFAULT '',     -- ISO 8601
    discovered_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    expires_at      TEXT    NOT NULL DEFAULT '',     -- when to re-check
    is_active       INTEGER NOT NULL DEFAULT 1,
    content_hash    TEXT    NOT NULL DEFAULT '',     -- for detecting changes
    UNIQUE(url)
);

-- Indexes for search queries
CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs (title);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs (company);
CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs (location);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs (source);
CREATE INDEX IF NOT EXISTS idx_jobs_scam_score ON jobs (scam_score);
CREATE INDEX IF NOT EXISTS idx_jobs_is_remote ON jobs (is_remote);
CREATE INDEX IF NOT EXISTS idx_jobs_is_active ON jobs (is_active);
CREATE INDEX IF NOT EXISTS idx_jobs_posted_at ON jobs (posted_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_discovered_at ON jobs (discovered_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs (content_hash);
-- Composite index for common filtered queries
CREATE INDEX IF NOT EXISTS idx_jobs_active_score ON jobs (is_active, scam_score);
CREATE INDEX IF NOT EXISTS idx_jobs_active_posted ON jobs (is_active, posted_at DESC);
