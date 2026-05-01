-- schema.sql — D1 (SQLite) database schema for JobSentinel Worker
--
-- Apply with:
--   wrangler d1 execute jobsentinel-db --file=src/schema.sql
--
-- For remote (deployed) database:
--   wrangler d1 execute jobsentinel-db --remote --file=src/schema.sql

-- ---------------------------------------------------------------------------
-- Scan history — one row per analysis request
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scan_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT    NOT NULL DEFAULT '',
    title       TEXT    NOT NULL DEFAULT '',
    company     TEXT    NOT NULL DEFAULT '',
    scam_score  REAL    NOT NULL DEFAULT 0.0,
    risk_level  TEXT    NOT NULL DEFAULT 'low',
    signal_count INTEGER NOT NULL DEFAULT 0,
    client_ip   TEXT    NOT NULL DEFAULT '',
    scanned_at  TEXT    NOT NULL  -- ISO 8601
);

CREATE INDEX IF NOT EXISTS idx_scan_history_scanned_at ON scan_history (scanned_at DESC);
CREATE INDEX IF NOT EXISTS idx_scan_history_risk_level ON scan_history (risk_level);
CREATE INDEX IF NOT EXISTS idx_scan_history_url        ON scan_history (url);

-- ---------------------------------------------------------------------------
-- Known scam patterns (seeded from Python knowledge base exports)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_id      TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    description     TEXT    NOT NULL DEFAULT '',
    category        TEXT    NOT NULL DEFAULT 'red_flag',
    status          TEXT    NOT NULL DEFAULT 'active',  -- active | deprecated | candidate
    regex           TEXT    NOT NULL DEFAULT '',
    keywords        TEXT    NOT NULL DEFAULT '[]',      -- JSON array
    bayesian_score  REAL    NOT NULL DEFAULT 0.5,
    observations    INTEGER NOT NULL DEFAULT 0,
    true_positives  INTEGER NOT NULL DEFAULT 0,
    false_positives INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_patterns_status   ON patterns (status);
CREATE INDEX IF NOT EXISTS idx_patterns_category ON patterns (category);

-- ---------------------------------------------------------------------------
-- User reports — crowd-sourced feedback to improve detection
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT    NOT NULL,
    is_scam     INTEGER NOT NULL DEFAULT 0,  -- 1 = scam, 0 = legitimate
    reason      TEXT    NOT NULL DEFAULT '',
    reported_at TEXT    NOT NULL,            -- ISO 8601
    processed   INTEGER NOT NULL DEFAULT 0  -- 1 = ingested into patterns
);

CREATE INDEX IF NOT EXISTS idx_user_reports_reported_at ON user_reports (reported_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_reports_is_scam     ON user_reports (is_scam);

-- ---------------------------------------------------------------------------
-- Seed: built-in high-confidence patterns
-- (mirrors Python knowledge.py seed_default_patterns)
-- ---------------------------------------------------------------------------
INSERT OR IGNORE INTO patterns (pattern_id, name, description, category, regex, keywords, bayesian_score)
VALUES
  ('upfront_payment',        'Upfront Payment Request',     'Requires fees, deposits, or equipment purchases before starting work', 'red_flag',
   '\b(fee required|send money|training fee|buy equipment|purchase (your |a )?equipment|starter kit fee|background check fee|pay (a |the )?deposit|wire (me|us)|upfront (cost|fee|payment)|advance fee)\b',
   '["fee required","send money","training fee","buy equipment","upfront payment","advance fee","wire transfer deposit"]', 0.92),

  ('personal_info_request',  'Personal Info Harvesting',    'Requests SSN, bank account, passport, or other sensitive PII before hiring', 'red_flag',
   '\b(social security|SSN|bank account|routing number|credit card|debit card|full (name and )?address|passport (number|copy)|drivers? licen[sc]e)\b',
   '["social security","SSN","bank account","routing number","passport number","drivers license"]', 0.90),

  ('guaranteed_income',      'Guaranteed Income Promise',   '"Guaranteed" salary or income claims — no legitimate employer does this', 'red_flag',
   '\b(guaranteed (salary|income|pay|earnings?|profit)|(guaranteed|promise[sd]) to (earn|make|pay))\b',
   '["guaranteed salary","guaranteed income","guaranteed earnings","promise to earn"]', 0.85),

  ('crypto_payment',         'Crypto / Untraceable Payment','Bitcoin, gift cards, wire transfers, or other untraceable payment methods', 'red_flag',
   '\b(bitcoin|btc|ethereum|eth|crypto(currency)?|gift card|western union|moneygram|wire transfer|zelle|cashapp|venmo)\b',
   '["bitcoin","cryptocurrency","gift card","western union","wire transfer","cashapp","venmo"]', 0.88),

  ('mlm_language',           'MLM / Pyramid Scheme',        'Multi-level marketing indicators: unlimited earnings, recruit others, residual income', 'red_flag',
   '\b(be your own boss|unlimited earning potential|residual income|network marketing|multi.?level marketing|mlm|downline|upline|recruit (others|people|friends)|financial freedom|passive income opportunity)\b',
   '["be your own boss","unlimited earning potential","residual income","network marketing","mlm","downline","recruit others"]', 0.82),

  ('reshipping',             'Reshipping / Package Scam',   'Receive, inspect, or forward packages from home — money mule vector', 'red_flag',
   '\b(receive (packages?|parcels?|shipments?)|reship(ping)?|re-ship(ping)?|forward (packages?|parcels?)|package (handler|inspector|coordinator) (from|at) home|inspect (packages?|items?) (at|from) home)\b',
   '["receive packages","reshipping","package handler from home","forward packages","parcel inspector"]', 0.88),

  ('interview_bypass',       'Interview / Screening Bypass','No interview required, hired on the spot, or start immediately without any process', 'red_flag',
   '\b(no interview (required|needed|necessary)|hired (on the spot|immediately|same day)|start (immediately|today|right away) no (questions?|interview)|no resume (required|needed|necessary)|no background (check|screening))\b',
   '["no interview required","hired on the spot","start immediately no interview","no background check"]', 0.80);
