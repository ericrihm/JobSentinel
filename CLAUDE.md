# Sentinel — LinkedIn Job Scam Detection & Validation Platform

AI-powered platform that detects scam job postings on LinkedIn, validates legitimate jobs, and helps job seekers avoid fraud. Uses multi-signal analysis, Bayesian scoring, and a self-improving flywheel to stay ahead of evolving scam tactics.

## Architecture

```
sentinel/
  __init__.py        — Package init, version
  models.py          — Data classes: JobPosting, ScamSignal, ValidationResult, CompanyProfile, ScamPattern
  scanner.py         — LinkedIn job scraping/ingestion: parse job pages, extract structured fields, batch import
  signals.py         — Signal extraction: 40+ scam indicators (salary anomaly, vague description, fake company, ghost job, upfront payment, urgency pressure, email domain, recruiter history)
  analyzer.py        — Multi-tier AI analysis: fast regex/heuristic pass → Haiku classification → Sonnet deep analysis for ambiguous cases
  scorer.py          — Bayesian Beta-distribution scam scoring with Thompson Sampling, confidence intervals, per-signal weight learning
  validator.py       — Company validation: cross-reference against business registries, LinkedIn company pages, domain WHOIS, Glassdoor, employer databases
  knowledge.py       — SQLite + FTS5 knowledge base: known scam patterns, verified companies, user reports, historical postings
  flywheel.py        — Self-improving loop: INGEST → SCORE → VALIDATE → LEARN → EVOLVE. CUSUM regression detection, pattern lifecycle (discovery → validated → deprecated)
  api.py             — FastAPI REST API + browser extension backend: /analyze endpoint, /report endpoint, /feed endpoint
  web/               — React frontend: job analysis dashboard, browser extension popup, scam pattern explorer
  cli.py             — CLI: analyze, scan, report, patterns, validate, stats, evolve, serve
  db.py              — SQLite persistence with FTS5 search, WAL mode
  config.py          — TOML config loading
```

## Key Design Decisions

- **Multi-tier detection**: Fast regex signals (< 10ms) → Bayesian scoring → AI classification only for ambiguous cases. Most scams caught at tier 1.
- **Bayesian over binary**: Every signal has a Beta(alpha, beta) posterior, not a boolean flag. Confidence grows with observations.
- **Self-improving**: Every user report (confirm scam / mark legitimate) updates signal weights via Thompson Sampling.
- **Privacy-first**: No LinkedIn credentials stored. Analysis works on job posting URLs/text, not account data.
- **Stdlib-first runtime**: Core detection runs on Python stdlib only. Optional: anthropic (AI tier), httpx (validation), fastapi (web API).

## Signal Categories (40+)

### Red Flags (High Weight)
- Upfront payment or fee required
- Requests for SSN/bank info before interview
- "Guaranteed" income claims
- Suspicious email domain (gmail/yahoo for "corporate" role)
- No company LinkedIn page or < 10 employees
- Job posted by individual, not company
- Cryptocurrency/wire transfer mentions

### Warning Signs (Medium Weight)
- Salary significantly above market rate for role/location
- Extremely vague job description
- No specific skills or qualifications listed
- "Work from home" with unrealistic pay
- Urgency language ("apply NOW", "limited spots")
- Recently created company LinkedIn page
- Job reposted many times with no hires
- Recruiter with < 50 connections
- Template/boilerplate description across many postings

### Ghost Job Indicators
- Posted > 30 days with no activity
- Company has hiring freeze but active postings
- Same role reposted monthly
- No interviewer assigned
- "Always hiring" pattern

### Structural Signals
- Grammar/spelling quality score
- Formatting consistency
- Contact info analysis (personal vs corporate)
- URL/link analysis in description
- Salary range width (too wide = suspicious)

## Scam Score Interpretation

| Score | Label | Action |
|-------|-------|--------|
| 0.0–0.2 | Verified Safe | Green badge, proceed |
| 0.2–0.4 | Likely Legitimate | Light check recommended |
| 0.4–0.6 | Suspicious | Review warnings before applying |
| 0.6–0.8 | Likely Scam | Strong warning, detailed breakdown |
| 0.8–1.0 | Almost Certainly Scam | Block, report suggestion |

## CLI Commands

```bash
sentinel analyze <url-or-text>           — Analyze a single job posting
sentinel analyze --file jobs.json        — Batch analyze from file
sentinel scan --query "software engineer" --location "remote" — Scan LinkedIn search results
sentinel report <url> --reason "asked for payment"  — Report a scam posting
sentinel validate <company-name>         — Validate a company
sentinel patterns [--type red-flag|warning|ghost]   — Show known scam patterns
sentinel stats                           — Detection statistics and accuracy
sentinel evolve                          — Run self-improvement cycle
sentinel serve [--port 8080]             — Start API server
sentinel extension-build                 — Build browser extension
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Storage

- Database: `~/.sentinel/sentinel.db`
- Config: `~/.config/sentinel/config.toml`
- Knowledge: `~/.sentinel/patterns/`
- Extension: `sentinel/web/extension/`

## Dependencies

Python 3.12+ stdlib only at runtime. Optional: `anthropic` (AI analysis), `httpx` (web validation), `fastapi` + `uvicorn` (API server), `beautifulsoup4` (HTML parsing).
