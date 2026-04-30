# JobSentinel

**AI-powered LinkedIn job scam detection and validation platform.**

JobSentinel analyzes job postings for 40+ scam indicators using multi-signal analysis and AI-powered classification. It helps job seekers identify fake listings, ghost jobs, and recruitment fraud before they apply.

## How It Works

```
Job Posting → Signal Extraction (26 detectors) → Risk Scoring → Classification
                                                       ↓
                                            AI Analysis (ambiguous cases)
                                                       ↓
                                            User Reports → Improved Accuracy
```

**Detection pipeline:**
1. **Signal extraction** — Pattern-based detectors scan for 40+ scam indicators across 5 categories
2. **Risk scoring** — Weighted signal combination produces a 0–1 scam probability score
3. **AI escalation** — Claude analyzes ambiguous cases that fall in the mid-range

## Quick Start

```bash
pip install -e .

# Initialize the database with default scam patterns
sentinel init --seed

# Analyze a job posting
sentinel analyze "We're hiring! No experience needed. Earn $5000/week guaranteed. Send $50 registration fee to start."

# Validate a company
sentinel validate "Google"

# Report a scam
sentinel report "https://linkedin.com/jobs/view/123" --reason "Asked for SSN before interview"

# View detection statistics
sentinel stats
```

## Signal Categories

| Category | Count | Examples |
|----------|-------|---------|
| Red Flags | 9 | Upfront payment, SSN request, guaranteed income, crypto payment, MLM, reshipping |
| Warnings | 9 | Salary anomaly, vague description, urgency language, no qualifications, phone anomaly |
| Ghost Job | 2 | Stale posting (>30 days), repeat reposting |
| Structural | 3 | Poor grammar/formatting, suspicious links, AI-generated content detection |
| Positive | 2 | Established company (1000+ employees), detailed requirements |

## Risk Levels

| Score | Level | Meaning |
|-------|-------|---------|
| 0.0–0.2 | Safe | Verified legitimate posting |
| 0.2–0.4 | Low | Likely legitimate, minor flags |
| 0.4–0.6 | Suspicious | Review warnings before applying |
| 0.6–0.8 | High | Strong scam indicators present |
| 0.8–1.0 | Scam | Almost certainly fraudulent |

## API Server

```bash
pip install fastapi uvicorn
sentinel serve --port 8080
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/analyze` | POST | Analyze a job posting |
| `/api/report` | POST | Submit scam/legitimate report |
| `/api/patterns` | GET | List detection patterns |
| `/api/stats` | GET | Detection statistics |
| `/api/health` | GET | Service health check |

## Architecture

```
sentinel/
├── models.py      — Data classes (JobPosting, ScamSignal, ValidationResult)
├── signals.py     — 26 signal extractors across 5 categories
├── scorer.py      — Weighted signal scoring engine
├── analyzer.py    — Multi-tier analysis pipeline (pattern matching → AI)
├── scanner.py     — Job posting parser (text, HTML, JSON)
├── validator.py   — Company validation (LinkedIn, WHOIS, known companies)
├── knowledge.py   — Pattern knowledge base with 20+ default scam patterns
├── db.py          — SQLite + FTS5 persistence
├── cli.py         — Click CLI (analyze, validate, report, patterns, stats)
└── api.py         — FastAPI REST API
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

154 tests covering signals, scoring, scanning, validation, persistence, and integration.

## Dependencies

- **Runtime**: Python 3.12+ stdlib only (no pip packages required)
- **Optional**: `anthropic` (AI analysis), `httpx` (web validation), `fastapi` + `uvicorn` (API server), `click` (CLI)

## License

MIT
