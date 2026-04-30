# JobSentinel

**AI-powered LinkedIn job scam detection and validation platform.**

JobSentinel analyzes job postings for 40+ scam indicators using multi-signal analysis, Bayesian scoring, and a self-improving detection flywheel. It helps job seekers identify fake listings, ghost jobs, and recruitment fraud before they apply.

## How It Works

```
Job Posting → Signal Extraction (18 detectors) → Bayesian Scoring → Risk Classification
                                                         ↓
                                              AI Escalation (ambiguous cases)
                                                         ↓
                                              User Reports → Flywheel Learning
```

**Three-tier detection:**
1. **Fast pass** (<10ms) — Regex-based signal extraction catches obvious scams
2. **Bayesian scoring** — Beta-distribution weighted combination with confidence intervals
3. **AI escalation** — Claude analyzes ambiguous cases (0.3-0.7 score range)

## Quick Start

```bash
pip install -e .

# Initialize the database with default scam patterns
sentinel init --seed

# Analyze a job posting
sentinel analyze "We're hiring! No experience needed. Earn $5000/week guaranteed. Send $50 registration fee to start."

# Validate a company
sentinel validate "Google"

# Report a scam (feeds the learning flywheel)
sentinel report "https://linkedin.com/jobs/view/123" --reason "Asked for SSN before interview"

# View detection statistics
sentinel stats

# Run self-improvement cycle
sentinel evolve
```

## Signal Categories

| Category | Count | Examples |
|----------|-------|---------|
| Red Flags | 6 | Upfront payment, SSN request, guaranteed income, crypto payment |
| Warnings | 6 | Salary anomaly, vague description, urgency language, no qualifications |
| Ghost Job | 2 | Stale posting (>30 days), repeat reposting |
| Structural | 2 | Poor grammar/formatting, suspicious links (bit.ly, telegram) |
| Positive | 2 | Established company (1000+ employees), detailed requirements |

## Risk Levels

| Score | Level | Meaning |
|-------|-------|---------|
| 0.0–0.2 | Safe | Verified legitimate posting |
| 0.2–0.4 | Low | Likely legitimate, minor flags |
| 0.4–0.6 | Suspicious | Review warnings before applying |
| 0.6–0.8 | High | Strong scam indicators present |
| 0.8–1.0 | Scam | Almost certainly fraudulent |

## Self-Improving Flywheel

JobSentinel gets smarter over time:

1. **INGEST** — Analyze job postings, extract signals
2. **SCORE** — Bayesian combination of weighted signals
3. **VALIDATE** — Users confirm or deny predictions
4. **LEARN** — Thompson Sampling updates signal weights based on accuracy
5. **EVOLVE** — Promote effective patterns, deprecate unreliable ones, detect regression via CUSUM

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
| `/api/health` | GET | Service health + flywheel grade |

## Architecture

```
sentinel/
├── models.py      — Data classes (JobPosting, ScamSignal, ValidationResult)
├── signals.py     — 18 signal extractors across 5 categories
├── scorer.py      — Bayesian Beta-distribution scoring + Thompson Sampling
├── analyzer.py    — Multi-tier analysis pipeline (regex → AI)
├── scanner.py     — Job posting parser (text, HTML, JSON)
├── validator.py   — Company validation (LinkedIn, WHOIS, known companies)
├── knowledge.py   — Pattern knowledge base with 20 default scam patterns
├── flywheel.py    — Self-improving detection loop + CUSUM regression
├── db.py          — SQLite + FTS5 persistence
├── cli.py         — Click CLI (analyze, validate, report, patterns, stats, evolve)
└── api.py         — FastAPI REST API
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

96 tests covering signals, scoring, scanning, validation, persistence, and integration.

## Dependencies

- **Runtime**: Python 3.12+ stdlib only (no pip packages required)
- **Optional**: `anthropic` (AI analysis), `httpx` (web validation), `fastapi` + `uvicorn` (API server), `click` (CLI)

## License

MIT
