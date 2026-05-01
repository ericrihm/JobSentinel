<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# JobSentinel

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-2789%20passing-brightgreen.svg)](#development)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-purple.svg)](https://github.com/astral-sh/ruff)

**AI-powered job scam detection and validation platform.**

*Protect yourself from fake listings, ghost jobs, and recruitment fraud before you apply.*

[Website](https://jobsentinel.org) | [API Docs](#rest-api) | [Browser Extension](#browser-extension-chrome)

</div>

---

## Why JobSentinel?

Job scams cost victims **$2 billion+ annually** (FTC, 2024). Platform moderation catches obvious fakes, but sophisticated scams slip through — MLM recruitment disguised as corporate roles, ghost jobs that waste months of your time, data harvesting posts that steal your identity.

JobSentinel catches what platforms miss.

## Quick Start

```bash
pip install -e .

# Analyze a suspicious posting
sentinel analyze "We're hiring! No experience needed. Earn $5000/week guaranteed. Send $50 registration fee to start."

# Analyze from a URL
sentinel analyze "https://linkedin.com/jobs/view/1234567890"

# Batch analyze from file
sentinel analyze --file jobs.json

# Validate a company
sentinel validate "Google"

# Report a scam (improves detection)
sentinel report "https://linkedin.com/jobs/view/123" --reason "Asked for SSN before interview"

# View detection statistics
sentinel stats
```

### Install Options

```bash
pip install -e ".[full]"    # Everything: AI + API server + web scraping
pip install -e ".[ai]"      # AI analysis (anthropic)
pip install -e ".[api]"     # API server (fastapi + uvicorn)
pip install -e ".[web]"     # Web scraping (httpx + beautifulsoup4)
```

## What It Detects

### Red Flags

| Signal | Example |
|--------|---------|
| Upfront payment | "Pay $99 registration fee to start" |
| Personal info harvesting | SSN, bank account before interview |
| Guaranteed income | "$5,000/week guaranteed" |
| Suspicious contact | Corporate role but gmail/yahoo |
| Crypto payment | "Paid in Bitcoin/crypto" |
| No company presence | No verifiable business presence |
| Interview bypass | "No interview needed, start immediately" |
| MLM / pyramid | "Build your team", "unlimited earning potential" |
| Reshipping scam | "Receive and forward packages" |

### Warnings

- Salary significantly above market rate
- Extremely vague job descriptions
- No qualifications listed
- Urgency / scarcity language
- Recently created company profiles
- Low-connection recruiters

### Ghost Jobs

- Stale postings (30+ days, no activity)
- Serial reposts (same role monthly)

### Risk Levels

| Score | Level | Meaning |
|-------|-------|---------|
| 0.0-0.2 | Safe | Proceed with confidence |
| 0.2-0.4 | Low | Likely legitimate |
| 0.4-0.6 | Suspicious | Review before applying |
| 0.6-0.8 | High | Strong scam indicators |
| 0.8-1.0 | Scam | Almost certainly fraudulent |

## Company Validation

Cross-reference companies against multiple sources:

- **Known employers database** — 300+ verified major employers
- **Domain WHOIS** — Check domain age (new domains = higher risk)
- **LinkedIn company page** — Verify follower count, employee count
- **Result caching** — 7-day TTL, `--refresh` to force re-check

## Browser Extension (Chrome)

Analyze job postings directly in your browser:

```bash
# Load the extension in Chrome developer mode
# Navigate to: chrome://extensions > Load unpacked > sentinel/web/extension/

# Start the API server
sentinel serve --port 8080
```

The extension adds a color-coded risk badge next to each job title with a detailed breakdown popup.

## REST API

```bash
pip install -e ".[api]"
sentinel serve --port 8080
# API docs at http://localhost:8080/docs
```

### `POST /api/analyze`

```bash
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Earn $5000/week guaranteed! Send $50 fee to start.", "title": "Data Entry", "company": ""}'
```

```json
{
  "scam_score": 0.95,
  "risk_level": "scam",
  "risk_label": "Almost Certainly Scam",
  "red_flags": [
    {"name": "upfront_payment", "detail": "Registration/training fee required"},
    {"name": "guaranteed_income", "detail": "Unrealistic income guarantee"}
  ],
  "warnings": [
    {"name": "vague_description", "detail": "No specific duties listed"}
  ],
  "signal_count": 7,
  "analysis_time_ms": 4.2
}
```

### `POST /api/report`

```bash
curl -X POST http://localhost:8080/api/report \
  -H "Content-Type: application/json" \
  -d '{"url": "https://linkedin.com/jobs/view/123", "is_scam": true, "reason": "Asked for payment"}'
```

### `GET /api/patterns`

```bash
curl http://localhost:8080/api/patterns?category=red_flag
```

### `GET /api/stats` | `GET /api/health`

Detection statistics and service health check.

## Configuration

```toml
# ~/.config/sentinel/config.toml

db_path = "~/.sentinel/sentinel.db"
ai_enabled = true
ai_model = "claude-haiku-4-5"
ai_model_deep = "claude-sonnet-4-6"
rate_limit_rpm = 60
cors_origins = ["http://localhost:3000"]
log_level = "INFO"
```

## Development

```bash
pip install -e ".[dev,full]"

# Run tests
python -m pytest tests/ -v

# Lint
ruff check sentinel/
```

### Security

- Input validation on all API fields (length limits, URL format)
- SQL injection prevention (parameterized queries throughout)
- Command injection prevention (domain validation before subprocess calls)
- HTML sanitization (script/style/event handler stripping)
- See `tests/test_security.py` for the full security test suite

## Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| Python 3.12+ | Runtime | Yes |
| `click` | CLI framework | Yes |
| `anthropic` | AI analysis | Optional |
| `fastapi` + `uvicorn` | API server | Optional |
| `httpx` | URL fetching | Optional |
| `beautifulsoup4` | HTML parsing | Optional |

Core detection runs on **Python stdlib only** — no pip packages required for the detection engine.

## Contributing

1. Fork the repo
2. Create a feature branch
3. Write tests for your changes
4. Run the test suite and linter
5. Submit a PR

## License

MIT
