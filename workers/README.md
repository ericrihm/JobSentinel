# JobSentinel — Cloudflare Workers Deployment

Free, globally-distributed job scam analysis API running on Cloudflare Workers free tier (100,000 requests/day).

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Compute | Cloudflare Workers | JS runtime, globally distributed |
| Rate limiting | Cloudflare KV | Sliding-window 10 req/min per IP |
| Persistence | Cloudflare D1 | Scan history, patterns, user reports |
| Signals | `src/signals.js` | 16 regex-based scam detectors |
| Scoring | `src/scorer.js` | Log-odds Bayesian scoring (matches Python) |

## Prerequisites

- [Node.js](https://nodejs.org/) 18+
- [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/install-and-update/) (`npm install -g wrangler`)
- A [Cloudflare account](https://dash.cloudflare.com/sign-up) (free tier is sufficient)

## Setup

### 1. Authenticate with Cloudflare

```bash
wrangler login
```

### 2. Install dependencies

```bash
cd workers
npm install
```

### 3. Create KV namespace for rate limiting

```bash
wrangler kv:namespace create "RATE_LIMIT"
# Copy the output IDs into wrangler.toml:
#   id = "..."
#   preview_id = "..."
```

### 4. Create D1 database

```bash
wrangler d1 create jobsentinel-db
# Copy the database_id into wrangler.toml
```

### 5. Apply database schema

```bash
# Local dev
npm run db:init

# Deployed (remote) database
npm run db:init:remote
```

### 6. Update wrangler.toml

Fill in the placeholder values in `wrangler.toml`:
- `RATE_LIMIT.id` — from step 3
- `RATE_LIMIT.preview_id` — from step 3
- `DB.database_id` — from step 4

### 7. Deploy

```bash
npm run deploy
```

Your worker URL will be printed:
```
https://jobsentinel-api.<your-subdomain>.workers.dev
```

## Local Development

```bash
npm run dev
# Worker available at http://localhost:8787
```

## API Reference

### POST /api/analyze

Analyze a job posting for scam signals.

**Request body (text mode):**
```json
{
  "text": "Job description text...",
  "title": "Software Engineer",
  "company": "Acme Corp",
  "url": "https://linkedin.com/jobs/view/123",
  "posted_date": "2024-01-15",
  "is_remote": false,
  "salary_min": 0,
  "salary_max": 0,
  "experience_level": ""
}
```

**Request body (structured mode — Chrome extension):**
```json
{
  "job_data": {
    "title": "Software Engineer",
    "company": "Acme Corp",
    "description": "Full job description...",
    "url": "https://linkedin.com/jobs/view/123",
    "company_linkedin_url": "https://linkedin.com/company/acme",
    "salary_min": 120000,
    "salary_max": 160000,
    "posted_date": "2024-01-01",
    "experience_level": "mid",
    "is_remote": true,
    "recruiter_connections": 150
  }
}
```

**Response:**
```json
{
  "job": { "url": "...", "title": "...", "company": "..." },
  "scam_score": 0.12,
  "confidence": 0.65,
  "risk_level": "safe",
  "risk_label": "Verified Safe",
  "red_flags": [],
  "warnings": [],
  "ghost_indicators": [],
  "positive_signals": [],
  "structural": [],
  "signal_count": 0,
  "ai_tier_used": "worker-regex",
  "analysis_time_ms": 2.1,
  "source": "cloudflare-worker"
}
```

**Risk levels:**

| risk_level | scam_score | Meaning |
|-----------|-----------|---------|
| `safe` | < 0.2 | Verified Safe |
| `low` | 0.2 – 0.4 | Likely Legitimate |
| `suspicious` | 0.4 – 0.6 | Review carefully |
| `high` | 0.6 – 0.8 | Likely Scam |
| `scam` | ≥ 0.8 | Almost Certainly Scam |

### POST /api/report

Submit crowd-sourced feedback.

```json
{
  "url": "https://linkedin.com/jobs/view/123",
  "is_scam": true,
  "reason": "Asked for SSN before interview"
}
```

### GET /api/patterns

List active scam patterns stored in D1.

### GET /api/stats

Detection statistics (total scans, avg score, risk breakdown).

### GET /api/health

Health check endpoint.

## Rate Limiting

- 10 requests per minute per IP address
- Uses Cloudflare KV for sliding-window tracking
- Returns `429` with `Retry-After` header when exceeded
- `X-RateLimit-Remaining` header on every response

## Signals Implemented (16 total)

| Signal | Category | Weight |
|--------|----------|--------|
| `upfront_payment` | red_flag | 0.95 |
| `personal_info_request` | red_flag | 0.92 |
| `crypto_payment` | red_flag | 0.90 |
| `reshipping` | red_flag | 0.90 |
| `guaranteed_income` | red_flag | 0.85 |
| `mlm_language` | red_flag | 0.80 |
| `interview_bypass` | red_flag | 0.75 |
| `suspicious_email_domain` | red_flag | 0.78 |
| `no_company` | red_flag | 0.70–0.85 |
| `wfh_unrealistic` | warning | 0.65 |
| `salary_anomaly` | warning | 0.55–0.70 |
| `urgency_language` | warning | 0.58 |
| `suspicious_links` | structural | 0.58 |
| `stale_posting` | ghost_job | 0.42–0.58 |
| `vague_description` | warning | 0.50–0.65 |
| `no_qualifications` | warning | 0.48 |

## Chrome Extension Integration

The worker accepts the same `job_data` JSON format that the existing Chrome extension sends to the Python API. To point the extension at the Worker, update the extension's API base URL:

```js
// In extension/background.js or config
const API_BASE = 'https://jobsentinel-api.<subdomain>.workers.dev';
```

No other changes required — the response schema is identical to the Python API.

## Cost

- Workers free tier: 100,000 requests/day, no credit card required
- KV free tier: 100,000 reads/day, 1,000 writes/day
- D1 free tier: 5 million reads/day, 100,000 writes/day, 5 GB storage
- All free tiers are more than sufficient for individual or small-team use

## Custom Domain (Optional)

To serve from your own domain instead of `workers.dev`, uncomment and fill in the `[routes]` section in `wrangler.toml`:

```toml
[routes]
pattern = "api.yourdomain.com/*"
zone_name = "yourdomain.com"
```

Then deploy again with `npm run deploy`.
