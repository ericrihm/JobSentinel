# CLAUDE.md — JobSentinel Development Guide

## Project Overview
JobSentinel is an open-source job aggregator with built-in scam detection (54+ signals). We aggregate jobs directly from company career pages (like hiring.cafe) and verify each listing against fraud signals before users see them. Serves three surfaces: Python backend, Cloudflare Worker API, and static website (docs/).

## Architecture

### Signal Engine (the core)
- Python: `sentinel/signals.py` (48 signals) + `sentinel/company_verifier.py` + `sentinel/adversarial.py`
- JavaScript: `workers/src/signals.js` (54 signals, ES module, no dependencies)
- **CRITICAL**: `workers/src/signals.js` is the source of truth for JS. It must be copied byte-for-byte to:
  - `docs/signals-engine.js` (website)
  - `sentinel/web/extension/signals-engine.js` (Chrome extension)
  After any change, run: `cp workers/src/signals.js docs/signals-engine.js && cp workers/src/signals.js sentinel/web/extension/signals-engine.js`

### Job Aggregation Pipeline
- **Source Adapters** (`sentinel/sources.py`): 9 adapters — Greenhouse, Lever, Ashby, SmartRecruiters, Adzuna, USAJobs, RemoteOK, Remotive, The Muse
- **Company Registry** (`sentinel/company_registry.json`): Maps companies to their ATS platform/slug. Only validated slugs.
- **Registry Loader** (`sentinel/registry.py`): Loads and caches the company registry
- **Pipeline** (`.github/workflows/scheduled-scan.yml`): Runs every 4 hours, fetches from all sources, scores each job, publishes to `docs/data/verified-jobs.json` and pushes to Worker D1

### Cloudflare Worker API (`workers/`)
- `POST /api/analyze` — Analyze a job posting for scam signals
- `GET /api/jobs` — Search/browse aggregated jobs (filters: q, location, remote, source, salary, risk, sort, pagination)
- `GET /api/jobs/:id` — Full job detail
- `POST /api/jobs/ingest` — Bulk ingest from pipeline (auth required, `INGEST_KEY`)
- `GET /api/jobs/stats` — Aggregated statistics
- `POST /api/report` — User scam reports
- `GET /api/patterns` — Known scam patterns
- D1 tables: `scan_history`, `patterns`, `user_reports`, `jobs`

### Website (docs/)
Static HTML/CSS/JS served via GitHub Pages at jobsentinel.org.
- **No build system** — plain HTML files, no bundler, no framework
- CSS: `docs/style.css` (shared) + page-specific inline `<style>` blocks
- Pages: index, analyze, demo, gallery, jobs, privacy, terms, 404
- `docs/jobs.html` — Dynamic job board with search, filters, API/static fallback
- `docs/data/verified-jobs.json` — Static fallback data updated every 4h by pipeline

## Cross-Page Consistency Rules

Every HTML page in docs/ MUST have:
1. **Nav links** to: index.html, analyze.html, gallery.html, jobs.html, demo.html
2. **Meta tags**: description, og:title, og:description, og:image, twitter:card, canonical
3. **Footer**: Full footer-grid (Product, Resources, Community columns) + copyright © 2026
4. **Includes**: `<link rel="stylesheet" href="style.css?v=HASH">`, favicon, viewport meta
5. **Accessibility**: aria-label on nav logo, aria-expanded on mobile toggle
6. **Script**: `<script type="module" src="analyze-bridge.js?v=HASH"></script>` before `</body>`
7. **Cache busting**: CSS/JS references use `?v=<commit-hash>` query strings, auto-updated by deploy workflow

When adding a new page: copy the structure from `docs/jobs.html` as a template. Add the page to `docs/sitemap.xml`.

When modifying shared elements (nav, footer): update ALL pages, not just one. Run `bash scripts/check-site.sh` to verify.

## Adding a New Source Adapter

1. Add class extending `JobSource` in `sentinel/sources.py`
2. For ATS adapters: add constructor taking `companies: list[str]`
3. Validate company slugs against the live API before adding to `sentinel/company_registry.json`
4. Add to `get_all_sources()` in sources.py
5. Add to the scan workflow import list in `.github/workflows/scheduled-scan.yml`
6. Add tests in `tests/test_sources.py`

## Signal Changes

When adding or modifying a signal:
1. Add/update in Python (`sentinel/signals.py`)
2. Add/update in JS (`workers/src/signals.js`)
3. Copy JS to website + extension (see Architecture above)
4. Add tests in `tests/test_core.py`
5. Verify signal name strings match exactly between Python and JS

## Testing
- Run: `python -m pytest tests/ -q`
- 2789+ tests, all must pass
- Site consistency: `bash scripts/check-site.sh`

## Design System

### Colors (CSS variables)
- `--green` (#3fb950): safe, verified, positive signals
- `--red` (#f85149): high risk, red flags, danger
- `--orange` (#db6d28): warnings, suspicious
- `--yellow` (#d29922): caution, ghost job indicators
- `--accent` (#58a6ff): links, CTAs, focus indicators
- `--bg` (#0a0c10): page background
- `--surface` (#12141a): card backgrounds

### Component Patterns
- Cards: `border-radius: 12px`, `border: 1px solid var(--border)`, `background: var(--surface)`
- Buttons: `.btn.btn-primary` for CTAs, `.btn.btn-secondary` for secondary actions
- Badges: `.badge.badge-green`, `.badge.badge-accent`, `.badge.badge-red`
- Signal pills: colored left-border on rows, dot indicators with category colors
- Fonts: Plus Jakarta Sans (display) + Inter (body)

### Tone
- Safety-first: lead with safety, not fear
- Specific over vague: "54 signals checked, 0 risk flags" not "verified safe"
- Plain language: explain what signals mean, don't just list them
- No dark patterns: extension nudges are dismissible, no gating content behind installs
- Direct from source: emphasize company career pages, not middlemen

## Git Conventions
- Commit format: `type: description` (feat, fix, chore, test, docs)
- Always include `Co-Authored-By:` trailer
- Don't push to main without tests passing

## Deployment
- **Website**: Push to main → `.github/workflows/deploy-pages.yml` → GitHub Pages (auto cache-busts CSS/JS with commit hash)
- **Worker**: `cd workers && npm run deploy` (requires `wrangler login`)
- **D1 migration**: `wrangler d1 execute jobsentinel-db --remote --file=src/migration-001-jobs.sql`
