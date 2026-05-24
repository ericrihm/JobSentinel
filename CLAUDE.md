# Sentinel — Job Scam Detection Platform

Repo: https://github.com/ericrihm/JobSentinel — v0.2.0, MIT. Three surfaces: Python backend, Cloudflare Worker API (`api.jobsentinel.org`), static website (`docs/` → GitHub Pages at jobsentinel.org).

## Build & Test

```bash
pip install -e ".[dev]"          # core + pytest/ruff/hypothesis/mutmut
pip install -e ".[full]"         # + anthropic, fastapi, httpx, bs4

python -m pytest tests/ -q       # 2819+ tests, all must pass
pytest tests/test_core.py        # single module
ruff check sentinel/             # lint (line-length 120, py312 target)
ruff format sentinel/

bash scripts/check-site.sh       # cross-page HTML consistency audit
```

## Architecture

```
sentinel/          Python package (entry: sentinel.cli:main)
workers/           Cloudflare Worker — Node.js/Wrangler, D1 + KV
  src/index.js     Router + all API handlers
  src/signals.js   54-signal JS engine (ES module, zero deps) ← SOURCE OF TRUTH
  src/scorer.js    JS scorer
  src/schema.sql   D1 schema: scan_history, patterns, user_reports, jobs
docs/              GitHub Pages static site (no build system — raw HTML/CSS/JS)
  signals-engine.js  ← COPY of workers/src/signals.js
  web/extension/signals-engine.js  ← COPY of workers/src/signals.js
tests/             60 test modules, conftest.py
.github/workflows/ ci.yml, deploy-pages.yml, deploy-worker.yml,
                   scheduled-scan.yml (every 4h), research-cycle.yml,
                   release.yml, site-check.yml
```

## Key Modules

- `signals.py` — 50+ heuristic scam signals (Python). TextNormalizer + EvasionDetector applied before every check.
- `scorer.py` — Bayesian scoring with Thompson Sampling weight learning. `SignalWeightTracker` (canonical). Falls back to static weights if DB unavailable.
- `nexus.py` — Unified entry point: `Nexus.deep_analyze(job) → NexusReport`. Gracefully degrades if any subsystem fails import. Also: `NexusLearner`, `NexusDashboard`, `NexusEvolver`.
- `flywheel.py` — INGEST → SCORE → VALIDATE → LEARN → EVOLVE loop. Imports `SignalWeightTracker` from scorer (backwards-compat re-export). `CUSUMDetector` for regression alarm.
- `daemon.py` — `SentinelDaemon`: blocking autonomous loop. Cycle phases: INGEST → SCORE → LEARN → EVOLVE → INNOVATE → RESEARCH → CORTEX → META-EVOLVE → SLEEP. Graceful SIGINT/SIGTERM.
- `cortex.py` — Meta-cognitive layer: observes all subsystems, routes signals, strategic resource allocation, tracks learning velocity.
- `autonomic.py` — Self-healing immune system: `CheckpointManager`, `RegressionGuard` (CUSUM+EWMA dual-monitor), `SelfIterator` (hill-climb + simulated annealing), `AutonomicController`.
- `innovation.py` — Thompson Sampling improvement engine: RESEARCH → GENERATE → TEST → MEASURE → PROMOTE cycle.
- `research.py` — Autonomous knowledge discovery flywheel. Targets weakest detection areas. Integrates findings into pattern DB.
- `meta_evolution.py` — Makes the flywheel system itself self-improving. `FitnessLandscape`, GP optimizer, `FlywheelSurgeon`.
- `sources.py` — 9 job source adapters: Greenhouse, Lever, Ashby, SmartRecruiters, Adzuna, USAJobs, RemoteOK, Remotive, The Muse.
- `company_registry.json` + `registry.py` — Maps companies → ATS platform/slug (validated only). Cached loader.
- `company_verifier.py` — Live company domain/existence verification.
- `adversarial.py` — `EvasionDetector` + `TextNormalizer` (char-dupe, char-drop, homoglyph normalization).
- `analyzer.py` — AI escalation tier: heuristics → LLM if heuristics are ambiguous.
- `llm_detect.py` — Anthropic SDK integration for AI-tier analysis (`sentinel[ai]`).
- `db.py` — SQLite state DB (WAL). Tables: patterns, scan_history, user_reports.
- `models.py` — Dataclasses: `JobPosting`, `ScamSignal`, `SignalCategory`, `RiskLevel`, `UserReport`, `ValidationResult`.
- `api.py` — FastAPI app (optional, `sentinel[api]`). Same endpoints as CF Worker for local dev.
- `mesh.py` — Interop mesh integration (fleet-level awareness).
- `shadow.py` — Shadow scorer for A/B weight testing.
- `ingest.py` / `scanner.py` — Pipeline glue for batch fetch+score.
- `throttle.py` — Request rate limiting for source adapters.
- `faik_patterns.py` / `fraud_handbook.py` / `scam_data.py` — Static pattern libraries and Benford analysis.
- `stylometry.py` — Writing-style fingerprinting for scam template detection.
- `graph.py` — Graph-based relationship signals (company/recruiter networks).
- `honeypot.py` — Honeypot job detection.
- `web/extension/` — Chrome extension (uses copied `signals-engine.js`).

## CLI Commands

```bash
sentinel analyze "job description text"          # heuristic analyze
sentinel analyze --file jobs.json                # batch from JSON
sentinel analyze --no-ai "..."                   # heuristics only, no LLM
sentinel validate --company "Acme" --domain acme.com  # verify company
sentinel report --url URL --reason "fake"        # submit scam report
sentinel patterns --type scam --status active    # browse pattern DB
sentinel stats                                   # DB/detection statistics
sentinel scan --query "remote engineer" --limit 20   # fetch + score live jobs
sentinel ingest --query "..." --sources greenhouse   # ingest from source adapters
sentinel ingest-history --limit 50               # browse ingest history
sentinel auto --queries "eng" --loop-count 5     # run flywheel N cycles
sentinel daemon --interval 3600 --max-cycles 0   # autonomous daemon (blocking)
sentinel evolve                                  # one NexusEvolver cycle
sentinel innovate                                # one Thompson Sampling strategy
sentinel innovation-report                       # show strategy effectiveness
sentinel needs-review --score-threshold 0.7      # surface uncertain jobs
sentinel cascade                                 # show/run cascade pipeline
sentinel mesh                                    # interop mesh status
sentinel plugins                                 # list loaded plugins
sentinel serve --port 8000                       # FastAPI local server
sentinel --json-output <cmd>                     # machine-readable JSON output
```

## Deployment

**Website (GitHub Pages)**
- Push to main → `deploy-pages.yml` → jobsentinel.org
- Auto cache-busts CSS/JS references with commit hash (`?v=<hash>`)
- Verify with `bash scripts/check-site.sh` before pushing

**Cloudflare Worker**
```bash
cd workers && npm run deploy          # requires wrangler login
wrangler d1 execute jobsentinel-db --remote --file=src/migration-001-jobs.sql
```
- Custom domain: `api.jobsentinel.org` (configure in CF dashboard → Workers & Pages → Triggers)
- Secrets: `wrangler secret put INGEST_KEY`
- KV namespace for rate limiting: `wrangler kv:namespace create "RATE_LIMIT"`

**Scheduled pipeline** — `scheduled-scan.yml` runs every 4h: fetch all sources → score → publish `docs/data/verified-jobs.json` → push to D1.

## Key Constraints

**JS signal sync (critical)** — `workers/src/signals.js` is the JS source of truth. After ANY change:
```bash
cp workers/src/signals.js docs/signals-engine.js
cp workers/src/signals.js sentinel/web/extension/signals-engine.js
```
Signal name strings must match exactly between Python (`signals.py`) and JS.

**Adding a signal** — Update Python AND JS, copy JS, add tests in `tests/test_core.py`.

**Adding a source adapter** — Extend `JobSource` in `sources.py`, validate company slugs against live API before adding to `company_registry.json`, add to `get_all_sources()`, add to workflow import list, add tests in `tests/test_sources.py`.

**HTML pages** — Every `docs/*.html` must have: nav links to all 5 pages, full meta/OG/twitter tags, `canonical`, footer-grid (Product/Resources/Community + © 2026), `style.css?v=HASH`, `analyze-bridge.js?v=HASH`. Template from `docs/jobs.html`. New pages → add to `docs/sitemap.xml`.

**Coverage gate** — `pytest-cov` enforces 80% minimum (`fail_under = 80`).

**Commit format** — `type: description` (feat/fix/chore/test/docs). Tests must pass before merge to main.

## Design System (docs/)

```
--bg:      #0a0c10   page background
--surface: #12141a   card backgrounds
--green:   #3fb950   safe / verified
--red:     #f85149   high risk / scam
--orange:  #db6d28   suspicious
--yellow:  #d29922   caution / ghost job
--accent:  #58a6ff   links / CTAs
Fonts: Plus Jakarta Sans (display) + Inter (body)
Cards: border-radius 12px, border 1px solid var(--border)
```
