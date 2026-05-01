# Task: Scaffold Browser Extension

## Task

Create a Chrome MV3 browser extension scaffold at `sentinel/web/extension/` that integrates with the Sentinel API server to analyze LinkedIn job postings.

## Context

- `CLAUDE.md` documents the extension at `sentinel/web/extension/` and references a `sentinel extension-build` CLI command.
- The API server runs at `http://localhost:8080` by default (via `sentinel serve`) with these endpoints:
  - `POST /api/analyze` -- accepts `{"text": "...", "url": "...", "title": "...", "company": "..."}`, returns `{"scam_score": 0.85, "risk_level": "scam", "risk_label": "Almost Certainly Scam", "red_flags": [...], "warnings": [...], "positive_signals": [...], "signal_count": 5, "confidence": 0.8, ...}`.
  - `POST /api/report` -- accepts `{"url": "...", "is_scam": true, "reason": "..."}`.
  - `GET /api/health` -- returns `{"status": "ok", "healthy": true, ...}`.
- The extension should work on `linkedin.com/jobs/*` pages.
- This is a scaffold -- it should be functional with the API server running locally but does not need to be production-ready.
- The `sentinel/web/` directory does not exist yet; it needs to be created.

## What To Do

### 1. Create directory structure

```
sentinel/web/
    __init__.py          (empty, makes it a package)
sentinel/web/extension/
    manifest.json
    popup.html
    popup.js
    content.js
    background.js
    styles.css
```

### 2. Create `manifest.json` (Chrome MV3)

```json
{
  "manifest_version": 3,
  "name": "Sentinel - Job Scam Detector",
  "version": "0.1.0",
  "description": "AI-powered LinkedIn job scam detection. Analyzes job postings for scam signals in real-time.",
  "permissions": ["activeTab", "storage"],
  "host_permissions": [
    "https://www.linkedin.com/jobs/*",
    "http://localhost:8080/*"
  ],
  "action": {
    "default_popup": "popup.html",
    "default_title": "Sentinel Scam Detector"
  },
  "content_scripts": [
    {
      "matches": ["https://www.linkedin.com/jobs/*"],
      "js": ["content.js"],
      "css": ["styles.css"],
      "run_at": "document_idle"
    }
  ],
  "background": {
    "service_worker": "background.js"
  }
}
```

### 3. Create `content.js` -- DOM extraction and badge injection

The content script should:

1. **Extract job posting data from LinkedIn DOM** using CSS selectors. LinkedIn uses class names like:
   - Title: `.job-details-jobs-unified-top-card__job-title`, `.topcard__title`, `h1.t-24`
   - Company: `.job-details-jobs-unified-top-card__company-name`, `.topcard__org-name-link`
   - Description: `.jobs-description__content`, `.description__text`, `.show-more-less-html__markup`
   - Location: `.job-details-jobs-unified-top-card__bullet`, `.topcard__flavor--bullet`

2. **Send extracted data to the Sentinel API** at `http://localhost:8080/api/analyze` via `fetch()`.

3. **Inject a scam score badge** near the job title on the page, color-coded by risk level:
   - Safe/Low: green
   - Suspicious: yellow
   - High: orange
   - Scam: red

4. **Handle LinkedIn's SPA navigation** -- LinkedIn is a single-page app that doesn't reload between job views. Use a `MutationObserver` on `document.body` to detect when the user navigates to a new job, then re-extract and re-analyze.

5. **Store the analysis result** in `chrome.storage.local` so the popup can access it.

### 4. Create `popup.html` + `popup.js` -- Results UI

**popup.html:**
- Header with "Sentinel" title and "Job Scam Detector" subtitle.
- Loading state (shown while waiting for analysis).
- Result view with:
  - Risk badge (color-coded by risk level).
  - Scam score as percentage.
  - Confidence value.
  - Red flags section with list of detected red flag signals.
  - Warnings section with list of warning signals.
  - Positive signals section.
- "No job detected" state for non-job pages.
- "API error" state with instructions to run `sentinel serve`.
- Footer with "Report as Scam" and "Mark as Legitimate" buttons.

**popup.js:**
- Load stored result from `chrome.storage.local`.
- Render the result into the popup HTML.
- Wire up the Report/Legitimate buttons to call `POST /api/report`.
- If no stored result, check API health and show appropriate state.

### 5. Create `background.js` -- Service worker

Minimal service worker:
- Listen for messages from content script via `chrome.runtime.onMessage`.
- Log installation event.
- Handle `OPEN_POPUP` messages by storing data.

### 6. Create `styles.css`

Clean, minimal CSS for both the in-page badge and the popup:

- **Badge styles:** Non-intrusive badge injected near job title. Use class names like `sentinel-badge`, `sentinel-safe`, `sentinel-suspicious`, `sentinel-high`, `sentinel-scam` for color coding.
- **Popup styles:** 350px wide popup with clear sections, readable typography.
- **Color scheme:**
  - Safe: `#22c55e` (green)
  - Low: `#22c55e` (green)
  - Suspicious: `#eab308` (yellow)
  - High: `#f97316` (orange)
  - Scam: `#ef4444` (red)
- **Buttons:** Clean button styles for report actions.
- **Signal lists:** Bulleted lists with category-appropriate colors.

### 7. Create `sentinel/web/__init__.py`

Empty file to make `sentinel/web/` a Python package.

## Acceptance Criteria

- [ ] `sentinel/web/extension/manifest.json` is a valid Chrome MV3 manifest.
- [ ] `content.js` extracts job title, company, description, URL from LinkedIn DOM.
- [ ] `content.js` calls `POST /api/analyze` and injects a color-coded scam score badge.
- [ ] `content.js` handles SPA navigation via MutationObserver.
- [ ] `popup.html` + `popup.js` display scam score, risk level, and signal breakdown.
- [ ] `popup.js` has "Report as Scam" and "Mark as Legitimate" buttons calling `/api/report`.
- [ ] `background.js` handles message passing between content script and popup.
- [ ] `styles.css` provides clean styling with risk-level color coding.
- [ ] All files are in `sentinel/web/extension/`.
- [ ] `sentinel/web/__init__.py` exists.
- [ ] No Python tests are broken.

## Constraints

- Chrome MV3 only (no MV2 deprecated APIs).
- No build tools required (no webpack, no TypeScript, no npm). Plain JS, HTML, CSS.
- API URL defaults to `http://localhost:8080`. No complex configuration UI needed.
- Content script must handle LinkedIn's SPA navigation (MutationObserver pattern).
- Do not use any JavaScript frameworks (React, Vue, etc.).
- The extension is a scaffold -- it should be loadable in Chrome as an unpacked extension via `chrome://extensions` developer mode.
- Do not create icon files (omit them from manifest or use inline data).

## Test Command

```bash
python -m pytest tests/ -v
```
