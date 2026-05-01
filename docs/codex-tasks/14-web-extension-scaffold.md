# Task: Scaffold Chrome Browser Extension

## Task

Create a Chrome MV3 browser extension that analyzes LinkedIn job postings in-page and shows scam risk scores.

## Context

- JobSentinel has a FastAPI backend at `sentinel/api.py` with POST `/api/analyze` endpoint
- The API accepts `{text, title, company}` and returns scam score, risk level, signals
- CLAUDE.md references `sentinel/web/extension/` but the directory doesn't exist
- The extension should work with the API server running locally (`sentinel serve --port 8080`)

## What To Do

### 1. Create directory structure

```
sentinel/web/extension/
├── manifest.json
├── popup.html
├── popup.js
├── content.js
├── background.js
├── styles.css
└── icons/
    └── (placeholder — use data URIs or simple SVG for now)
```

### 2. manifest.json — Chrome MV3

```json
{
  "manifest_version": 3,
  "name": "JobSentinel",
  "version": "0.1.0",
  "description": "Detect scam job postings on LinkedIn",
  "permissions": ["activeTab", "storage"],
  "host_permissions": ["https://www.linkedin.com/*"],
  "action": {
    "default_popup": "popup.html"
  },
  "content_scripts": [{
    "matches": ["https://www.linkedin.com/jobs/*"],
    "js": ["content.js"]
  }],
  "background": {
    "service_worker": "background.js"
  }
}
```

### 3. content.js — DOM extraction

Extract from LinkedIn job page:
- Job title: `h1.t-24`, `h1.job-details-jobs-unified-top-card__job-title`, or first `h1`
- Company: `.job-details-jobs-unified-top-card__company-name`, `a[data-tracking-control-name="public_jobs_topcard-org-name"]`
- Description: `.jobs-description-content`, `.show-more-less-html__markup`, or `.description__text`
- Location: `.job-details-jobs-unified-top-card__bullet`

Send extracted data to background script via `chrome.runtime.sendMessage`.

### 4. background.js — API communication

Listen for messages from content script. POST to configurable API URL (default `http://localhost:8080/api/analyze`). Store API URL in `chrome.storage.local`. Return results to popup.

### 5. popup.html + popup.js — Results UI

Simple popup showing:
- Scan button (triggers content script extraction)
- Risk score with color-coded badge (green/yellow/orange/red)
- Risk level label
- Signal list (red flags, warnings, positive signals)
- Settings link to configure API URL
- Loading state while analysis runs

### 6. styles.css

Clean styling:
- Risk colors: safe=#22c55e, low=#84cc16, suspicious=#eab308, high=#f97316, scam=#ef4444
- Compact layout (popup is 350px wide, auto height)
- Signal list with category icons (! for red flags, ~ for warnings, + for positive)

## Acceptance Criteria

- [ ] Extension loads in Chrome via `chrome://extensions` (developer mode, load unpacked)
- [ ] Content script extracts job data from LinkedIn job pages
- [ ] Popup displays scam analysis results from local API
- [ ] Risk score is color-coded
- [ ] Signals are listed by category
- [ ] Graceful error handling when API is not running
- [ ] No external dependencies — pure HTML/CSS/JS

## Constraints

- Chrome MV3 only (no MV2 deprecated APIs)
- No build tools required (no webpack, no npm)
- API URL configurable via popup settings
- Do not store user data beyond settings
- Content script must not break LinkedIn page functionality

## Test Command

```bash
python -m pytest tests/ -v
```
