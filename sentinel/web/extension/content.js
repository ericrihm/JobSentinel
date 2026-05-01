/**
 * JobSentinel Content Script
 * Extracts job details from LinkedIn job pages, delegates analysis to
 * background.js, and injects a risk badge next to the job title.
 */

const BADGE_ID = "jobsentinel-badge";
const POPUP_ID = "jobsentinel-detail-popup";

// ---------------------------------------------------------------------------
// Extraction helpers
// ---------------------------------------------------------------------------

function queryText(selectors) {
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim()) return el.textContent.trim();
  }
  return "";
}

function extractTitle() {
  return queryText([
    "h1.t-24",
    ".job-details-jobs-unified-top-card__job-title",
    ".jobs-unified-top-card__job-title",
    "h1",
  ]);
}

function extractCompany() {
  return queryText([
    ".job-details-jobs-unified-top-card__company-name",
    ".jobs-unified-top-card__company-name",
    ".topcard__org-name-link",
    ".jobs-details-top-card__company-url",
  ]);
}

function extractLocation() {
  return queryText([
    ".job-details-jobs-unified-top-card__bullet",
    ".jobs-unified-top-card__bullet",
    ".topcard__flavor--bullet",
  ]);
}

function extractSalary() {
  return queryText([
    ".job-details-jobs-unified-top-card__job-insight--highlight",
    ".compensation__salary",
    ".salary-main-rail__salary-info",
  ]);
}

function extractDescription() {
  return queryText([
    ".jobs-description-content",
    ".show-more-less-html__markup",
    ".description__text",
    ".jobs-box__html-content",
    "#job-details",
  ]);
}

function extractRecruiter() {
  return queryText([
    ".hirer-card__hirer-information",
    ".message-the-recruiter .artdeco-entity-lockup__title",
    ".jobs-poster__name",
  ]);
}

function findTitleElement() {
  const selectors = [
    "h1.t-24",
    ".job-details-jobs-unified-top-card__job-title",
    ".jobs-unified-top-card__job-title",
    "h1",
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el) return el;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Risk helpers
// ---------------------------------------------------------------------------

function riskLabel(score) {
  if (score < 0.2) return { label: "Safe",        level: "safe" };
  if (score < 0.4) return { label: "Likely Legit", level: "likely-legit" };
  if (score < 0.6) return { label: "Suspicious",   level: "suspicious" };
  if (score < 0.8) return { label: "Likely Scam",  level: "high" };
  return               { label: "Scam",         level: "scam" };
}

// ---------------------------------------------------------------------------
// Detail popup (shown on badge hover / click)
// ---------------------------------------------------------------------------

function buildDetailPopup(result) {
  const existing = document.getElementById(POPUP_ID);
  if (existing) existing.remove();

  const score   = result.scam_score ?? result.score ?? 0;
  const pct     = Math.round(score * 100);
  const { label, level } = riskLabel(score);
  const colorMap = { safe: "#22c55e", "likely-legit": "#86efac", suspicious: "#eab308", high: "#f97316", scam: "#ef4444" };
  const color    = colorMap[level] || "#94a3b8";

  const popup = document.createElement("div");
  popup.id = POPUP_ID;
  popup.style.cssText = [
    "position:absolute", "z-index:99999", "background:#1e293b",
    "color:#f1f5f9", "border:1px solid #334155", "border-radius:10px",
    "padding:14px 16px", "width:300px", "box-shadow:0 8px 32px rgba(0,0,0,0.5)",
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif",
    "font-size:13px", "line-height:1.5",
  ].join(";");

  // Score header
  const header = document.createElement("div");
  header.style.cssText = "display:flex;align-items:center;gap:10px;margin-bottom:10px;";
  header.innerHTML = `
    <span style="font-size:22px;font-weight:800;color:${color}">${pct}%</span>
    <span style="font-size:13px;font-weight:700;color:${color}">${label}</span>
    <span style="margin-left:auto;font-size:11px;color:#64748b">${result.ai_tier_used || ""}</span>
  `;
  popup.appendChild(header);

  // Signal groups
  const groups = [
    { key: "red_flags",       heading: "Red Flags",        color: "#ef4444", prefix: "✕" },
    { key: "warnings",        heading: "Warnings",         color: "#eab308", prefix: "⚠" },
    { key: "positive_signals", heading: "Positive Signals", color: "#22c55e", prefix: "✓" },
  ];

  for (const { key, heading, color: gc, prefix } of groups) {
    const items = result[key] ?? [];
    if (!items.length) continue;
    const section = document.createElement("div");
    section.style.cssText = "margin-bottom:8px;";
    section.innerHTML = `<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:${gc};margin-bottom:4px">${heading}</div>`;
    const ul = document.createElement("ul");
    ul.style.cssText = "list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:3px;";
    for (const sig of items) {
      const li = document.createElement("li");
      li.style.cssText = `font-size:12px;color:#94a3b8;padding:3px 6px;border-radius:4px;background:rgba(255,255,255,0.04);`;
      li.textContent = `${prefix} ${sig.detail || sig.name || ""}`;
      ul.appendChild(li);
    }
    section.appendChild(ul);
    popup.appendChild(section);
  }

  // Footer link
  const footer = document.createElement("div");
  footer.style.cssText = "margin-top:10px;padding-top:8px;border-top:1px solid #334155;text-align:right;";
  footer.innerHTML = `<a href="https://jobsentinel.org/analyze.html" target="_blank" rel="noopener"
    style="font-size:11px;color:#60a5fa;text-decoration:none;">View full report →</a>`;
  popup.appendChild(footer);

  // Close on outside click
  const closePopup = (e) => {
    if (!popup.contains(e.target) && e.target.id !== BADGE_ID) {
      popup.remove();
      document.removeEventListener("click", closePopup, true);
    }
  };
  setTimeout(() => document.addEventListener("click", closePopup, true), 0);

  return popup;
}

function showDetailPopup(result, badgeEl) {
  const popup = buildDetailPopup(result);
  document.body.appendChild(popup);

  // Position below badge
  const rect = badgeEl.getBoundingClientRect();
  const scrollY = window.scrollY;
  const scrollX = window.scrollX;
  let top  = rect.bottom + scrollY + 6;
  let left = rect.left  + scrollX;

  // Clamp to viewport width
  const popupW = 300;
  if (left + popupW > window.innerWidth + scrollX - 8) {
    left = window.innerWidth + scrollX - popupW - 8;
  }
  popup.style.top  = `${top}px`;
  popup.style.left = `${left}px`;
}

// ---------------------------------------------------------------------------
// Badge injection
// ---------------------------------------------------------------------------

function injectStyles() {
  if (document.getElementById("jobsentinel-styles")) return;
  const link = document.createElement("link");
  link.id    = "jobsentinel-styles";
  link.rel   = "stylesheet";
  link.href  = chrome.runtime.getURL("styles.css");
  document.head.appendChild(link);
}

function injectBadge(result) {
  const existing = document.getElementById(BADGE_ID);
  if (existing) existing.remove();

  const titleEl = findTitleElement();
  if (!titleEl) return;

  const score = result.scam_score ?? result.score ?? 0;
  const { label, level } = riskLabel(score);
  const pct = Math.round(score * 100);

  const badge = document.createElement("span");
  badge.id        = BADGE_ID;
  badge.className = `jobsentinel-badge jobsentinel-badge--${level}`;
  badge.title     = `JobSentinel: ${pct}% scam probability — click for details`;
  badge.textContent = `🛡 ${label} (${pct}%)`;
  badge.style.cursor = "pointer";

  badge.addEventListener("click", (e) => {
    e.stopPropagation();
    const existingPopup = document.getElementById(POPUP_ID);
    if (existingPopup) { existingPopup.remove(); return; }
    showDetailPopup(result, badge);
  });

  titleEl.insertAdjacentElement("afterend", badge);
}

function injectLoadingBadge() {
  const existing = document.getElementById(BADGE_ID);
  if (existing) existing.remove();

  const titleEl = findTitleElement();
  if (!titleEl) return;

  const badge = document.createElement("span");
  badge.id        = BADGE_ID;
  badge.className = "jobsentinel-badge jobsentinel-badge--loading";
  badge.textContent = "🛡 Analyzing…";
  titleEl.insertAdjacentElement("afterend", badge);
}

// ---------------------------------------------------------------------------
// Analysis — delegates to background.js via message passing
// ---------------------------------------------------------------------------

let lastUrl      = "";
let analysisTimer = null;

async function analyzeJob() {
  const url = window.location.href;

  if (!url.includes("/jobs/view/") && !url.includes("/jobs/collections/")) return;

  const title       = extractTitle();
  const company     = extractCompany();
  const description = extractDescription();
  const location    = extractLocation();
  const salary      = extractSalary();
  const recruiter   = extractRecruiter();

  if (url === lastUrl && title && company) return;
  lastUrl = url;

  injectStyles();
  injectLoadingBadge();

  const payload = { url, title, company, description, text: description, location, salary, recruiter_name: recruiter };

  try {
    const response = await chrome.runtime.sendMessage({ type: "ANALYZE_JOB", payload });

    if (!response || !response.ok) {
      throw new Error(response?.error || "Analysis failed");
    }

    injectBadge(response.result);

    chrome.runtime.sendMessage({ type: "ANALYSIS_COMPLETE", result: response.result, url });

  } catch (err) {
    const badge = document.getElementById(BADGE_ID);
    if (badge) {
      badge.className  = "jobsentinel-badge jobsentinel-badge--error";
      badge.textContent = "🛡 Error";
      badge.title       = err.message;
    }
    chrome.storage.local.set({ lastResult: null, lastUrl: url, lastError: err.message, lastUpdated: Date.now() });
  }
}

// ---------------------------------------------------------------------------
// SPA navigation via MutationObserver
// ---------------------------------------------------------------------------

function scheduleAnalysis() {
  if (analysisTimer) clearTimeout(analysisTimer);
  analysisTimer = setTimeout(analyzeJob, 800);
}

const observer = new MutationObserver(() => {
  if (window.location.href !== lastUrl) scheduleAnalysis();
});

observer.observe(document.body, { childList: true, subtree: true });

scheduleAnalysis();
