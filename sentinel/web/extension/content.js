/**
 * JobSentinel Content Script
 * Extracts job data from LinkedIn and sends it to the analysis API.
 */

const API_BASE = "http://localhost:8080";
const BADGE_ID = "jobsentinel-badge";

// --- Selectors ---

function extractTitle() {
  const selectors = [
    "h1.t-24",
    ".job-details-jobs-unified-top-card__job-title",
    ".jobs-unified-top-card__job-title",
    "h1",
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim()) {
      return el.textContent.trim();
    }
  }
  return "";
}

function extractCompany() {
  const selectors = [
    ".job-details-jobs-unified-top-card__company-name",
    ".jobs-unified-top-card__company-name",
    ".topcard__org-name-link",
    ".jobs-details-top-card__company-url",
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim()) {
      return el.textContent.trim();
    }
  }
  return "";
}

function extractDescription() {
  const selectors = [
    ".jobs-description-content",
    ".show-more-less-html__markup",
    ".description__text",
    ".jobs-box__html-content",
    "#job-details",
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && el.textContent.trim()) {
      return el.textContent.trim();
    }
  }
  return "";
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

// --- Badge injection ---

function riskLabel(score) {
  if (score < 0.2) return { label: "Safe", level: "safe" };
  if (score < 0.4) return { label: "Likely Legit", level: "likely-legit" };
  if (score < 0.6) return { label: "Suspicious", level: "suspicious" };
  if (score < 0.8) return { label: "Likely Scam", level: "high" };
  return { label: "Scam", level: "scam" };
}

function injectBadge(result) {
  // Remove existing badge
  const existing = document.getElementById(BADGE_ID);
  if (existing) existing.remove();

  const titleEl = findTitleElement();
  if (!titleEl) return;

  const score = result.scam_score ?? result.score ?? 0;
  const { label, level } = riskLabel(score);
  const pct = Math.round(score * 100);

  const badge = document.createElement("span");
  badge.id = BADGE_ID;
  badge.className = `jobsentinel-badge jobsentinel-badge--${level}`;
  badge.title = `JobSentinel: ${pct}% scam probability`;
  badge.textContent = `🛡 ${label} (${pct}%)`;

  // Insert after the title element
  titleEl.insertAdjacentElement("afterend", badge);
}

function injectLoadingBadge() {
  const existing = document.getElementById(BADGE_ID);
  if (existing) existing.remove();

  const titleEl = findTitleElement();
  if (!titleEl) return;

  const badge = document.createElement("span");
  badge.id = BADGE_ID;
  badge.className = "jobsentinel-badge jobsentinel-badge--loading";
  badge.textContent = "🛡 Analyzing…";
  titleEl.insertAdjacentElement("afterend", badge);
}

function injectStyles() {
  if (document.getElementById("jobsentinel-styles")) return;
  const link = document.createElement("link");
  link.id = "jobsentinel-styles";
  link.rel = "stylesheet";
  link.href = chrome.runtime.getURL("styles.css");
  document.head.appendChild(link);
}

// --- Analysis ---

let lastUrl = "";
let analysisTimer = null;

async function analyzeJob() {
  const url = window.location.href;

  // Only run on job detail pages
  if (!url.includes("/jobs/view/") && !url.includes("/jobs/collections/")) {
    return;
  }

  const title = extractTitle();
  const company = extractCompany();
  const text = extractDescription();

  // Don't re-analyze the same job
  if (url === lastUrl && title && company) return;
  lastUrl = url;

  injectStyles();
  injectLoadingBadge();

  const payload = { text, title, company, url };

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();

    // Store result for popup
    chrome.storage.local.set({
      lastResult: result,
      lastUrl: url,
      lastTitle: title,
      lastCompany: company,
      lastUpdated: Date.now(),
    });

    injectBadge(result);

    // Notify background
    chrome.runtime.sendMessage({
      type: "ANALYSIS_COMPLETE",
      result,
      url,
    });
  } catch (err) {
    const badge = document.getElementById(BADGE_ID);
    if (badge) {
      badge.className = "jobsentinel-badge jobsentinel-badge--error";
      badge.textContent = "🛡 Error";
      badge.title = err.message;
    }

    chrome.storage.local.set({
      lastResult: null,
      lastUrl: url,
      lastError: err.message,
      lastUpdated: Date.now(),
    });
  }
}

// --- SPA Navigation via MutationObserver ---

function scheduleAnalysis() {
  if (analysisTimer) clearTimeout(analysisTimer);
  // Debounce: wait for LinkedIn's SPA to finish rendering
  analysisTimer = setTimeout(analyzeJob, 800);
}

const observer = new MutationObserver((mutations) => {
  const currentUrl = window.location.href;
  if (currentUrl !== lastUrl) {
    scheduleAnalysis();
  }
});

observer.observe(document.body, { childList: true, subtree: true });

// Run on initial load
scheduleAnalysis();
