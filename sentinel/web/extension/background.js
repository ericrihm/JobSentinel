/**
 * JobSentinel Background Service Worker
 *
 * Responsibilities:
 *  - Receive ANALYZE_JOB messages from content.js / popup.js
 *  - Call https://api.jobsentinel.org/api/analyze (or user-configured endpoint)
 *  - Cache results in chrome.storage.local with a 24-hour TTL
 *  - Fallback to full 54-signal client-side detection when the API is unreachable
 */

import { extractSignals } from './signals-engine.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_API_BASE = "https://api.jobsentinel.org";
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

// ---------------------------------------------------------------------------
// Client-side fallback signal detection (subset of workers/src/signals.js)
// ---------------------------------------------------------------------------


function clientSideAnalyze(job) {
  const signals = extractSignals(job);

  const EPSILON = 1e-6;
  let logOdds = 0;
  for (const s of signals) {
    const w = Math.max(EPSILON, Math.min(1 - EPSILON, s.weight));
    logOdds += s.category === "positive"
      ? -Math.log((1 - w) / w)
      : Math.log(w / (1 - w));
  }
  const scam_score = Math.round((1 / (1 + Math.exp(-logOdds))) * 1000) / 1000;
  const nTotal = signals.length;
  const nScam  = signals.filter(s => s.category !== "positive").length;
  const baseConf = 1 - Math.exp(-0.3 * nTotal);
  const agreement = nTotal > 0 ? Math.max(nScam, nTotal - nScam) / nTotal : 1;
  const confidence = Math.round(baseConf * agreement * 1000) / 1000;

  function riskLabel(score) {
    if (score < 0.2) return { risk_level: "safe",       risk_label: "Safe" };
    if (score < 0.4) return { risk_level: "low",        risk_label: "Likely Legitimate" };
    if (score < 0.6) return { risk_level: "suspicious", risk_label: "Suspicious" };
    if (score < 0.8) return { risk_level: "high",       risk_label: "Likely Scam" };
    return              { risk_level: "scam",       risk_label: "Almost Certainly Scam" };
  }

  return {
    scam_score,
    confidence,
    ...riskLabel(scam_score),
    signals,
    red_flags:       signals.filter(s => s.category === "red_flag"),
    warnings:        signals.filter(s => s.category === "warning"),
    positive_signals: signals.filter(s => s.category === "positive"),
    signal_count: signals.length,
    ai_tier_used: "client-full-engine",
    source: "extension-local",
  };
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

function cacheKey(url) {
  return `cache_${url}`;
}

async function getCached(url) {
  const key = cacheKey(url);
  const data = await chrome.storage.local.get(key);
  const entry = data[key];
  if (!entry) return null;
  if (Date.now() - entry.cachedAt > CACHE_TTL_MS) {
    chrome.storage.local.remove(key);
    return null;
  }
  return entry;
}

async function setCache(url, result, jobMeta) {
  const key = cacheKey(url);
  await chrome.storage.local.set({
    [key]: { result, jobMeta, cachedAt: Date.now() },
  });
}

// ---------------------------------------------------------------------------
// API call with client-side fallback
// ---------------------------------------------------------------------------

async function analyzeJob(payload) {
  const settings = await chrome.storage.local.get(["apiBase"]);
  const apiBase = settings.apiBase || DEFAULT_API_BASE;

  // Check cache first
  const cached = await getCached(payload.url);
  if (cached) {
    console.log("[JobSentinel] Cache hit for:", payload.url);
    return { result: cached.result, jobMeta: cached.jobMeta, fromCache: true };
  }

  // Try remote API
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(`${apiBase}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) throw new Error(`API HTTP ${response.status}`);
    const result = await response.json();

    const jobMeta = { title: payload.title, company: payload.company, location: payload.location };
    await setCache(payload.url, result, jobMeta);

    console.log("[JobSentinel] API result for:", payload.url, "score:", result.scam_score);
    return { result, jobMeta, fromCache: false, source: "api" };

  } catch (apiErr) {
    console.warn("[JobSentinel] API unreachable, falling back to client-side analysis:", apiErr.message);

    const result = clientSideAnalyze(payload);
    const jobMeta = { title: payload.title, company: payload.company, location: payload.location };
    await setCache(payload.url, result, jobMeta);

    return { result, jobMeta, fromCache: false, source: "fallback" };
  }
}

// ---------------------------------------------------------------------------
// Extension lifecycle
// ---------------------------------------------------------------------------

chrome.runtime.onInstalled.addListener((details) => {
  console.log("[JobSentinel] Extension installed:", details.reason);

  if (details.reason === "install") {
    chrome.storage.local.clear(() => {
      // Set defaults
      chrome.storage.local.set({ apiBase: DEFAULT_API_BASE, autoScan: true });
      console.log("[JobSentinel] Storage initialized with defaults.");
    });
  }
});

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  // Content script or popup requesting analysis
  if (message.type === "ANALYZE_JOB") {
    analyzeJob(message.payload)
      .then(({ result, jobMeta, fromCache, source }) => {
        // Persist to storage so popup can read it
        chrome.storage.local.set({
          lastResult:  result,
          lastUrl:     message.payload.url,
          lastTitle:   jobMeta.title || message.payload.title || "",
          lastCompany: jobMeta.company || message.payload.company || "",
          lastUpdated: Date.now(),
          lastError:   null,
        });
        sendResponse({ ok: true, result, fromCache, source });
      })
      .catch((err) => {
        chrome.storage.local.set({
          lastResult:  null,
          lastUrl:     message.payload.url,
          lastError:   err.message,
          lastUpdated: Date.now(),
        });
        sendResponse({ ok: false, error: err.message });
      });
    return true; // keep channel open for async
  }

  // Popup requesting current stored status
  if (message.type === "GET_STATUS") {
    chrome.storage.local.get(
      ["lastResult", "lastUrl", "lastTitle", "lastCompany", "lastError", "lastUpdated"],
      (data) => sendResponse(data)
    );
    return true;
  }

  // Cache clear request (from options page)
  if (message.type === "CLEAR_CACHE") {
    chrome.storage.local.get(null, (all) => {
      const cacheKeys = Object.keys(all).filter(k => k.startsWith("cache_"));
      chrome.storage.local.remove(cacheKeys, () => {
        sendResponse({ ok: true, cleared: cacheKeys.length });
      });
    });
    return true;
  }

  // Log completed analyses (from content script)
  if (message.type === "ANALYSIS_COMPLETE") {
    console.log(
      "[JobSentinel] Analysis complete for:", message.url,
      "| Score:", message.result?.scam_score ?? message.result?.score
    );
  }
});
