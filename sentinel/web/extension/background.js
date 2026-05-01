/**
 * JobSentinel Background Service Worker
 *
 * Responsibilities:
 *  - Receive ANALYZE_JOB messages from content.js / popup.js
 *  - Call https://api.jobsentinel.org/api/analyze (or user-configured endpoint)
 *  - Cache results in chrome.storage.local with a 24-hour TTL
 *  - Fallback to client-side signal detection when the API is unreachable
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_API_BASE = "https://api.jobsentinel.org";
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

// ---------------------------------------------------------------------------
// Client-side fallback signal detection (subset of workers/src/signals.js)
// ---------------------------------------------------------------------------

const RE_UPFRONT_PAY     = /\b(fee required|send money|training fee|buy equipment|purchase (your |a )?equipment|starter kit fee|background check fee|pay (a |the )?deposit|wire (me|us)|upfront (cost|fee|payment)|advance fee)\b/i;
const RE_PERSONAL_INFO   = /\b(social security|SSN|bank account( number)?|routing number|credit card( number)?|full (name and )?address|passport (number|copy)|drivers? licen[sc]e)\b/i;
const RE_GUARANTEED      = /\b(guaranteed (salary|income|pay|earnings?|profit)|earn \$[\d,]+\s*(a |per )?(day|daily|week|hour|hr) guaranteed)\b/i;
const RE_CRYPTO          = /\b(bitcoin|btc|ethereum|eth|crypto(currency)?|gift card|western union|moneygram|wire transfer|zelle|cashapp|venmo)\b/i;
const RE_URGENCY         = /\b(apply (now|immediately|today|asap)|limited (spots?|openings?|positions?)|hiring (immediately|now|today)|urgent(ly)? (hiring|needed)|act (now|fast|quickly))\b/i;
const RE_MLM             = /\b(be your own boss|unlimited earning potential|residual income|network marketing|mlm|passive income opportunity)\b/i;
const RE_RESHIPPING      = /\b(receive (packages?|parcels?)|reship(ping)?|re-ship(ping)?|forward (packages?|parcels?)|package (handler|inspector) (from|at) home)\b/i;
const RE_INTERVIEW_SKIP  = /\b(no interview (required|needed|necessary)|hired (on the spot|immediately|same day)|no resume (required|needed))\b/i;
const RE_PERSONAL_DOMAIN = /\b@(gmail|yahoo|hotmail|outlook|aol|icloud|protonmail)\./i;
const RE_SUSPICIOUS_LINK = /(bit\.ly|tinyurl\.com|t\.me\/|telegram\.me\/|wa\.me\/|typeform\.com)/i;
const RE_TECH_STACK      = /\b(python|javascript|typescript|golang|rust|react|angular|vue|django|flask|spring|postgres|mysql|mongodb|kubernetes|docker|terraform|aws|azure|gcp|sql|linux|git)\b/i;
const RE_EXPERIENCE_YRS  = /\b(\d+\+?\s*(?:years?|yrs?)\s*(of\s+)?(experience|exp))\b/i;
const RE_DEGREE          = /\b(bachelor|master|mba|ph\.?d|associate|degree (in|required)|diploma)\b/i;

function clientSideAnalyze(job) {
  const text = `${job.title || ""} ${job.description || ""}`;
  const signals = [];

  function flag(name, category, weight, confidence, detail, evidence) {
    signals.push({ name, category, weight, confidence, detail, evidence, triggered: true });
  }

  let m;
  if ((m = text.match(RE_UPFRONT_PAY)))   flag("upfront_payment",      "red_flag", 0.95, 0.90, "Requests upfront payment or equipment purchase", m[0]);
  if ((m = text.match(RE_PERSONAL_INFO))) flag("personal_info_request", "red_flag", 0.92, 0.88, "Requests sensitive personal/financial info before interview", m[0]);
  if ((m = text.match(RE_GUARANTEED)))    flag("guaranteed_income",     "red_flag", 0.85, 0.82, "Promises guaranteed income — legitimate employers never do this", m[0]);
  if ((m = text.match(RE_CRYPTO)))        flag("crypto_payment",        "red_flag", 0.90, 0.87, "Untraceable payment method mentioned (crypto/wire/gift card)", m[0]);
  if ((m = text.match(RE_RESHIPPING)))    flag("reshipping",            "red_flag", 0.90, 0.85, "Reshipping / package forwarding — classic money-mule vector", m[0]);
  if ((m = text.match(RE_MLM)))           flag("mlm_language",          "red_flag", 0.80, 0.82, "MLM / pyramid scheme language detected", m[0]);
  if ((m = text.match(RE_INTERVIEW_SKIP))) flag("interview_bypass",     "red_flag", 0.75, 0.80, "Posting explicitly skips standard hiring steps", m[0]);
  if ((m = text.match(RE_PERSONAL_DOMAIN))) flag("suspicious_email",   "red_flag", 0.78, 0.75, "Corporate role advertised with personal email domain", m[0]);
  if ((m = text.match(RE_SUSPICIOUS_LINK))) flag("suspicious_links",   "warning",  0.58, 0.65, "Description contains shortened or third-party form links", m[0]);
  if ((m = text.match(RE_URGENCY)))       flag("urgency_language",      "warning",  0.58, 0.65, "Artificial urgency language used to pressure applicants", m[0]);

  if (!job.company || !String(job.company).trim()) {
    flag("no_company", "red_flag", 0.85, 0.80, "No company name listed", "company field is empty");
  }

  const words = String(job.description || "").trim().split(/\s+/).filter(Boolean);
  if (words.length < 30) {
    flag("vague_description", "warning", words.length < 10 ? 0.65 : 0.50, 0.70,
      `Job description is extremely sparse (${words.length} words)`, "");
  } else if (words.length >= 20 && !RE_TECH_STACK.test(text) && !RE_EXPERIENCE_YRS.test(text) && !RE_DEGREE.test(text) && !/\b(require[sd]?|must have|qualif|proficien|skill)\b/i.test(text)) {
    flag("no_qualifications", "warning", 0.48, 0.55, "No skills, degree, or experience requirements listed", "");
  }

  // Bayesian log-odds scoring (matches workers/src/scorer.js)
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
    positive_signals: [],
    signal_count: signals.length,
    ai_tier_used: "client-fallback",
    source: "extension-fallback",
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
