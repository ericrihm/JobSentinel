/**
 * JobSentinel Popup Script
 */

const API_BASE = "http://localhost:8080";

// --- DOM refs ---
const states = {
  loading: document.getElementById("state-loading"),
  noJob: document.getElementById("state-no-job"),
  error: document.getElementById("state-error"),
  result: document.getElementById("state-result"),
};

function showState(name) {
  Object.entries(states).forEach(([key, el]) => {
    el.classList.toggle("hidden", key !== name);
  });
}

// --- Risk helpers ---

function riskInfo(score) {
  if (score < 0.2) return { label: "Safe", level: "safe", color: "#22c55e" };
  if (score < 0.4) return { label: "Likely Legit", level: "likely-legit", color: "#86efac" };
  if (score < 0.6) return { label: "Suspicious", level: "suspicious", color: "#eab308" };
  if (score < 0.8) return { label: "Likely Scam", level: "high", color: "#f97316" };
  return { label: "Scam", level: "scam", color: "#ef4444" };
}

// --- Render result ---

function renderResult(data) {
  const result = data.lastResult;
  const score = result.scam_score ?? result.score ?? 0;
  const { label, level, color } = riskInfo(score);
  const pct = Math.round(score * 100);

  // Badge
  const badge = document.getElementById("risk-badge");
  badge.className = `risk-badge risk-badge--${level}`;
  badge.style.borderColor = color;
  document.getElementById("risk-label").textContent = label;
  document.getElementById("risk-label").style.color = color;

  // Score %
  const scorePct = document.getElementById("score-pct");
  scorePct.textContent = `${pct}%`;
  scorePct.style.color = color;

  // Job meta
  document.getElementById("job-title").textContent = data.lastTitle || result.title || "";
  document.getElementById("job-company").textContent = data.lastCompany || result.company || "";

  // Confidence
  const conf = result.confidence ?? result.confidence_interval;
  if (conf !== undefined && conf !== null) {
    const confPct = typeof conf === "number" ? `${Math.round(conf * 100)}%` : JSON.stringify(conf);
    document.getElementById("confidence-val").textContent = confPct;
  }

  // Signals
  const signals = result.signals ?? result.signal_results ?? [];
  const redFlags = signals.filter((s) => s.category === "red_flag" && s.triggered);
  const warnings = signals.filter((s) => s.category === "warning" && s.triggered);
  const positives = signals.filter((s) => s.category === "positive" && s.triggered);

  renderSignalGroup("red-flags-section", "red-flags-list", redFlags);
  renderSignalGroup("warnings-section", "warnings-list", warnings);
  renderSignalGroup("positives-section", "positives-list", positives);

  // AI summary
  const summary = result.summary ?? result.explanation ?? result.ai_analysis?.summary;
  const summarySection = document.getElementById("summary-section");
  if (summary) {
    document.getElementById("summary-text").textContent = summary;
    summarySection.classList.remove("hidden");
  } else {
    summarySection.classList.add("hidden");
  }

  // Timestamp
  if (data.lastUpdated) {
    const ts = new Date(data.lastUpdated);
    document.getElementById("timestamp").textContent =
      `Analyzed ${ts.toLocaleTimeString()}`;
  }

  showState("result");
}

function renderSignalGroup(sectionId, listId, signals) {
  const section = document.getElementById(sectionId);
  const list = document.getElementById(listId);

  if (!signals || signals.length === 0) {
    section.classList.add("hidden");
    return;
  }

  list.innerHTML = "";
  signals.forEach((sig) => {
    const li = document.createElement("li");
    li.textContent = sig.description ?? sig.name ?? sig.signal_name ?? JSON.stringify(sig);
    list.appendChild(li);
  });

  section.classList.remove("hidden");
}

// --- Report actions ---

async function sendReport(verdict) {
  const data = await chrome.storage.local.get(["lastUrl", "lastResult"]);
  if (!data.lastUrl) return;

  const btn = verdict === "scam"
    ? document.getElementById("btn-report")
    : document.getElementById("btn-legit");

  btn.disabled = true;
  btn.textContent = "Sending…";

  try {
    const response = await fetch(`${API_BASE}/api/report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: data.lastUrl,
        verdict,
        scam_score: data.lastResult?.scam_score ?? data.lastResult?.score,
      }),
    });

    if (response.ok) {
      btn.textContent = verdict === "scam" ? "Reported ✓" : "Marked ✓";
      btn.classList.add("btn--success");
    } else {
      throw new Error(`HTTP ${response.status}`);
    }
  } catch (err) {
    btn.textContent = "Error — retry?";
    btn.disabled = false;
    console.error("[JobSentinel] Report failed:", err);
  }
}

document.getElementById("btn-report").addEventListener("click", () => sendReport("scam"));
document.getElementById("btn-legit").addEventListener("click", () => sendReport("legitimate"));

// --- Init ---

async function init() {
  showState("loading");

  // Get active tab
  let activeTab;
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    activeTab = tab;
  } catch (_) {
    // tabs permission not granted (shouldn't happen with activeTab)
  }

  const url = activeTab?.url ?? "";
  const isLinkedInJob =
    url.includes("linkedin.com/jobs/view/") ||
    url.includes("linkedin.com/jobs/collections/");

  if (!isLinkedInJob) {
    showState("noJob");
    return;
  }

  // Read from storage
  const data = await chrome.storage.local.get([
    "lastResult",
    "lastUrl",
    "lastTitle",
    "lastCompany",
    "lastError",
    "lastUpdated",
  ]);

  // Check if stored result matches current URL
  const resultFresh =
    data.lastResult &&
    data.lastUrl === url &&
    data.lastUpdated &&
    Date.now() - data.lastUpdated < 5 * 60 * 1000; // 5 min TTL

  if (resultFresh) {
    renderResult(data);
    return;
  }

  if (data.lastError && data.lastUrl === url) {
    document.getElementById("error-msg").textContent =
      `API error: ${data.lastError}`;
    showState("error");
    return;
  }

  // Still waiting for content script to finish
  showState("loading");

  // Poll storage briefly
  let attempts = 0;
  const poll = setInterval(async () => {
    attempts++;
    const fresh = await chrome.storage.local.get([
      "lastResult", "lastUrl", "lastTitle", "lastCompany", "lastError", "lastUpdated",
    ]);

    if (fresh.lastResult && fresh.lastUrl === url) {
      clearInterval(poll);
      renderResult(fresh);
    } else if (fresh.lastError && fresh.lastUrl === url) {
      clearInterval(poll);
      document.getElementById("error-msg").textContent =
        `API error: ${fresh.lastError}`;
      showState("error");
    } else if (attempts >= 10) {
      clearInterval(poll);
      // Timeout — show no-job or generic error
      document.getElementById("error-msg").textContent =
        "Analysis timed out. Make sure sentinel serve is running.";
      showState("error");
    }
  }, 500);
}

init();
