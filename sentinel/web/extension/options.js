/**
 * JobSentinel Options Page Script
 */

const DEFAULT_API_BASE = "https://api.jobsentinel.org";

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const inputApiBase    = document.getElementById("input-api-base");
const toggleAutoScan  = document.getElementById("toggle-auto-scan");
const btnClearCache   = document.getElementById("btn-clear-cache");
const btnSave         = document.getElementById("btn-save");
const saveStatus      = document.getElementById("save-status");
const cacheStatus     = document.getElementById("cache-status");

// ---------------------------------------------------------------------------
// Load saved settings
// ---------------------------------------------------------------------------

async function loadSettings() {
  const data = await chrome.storage.local.get(["apiBase", "autoScan"]);
  inputApiBase.value      = data.apiBase   ?? DEFAULT_API_BASE;
  toggleAutoScan.checked  = data.autoScan  !== false; // default true
}

// ---------------------------------------------------------------------------
// Save settings
// ---------------------------------------------------------------------------

async function saveSettings() {
  const apiBase  = inputApiBase.value.trim() || DEFAULT_API_BASE;
  const autoScan = toggleAutoScan.checked;

  // Basic URL validation
  try {
    new URL(apiBase);
  } catch {
    showStatus(saveStatus, "Invalid URL — please enter a valid API endpoint.", false);
    return;
  }

  await chrome.storage.local.set({ apiBase, autoScan });
  showStatus(saveStatus, "Settings saved.", true);
}

// ---------------------------------------------------------------------------
// Clear cache
// ---------------------------------------------------------------------------

async function clearCache() {
  btnClearCache.disabled    = true;
  btnClearCache.textContent = "Clearing…";

  // Ask background to clear cache keys
  try {
    const response = await chrome.runtime.sendMessage({ type: "CLEAR_CACHE" });
    const count = response?.cleared ?? 0;
    showStatus(cacheStatus, `Cache cleared — removed ${count} cached result${count !== 1 ? "s" : ""}.`, true);
  } catch (err) {
    // Fallback: clear directly if background isn't running
    const all = await chrome.storage.local.get(null);
    const keys = Object.keys(all).filter(k => k.startsWith("cache_"));
    await chrome.storage.local.remove(keys);
    showStatus(cacheStatus, `Cache cleared — removed ${keys.length} cached result${keys.length !== 1 ? "s" : ""}.`, true);
  }

  btnClearCache.textContent = "Clear Cache";
  btnClearCache.disabled    = false;
}

// ---------------------------------------------------------------------------
// Status helper
// ---------------------------------------------------------------------------

function showStatus(el, msg, success) {
  el.textContent  = msg;
  el.style.color  = success ? "var(--color-safe)" : "var(--color-scam)";
  setTimeout(() => { el.textContent = ""; }, 3000);
}

// ---------------------------------------------------------------------------
// Event bindings
// ---------------------------------------------------------------------------

btnSave.addEventListener("click", saveSettings);
btnClearCache.addEventListener("click", clearCache);

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadSettings();
