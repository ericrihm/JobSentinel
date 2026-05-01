/**
 * JobSentinel Background Service Worker
 */

chrome.runtime.onInstalled.addListener((details) => {
  console.log("[JobSentinel] Extension installed:", details.reason);

  if (details.reason === "install") {
    // Clear any stale storage on fresh install
    chrome.storage.local.clear(() => {
      console.log("[JobSentinel] Storage cleared on install.");
    });
  }
});

// Listen for messages from content script or popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ANALYSIS_COMPLETE") {
    console.log(
      "[JobSentinel] Analysis complete for:",
      message.url,
      "| Score:",
      message.result?.scam_score ?? message.result?.score
    );
  }

  if (message.type === "GET_STATUS") {
    chrome.storage.local.get(["lastResult", "lastUrl", "lastError", "lastUpdated"], (data) => {
      sendResponse(data);
    });
    return true; // Keep channel open for async response
  }
});
