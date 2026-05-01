/**
 * analyze-bridge.js — Bridge between the full 54-signal engine and the web UI.
 *
 * Imports the same signal engine used by the Cloudflare Worker and browser
 * extension, so the website delivers identical detection quality client-side.
 *
 * Usage:
 *   <script type="module" src="analyze-bridge.js"></script>
 *   Then call window.sentinelAnalyze(text) from any inline script.
 */

import { extractSignals } from './signals-engine.js';

/**
 * Build a minimal JobPosting-like object from raw text input.
 * The signal engine expects fields like title, description, company, etc.
 * When analyzing pasted text, we put everything in description and let
 * the signals parse what they can.
 */
function buildJobFromText(text) {
  const lines = text.trim().split('\n').filter(l => l.trim());

  let title = '';
  let company = '';
  let description = text;
  let url = '';
  let salary_min = 0;
  let salary_max = 0;

  // Heuristic: first line might be the job title if it's short
  if (lines.length > 1 && lines[0].length < 100) {
    title = lines[0].trim();
    description = lines.slice(1).join('\n');
  }

  // Try to extract company name from common patterns
  const companyMatch = text.match(/(?:company|employer|organization|firm):\s*(.+)/i)
    || text.match(/(?:at|for|with)\s+([A-Z][A-Za-z\s&.,]+?)(?:\s*[-–—|]\s|\.\s|\n)/);
  if (companyMatch) {
    company = companyMatch[1].trim().slice(0, 100);
  }

  // Try to extract URL
  const urlMatch = text.match(/https?:\/\/[^\s<>"']+/);
  if (urlMatch) {
    url = urlMatch[0];
  }

  // Try to extract salary range
  const salaryMatch = text.match(/\$[\d,]+[kK]?\s*[-–—to]+\s*\$?([\d,]+)[kK]?/);
  if (salaryMatch) {
    const nums = text.match(/\$([\d,]+)/g);
    if (nums && nums.length >= 2) {
      salary_min = parseInt(nums[0].replace(/[$,]/g, ''), 10);
      salary_max = parseInt(nums[1].replace(/[$,]/g, ''), 10);
      // Handle 'k' suffix
      if (text.match(/\$[\d,]+[kK]/)) {
        if (salary_min < 1000) salary_min *= 1000;
        if (salary_max < 1000) salary_max *= 1000;
      }
    }
  }

  return {
    url,
    title,
    description,
    company,
    company_linkedin_url: '',
    salary_min,
    salary_max,
    posted_date: '',
    recruiter_connections: 0,
    recruiter_name: '',
    is_remote: /\b(remote|work from home|wfh)\b/i.test(text),
    is_repost: false,
    repost_count: 0,
    applicant_count: 0,
    days_posted: 0,
    location: '',
  };
}

/**
 * Analyze text using the full 54-signal engine.
 * Returns { scamScore (0-100), signals, riskLevel, riskLabel }.
 */
function analyze(text) {
  if (!text || !text.trim()) {
    return { scamScore: 0, signals: [], riskLevel: 'safe', riskLabel: 'No text provided' };
  }

  const job = buildJobFromText(text);
  const rawSignals = extractSignals(job);

  // Calculate weighted score (same algorithm as Python analyzer)
  let positiveWeight = 0;
  let negativeWeight = 0;

  const signals = rawSignals.map(s => {
    const w = s.weight || 0;
    const conf = s.confidence || 0.5;
    const effectiveWeight = w * conf;

    if (s.category === 'positive') {
      positiveWeight += effectiveWeight;
    } else {
      negativeWeight += effectiveWeight;
    }

    return {
      id: s.name,
      label: s.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      category: s.category,
      weight: Math.round(w * 100),
      confidence: Math.round(conf * 100),
      detail: s.detail || '',
      evidence: s.evidence || '',
    };
  });

  // Compute score: negative signals push up, positive signals pull down
  const rawScore = Math.max(0, negativeWeight - positiveWeight * 0.5);
  const scamScore = Math.round(Math.min(100, rawScore * 100));

  let riskLevel, riskLabel;
  if (scamScore >= 65) {
    riskLevel = 'high';
    riskLabel = 'HIGH RISK — Multiple strong scam indicators detected';
  } else if (scamScore >= 40) {
    riskLevel = 'suspicious';
    riskLabel = 'SUSPICIOUS — Several warning signs present';
  } else if (scamScore >= 20) {
    riskLevel = 'caution';
    riskLabel = 'SOME WARNINGS — Minor concerns detected';
  } else {
    riskLevel = 'safe';
    riskLabel = 'LIKELY SAFE — No significant scam signals found';
  }

  return {
    scamScore,
    signals,
    riskLevel,
    riskLabel,
    signalCount: signals.length,
    source: 'client-side (full engine)',
  };
}

// Expose to non-module scripts
window.sentinelAnalyze = analyze;
window.sentinelBuildJob = buildJobFromText;
window.sentinelExtractSignals = extractSignals;

export { analyze, buildJobFromText };
