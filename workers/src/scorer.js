/**
 * scorer.js — Weighted Bayesian scoring engine (JS port of Python scorer)
 *
 * Uses log-odds accumulation matching the Python implementation:
 *   - Starts at log-odds 0 (50% prior)
 *   - Positive signals push toward legitimate (reduce score)
 *   - All other categories push toward scam (increase score)
 *   - Final score = sigmoid(log_odds)
 *
 * Risk thresholds match Python _RISK_THRESHOLDS:
 *   SAFE        < 0.2
 *   LOW         0.2 – 0.4
 *   SUSPICIOUS  0.4 – 0.6
 *   HIGH        0.6 – 0.8
 *   SCAM        >= 0.8
 */

// ---------------------------------------------------------------------------
// Core scoring
// ---------------------------------------------------------------------------

/**
 * Compute scam score and confidence from extracted signals.
 * @param {Array<{name: string, category: string, weight: number, confidence: number}>} signals
 * @returns {{ scam_score: number, confidence: number }}
 */
export function scoreSignals(signals) {
  if (!signals || signals.length === 0) {
    return { scam_score: 0.0, confidence: 0.0 };
  }

  const EPSILON = 1e-6;
  let logOdds = 0.0;

  for (const s of signals) {
    const w = Math.max(EPSILON, Math.min(1.0 - EPSILON, s.weight));

    if (s.category === 'positive') {
      // Positive signals: shift log-odds toward legitimate
      logOdds -= Math.log((1.0 - w) / w);
    } else {
      // Scam signals: shift log-odds toward scam
      logOdds += Math.log(w / (1.0 - w));
    }
  }

  // Sigmoid to get probability
  const scam_score = Math.round((1.0 / (1.0 + Math.exp(-logOdds))) * 10000) / 10000;

  // Confidence: grows with signal count, penalised when signals disagree
  const nScam = signals.filter(s => s.category !== 'positive').length;
  const nPos = signals.filter(s => s.category === 'positive').length;
  const total = signals.length;

  // Base confidence from signal count (asymptote ~0.95)
  const baseConf = 1.0 - Math.exp(-0.3 * total);

  // Agreement factor: 1.0 when all same-direction, lower when mixed
  const majority = Math.max(nScam, nPos);
  const agreement = total > 0 ? majority / total : 1.0;

  const confidence = Math.round(baseConf * agreement * 10000) / 10000;

  return { scam_score, confidence };
}

// ---------------------------------------------------------------------------
// Risk classification
// ---------------------------------------------------------------------------

const RISK_THRESHOLDS = {
  safe: 0.2,
  low: 0.4,
  suspicious: 0.6,
  high: 0.8,
};

/**
 * Map a scam score to a risk level string.
 * @param {number} scamScore
 * @returns {string}
 */
export function classifyRisk(scamScore) {
  if (scamScore < RISK_THRESHOLDS.safe) return 'safe';
  if (scamScore < RISK_THRESHOLDS.low) return 'low';
  if (scamScore < RISK_THRESHOLDS.suspicious) return 'suspicious';
  if (scamScore < RISK_THRESHOLDS.high) return 'high';
  return 'scam';
}

/**
 * Map a scam score to a human-readable risk label.
 * @param {number} scamScore
 * @returns {string}
 */
export function riskLabel(scamScore) {
  if (scamScore < 0.2) return 'Verified Safe';
  if (scamScore < 0.4) return 'Likely Legitimate';
  if (scamScore < 0.6) return 'Suspicious';
  if (scamScore < 0.8) return 'Likely Scam';
  return 'Almost Certainly Scam';
}

// ---------------------------------------------------------------------------
// Full result builder
// ---------------------------------------------------------------------------

/**
 * Build the complete analysis result object (matches Python ValidationResult.to_dict).
 * @param {object} job
 * @param {Array<object>} signals
 * @param {number} analysisTimeMs
 * @returns {object}
 */
export function buildResult(job, signals, analysisTimeMs = 0) {
  const { scam_score, confidence } = scoreSignals(signals);
  const risk_level = classifyRisk(scam_score);
  const risk_label = riskLabel(scam_score);

  // Partition signals by category
  const red_flags = signals
    .filter(s => s.category === 'red_flag')
    .map(s => ({ name: s.name, detail: s.detail, evidence: s.evidence }));

  const warnings = signals
    .filter(s => s.category === 'warning')
    .map(s => ({ name: s.name, detail: s.detail, evidence: s.evidence }));

  const ghost_indicators = signals
    .filter(s => s.category === 'ghost_job')
    .map(s => ({ name: s.name, detail: s.detail, evidence: s.evidence }));

  const positive_signals = signals
    .filter(s => s.category === 'positive')
    .map(s => ({ name: s.name, detail: s.detail, evidence: s.evidence }));

  const structural = signals
    .filter(s => s.category === 'structural')
    .map(s => ({ name: s.name, detail: s.detail, evidence: s.evidence }));

  return {
    job: {
      url: job.url || '',
      title: job.title || '',
      company: job.company || '',
      location: job.location || '',
      salary_min: job.salary_min || 0,
      salary_max: job.salary_max || 0,
      posted_date: job.posted_date || '',
      experience_level: job.experience_level || '',
      employment_type: job.employment_type || '',
      is_remote: Boolean(job.is_remote),
    },
    scam_score: Math.round(scam_score * 1000) / 1000,
    confidence: Math.round(confidence * 1000) / 1000,
    risk_level,
    risk_label,
    red_flags,
    warnings,
    ghost_indicators,
    positive_signals,
    structural,
    signal_count: signals.length,
    ai_tier_used: 'worker-regex',
    analysis_time_ms: Math.round(analysisTimeMs * 10) / 10,
    source: 'cloudflare-worker',
  };
}
