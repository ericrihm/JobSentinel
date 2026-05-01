/**
 * signals.js — 21 scam signal detectors ported from Python
 *
 * Each detector receives a job object and returns a signal object or null.
 * Signal object shape: { name, category, weight, confidence, detail, evidence }
 *
 * Categories: "red_flag" | "warning" | "ghost_job" | "structural" | "positive"
 */

// ---------------------------------------------------------------------------
// Compiled regex patterns (module-level, reused across requests)
// ---------------------------------------------------------------------------

const RE_PERSONAL_DOMAINS = /\b@(gmail|yahoo|hotmail|outlook|aol|icloud|protonmail|mail|ymail|live|msn|me|googlemail)\./i;

const RE_UPFRONT_PAY = /\b(fee required|send money|training fee|buy equipment|purchase (your |a )?equipment|starter kit fee|background check fee|pay (a |the )?deposit|wire (me|us)|upfront (cost|fee|payment)|advance fee)\b/i;

const RE_PERSONAL_INFO = /\b(social security|SSN|bank account( number)?|routing number|credit card( number)?|debit card|full (name and )?address|passport (number|copy)|drivers? licen[sc]e)\b/i;

const RE_GUARANTEED_INCOME = /\b(guaranteed (salary|income|pay|earnings?|profit)|earn \$[\d,]+\s*(a |per )?(day|daily|week|hour|hr) guaranteed|(guaranteed|promise[sd]) to (earn|make|pay))\b/i;

const RE_CRYPTO = /\b(bitcoin|btc|ethereum|eth|crypto(currency)?|gift card|western union|moneygram|wire transfer|zelle|cashapp|venmo)\b/i;

const RE_URGENCY = /\b(apply (now|immediately|today|asap)|limited (spots?|openings?|positions?)|hiring (immediately|now|today|asap)|urgent(ly)? (hiring|needed)|positions? (filling|fill) fast|don'?t (miss|wait)|act (now|fast|quickly)|only \d+ (spots?|seats?|positions?) (left|remaining|available))\b/i;

const RE_NO_EXPERIENCE = /\b(no experience (required|needed|necessary)|no (skills?|qualifications?|background) (required|needed)|anyone can|so easy|simple (job|work|tasks?))\b/i;

const RE_INTERVIEW_BYPASS = /\b(no interview (required|needed|necessary)|hired (on the spot|immediately|same day)|start (immediately|today|right away) no (questions?|interview)|no resume (required|needed|necessary)|no background (check|screening))\b/i;

const RE_MLM_LANGUAGE = /\b(be your own boss|unlimited earning potential|residual income|network marketing|multi.?level marketing|mlm|downline|upline|recruit (others|people|friends)|financial freedom|work when you want|set your own (hours|schedule)|passive income opportunity)\b/i;

const RE_RESHIPPING = /\b(receive (packages?|parcels?|shipments?)|reship(ping)?|re-ship(ping)?|forward (packages?|parcels?)|package (handler|inspector|coordinator) (from|at) home|quality control inspector (from|at) home|inspect (packages?|items?) (at|from) home)\b/i;

const RE_TECH_STACK = /\b(python|java(?:script)?|typescript|golang|rust|ruby|php|swift|kotlin|c\+\+|c#|scala|react|angular|vue|django|flask|fastapi|spring|rails|node(?:\.js)?|express|postgres(?:ql)?|mysql|mongodb|redis|kafka|kubernetes|docker|terraform|aws|azure|gcp|sql|linux|bash|git)\b/i;

const RE_EXPERIENCE_YRS = /\b(\d+\+?\s*(?:years?|yrs?)\s*(of\s+)?(experience|exp))\b/i;

const RE_DEGREE = /\b(bachelor|b\.s\.|b\.a\.|master|m\.s\.|m\.a\.|mba|ph\.?d|associate|degree (in|required)|diploma)\b/i;

const RE_SUSPICIOUS_LINKS = /(bit\.ly|tinyurl\.com|t\.co\/|forms\.gle|docs\.google\.com\/forms|t\.me\/|telegram\.me\/|wa\.me\/|whatsapp\.com\/|typeform\.com)/i;

const RE_GRAMMAR_CAPS = /\b[A-Z]{5,}\b/g;
const RE_GRAMMAR_PUNCT = /[!?]{2,}/g;

const RE_STALE_DATE_YMD = /^(\d{4})-(\d{2})-(\d{2})(?:T[\d:]+Z?)?$/;

// ---------------------------------------------------------------------------
// Helper: get full text combining title + description
// ---------------------------------------------------------------------------

function fullText(job) {
  return `${job.title || ''} ${job.description || ''}`.trim();
}

// ---------------------------------------------------------------------------
// Helper: parse days since posted date
// ---------------------------------------------------------------------------

function daysSincePosted(postedDate) {
  if (!postedDate) return null;
  const m = String(postedDate).match(RE_STALE_DATE_YMD);
  if (!m) return null;
  const dt = new Date(`${m[1]}-${m[2]}-${m[3]}T00:00:00Z`);
  if (isNaN(dt.getTime())) return null;
  return Math.floor((Date.now() - dt.getTime()) / 86400000);
}

// ---------------------------------------------------------------------------
// Signal 1: upfront_payment  (RED FLAG, weight 0.95)
// ---------------------------------------------------------------------------

export function checkUpfrontPayment(job) {
  const text = fullText(job);
  const m = text.match(RE_UPFRONT_PAY);
  if (!m) return null;
  return {
    name: 'upfront_payment',
    category: 'red_flag',
    weight: 0.95,
    confidence: 0.90,
    detail: 'Posting requests upfront payment or equipment purchase',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 2: personal_info_request  (RED FLAG, weight 0.92)
// ---------------------------------------------------------------------------

export function checkPersonalInfoRequest(job) {
  const text = fullText(job);
  const m = text.match(RE_PERSONAL_INFO);
  if (!m) return null;
  return {
    name: 'personal_info_request',
    category: 'red_flag',
    weight: 0.92,
    confidence: 0.88,
    detail: 'Posting requests sensitive personal/financial info before interview',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 3: guaranteed_income  (RED FLAG, weight 0.85)
// ---------------------------------------------------------------------------

export function checkGuaranteedIncome(job) {
  const text = fullText(job);
  const m = text.match(RE_GUARANTEED_INCOME);
  if (!m) return null;
  return {
    name: 'guaranteed_income',
    category: 'red_flag',
    weight: 0.85,
    confidence: 0.82,
    detail: 'Posting promises guaranteed income — legitimate employers never do this',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 4: suspicious_email_domain  (RED FLAG, weight 0.78)
// ---------------------------------------------------------------------------

export function checkSuspiciousEmailDomain(job) {
  const combined = `${fullText(job)} ${job.recruiter_name || ''}`;
  const m = combined.match(RE_PERSONAL_DOMAINS);
  if (!m) return null;
  return {
    name: 'suspicious_email_domain',
    category: 'red_flag',
    weight: 0.78,
    confidence: 0.75,
    detail: 'Corporate role advertised with personal email domain',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 5: crypto_payment  (RED FLAG, weight 0.90)
// ---------------------------------------------------------------------------

export function checkCryptoPayment(job) {
  const text = fullText(job);
  const m = text.match(RE_CRYPTO);
  if (!m) return null;
  return {
    name: 'crypto_payment',
    category: 'red_flag',
    weight: 0.90,
    confidence: 0.87,
    detail: 'Untraceable payment method mentioned (crypto/wire/gift card)',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 6: no_company  (RED FLAG, weight 0.85/0.70)
// ---------------------------------------------------------------------------

export function checkNoCompanyPresence(job) {
  if (!String(job.company || '').trim()) {
    return {
      name: 'no_company',
      category: 'red_flag',
      weight: 0.85,
      confidence: 0.80,
      detail: 'No company name listed',
      evidence: 'company field is empty',
    };
  }
  if (!String(job.company_linkedin_url || '').trim()) {
    return {
      name: 'no_company',
      category: 'red_flag',
      weight: 0.70,
      confidence: 0.65,
      detail: 'No company LinkedIn page linked',
      evidence: 'company_linkedin_url is empty',
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal 7: interview_bypass  (RED FLAG, weight 0.75)
// ---------------------------------------------------------------------------

export function checkInterviewBypass(job) {
  const text = fullText(job);
  const m = text.match(RE_INTERVIEW_BYPASS);
  if (!m) return null;
  return {
    name: 'interview_bypass',
    category: 'red_flag',
    weight: 0.75,
    confidence: 0.80,
    detail: 'Posting explicitly skips standard hiring steps (interview, resume, background check)',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 8: mlm_language  (RED FLAG, weight 0.80)
// ---------------------------------------------------------------------------

export function checkMlmLanguage(job) {
  const text = fullText(job);
  const m = text.match(RE_MLM_LANGUAGE);
  if (!m) return null;
  return {
    name: 'mlm_language',
    category: 'red_flag',
    weight: 0.80,
    confidence: 0.82,
    detail: 'Multi-level marketing / pyramid scheme language detected',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 9: reshipping  (RED FLAG, weight 0.90)
// ---------------------------------------------------------------------------

export function checkReshipping(job) {
  const text = fullText(job);
  const m = text.match(RE_RESHIPPING);
  if (!m) return null;
  return {
    name: 'reshipping',
    category: 'red_flag',
    weight: 0.90,
    confidence: 0.85,
    detail: 'Reshipping / package forwarding job — classic money-mule vector',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 10: salary_anomaly  (WARNING, weight 0.55/0.85)
// ---------------------------------------------------------------------------

export function checkSalaryAnomaly(job) {
  const lo = Number(job.salary_min) || 0;
  const hi = Number(job.salary_max) || 0;

  // Wide range check
  if (lo > 0 && hi > 0 && hi / lo > 3.0) {
    return {
      name: 'salary_anomaly',
      category: 'warning',
      weight: 0.55,
      confidence: 0.60,
      detail: `Salary range is suspiciously wide: $${lo.toLocaleString()}–$${hi.toLocaleString()}`,
      evidence: `${lo}–${hi}`,
    };
  }

  // Entry-level unrealistically high ceiling
  const levelRaw = String(job.experience_level || '').toLowerCase();
  const ceiling = hi > 0 ? hi : lo;
  const entryLevels = new Set(['entry', 'entry level', 'entry-level', 'internship', 'junior']);
  if (entryLevels.has(levelRaw) && ceiling > 500000) {
    return {
      name: 'salary_anomaly',
      category: 'warning',
      weight: 0.70,
      confidence: 0.72,
      detail: `Unrealistically high salary for ${levelRaw} role: $${ceiling.toLocaleString()}`,
      evidence: String(ceiling),
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Signal 11: vague_description  (WARNING, weight 0.50/0.65)
// ---------------------------------------------------------------------------

export function checkVagueDescription(job) {
  const words = String(job.description || '').trim().split(/\s+/).filter(Boolean);
  if (words.length >= 30) return null;
  const weight = words.length < 10 ? 0.65 : 0.50;
  return {
    name: 'vague_description',
    category: 'warning',
    weight,
    confidence: 0.70,
    detail: `Job description is extremely sparse (${words.length} words)`,
    evidence: String(job.description || '').slice(0, 120),
  };
}

// ---------------------------------------------------------------------------
// Signal 12: no_qualifications  (WARNING, weight 0.48)
// ---------------------------------------------------------------------------

export function checkNoQualifications(job) {
  const desc = String(job.description || '');
  const words = desc.trim().split(/\s+/).filter(Boolean);
  if (words.length < 20) return null;

  const hasQual =
    RE_TECH_STACK.test(desc) ||
    RE_EXPERIENCE_YRS.test(desc) ||
    RE_DEGREE.test(desc) ||
    /\b(require[sd]?|must have|qualif|proficien|skill)\b/i.test(desc);

  if (hasQual) return null;

  return {
    name: 'no_qualifications',
    category: 'warning',
    weight: 0.48,
    confidence: 0.55,
    detail: 'Non-trivial description but no skills, degree, or experience requirements',
    evidence: '',
  };
}

// ---------------------------------------------------------------------------
// Signal 13: urgency_language  (WARNING, weight 0.58)
// ---------------------------------------------------------------------------

export function checkUrgencyLanguage(job) {
  const text = fullText(job);
  const m = text.match(RE_URGENCY);
  if (!m) return null;
  return {
    name: 'urgency_language',
    category: 'warning',
    weight: 0.58,
    confidence: 0.65,
    detail: 'Artificial urgency language used to pressure applicants',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 14: wfh_unrealistic  (WARNING, weight 0.65)
// ---------------------------------------------------------------------------

export function checkWfhUnrealistic(job) {
  const text = fullText(job);
  const isWfh = job.is_remote || /\b(work from home|remote|wfh|work at home)\b/i.test(text);
  if (!isWfh) return null;
  const m = text.match(RE_NO_EXPERIENCE);
  if (!m) return null;
  return {
    name: 'wfh_unrealistic',
    category: 'warning',
    weight: 0.65,
    confidence: 0.68,
    detail: "Remote 'no-experience-needed' role — common reshipping/money-mule vector",
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 15: stale_posting  (GHOST JOB, weight 0.42/0.58)
// ---------------------------------------------------------------------------

export function checkStalePosting(job) {
  const days = daysSincePosted(job.posted_date);
  if (days === null) return null;

  if (days > 60) {
    return {
      name: 'stale_posting',
      category: 'ghost_job',
      weight: 0.58,
      confidence: 0.70,
      detail: `Posting is ${days} days old with no visible activity`,
      evidence: `${days} days since posted`,
    };
  }
  if (days > 30) {
    return {
      name: 'stale_posting',
      category: 'ghost_job',
      weight: 0.42,
      confidence: 0.55,
      detail: `Posting is ${days} days old — may be a ghost job`,
      evidence: `${days} days since posted`,
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Bonus: suspicious_links  (STRUCTURAL, weight 0.58) — high signal-to-noise
// ---------------------------------------------------------------------------

export function checkSuspiciousLinks(job) {
  const text = fullText(job);
  const m = text.match(RE_SUSPICIOUS_LINKS);
  if (!m) return null;
  return {
    name: 'suspicious_links',
    category: 'structural',
    weight: 0.58,
    confidence: 0.65,
    detail: 'Description contains shortened or third-party form links',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Registry: all signal checkers in priority order
// ---------------------------------------------------------------------------

export const ALL_SIGNALS = [
  // Red flags
  checkUpfrontPayment,
  checkPersonalInfoRequest,
  checkGuaranteedIncome,
  checkSuspiciousEmailDomain,
  checkCryptoPayment,
  checkNoCompanyPresence,
  checkInterviewBypass,
  checkMlmLanguage,
  checkReshipping,
  // Warnings
  checkSalaryAnomaly,
  checkVagueDescription,
  checkNoQualifications,
  checkUrgencyLanguage,
  checkWfhUnrealistic,
  checkGrammarQuality,
  // Ghost job
  checkStalePosting,
  // Structural
  checkSuspiciousLinks,
  // Positive
  checkCompanyDetails,
  checkStructuredInterview,
  checkBenefitsListed,
  checkSalaryRange,
];

/**
 * Run all signals against a job object, return array of fired signals.
 * @param {object} job
 * @returns {Array<object>}
 */
export function extractSignals(job) {
  const signals = [];
  for (const check of ALL_SIGNALS) {
    const signal = check(job);
    if (signal !== null) signals.push(signal);
  }
  return signals;
}


// ---------------------------------------------------------------------------
// Signal 17: grammar_quality  (WARNING, weight 0.45)
// ---------------------------------------------------------------------------

export function checkGrammarQuality(job) {
  const text = fullText(job);
  if (text.length < 100) return null;

  let score = 0;
  const capsWords = (text.match(RE_GRAMMAR_CAPS) || []).length;
  if (capsWords > 3) score += capsWords;
  const excessPunct = (text.match(RE_GRAMMAR_PUNCT) || []).length;
  if (excessPunct > 0) score += excessPunct * 3;

  if (score < 5) return null;
  return {
    name: 'grammar_quality',
    category: 'warning',
    weight: 0.45,
    confidence: 0.55,
    detail: 'Excessive capitalization and punctuation — common in scam postings',
    evidence: `${capsWords} all-caps words, ${excessPunct} repeated punctuation`,
  };
}

// ---------------------------------------------------------------------------
// Signal 18: company_details  (POSITIVE, weight 0.35)
// ---------------------------------------------------------------------------

const RE_COMPANY_DETAILS = /\b(founded in \d{4}|our team of \d+|series [a-c]|publicly traded|nasdaq|nyse|fortune \d+|inc\.? 5000|verified employer|employees?\b.*\d{2,})\b/i;

export function checkCompanyDetails(job) {
  const text = fullText(job);
  const m = text.match(RE_COMPANY_DETAILS);
  if (!m) return null;
  return {
    name: 'company_details',
    category: 'positive',
    weight: 0.35,
    confidence: 0.70,
    detail: 'Posting includes verifiable company details (founding year, size, funding)',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 19: structured_interview  (POSITIVE, weight 0.30)
// ---------------------------------------------------------------------------

export function checkStructuredInterview(job) {
  const text = fullText(job);
  const matches = [];
  const re = /\b(phone screen|technical interview|panel interview|coding challenge|take.?home (test|assignment)|onsite interview|background check|reference check|offer letter|onboarding)\b/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    matches.push(m[0]);
  }
  if (matches.length < 2) return null;
  return {
    name: 'structured_interview',
    category: 'positive',
    weight: 0.30,
    confidence: 0.75,
    detail: 'Multi-step interview process described — strong legitimacy indicator',
    evidence: matches.slice(0, 3).join(', '),
  };
}

// ---------------------------------------------------------------------------
// Signal 20: benefits_listed  (POSITIVE, weight 0.28)
// ---------------------------------------------------------------------------

export function checkBenefitsListed(job) {
  const text = fullText(job);
  const matches = [];
  const re = /\b(health insurance|dental|vision|401\(?k\)?|paid time off|pto|parental leave|stock options|equity|rsu|espp|tuition reimbursement|retirement|pension)\b/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    matches.push(m[0]);
  }
  if (matches.length < 2) return null;
  return {
    name: 'benefits_listed',
    category: 'positive',
    weight: 0.28,
    confidence: 0.72,
    detail: 'Legitimate employment benefits described',
    evidence: matches.slice(0, 4).join(', '),
  };
}

// ---------------------------------------------------------------------------
// Signal 21: salary_range_present  (POSITIVE, weight 0.20)
// ---------------------------------------------------------------------------

const RE_SALARY_RANGE = /\$[\d,]+[kK]?\s*[-–—to]+\s*\$[\d,]+[kK]?/;

export function checkSalaryRange(job) {
  const text = fullText(job);
  if (job.salary_min > 0 && job.salary_max > 0) {
    return {
      name: 'salary_range',
      category: 'positive',
      weight: 0.20,
      confidence: 0.65,
      detail: 'Specific salary range provided',
      evidence: '$' + job.salary_min.toLocaleString() + '-$' + job.salary_max.toLocaleString(),
    };
  }
  const m2 = text.match(RE_SALARY_RANGE);
  if (!m2) return null;
  return {
    name: 'salary_range',
    category: 'positive',
    weight: 0.20,
    confidence: 0.65,
    detail: 'Specific salary range provided',
    evidence: m2[0],
  };
}
