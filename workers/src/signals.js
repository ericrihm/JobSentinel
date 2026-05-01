/**
 * signals.js — 54 scam signal detectors ported from Python
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
  checkDataHarvesting,
  checkPigButcheringJob,
  checkVisaSponsorshipScam,
  checkGovernmentImpersonation,
  checkEvolvedMlm,
  checkKnownScamEntity,
  checkBrandImpersonationUrl,
  checkUrlReputationBad,
  checkSuspiciousRedirectChain,
  checkCompanyNotFound,
  // Warnings
  checkSalaryAnomaly,
  checkVagueDescription,
  checkNoQualifications,
  checkUrgencyLanguage,
  checkWfhUnrealistic,
  checkGrammarQuality,
  checkLowRecruiterConnections,
  checkNewRecruiterAccount,
  checkPhoneAnomaly,
  checkCompensationRedFlags,
  checkCompanyNameSuspicious,
  checkSuspiciousCompanyNameEnhanced,
  checkSurveyClickfarm,
  checkContactChannelSuspicious,
  checkFakeStaffingAgency,
  checkHighRiskTld,
  checkVirtualOfficeAddress,
  checkCompanyDomainMismatch,
  // Ghost job
  checkStalePosting,
  checkRepostPattern,
  checkTalentPoolLanguage,
  checkHighApplicantCount,
  checkRoleTitleGeneric,
  // Structural
  checkSuspiciousLinks,
  checkShortenedUrl,
  checkAiGeneratedContent,
  // Positive
  checkCompanyDetails,
  checkStructuredInterview,
  checkBenefitsListed,
  checkSalaryRange,
  checkVerifiedCompanyWebsite,
  checkProfessionalApplicationProcess,
  checkEstablishedCompany,
  checkDetailedRequirements,
  checkCompanyVerified,
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

// ---------------------------------------------------------------------------
// Signal 22: low_recruiter_connections  (WARNING, weight 0.45/0.62)
// ---------------------------------------------------------------------------

export function checkLowRecruiterConnections(job) {
  const n = Number(job.recruiter_connections) || 0;
  if (n <= 0) return null;
  if (n >= 50) return null;
  const weight = n < 20 ? 0.62 : 0.45;
  return {
    name: 'low_recruiter_connections',
    category: 'warning',
    weight,
    confidence: 0.60,
    detail: `Recruiter has only ${n} LinkedIn connections — likely a fake profile`,
    evidence: String(n),
  };
}

// ---------------------------------------------------------------------------
// Signal 23: new_recruiter_account  (WARNING, weight 0.55)
// Convention: recruiter_connections === -1 encodes "account age < 7 days"
// ---------------------------------------------------------------------------

export function checkNewRecruiterAccount(job) {
  if (Number(job.recruiter_connections) !== -1) return null;
  return {
    name: 'new_recruiter_account',
    category: 'warning',
    weight: 0.55,
    confidence: 0.65,
    detail: 'Recruiter account was created less than 7 days ago — strong scam indicator',
    evidence: 'recruiter_account_age < 7 days',
  };
}

// ---------------------------------------------------------------------------
// Signal 24: repost_pattern  (GHOST JOB, weight 0.50)
// ---------------------------------------------------------------------------

export function checkRepostPattern(job) {
  if (!job.is_repost) return null;
  return {
    name: 'repost_pattern',
    category: 'ghost_job',
    weight: 0.50,
    confidence: 0.60,
    detail: 'Role has been reposted — no hires made from previous postings',
    evidence: 'is_repost=true',
  };
}

// ---------------------------------------------------------------------------
// Signal 25: talent_pool_language  (GHOST JOB, weight 0.55)
// ---------------------------------------------------------------------------

const RE_TALENT_POOL = /\b(talent pipeline|talent community|talent pool|future openings?|expressions? of interest|join our (talent|candidate) network|pipeline (of candidates|for future)|upcoming opportunities)\b/i;

export function checkTalentPoolLanguage(job) {
  const text = fullText(job);
  const m = text.match(RE_TALENT_POOL);
  if (!m) return null;
  return {
    name: 'talent_pool_language',
    category: 'ghost_job',
    weight: 0.55,
    confidence: 0.65,
    detail: 'Posting uses talent-pool / future-openings language — likely not a real open role',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 26: high_applicant_count  (GHOST JOB, weight 0.48)
// Fires only when >500 applicants AND posting is >30 days old
// ---------------------------------------------------------------------------

export function checkHighApplicantCount(job) {
  const count = Number(job.applicant_count) || 0;
  if (count <= 500) return null;
  const days = daysSincePosted(job.posted_date);
  if (days === null || days <= 30) return null;
  return {
    name: 'high_applicant_count',
    category: 'ghost_job',
    weight: 0.48,
    confidence: 0.55,
    detail: `Over ${count} applicants on a ${days}-day-old posting — likely a ghost job or perpetually open pipeline role`,
    evidence: `${count} applicants, ${days} days old`,
  };
}

// ---------------------------------------------------------------------------
// Signal 27: role_title_generic  (GHOST JOB, weight 0.42)
// ---------------------------------------------------------------------------

const RE_GENERIC_TITLES = /^\s*(various positions?|multiple openings?|multiple positions?|team member|general (applicant|application)|open application|general (hire|hiring)|various roles?|multiple roles?)\s*$/i;

export function checkRoleTitleGeneric(job) {
  const title = String(job.title || '');
  if (!RE_GENERIC_TITLES.test(title)) return null;
  return {
    name: 'role_title_generic',
    category: 'ghost_job',
    weight: 0.42,
    confidence: 0.55,
    detail: 'Role title is extremely generic — indicative of a catch-all pipeline posting',
    evidence: title,
  };
}

// ---------------------------------------------------------------------------
// Signal 28: ai_generated_content  (STRUCTURAL, weight 0.45)
// ---------------------------------------------------------------------------

const RE_GENERIC_CULTURE = /\b(dynamic team|fast.paced environment|passionate individuals?|results.driven|go-getter|self.starter|rockstar|ninja|guru|collaborative culture|inclusive workplace|family.like (team|culture)|work hard play hard|make a (difference|impact))\b/i;

const RE_SPECIFIC_DETAILS = /\b(team [A-Z][a-z]+|project [A-Z][a-z]+|[A-Z][a-z]+ (squad|pod|team|org)|using [A-Z][a-zA-Z]+|built (on|with) [A-Z]|our (codebase|stack|platform|product|API|service))\b/;

export function checkAiGeneratedContent(job) {
  const desc = String(job.description || '');
  const words = desc.trim().split(/\s+/).filter(Boolean);
  if (words.length < 30) return null;

  const genericMatches = desc.match(new RegExp(RE_GENERIC_CULTURE.source, 'gi')) || [];
  const genericHits = genericMatches.length;
  if (genericHits === 0) return null;

  const specificMatches = desc.match(new RegExp(RE_SPECIFIC_DETAILS.source, 'g')) || [];
  const techMatches = desc.match(new RegExp(RE_TECH_STACK.source, 'gi')) || [];
  const expMatches = desc.match(new RegExp(RE_EXPERIENCE_YRS.source, 'gi')) || [];
  const specificHits = specificMatches.length + techMatches.length + expMatches.length;

  const ratio = genericHits / Math.max(specificHits, 1);
  if (ratio <= 3.0) return null;

  return {
    name: 'ai_generated_content',
    category: 'structural',
    weight: 0.45,
    confidence: 0.55,
    detail: `Description has ${genericHits} generic culture phrase(s) but only ${specificHits} specific detail(s) — ratio ${ratio.toFixed(1)}:1 suggests AI filler`,
    evidence: `${genericHits} generic vs ${specificHits} specific`,
  };
}

// ---------------------------------------------------------------------------
// Signal 29: phone_anomaly  (WARNING, weight 0.50)
// ---------------------------------------------------------------------------

const RE_PREMIUM_PHONE = /\b(900|976)[-.\s]?\d{3}[-.\s]?\d{4}\b/;
const RE_PHONE_GENERAL = /\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/;

export function checkPhoneAnomaly(job) {
  const text = fullText(job);
  const premium = text.match(RE_PREMIUM_PHONE);
  if (premium) {
    return {
      name: 'phone_anomaly',
      category: 'warning',
      weight: 0.50,
      confidence: 0.75,
      detail: 'Premium-rate phone number detected (900/976 prefix)',
      evidence: premium[0],
    };
  }
  const general = text.match(RE_PHONE_GENERAL);
  if (general) {
    return {
      name: 'phone_anomaly',
      category: 'warning',
      weight: 0.50,
      confidence: 0.50,
      detail: 'Phone number embedded in job description — unusual for LinkedIn postings',
      evidence: general[0],
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal 30: data_harvesting  (RED FLAG, weight 0.85)
// ---------------------------------------------------------------------------

const RE_DATA_HARVESTING = /\b(complete (our|the) (application|form|survey) (at|on|via)|fill out (our|the) (application|form) (at|on|via)|apply (at|via|through|on) (our )?(external|separate|outside)|submit (your )?(application|info|details) (at|to|via) (http|www|forms?\.|typeform|google))\b/i;

export function checkDataHarvesting(job) {
  const text = fullText(job);
  const m = text.match(RE_DATA_HARVESTING);
  if (m) {
    return {
      name: 'data_harvesting',
      category: 'red_flag',
      weight: 0.85,
      confidence: 0.78,
      detail: 'Posting redirects applicants to an external form/site to collect personal data',
      evidence: m[0],
    };
  }
  // Bare Google Forms / Typeform links without redirect phrasing
  const linkM = text.match(/(forms\.gle|docs\.google\.com\/forms|typeform\.com|airtable\.com\/shr)/i);
  if (linkM) {
    return {
      name: 'data_harvesting',
      category: 'red_flag',
      weight: 0.85,
      confidence: 0.70,
      detail: 'External data-collection form linked in job description',
      evidence: linkM[0],
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal 31: compensation_red_flags  (WARNING, weight 0.55)
// ---------------------------------------------------------------------------

const RE_COMPENSATION_RED_FLAGS = /\b(commission.only|100% commission|1099 only|independent contractor (only|position|role)|performance.based pay only|training (period|pay) (is )?unpaid|no base (salary|pay)|draw against commission)\b/i;

export function checkCompensationRedFlags(job) {
  const text = fullText(job);
  const m = text.match(RE_COMPENSATION_RED_FLAGS);
  if (!m) return null;
  return {
    name: 'compensation_red_flags',
    category: 'warning',
    weight: 0.55,
    confidence: 0.65,
    detail: 'Compensation structure raises red flags (commission-only, unpaid training, 1099-only)',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 32: company_name_suspicious  (WARNING, weight 0.50)
// ---------------------------------------------------------------------------

const RE_SUSPICIOUS_COMPANY_SUFFIX = /\b(\w+\s+(Solutions|Global|International|Enterprises?|Worldwide|Unlimited|Ventures?|Associates?|Consulting|Services?|Group|Partners?))\s*$/i;

export function checkCompanyNameSuspicious(job) {
  const name = String(job.company || '').trim();
  if (!name) return null;

  // All-caps company name
  if (name === name.toUpperCase() && name.length > 3 && /[A-Z]{4,}/.test(name)) {
    return {
      name: 'company_name_suspicious',
      category: 'warning',
      weight: 0.50,
      confidence: 0.50,
      detail: 'Company name is entirely uppercase — unusual for legitimate organisations',
      evidence: name,
    };
  }

  // Single-word + generic suffix (e.g. "Horizon Solutions", "Apex Global")
  const m = name.match(RE_SUSPICIOUS_COMPANY_SUFFIX);
  if (m) {
    const parts = name.split(/\s+/);
    if (parts.length <= 3) {
      return {
        name: 'company_name_suspicious',
        category: 'warning',
        weight: 0.50,
        confidence: 0.45,
        detail: 'Company name matches common scam pattern (generic word + Solutions/Global/etc.)',
        evidence: name,
      };
    }
  }

  return null;
}

// ---------------------------------------------------------------------------
// Signal 33: known_scam_entity  (RED FLAG, weight 0.88)
// No DB in Workers — use regex patterns for well-known scam entity names
// ---------------------------------------------------------------------------

const RE_KNOWN_SCAM_ENTITIES = /\b(amazom|amazzon|faceboook|googlehire|linkedln|linkediin|workathomejobs|easycashjobs|quickhire247|myjobsearch247|nationaljobfinder|workfromhomejobs247|nationalhiringcenter|careersatgoogle|amazonhirenow)\b/i;

export function checkKnownScamEntity(job) {
  const text = `${fullText(job)} ${String(job.company || '')}`;
  const m = text.match(RE_KNOWN_SCAM_ENTITIES);
  if (!m) return null;
  return {
    name: 'known_scam_entity',
    category: 'red_flag',
    weight: 0.88,
    confidence: 0.85,
    detail: `Text matches a known scam entity pattern: '${m[0]}'`,
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 34: pig_butchering_job  (RED FLAG, weight 0.88)
// ---------------------------------------------------------------------------

const RE_PIG_BUTCHERING = /\b(cryptocurrency trader|digital asset (manager|analyst|trader)|DeFi (analyst|trader|specialist)|liquidity provider|investment opportunity.{0,60}(guaranteed|high) return|forex (trader|analyst|manager)|crypto exchange (analyst|specialist)|blockchain investment (analyst|manager))\b/i;

export function checkPigButcheringJob(job) {
  const text = fullText(job);
  const m = text.match(RE_PIG_BUTCHERING);
  if (!m) return null;
  return {
    name: 'pig_butchering_job',
    category: 'red_flag',
    weight: 0.88,
    confidence: 0.80,
    detail: 'Posting matches pig butchering scam pattern (fake crypto/investment role)',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 35: survey_clickfarm  (WARNING, weight 0.75)
// ---------------------------------------------------------------------------

const RE_SURVEY_CLICKFARM = /\b(online survey.{0,40}(earn|paid|money)|product reviewer.{0,40}(earn|paid|money|from home)|social media evaluator|paid per click|earn per review|get paid.{0,30}(surveys?|reviews?|clicks?)|review products.{0,30}(earn|paid|keep))\b/i;

export function checkSurveyClickfarm(job) {
  const text = fullText(job);
  const m = text.match(RE_SURVEY_CLICKFARM);
  if (!m) return null;
  return {
    name: 'survey_clickfarm',
    category: 'warning',
    weight: 0.75,
    confidence: 0.72,
    detail: 'Posting matches survey/click-farm scam pattern',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 36: visa_sponsorship_scam  (RED FLAG, weight 0.82)
// ---------------------------------------------------------------------------

const RE_VISA_KEYWORDS = /\b(guaranteed visa|guaranteed H-?1B|H-?1B sponsor.{0,40}(fee|payment|deposit|cost)|immigration fee|visa processing fee|work permit fee|visa (application|processing).{0,40}(pay|fee|cost|deposit))\b/i;

export function checkVisaSponsorshipScam(job) {
  const text = fullText(job);
  const m = text.match(RE_VISA_KEYWORDS);
  if (!m) return null;
  return {
    name: 'visa_sponsorship_scam',
    category: 'red_flag',
    weight: 0.82,
    confidence: 0.78,
    detail: 'Posting promises visa sponsorship with upfront fees — legitimate employers never charge',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 37: government_impersonation  (RED FLAG, weight 0.72/0.88)
// Requires BOTH a govt agency mention AND a personal-info request
// ---------------------------------------------------------------------------

const RE_GOVT_IMPERSONATION = /\b(Department of Defense|DOD|FBI|CIA|DHS|NATO|Department of Homeland Security|NSA|Secret Service|Federal Bureau of Investigation|Central Intelligence Agency)\b/i;

const RE_PERSONAL_INFO_REQUEST_GOV = /\b(provide (your )?(SSN|social security|passport|bank account|date of birth)|send (your )?(SSN|passport|bank|personal (info|information|details))|submit (your )?(personal|banking|financial) (info|information|details))\b/i;

export function checkGovernmentImpersonation(job) {
  const text = fullText(job);
  const govtM = text.match(RE_GOVT_IMPERSONATION);
  if (!govtM) return null;
  const infoM = text.match(RE_PERSONAL_INFO_REQUEST_GOV);
  if (!infoM) return null;
  const hasCompany = Boolean(String(job.company_linkedin_url || '').trim());
  const weight = hasCompany ? 0.72 : 0.88;
  return {
    name: 'government_impersonation',
    category: 'red_flag',
    weight,
    confidence: 0.82,
    detail: `Posting claims government/military affiliation (${govtM[0]}) while requesting personal info`,
    evidence: `${govtM[0]} + ${infoM[0]}`,
  };
}

// ---------------------------------------------------------------------------
// Signal 38: fake_staffing_agency  (WARNING, weight 0.68)
// Fires when a known brand is mentioned by a different (unlinked) company
// ---------------------------------------------------------------------------

const RE_KNOWN_BRANDS = /\b(Google|Amazon|Microsoft|Apple|Meta|Facebook|Netflix|Tesla|Goldman Sachs|JPMorgan|McKinsey|Deloitte|PwC|EY|KPMG|Coca.Cola|Nike|Disney|Boeing|Lockheed Martin|Raytheon)\b/i;

export function checkFakeStaffingAgency(job) {
  const text = fullText(job);
  const brandM = text.match(RE_KNOWN_BRANDS);
  if (!brandM) return null;
  const company = String(job.company || '').trim().toLowerCase();
  const brandLower = brandM[0].toLowerCase();
  if (brandLower.split(' ').some(w => company.includes(w))) return null;
  if (String(job.company_linkedin_url || '').trim()) return null;
  return {
    name: 'fake_staffing_agency',
    category: 'warning',
    weight: 0.68,
    confidence: 0.60,
    detail: `Posting references ${brandM[0]} but is posted by unknown '${job.company}' with no LinkedIn page`,
    evidence: `brand=${brandM[0]}, poster=${job.company}`,
  };
}

// ---------------------------------------------------------------------------
// Signal 39: evolved_mlm  (RED FLAG, weight 0.78)
// ---------------------------------------------------------------------------

const RE_EVOLVED_MLM = /\b(brand ambassador.{0,60}(earn|income|commission|own boss)|independent business owner|wellness consultant.{0,60}(earn|income|opportunity)|starter inventory|authorized distributor.{0,60}(earn|income|opportunity)|health and wellness.{0,40}(opportunity|income|earn)|purchase (your |a )?starter (kit|package|inventory))\b/i;

export function checkEvolvedMlm(job) {
  const text = fullText(job);
  const m = text.match(RE_EVOLVED_MLM);
  if (!m) return null;
  return {
    name: 'evolved_mlm',
    category: 'red_flag',
    weight: 0.78,
    confidence: 0.74,
    detail: 'Posting uses evolved MLM language (brand ambassador, wellness consultant, starter inventory)',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 40: contact_channel_suspicious  (WARNING, weight 0.55/0.65)
// ---------------------------------------------------------------------------

const RE_CONTACT_SUSPICIOUS = /\b(apply (via|through|on|at) (telegram|whatsapp)|contact (us |me )?(on|via|at|through) (telegram|whatsapp)|(telegram|whatsapp) (only|to apply|for (details|info|more))|send (message|msg|text|DM) (on|via|to) (telegram|whatsapp))\b/i;

const RE_CELL_ONLY_CONTACT = /\b(call|text|contact|reach) (me|us) (at|on) \(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/i;

export function checkContactChannelSuspicious(job) {
  const text = fullText(job);
  const m = text.match(RE_CONTACT_SUSPICIOUS);
  if (m) {
    return {
      name: 'contact_channel_suspicious',
      category: 'warning',
      weight: 0.65,
      confidence: 0.68,
      detail: 'Posting directs applicants to Telegram/WhatsApp for application',
      evidence: m[0],
    };
  }
  const cellM = text.match(RE_CELL_ONLY_CONTACT);
  if (cellM && !String(job.company_linkedin_url || '').trim()) {
    return {
      name: 'contact_channel_suspicious',
      category: 'warning',
      weight: 0.55,
      confidence: 0.55,
      detail: 'Sole contact method is a personal phone number with no company LinkedIn',
      evidence: cellM[0],
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal 41: verified_company_website  (POSITIVE, weight 0.32)
// ---------------------------------------------------------------------------

const RE_COMPANY_WEBSITE_URL = /(?:https?:\/\/)?([a-z0-9][-a-z0-9]*\.)+[a-z]{2,}/gi;

export function checkVerifiedCompanyWebsite(job) {
  const text = fullText(job);
  const company = String(job.company || '').trim().toLowerCase();
  if (!company || company.length < 3) return null;

  const urls = text.match(RE_COMPANY_WEBSITE_URL) || [];
  if (urls.length === 0) return null;

  const companyWord = company.replace(/[^a-z0-9]/g, '');
  for (const url of urls) {
    const urlNorm = url.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (urlNorm.includes(companyWord)) {
      return {
        name: 'verified_company_website',
        category: 'positive',
        weight: 0.32,
        confidence: 0.60,
        detail: `Posting references a website matching company name '${job.company}'`,
        evidence: url,
      };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal 42: professional_application_process  (POSITIVE, weight 0.30)
// ---------------------------------------------------------------------------

const RE_ATS_SYSTEMS = /\b(greenhouse|lever|workday|taleo|icims|smartrecruiters|brassring|jobvite|bamboohr|ashby|jazz ?hr|applicant tracking|apply (on|at|through|via) (our |the )?(careers?|jobs?) (page|portal|site|website)|careers?\.(company|com|org|io|co)|\/careers?\/|\/jobs?\/)\b/i;

export function checkProfessionalApplicationProcess(job) {
  const text = fullText(job);
  const m = text.match(RE_ATS_SYSTEMS);
  if (!m) return null;
  return {
    name: 'professional_application_process',
    category: 'positive',
    weight: 0.30,
    confidence: 0.62,
    detail: 'Posting references a professional application process (ATS or company careers page)',
    evidence: m[0],
  };
}

// ---------------------------------------------------------------------------
// Signal 43: established_company  (POSITIVE, weight 0.35)
// Fires when company_size indicates >= 100 employees
// ---------------------------------------------------------------------------

export function checkEstablishedCompany(job) {
  const sizeStr = String(job.company_size || '').replace(/,/g, '').toLowerCase();
  const m = sizeStr.match(/(\d+)/);
  if (!m) return null;
  if (parseInt(m[1], 10) < 100) return null;
  return {
    name: 'established_company',
    category: 'positive',
    weight: 0.35,
    confidence: 0.65,
    detail: `Company has ${job.company_size} employees — established organisation`,
    evidence: String(job.company_size),
  };
}

// ---------------------------------------------------------------------------
// Signal 44: detailed_requirements  (POSITIVE, weight 0.38)
// Fires when description has >= 2 of: 3+ techs, experience years, degree, benefits
// ---------------------------------------------------------------------------

const RE_BENEFITS = /\b(health insurance|dental|vision|401k|equity|stock options?|pto|paid time off|parental leave|bonus|pension|retirement)\b/i;

export function checkDetailedRequirements(job) {
  const desc = String(job.description || '');
  const words = desc.trim().split(/\s+/).filter(Boolean);
  if (words.length < 20) return null;

  const techMatches = desc.match(new RegExp(RE_TECH_STACK.source, 'gi')) || [];
  const uniqueTechs = new Set(techMatches.map(t => t.toLowerCase())).size;

  const qualifiers = [
    uniqueTechs >= 3,
    RE_EXPERIENCE_YRS.test(desc),
    RE_DEGREE.test(desc),
    RE_BENEFITS.test(desc),
  ];

  if (qualifiers.filter(Boolean).length < 2) return null;

  const evidenceParts = [];
  if (qualifiers[0]) evidenceParts.push(`${uniqueTechs} technologies listed`);
  if (qualifiers[1]) evidenceParts.push('experience years specified');
  if (qualifiers[2]) evidenceParts.push('degree requirement stated');
  if (qualifiers[3]) evidenceParts.push('benefits mentioned');

  return {
    name: 'detailed_requirements',
    category: 'positive',
    weight: 0.38,
    confidence: 0.70,
    detail: 'Detailed, specific requirements indicate a genuine role',
    evidence: evidenceParts.join(', '),
  };
}

// ---------------------------------------------------------------------------
// Link Analyzer Signals (ported from sentinel/link_analyzer.py)
// ---------------------------------------------------------------------------

// URL shortener domains (offline — no network calls)
const SHORTENER_DOMAINS = new Set([
  'bit.ly', 'tinyurl.com', 't.co', 'ow.ly', 'goo.gl',
  'buff.ly', 'short.io', 'rebrand.ly', 'cutt.ly', 'tiny.cc',
  'is.gd', 'v.gd', 'shrtco.de', 'shorturl.at', 'clck.ru', 'lnkd.in',
]);

// High-risk TLDs (Spamhaus / SURBL research)
const HIGH_RISK_TLDS = new Set([
  '.xyz', '.top', '.club', '.work', '.click', '.link',
  '.info', '.biz', '.online', '.site', '.live',
]);

// Known brands for impersonation detection (Levenshtein)
const BRAND_NAMES = [
  'google', 'linkedin', 'amazon', 'microsoft', 'apple',
  'facebook', 'meta', 'netflix', 'twitter', 'instagram',
  'paypal', 'ebay', 'walmart', 'indeed', 'glassdoor',
  'monster', 'ziprecruiter', 'workday', 'salesforce',
];

// Local blocklist patterns for url_reputation_bad (static, no network)
const LOCAL_BLOCKLIST = [
  /jobs?[-.]?(apply|offer|hire|work)\.(?:xyz|top|click|live|site|online)/i,
  /work[-.]?from[-.]?home\d*\./i,
  /earn[-.]?\$\d+/i,
  /easy[-.]?money\d*\./i,
  /get[-.]?hired[-.]?(now|today|fast)/i,
  /career[-.]?opportunity\d+\./i,
  /(apply|recruit)[-_]?[a-z]{6,12}\.(?:xyz|top|click|biz|info)/i,
];

// URL extraction regex (matches full URLs, shortener paths, and bare domains)
const RE_URL_EXTRACT = /https?:\/\/[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=%]+|(?:bit\.ly|tinyurl\.com|t\.co|ow\.ly|goo\.gl|buff\.ly|short\.io|rebrand\.ly|cutt\.ly|tiny\.cc|is\.gd|v\.gd|shrtco\.de|lnkd\.in)\/[a-zA-Z0-9_\-/?=&%+.#]+|(?<![/@])(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+(?:com|net|org|io|co|ai|dev|app|jobs|work|xyz|top|club|click|link|info|biz|online|site|live|me|us|uk|ca|au|de)(?:\/[a-zA-Z0-9_\-/?=&%+.#]*)?/gi;

/**
 * Levenshtein distance — same algorithm used in Python signals.py
 * @param {string} a
 * @param {string} b
 * @returns {number}
 */
function levenshtein(a, b) {
  if (a === b) return 0;
  if (!a) return b.length;
  if (!b) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const curr = [i];
    for (let j = 1; j <= b.length; j++) {
      curr.push(Math.min(
        prev[j] + 1,
        curr[j - 1] + 1,
        prev[j - 1] + (a[i - 1] !== b[j - 1] ? 1 : 0),
      ));
    }
    prev = curr;
  }
  return prev[prev.length - 1];
}

/**
 * Parse the hostname from a URL, lowercase, strip www.
 * @param {string} url
 * @returns {string}
 */
function parseDomain(url) {
  if (!url) return '';
  let u = url;
  if (!u.startsWith('http://') && !u.startsWith('https://')) u = 'http://' + u;
  try {
    const host = new URL(u).hostname || '';
    return host.toLowerCase().replace(/^www\./, '');
  } catch {
    return '';
  }
}

/**
 * Return the TLD (with leading dot) from a domain, e.g. '.xyz'
 * @param {string} domain
 * @returns {string}
 */
function getTld(domain) {
  const idx = domain.lastIndexOf('.');
  return idx >= 0 ? domain.slice(idx) : '';
}

/**
 * Extract deduplicated URLs from text.
 * @param {string} text
 * @returns {string[]}
 */
function extractUrls(text) {
  const raw = text.match(RE_URL_EXTRACT) || [];
  const seen = new Set();
  const results = [];
  for (let url of raw) {
    url = url.replace(/[.,;:!?"')]+$/, '');
    if (url && !seen.has(url)) {
      seen.add(url);
      results.push(url);
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Signal LA-1: shortened_url  (STRUCTURAL, weight 0.45)
// ---------------------------------------------------------------------------

export function checkShortenedUrl(job) {
  const text = fullText(job);
  const urls = extractUrls(text);
  for (const url of urls) {
    const domain = parseDomain(url);
    if (SHORTENER_DOMAINS.has(domain)) {
      return {
        name: 'shortened_url',
        category: 'structural',
        weight: 0.45,
        confidence: 0.80,
        detail: 'Job posting contains a URL shortener link — destination is hidden',
        evidence: url,
      };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal LA-2: high_risk_tld  (WARNING, weight 0.55)
// ---------------------------------------------------------------------------

export function checkHighRiskTld(job) {
  const text = fullText(job);
  const urls = extractUrls(text);
  for (const url of urls) {
    const domain = parseDomain(url);
    if (!domain) continue;
    const tld = getTld(domain);
    if (HIGH_RISK_TLDS.has(tld)) {
      return {
        name: 'high_risk_tld',
        category: 'warning',
        weight: 0.55,
        confidence: 0.70,
        detail: `URL uses a high-risk TLD (${tld})`,
        evidence: url,
      };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal LA-3: brand_impersonation_url  (RED FLAG, weight 0.85)
// Offline Levenshtein check — no network needed
// ---------------------------------------------------------------------------

export function checkBrandImpersonationUrl(job) {
  const text = fullText(job);
  const urls = extractUrls(text);
  for (const url of urls) {
    const domain = parseDomain(url);
    if (!domain) continue;

    // Registered label = second-to-last part (strip TLD)
    const parts = domain.split('.');
    const registeredLabel = parts.length >= 2 ? parts[parts.length - 2] : parts[0];

    for (const brand of BRAND_NAMES) {
      if (registeredLabel === brand) break; // exact match = real brand
      const dist = levenshtein(registeredLabel, brand);
      if (dist > 0 && dist <= 2) {
        return {
          name: 'brand_impersonation_url',
          category: 'red_flag',
          weight: 0.85,
          confidence: 0.78,
          detail: `URL domain appears to impersonate '${brand}' (domain: ${domain})`,
          evidence: url,
        };
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal LA-4: url_reputation_bad  (RED FLAG, weight 0.90)
// Offline local blocklist only (no Safe Browsing / PhishTank API calls)
// ---------------------------------------------------------------------------

export function checkUrlReputationBad(job) {
  const text = fullText(job);
  const urls = extractUrls(text);
  for (const url of urls) {
    for (const pattern of LOCAL_BLOCKLIST) {
      if (pattern.test(url)) {
        return {
          name: 'url_reputation_bad',
          category: 'red_flag',
          weight: 0.90,
          confidence: 0.75,
          detail: 'URL matches a local scam-site pattern',
          evidence: url,
        };
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal LA-5: suspicious_redirect_chain  (RED FLAG, weight 0.72)
// Offline heuristic: detect multi-hop redirect patterns in text
// (full redirect following is only possible with network; here we detect
// explicit redirect/tracking language combined with shortener links)
// ---------------------------------------------------------------------------

const RE_REDIRECT_LANGUAGE = /\b(click here to (apply|continue|proceed)|follow (this |the )?link to (apply|continue|access)|you will be (redirected|forwarded)|apply (via|through|at) the (link|url) (below|above|here))\b/i;

export function checkSuspiciousRedirectChain(job) {
  const text = fullText(job);
  const urls = extractUrls(text);

  // Fire if there's a shortener AND explicit redirect/apply language
  const hasShortener = urls.some(url => SHORTENER_DOMAINS.has(parseDomain(url)));
  const hasRedirectLanguage = RE_REDIRECT_LANGUAGE.test(text);

  if (hasShortener && hasRedirectLanguage) {
    const shortenerUrl = urls.find(url => SHORTENER_DOMAINS.has(parseDomain(url)));
    return {
      name: 'suspicious_redirect_chain',
      category: 'red_flag',
      weight: 0.72,
      confidence: 0.60,
      detail: 'Posting uses a URL shortener combined with redirect/apply language — hides true destination',
      evidence: shortenerUrl || 'shortener + redirect language detected',
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Company Verifier Signals (ported from sentinel/company_verifier.py)
// ---------------------------------------------------------------------------

// Known legitimate companies (subset of Python's _KNOWN_COMPANIES)
const KNOWN_COMPANIES = new Set([
  // Big Tech
  'google', 'alphabet', 'meta', 'facebook', 'apple', 'amazon', 'microsoft',
  'netflix', 'nvidia', 'intel', 'ibm', 'oracle', 'salesforce', 'adobe',
  'sap', 'vmware', 'cisco', 'qualcomm', 'broadcom', 'amd',
  // Cloud & Infra
  'aws', 'cloudflare', 'fastly', 'twilio', 'okta', 'datadog', 'splunk',
  'pagerduty', 'elastic', 'mongodb', 'hashicorp', 'confluent',
  'databricks', 'snowflake', 'palantir', 'digitalocean',
  // Fintech & Finance
  'stripe', 'square', 'block', 'paypal', 'coinbase', 'robinhood', 'plaid',
  'jpmorgan', 'jp morgan', 'goldman sachs', 'morgan stanley', 'wells fargo',
  'bank of america', 'citibank', 'citi', 'blackrock', 'fidelity', 'vanguard',
  'american express', 'visa', 'mastercard', 'capital one',
  // E-commerce & Delivery
  'shopify', 'ebay', 'etsy', 'wayfair', 'doordash', 'instacart', 'uber eats',
  'uber', 'lyft', 'airbnb',
  // Enterprise SaaS
  'workday', 'servicenow', 'zendesk', 'hubspot', 'atlassian', 'slack',
  'zoom', 'dropbox', 'box', 'docusign', 'asana', 'figma',
  // Security
  'crowdstrike', 'palo alto networks', 'fortinet', 'zscaler', 'sentinelone',
  'cyberark', 'rapid7', 'qualys', 'tenable', 'proofpoint',
  // Dev Tools
  'github', 'gitlab', 'jfrog', 'jetbrains', 'postman', 'new relic',
  // Healthcare
  'unitedhealth', 'anthem', 'aetna', 'cigna', 'humana', 'cvs health',
  'johnson & johnson', 'pfizer', 'moderna', 'gilead', 'biogen',
  'abbvie', 'merck', 'eli lilly', 'amgen', 'roche', 'novartis',
  // Media
  'disney', 'comcast', 'spotify', 'tiktok', 'bytedance', 'snapchat',
  'twitter', 'reddit', 'pinterest', 'linkedin',
  // Consulting
  'mckinsey', 'bain', 'bcg', 'deloitte', 'pwc', 'kpmg', 'ey',
  'ernst & young', 'accenture', 'booz allen', 'infosys', 'cognizant',
  // Retail
  'walmart', 'target', 'costco', 'kroger', 'home depot',
  'best buy', 'walgreens', 'cvs',
  // Telecom
  'at&t', 'verizon', 't-mobile', 'charter',
  // Aerospace & Defense
  'lockheed martin', 'raytheon', 'boeing', 'northrop grumman', 'spacex',
  // Automotive
  'tesla', 'ford', 'general motors', 'gm', 'toyota', 'honda',
  // AI / ML
  'openai', 'anthropic', 'cohere', 'hugging face', 'scale ai', 'deepmind',
  // Logistics
  'fedex', 'ups', 'dhl', 'usps',
  // Travel
  'marriott', 'hilton', 'hyatt', 'united airlines', 'delta', 'american airlines',
  // HR / Recruiting
  'adp', 'paychex', 'bamboohr', 'greenhouse', 'lever',
]);

// Brand names for company misspelling detection
const CV_BRAND_NAMES = [
  'google', 'amazon', 'microsoft', 'apple', 'facebook', 'meta',
  'netflix', 'tesla', 'paypal', 'walmart', 'target', 'costco',
  'deloitte', 'accenture', 'ibm', 'oracle', 'salesforce', 'nvidia',
];

// Virtual office / PO box patterns
const RE_VIRTUAL_OFFICE = /\b(regus|wework|we work|iws|intelligent office|davinci|opus virtual|servcorp|alliance virtual|virtual office|shared office|coworking|co-working|executive suites|postal connections|ups store|mailbox|pmb \d|suite \d{3,4}(?![a-z]))\b/i;
const RE_PO_BOX = /\b(p\.?\s*o\.?\s*box|post\s+office\s+box|po\s+box)\b/i;

// Generic buzzword suffix (company name heuristic)
const RE_GENERIC_SUFFIX_CV = /^[A-Za-z]{3,20}\s+(Solutions|Global|International|Enterprises?|Worldwide|Unlimited|Ventures?|Associates?|Consulting|Services?|Group|Partners?|Network|Systems|Resources|Staffing|Placement|Opportunities|Connections)$/i;

// Random character patterns (scam company names)
const RE_RANDOM_CHARS = /[A-Z]{4,}\d|[A-Z]\d[A-Z]\d|[0-9]{3,}[A-Za-z]/;

// Excessive entity suffixing (e.g. "MyBiz LLC LLC")
const RE_EXCESSIVE_ENTITY = /\b(LLC|Inc|Corp|Ltd|Co\.?)\s*,?\s*(LLC|Inc|Corp|Ltd|Co\.?)\b/i;

/**
 * Return the closest brand if company name looks like a misspelling (Levenshtein).
 * @param {string} name
 * @returns {string|null}
 */
function isMisspelledBrand(name) {
  // Strip common entity suffixes (LLC, Inc, Corp, Ltd, Co) before comparing
  const stripped = name.replace(/\s+(llc|inc\.?|corp\.?|ltd\.?|co\.?|plc|llp|lp)\s*$/i, '').trim();
  const normalized = stripped.toLowerCase().replace(/[^a-z0-9]/g, '');
  for (const brand of CV_BRAND_NAMES) {
    const threshold = Math.max(1, Math.floor(brand.length / 4));
    const dist = levenshtein(normalized, brand);
    if (dist > 0 && dist <= threshold) return brand;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal CV-1: company_not_found  (RED FLAG, weight 0.75)
// Offline heuristic: company name fails all legitimacy checks
// ---------------------------------------------------------------------------

export function checkCompanyNotFound(job) {
  const name = String(job.company || '').trim();
  if (!name) return null;

  // If it's in our known-good list, definitely not "not found"
  const nameLower = name.toLowerCase();
  if (KNOWN_COMPANIES.has(nameLower)) return null;
  // Prefix/suffix match for variants like "Google LLC"
  for (const known of KNOWN_COMPANIES) {
    if (nameLower.startsWith(known + ' ') || nameLower.endsWith(' ' + known)) return null;
  }

  // Fire if the name has multiple strong suspicious indicators
  let suspicionScore = 0;
  if (RE_RANDOM_CHARS.test(name)) suspicionScore += 2;
  if (RE_EXCESSIVE_ENTITY.test(name)) suspicionScore += 1;
  if (name === name.toUpperCase() && name.length > 4 && /[A-Z]{5,}/.test(name)) suspicionScore += 1;
  if (!job.company_linkedin_url || !String(job.company_linkedin_url).trim()) suspicionScore += 1;

  if (suspicionScore >= 3) {
    return {
      name: 'company_not_found',
      category: 'red_flag',
      weight: 0.75,
      confidence: 0.65,
      detail: `Company '${name}' shows multiple indicators of being non-existent or fabricated`,
      evidence: name,
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal CV-2: company_domain_mismatch  (WARNING, weight 0.65)
// Offline: check if description contains a URL whose domain doesn't
// correspond to the company name (fuzzy match, no DNS)
// ---------------------------------------------------------------------------

export function checkCompanyDomainMismatch(job) {
  const name = String(job.company || '').trim();
  if (!name || name.length < 3) return null;

  const text = fullText(job);
  const urls = extractUrls(text);
  if (urls.length === 0) return null;

  // Normalize company name for comparison
  const companyNorm = name.toLowerCase().replace(/[^a-z0-9]/g, '');

  // Count URLs that don't match the company name
  let mismatchCount = 0;
  let mismatchExample = '';
  for (const url of urls) {
    const domain = parseDomain(url);
    if (!domain) continue;
    // Skip known shorteners and personal email domains
    if (SHORTENER_DOMAINS.has(domain)) continue;
    const domainNorm = domain.replace(/[^a-z0-9]/g, '');
    if (!domainNorm.includes(companyNorm) && !companyNorm.includes(domainNorm.slice(0, Math.max(4, companyNorm.length)))) {
      mismatchCount++;
      if (!mismatchExample) mismatchExample = domain;
    }
  }

  // Only fire if there are URLs AND none match the company
  if (mismatchCount > 0 && mismatchCount === urls.length) {
    return {
      name: 'company_domain_mismatch',
      category: 'warning',
      weight: 0.65,
      confidence: 0.55,
      detail: `Company name '${name}' does not match any URLs in the posting (e.g. ${mismatchExample})`,
      evidence: `name=${name}, domain=${mismatchExample}`,
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal CV-3: virtual_office_address  (WARNING, weight 0.45)
// ---------------------------------------------------------------------------

export function checkVirtualOfficeAddress(job) {
  const location = String(job.location || '').trim();
  if (!location) return null;

  if (RE_VIRTUAL_OFFICE.test(location)) {
    return {
      name: 'virtual_office_address',
      category: 'warning',
      weight: 0.45,
      confidence: 0.65,
      detail: `Location '${location}' appears to be a virtual office address commonly used by scammers`,
      evidence: location,
    };
  }
  if (RE_PO_BOX.test(location)) {
    return {
      name: 'virtual_office_address',
      category: 'warning',
      weight: 0.45,
      confidence: 0.55,
      detail: `Location '${location}' is a PO Box — no physical office`,
      evidence: location,
    };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Signal CV-4: suspicious_company_name  (WARNING, weight 0.60)
// Enhanced version with Levenshtein brand-misspelling detection
// ---------------------------------------------------------------------------

export function checkSuspiciousCompanyNameEnhanced(job) {
  const name = String(job.company || '').trim();
  if (!name) return null;

  // Skip if it's a known legitimate company
  const nameLower = name.toLowerCase();
  if (KNOWN_COMPANIES.has(nameLower)) return null;

  // Misspelled brand name (Levenshtein)
  const misspelled = isMisspelledBrand(name);
  if (misspelled) {
    return {
      name: 'suspicious_company_name',
      category: 'warning',
      weight: 0.60,
      confidence: 0.70,
      detail: `Company name '${name}' looks like a misspelling of '${misspelled}'`,
      evidence: `${name} ≈ ${misspelled}`,
    };
  }

  // Random character pattern
  if (RE_RANDOM_CHARS.test(name)) {
    return {
      name: 'suspicious_company_name',
      category: 'warning',
      weight: 0.60,
      confidence: 0.55,
      detail: `Company name '${name}' contains random-character patterns`,
      evidence: 'random_char_pattern',
    };
  }

  // Excessive entity suffixing
  if (RE_EXCESSIVE_ENTITY.test(name)) {
    return {
      name: 'suspicious_company_name',
      category: 'warning',
      weight: 0.60,
      confidence: 0.55,
      detail: `Company name '${name}' has duplicate entity suffixes (e.g. LLC LLC)`,
      evidence: 'excessive_entity_suffix',
    };
  }

  // Generic single-word + buzzword suffix
  const parts = name.split(/\s+/);
  if (RE_GENERIC_SUFFIX_CV.test(name) && parts.length <= 3) {
    return {
      name: 'suspicious_company_name',
      category: 'warning',
      weight: 0.60,
      confidence: 0.50,
      detail: `Company name '${name}' matches generic buzzword pattern`,
      evidence: 'generic_buzzword_name',
    };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Signal CV-5: company_verified  (POSITIVE, weight 0.30)
// Offline: fires when company is in the known-companies list
// ---------------------------------------------------------------------------

export function checkCompanyVerified(job) {
  const name = String(job.company || '').trim();
  if (!name) return null;

  const nameLower = name.toLowerCase();

  if (KNOWN_COMPANIES.has(nameLower)) {
    return {
      name: 'company_verified',
      category: 'positive',
      weight: 0.30,
      confidence: 0.75,
      detail: `Company '${name}' is in the known-legitimate companies list`,
      evidence: 'verification_source=known_companies_list',
    };
  }
  // Prefix/suffix match for variants like "Google LLC"
  for (const known of KNOWN_COMPANIES) {
    if (nameLower.startsWith(known + ' ') || nameLower.endsWith(' ' + known)) {
      return {
        name: 'company_verified',
        category: 'positive',
        weight: 0.30,
        confidence: 0.70,
        detail: `Company '${name}' matches known-legitimate company '${known}'`,
        evidence: `verification_source=known_companies_list, matched=${known}`,
      };
    }
  }

  return null;
}
