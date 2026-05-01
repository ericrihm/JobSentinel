/**
 * index.js — JobSentinel Cloudflare Worker entry point
 *
 * Routes:
 *   POST /api/analyze      — analyze job posting text
 *   POST /api/report       — submit scam/legitimate verdict
 *   GET  /api/patterns     — list known patterns from D1
 *   GET  /api/stats        — scan statistics from D1
 *   GET  /api/health       — health check
 *   GET  /                 — API info
 *
 * Bindings required (wrangler.toml):
 *   RATE_LIMIT  — KV namespace for per-IP rate limiting
 *   DB          — D1 database for scan history + patterns
 */

import { extractSignals } from './signals.js';
import { buildResult } from './scorer.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RATE_LIMIT_RPM = 10;           // requests per minute per IP
const RATE_LIMIT_WINDOW_MS = 60000;  // 60-second sliding window
const MAX_TEXT_LENGTH = 50000;
const MAX_TITLE_LENGTH = 500;
const MAX_COMPANY_LENGTH = 500;

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Max-Age': '86400',
};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function jsonResponse(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...CORS_HEADERS,
      ...extraHeaders,
    },
  });
}

function errorResponse(message, status = 400) {
  return jsonResponse({ error: message, status }, status);
}

function getClientIP(request) {
  return (
    request.headers.get('CF-Connecting-IP') ||
    request.headers.get('X-Forwarded-For')?.split(',')[0]?.trim() ||
    '0.0.0.0'
  );
}

// ---------------------------------------------------------------------------
// Rate limiter using Cloudflare KV
// Sliding-window: stores a JSON array of timestamps per IP.
// ---------------------------------------------------------------------------

async function checkRateLimit(env, ip) {
  const key = `rl:${ip}`;
  const now = Date.now();
  const windowStart = now - RATE_LIMIT_WINDOW_MS;

  let timestamps = [];
  try {
    const stored = await env.RATE_LIMIT.get(key);
    if (stored) {
      timestamps = JSON.parse(stored).filter(t => t > windowStart);
    }
  } catch {
    // If KV is unavailable, allow the request (fail open)
    return { allowed: true, remaining: RATE_LIMIT_RPM };
  }

  if (timestamps.length >= RATE_LIMIT_RPM) {
    const resetAt = Math.min(...timestamps) + RATE_LIMIT_WINDOW_MS;
    return { allowed: false, remaining: 0, resetAt };
  }

  timestamps.push(now);

  // Store with 70-second TTL (slightly longer than window to handle clock skew)
  try {
    await env.RATE_LIMIT.put(key, JSON.stringify(timestamps), {
      expirationTtl: 70,
    });
  } catch {
    // Best-effort — don't fail the request if KV write fails
  }

  return { allowed: true, remaining: RATE_LIMIT_RPM - timestamps.length };
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

function validateAnalyzeRequest(body) {
  const errors = [];

  if (!body.text && !body.job_data) {
    errors.push('Provide either "text" (string) or "job_data" (object)');
  }

  if (body.text !== undefined) {
    if (typeof body.text !== 'string') errors.push('"text" must be a string');
    else if (body.text.length > MAX_TEXT_LENGTH)
      errors.push(`"text" exceeds ${MAX_TEXT_LENGTH} characters`);
    else if (/<script/i.test(body.text)) errors.push('"text" may not contain script tags');
  }

  if (body.title !== undefined) {
    if (typeof body.title !== 'string') errors.push('"title" must be a string');
    else if (body.title.length > MAX_TITLE_LENGTH)
      errors.push(`"title" exceeds ${MAX_TITLE_LENGTH} characters`);
    else if (/<script/i.test(body.title)) errors.push('"title" may not contain script tags');
  }

  if (body.company !== undefined) {
    if (typeof body.company !== 'string') errors.push('"company" must be a string');
    else if (body.company.length > MAX_COMPANY_LENGTH)
      errors.push(`"company" exceeds ${MAX_COMPANY_LENGTH} characters`);
  }

  if (body.url !== undefined && body.url !== null) {
    if (typeof body.url !== 'string') errors.push('"url" must be a string');
    else if (!/^https?:\/\//i.test(body.url)) errors.push('"url" must start with http:// or https://');
  }

  return errors;
}

// ---------------------------------------------------------------------------
// Build a JobPosting-like object from the request body
// ---------------------------------------------------------------------------

function buildJobFromRequest(body) {
  // If job_data provided, use it directly (structured input from extension)
  if (body.job_data && typeof body.job_data === 'object') {
    return {
      url: body.job_data.url || body.url || '',
      title: String(body.job_data.title || body.title || '').slice(0, MAX_TITLE_LENGTH),
      company: String(body.job_data.company || body.company || '').slice(0, MAX_COMPANY_LENGTH),
      location: String(body.job_data.location || ''),
      description: String(body.job_data.description || body.text || '').slice(0, MAX_TEXT_LENGTH),
      salary_min: Number(body.job_data.salary_min) || 0,
      salary_max: Number(body.job_data.salary_max) || 0,
      salary_currency: body.job_data.salary_currency || 'USD',
      posted_date: body.job_data.posted_date || '',
      applicant_count: Number(body.job_data.applicant_count) || 0,
      experience_level: String(body.job_data.experience_level || ''),
      employment_type: String(body.job_data.employment_type || ''),
      industry: String(body.job_data.industry || ''),
      company_size: String(body.job_data.company_size || ''),
      company_linkedin_url: String(body.job_data.company_linkedin_url || ''),
      recruiter_name: String(body.job_data.recruiter_name || ''),
      recruiter_connections: Number(body.job_data.recruiter_connections) || 0,
      is_remote: Boolean(body.job_data.is_remote),
      is_repost: Boolean(body.job_data.is_repost),
    };
  }

  // Text-mode: flat fields from request body
  return {
    url: body.url || '',
    title: String(body.title || '').slice(0, MAX_TITLE_LENGTH),
    company: String(body.company || '').slice(0, MAX_COMPANY_LENGTH),
    location: String(body.location || ''),
    description: String(body.text || '').slice(0, MAX_TEXT_LENGTH),
    salary_min: Number(body.salary_min) || 0,
    salary_max: Number(body.salary_max) || 0,
    salary_currency: body.salary_currency || 'USD',
    posted_date: body.posted_date || '',
    applicant_count: Number(body.applicant_count) || 0,
    experience_level: String(body.experience_level || ''),
    employment_type: String(body.employment_type || ''),
    industry: String(body.industry || ''),
    company_size: String(body.company_size || ''),
    company_linkedin_url: String(body.company_linkedin_url || ''),
    recruiter_name: String(body.recruiter_name || ''),
    recruiter_connections: Number(body.recruiter_connections) || 0,
    is_remote: Boolean(body.is_remote),
    is_repost: Boolean(body.is_repost),
  };
}

// ---------------------------------------------------------------------------
// D1 helpers
// ---------------------------------------------------------------------------

async function logScanToD1(env, job, result, clientIP) {
  try {
    await env.DB.prepare(
      `INSERT INTO scan_history
         (url, title, company, scam_score, risk_level, signal_count, client_ip, scanned_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    )
      .bind(
        job.url || '',
        (job.title || '').slice(0, 500),
        (job.company || '').slice(0, 500),
        result.scam_score,
        result.risk_level,
        result.signal_count,
        clientIP,
        new Date().toISOString(),
      )
      .run();
  } catch {
    // Non-critical — don't fail the response if D1 write fails
  }
}

async function getStatsFromD1(env) {
  try {
    const [totals, riskBreakdown] = await Promise.all([
      env.DB.prepare(
        'SELECT COUNT(*) as total, AVG(scam_score) as avg_score FROM scan_history'
      ).first(),
      env.DB.prepare(
        `SELECT risk_level, COUNT(*) as count
         FROM scan_history
         GROUP BY risk_level`
      ).all(),
    ]);

    const byRisk = {};
    for (const row of (riskBreakdown?.results || [])) {
      byRisk[row.risk_level] = row.count;
    }

    return {
      total_jobs_analyzed: totals?.total || 0,
      avg_scam_score: totals?.avg_score ? Math.round(totals.avg_score * 1000) / 1000 : 0,
      risk_breakdown: byRisk,
    };
  } catch {
    return { total_jobs_analyzed: 0, avg_scam_score: 0, risk_breakdown: {} };
  }
}

async function getPatternsFromD1(env) {
  try {
    const result = await env.DB.prepare(
      `SELECT pattern_id, name, description, category, status, keywords, bayesian_score
       FROM patterns
       WHERE status = 'active'
       ORDER BY bayesian_score DESC
       LIMIT 100`
    ).all();

    return (result?.results || []).map(row => ({
      ...row,
      keywords: row.keywords ? JSON.parse(row.keywords) : [],
    }));
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async function handleAnalyze(request, env) {
  const clientIP = getClientIP(request);

  // Rate limiting
  const rateCheck = await checkRateLimit(env, clientIP);
  if (!rateCheck.allowed) {
    return jsonResponse(
      { error: 'Rate limit exceeded. Max 10 requests/minute.', retry_after_ms: rateCheck.resetAt - Date.now() },
      429,
      {
        'X-RateLimit-Limit': String(RATE_LIMIT_RPM),
        'X-RateLimit-Remaining': '0',
        'Retry-After': String(Math.ceil((rateCheck.resetAt - Date.now()) / 1000)),
      }
    );
  }

  // Parse body
  let body;
  try {
    body = await request.json();
  } catch {
    return errorResponse('Invalid JSON body');
  }

  // Validate
  const validationErrors = validateAnalyzeRequest(body);
  if (validationErrors.length > 0) {
    return errorResponse(validationErrors.join('; '));
  }

  // Build job and run signals
  const startMs = Date.now();
  const job = buildJobFromRequest(body);
  const signals = extractSignals(job);
  const result = buildResult(job, signals, Date.now() - startMs);

  // Log to D1 (fire-and-forget, non-blocking)
  env.DB && logScanToD1(env, job, result, clientIP);

  return jsonResponse(result, 200, {
    'X-RateLimit-Limit': String(RATE_LIMIT_RPM),
    'X-RateLimit-Remaining': String(rateCheck.remaining),
  });
}

async function handleReport(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return errorResponse('Invalid JSON body');
  }

  if (!body.url || typeof body.url !== 'string') return errorResponse('"url" is required');
  if (typeof body.is_scam !== 'boolean') return errorResponse('"is_scam" (boolean) is required');
  if (body.reason && /<script/i.test(body.reason)) return errorResponse('"reason" may not contain script tags');
  if (!body.url.match(/^https?:\/\//i)) return errorResponse('"url" must start with http:// or https://');

  try {
    await env.DB.prepare(
      `INSERT INTO user_reports (url, is_scam, reason, reported_at)
       VALUES (?, ?, ?, ?)`
    )
      .bind(
        body.url.slice(0, 2048),
        body.is_scam ? 1 : 0,
        String(body.reason || '').slice(0, 5000),
        new Date().toISOString(),
      )
      .run();
  } catch {
    // Non-critical path — still return success so extension doesn't error
  }

  return jsonResponse({
    url: body.url,
    verdict: body.is_scam ? 'scam' : 'legitimate',
    reason: body.reason || '',
    recorded: true,
    message: 'Report received. Thank you for helping improve JobSentinel.',
  });
}

async function handlePatterns(env) {
  const patterns = await getPatternsFromD1(env);
  return jsonResponse({ patterns, count: patterns.length });
}

async function handleStats(env) {
  const stats = await getStatsFromD1(env);
  return jsonResponse(stats);
}

function handleHealth(env) {
  return jsonResponse({
    status: 'ok',
    service: 'jobsentinel-worker',
    version: env.API_VERSION || '1.0.0',
    timestamp: new Date().toISOString(),
  });
}

function handleRoot(env) {
  return jsonResponse({
    service: 'JobSentinel API',
    version: env.API_VERSION || '1.0.0',
    description: 'Free, globally-distributed job scam analysis API',
    endpoints: {
      'POST /api/analyze': 'Analyze a job posting for scam signals',
      'POST /api/report': 'Report a job as scam or legitimate',
      'GET /api/patterns': 'List known scam patterns',
      'GET /api/stats': 'Detection statistics',
      'GET /api/health': 'Health check',
    },
    rate_limit: `${RATE_LIMIT_RPM} requests/minute`,
    docs: 'https://github.com/your-org/sentinel#api',
  });
}

// ---------------------------------------------------------------------------
// Main fetch handler
// ---------------------------------------------------------------------------

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const method = request.method.toUpperCase();

    // Handle CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // Route dispatch
    try {
      if (url.pathname === '/api/analyze' && method === 'POST') {
        return await handleAnalyze(request, env);
      }
      if (url.pathname === '/api/report' && method === 'POST') {
        return await handleReport(request, env);
      }
      if (url.pathname === '/api/patterns' && method === 'GET') {
        return await handlePatterns(env);
      }
      if (url.pathname === '/api/stats' && method === 'GET') {
        return await handleStats(env);
      }
      if (url.pathname === '/api/health' && method === 'GET') {
        return handleHealth(env);
      }
      if (url.pathname === '/' && method === 'GET') {
        return handleRoot(env);
      }

      return errorResponse(`Not found: ${method} ${url.pathname}`, 404);
    } catch (err) {
      console.error('Unhandled error:', err);
      return errorResponse('Internal server error', 500);
    }
  },
};
