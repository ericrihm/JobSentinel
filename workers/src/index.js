/**
 * index.js — JobSentinel Cloudflare Worker entry point
 *
 * Routes:
 *   POST /api/analyze      — analyze job posting text
 *   POST /api/report       — submit scam/legitimate verdict
 *   GET  /api/patterns     — list known patterns from D1
 *   GET  /api/stats        — scan statistics from D1
 *   GET  /api/jobs         — search and browse aggregated jobs
 *   GET  /api/jobs/stats   — job aggregation stats
 *   GET  /api/jobs/:id     — full job detail
 *   POST /api/jobs/ingest  — bulk ingest jobs (pipeline, auth required)
 *   GET  /api/health       — health check
 *   GET  /                 — API info
 *
 * Bindings required (wrangler.toml):
 *   RATE_LIMIT  — KV namespace for per-IP rate limiting
 *   DB          — D1 database for scan history + patterns + jobs
 *   INGEST_KEY  — env var for authenticating ingest requests
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

const JOBS_MAX_PER_PAGE = 100;
const JOBS_DEFAULT_PER_PAGE = 20;
const DESCRIPTION_PREVIEW_LENGTH = 200;

const RISK_SCORE_THRESHOLDS = {
  safe:       { min: 0,    max: 0.19 },
  low:        { min: 0.20, max: 0.39 },
  suspicious: { min: 0.40, max: 0.59 },
  high:       { min: 0.60, max: 1.0  },
};

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
// Jobs API helpers
// ---------------------------------------------------------------------------

/**
 * Build WHERE clause and params array from search query parameters.
 * All user input goes through parameterized queries to prevent SQL injection.
 */
function buildJobSearchQuery(searchParams) {
  const conditions = [];
  const params = [];

  // Full-text search across title, company, description
  const q = searchParams.get('q');
  if (q) {
    const term = '%' + q + '%';
    conditions.push('(title LIKE ? OR company LIKE ? OR description LIKE ?)');
    params.push(term, term, term);
  }

  // Location filter
  const location = searchParams.get('location');
  if (location) {
    conditions.push('location LIKE ?');
    params.push('%' + location + '%');
  }

  // Remote filter
  const remote = searchParams.get('remote');
  if (remote === 'true') {
    conditions.push('is_remote = 1');
  }

  // Source filter
  const source = searchParams.get('source');
  if (source) {
    conditions.push('source = ?');
    params.push(source);
  }

  // Salary filters
  const minSalary = searchParams.get('min_salary');
  if (minSalary && !isNaN(Number(minSalary))) {
    conditions.push('salary_max >= ?');
    params.push(Number(minSalary));
  }

  const maxSalary = searchParams.get('max_salary');
  if (maxSalary && !isNaN(Number(maxSalary))) {
    conditions.push('salary_min <= ?');
    params.push(Number(maxSalary));
  }

  // Risk level filter — exclude jobs above the specified threshold
  const risk = searchParams.get('risk');
  const maxScore = risk && RISK_SCORE_THRESHOLDS[risk]
    ? RISK_SCORE_THRESHOLDS[risk].max
    : RISK_SCORE_THRESHOLDS.suspicious.max;  // default: exclude high-risk
  conditions.push('scam_score <= ?');
  params.push(maxScore);

  // Always filter to active jobs only
  conditions.push('is_active = 1');

  const whereClause = conditions.length > 0
    ? 'WHERE ' + conditions.join(' AND ')
    : '';

  return { whereClause, params };
}

/**
 * Determine ORDER BY clause from sort parameter.
 */
function buildJobSortClause(sort) {
  switch (sort) {
    case 'salary':
      return 'ORDER BY salary_max DESC, salary_min DESC';
    case 'score':
      return 'ORDER BY scam_score ASC';
    case 'newest':
    default:
      return 'ORDER BY posted_at DESC, discovered_at DESC';
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
      'GET /api/jobs': 'Search and browse aggregated jobs',
      'GET /api/jobs/stats': 'Aggregated job statistics',
      'GET /api/jobs/:id': 'Full job detail by ID',
      'POST /api/jobs/ingest': 'Bulk ingest jobs (auth required)',
      'GET /api/health': 'Health check',
    },
    rate_limit: `${RATE_LIMIT_RPM} requests/minute`,
    docs: 'https://github.com/ericrihm/JobSentinel#api',
  });
}

// ---------------------------------------------------------------------------
// Route handlers — Jobs API
// ---------------------------------------------------------------------------

/**
 * GET /api/jobs — Search and browse aggregated job listings.
 *
 * Query params: q, location, remote, source, min_salary, max_salary,
 *               risk, sort, page, per_page
 */
async function handleJobSearch(request, env) {
  const url = new URL(request.url);
  const searchParams = url.searchParams;

  // Pagination
  const page = Math.max(1, parseInt(searchParams.get('page') || '1', 10) || 1);
  const perPage = Math.min(
    JOBS_MAX_PER_PAGE,
    Math.max(1, parseInt(searchParams.get('per_page') || String(JOBS_DEFAULT_PER_PAGE), 10) || JOBS_DEFAULT_PER_PAGE)
  );
  const offset = (page - 1) * perPage;

  // Build query
  const { whereClause, params } = buildJobSearchQuery(searchParams);
  const sortClause = buildJobSortClause(searchParams.get('sort'));

  // Run data + count queries in parallel
  const dataSQL = `SELECT id, external_id, url, title, company, location,
                          SUBSTR(description, 1, ${DESCRIPTION_PREVIEW_LENGTH}) AS description_preview,
                          salary_min, salary_max, salary_currency,
                          employment_type, experience_level, is_remote,
                          source, source_company, scam_score, risk_level,
                          signal_count, posted_at, discovered_at, expires_at,
                          is_active, content_hash
                   FROM jobs ${whereClause} ${sortClause}
                   LIMIT ? OFFSET ?`;

  const countSQL = `SELECT COUNT(*) as total FROM jobs ${whereClause}`;

  const dataParams = [...params, perPage, offset];
  const countParams = [...params];

  try {
    const [dataResult, countResult] = await Promise.all([
      env.DB.prepare(dataSQL).bind(...dataParams).all(),
      env.DB.prepare(countSQL).bind(...countParams).first(),
    ]);

    const jobs = dataResult?.results || [];
    const total = countResult?.total || 0;
    const totalPages = Math.ceil(total / perPage);

    // Build filters echo for the response
    const filters = {};
    for (const key of ['q', 'location', 'remote', 'source', 'min_salary', 'max_salary', 'risk', 'sort']) {
      const val = searchParams.get(key);
      if (val !== null) filters[key] = val;
    }

    return jsonResponse({
      jobs,
      total,
      page,
      per_page: perPage,
      total_pages: totalPages,
      filters,
    });
  } catch (err) {
    console.error('Job search error:', err);
    return errorResponse('Failed to search jobs', 500);
  }
}

/**
 * GET /api/jobs/:id — Full job detail including complete description.
 */
async function handleJobDetail(env, jobId) {
  const id = parseInt(jobId, 10);
  if (isNaN(id) || id < 1) {
    return errorResponse('Invalid job ID', 400);
  }

  try {
    const job = await env.DB.prepare(
      'SELECT * FROM jobs WHERE id = ?'
    ).bind(id).first();

    if (!job) {
      return errorResponse('Job not found', 404);
    }

    return jsonResponse(job);
  } catch (err) {
    console.error('Job detail error:', err);
    return errorResponse('Failed to fetch job', 500);
  }
}

/**
 * POST /api/jobs/ingest — Bulk ingest jobs from the pipeline.
 *
 * Protected by INGEST_KEY bearer token.
 * Accepts { "jobs": [...] } and upserts on URL.
 */
async function handleJobIngest(request, env) {
  // Auth check
  const authHeader = request.headers.get('Authorization') || '';
  const token = authHeader.replace(/^Bearer\s+/i, '');

  if (!env.INGEST_KEY || token !== env.INGEST_KEY) {
    return errorResponse('Unauthorized', 401);
  }

  // Parse body
  let body;
  try {
    body = await request.json();
  } catch {
    return errorResponse('Invalid JSON body');
  }

  if (!Array.isArray(body.jobs)) {
    return errorResponse('"jobs" must be an array');
  }

  if (body.jobs.length === 0) {
    return jsonResponse({ inserted: 0, updated: 0, errors: [] });
  }

  if (body.jobs.length > 1000) {
    return errorResponse('Maximum 1000 jobs per ingest request');
  }

  let inserted = 0;
  let updated = 0;
  const errors = [];

  const upsertSQL = `INSERT OR REPLACE INTO jobs
    (external_id, url, title, company, location, description,
     salary_min, salary_max, salary_currency,
     employment_type, experience_level, is_remote,
     source, source_company, scam_score, risk_level, signal_count,
     posted_at, discovered_at, expires_at, is_active, content_hash)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`;

  for (let i = 0; i < body.jobs.length; i++) {
    const j = body.jobs[i];
    try {
      if (!j.url) {
        errors.push({ index: i, error: 'Missing required field: url' });
        continue;
      }

      // Check if URL already exists to track insert vs update
      const existing = await env.DB.prepare(
        'SELECT id FROM jobs WHERE url = ?'
      ).bind(j.url).first();

      await env.DB.prepare(upsertSQL).bind(
        String(j.external_id || ''),
        String(j.url),
        String(j.title || ''),
        String(j.company || ''),
        String(j.location || ''),
        String(j.description || ''),
        Number(j.salary_min) || 0,
        Number(j.salary_max) || 0,
        String(j.salary_currency || 'USD'),
        String(j.employment_type || ''),
        String(j.experience_level || ''),
        j.is_remote ? 1 : 0,
        String(j.source || ''),
        String(j.source_company || ''),
        Number(j.scam_score) || 0,
        String(j.risk_level || 'safe'),
        Number(j.signal_count) || 0,
        String(j.posted_at || ''),
        String(j.discovered_at || new Date().toISOString()),
        String(j.expires_at || ''),
        j.is_active !== undefined ? (j.is_active ? 1 : 0) : 1,
        String(j.content_hash || ''),
      ).run();

      if (existing) {
        updated++;
      } else {
        inserted++;
      }
    } catch (err) {
      errors.push({ index: i, url: j.url || '', error: String(err.message || err) });
    }
  }

  return jsonResponse({ inserted, updated, errors });
}

/**
 * GET /api/jobs/stats — Aggregated statistics for the jobs table.
 */
async function handleJobStats(env) {
  try {
    const [
      totals,
      bySource,
      byRisk,
      salaryStats,
      dateRange,
    ] = await Promise.all([
      env.DB.prepare(
        'SELECT COUNT(*) as total FROM jobs WHERE is_active = 1'
      ).first(),

      env.DB.prepare(
        `SELECT source, COUNT(*) as count
         FROM jobs WHERE is_active = 1
         GROUP BY source ORDER BY count DESC`
      ).all(),

      env.DB.prepare(
        `SELECT risk_level, COUNT(*) as count
         FROM jobs WHERE is_active = 1
         GROUP BY risk_level ORDER BY count DESC`
      ).all(),

      env.DB.prepare(
        `SELECT AVG(salary_min) as avg_salary_min, AVG(salary_max) as avg_salary_max
         FROM jobs WHERE is_active = 1 AND salary_max > 0`
      ).first(),

      env.DB.prepare(
        `SELECT MIN(posted_at) as oldest, MAX(posted_at) as newest
         FROM jobs WHERE is_active = 1 AND posted_at != ''`
      ).first(),
    ]);

    const sourceBreakdown = {};
    for (const row of (bySource?.results || [])) {
      if (row.source) sourceBreakdown[row.source] = row.count;
    }

    const riskBreakdown = {};
    for (const row of (byRisk?.results || [])) {
      riskBreakdown[row.risk_level] = row.count;
    }

    return jsonResponse({
      total_active_jobs: totals?.total || 0,
      by_source: sourceBreakdown,
      by_risk_level: riskBreakdown,
      avg_salary_min: salaryStats?.avg_salary_min
        ? Math.round(salaryStats.avg_salary_min)
        : 0,
      avg_salary_max: salaryStats?.avg_salary_max
        ? Math.round(salaryStats.avg_salary_max)
        : 0,
      oldest_posting: dateRange?.oldest || null,
      newest_posting: dateRange?.newest || null,
    });
  } catch (err) {
    console.error('Job stats error:', err);
    return errorResponse('Failed to fetch job stats', 500);
  }
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

      // --- Jobs API routes ---
      if (url.pathname === '/api/jobs' && method === 'GET') {
        return await handleJobSearch(request, env);
      }
      if (url.pathname === '/api/jobs/stats' && method === 'GET') {
        return await handleJobStats(env);
      }
      if (url.pathname === '/api/jobs/ingest' && method === 'POST') {
        return await handleJobIngest(request, env);
      }
      // /api/jobs/:id — match numeric ID
      const jobDetailMatch = url.pathname.match(/^\/api\/jobs\/(\d+)$/);
      if (jobDetailMatch && method === 'GET') {
        return await handleJobDetail(env, jobDetailMatch[1]);
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
