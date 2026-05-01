/* ============================================================
   JobSentinel — app.js
   Interactive features: live demo, smooth scroll, accordion,
   animated counters, copy buttons
   ============================================================ */

'use strict';

/* ---- Smooth scroll nav links ---- */
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', e => {
    const id = link.getAttribute('href');
    const target = document.querySelector(id);
    if (!target) return;
    e.preventDefault();
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // close mobile menu if open
    document.querySelector('.nav-links')?.classList.remove('open');
  });
});

/* ---- Mobile nav toggle ---- */
const toggle = document.querySelector('.nav-mobile-toggle');
const navLinks = document.querySelector('.nav-links');
toggle?.addEventListener('click', () => navLinks?.classList.toggle('open'));

/* ---- Reveal on scroll ---- */
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      revealObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.12 });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

/* ---- Animated counters ---- */
function animateCounter(el) {
  const raw = el.dataset.target;       // e.g. "40+", "1500+", "<10"
  const numMatch = raw.match(/[\d.]+/);
  if (!numMatch) { el.textContent = raw; return; }
  const target = parseFloat(numMatch[0]);
  const prefix = raw.match(/^[^0-9]*/)?.[0] || '';
  const suffix = raw.replace(/^[^0-9]*[\d.]+/, '');
  const duration = 1400;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
    const value = Math.round(eased * target);
    el.textContent = prefix + value + suffix;
    if (progress < 1) requestAnimationFrame(tick);
    else el.textContent = raw; // ensure exact final value
  }
  requestAnimationFrame(tick);
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      animateCounter(e.target);
      counterObserver.unobserve(e.target);
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('.stat-num[data-target]').forEach(el => counterObserver.observe(el));

/* ---- Accordion ---- */
document.querySelectorAll('.accordion-header').forEach(header => {
  header.addEventListener('click', () => {
    const item = header.closest('.accordion-item');
    const isOpen = item.classList.contains('open');
    // Close all
    document.querySelectorAll('.accordion-item.open').forEach(i => i.classList.remove('open'));
    // Open clicked (if it wasn't already open)
    if (!isOpen) item.classList.add('open');
  });
});

/* ---- Copy buttons ---- */
document.querySelectorAll('.copy-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const block = btn.closest('.code-block');
    const pre = block?.querySelector('pre');
    if (!pre) return;
    navigator.clipboard.writeText(pre.innerText.trim()).then(() => {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      btn.style.color = 'var(--green)';
      setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 1800);
    });
  });
});

/* ================================================================
   LIVE DEMO — client-side scam detection
   ================================================================ */

/** Signal definitions: [id, label, category, weight, keywords/test] */
const SIGNALS = [
  // --- Red Flags (high weight) ---
  {
    id: 'upfront_payment',
    label: 'Upfront Payment Required',
    category: 'red_flag',
    weight: 35,
    keywords: ['pay a fee', 'upfront fee', 'registration fee', 'starter kit', 'initial deposit',
                'purchase equipment', 'buy your kit', 'send money', 'wire transfer',
                'pay to start', 'investment required', 'fee to apply'],
  },
  {
    id: 'ssn_early',
    label: 'SSN / Bank Info Requested',
    category: 'red_flag',
    weight: 40,
    keywords: ['social security', 'ssn', 'bank account number', 'routing number',
                'credit card number', 'wire funds', 'western union', 'money order',
                'zelle', 'venmo', 'cash app'],
  },
  {
    id: 'guaranteed_income',
    label: 'Guaranteed Income Claims',
    category: 'red_flag',
    weight: 30,
    keywords: ['guaranteed income', 'guaranteed salary', 'earn up to', 'make $1000',
                'make $500 a day', 'earn thousands', 'six figures guaranteed',
                'unlimited earning potential', 'financial freedom', 'passive income stream'],
  },
  {
    id: 'work_from_home_easy',
    label: 'Unrealistic Remote Promise',
    category: 'red_flag',
    weight: 20,
    keywords: ['work from home no experience', 'no experience needed', 'no skills required',
                'be your own boss', 'set your own hours unlimited',
                'anyone can do this', 'no interviews required'],
  },
  {
    id: 'urgency_pressure',
    label: 'High-Pressure Urgency',
    category: 'red_flag',
    weight: 22,
    keywords: ['act now', 'limited spots', 'apply immediately', 'offer expires today',
                'don\'t miss out', 'hurry', 'urgent hiring', 'instant hire',
                'hired on the spot', 'no waiting'],
  },
  {
    id: 'personal_info_excessive',
    label: 'Excessive Personal Info Demand',
    category: 'red_flag',
    weight: 28,
    keywords: ['copy of id', 'copy of passport', 'birth certificate', 'background check fee',
                'credit check fee', 'send a photo', 'selfie with id'],
  },

  // --- Warnings (medium weight) ---
  {
    id: 'vague_company',
    label: 'Vague Company Description',
    category: 'warning',
    weight: 15,
    keywords: ['fast-growing company', 'leading company', 'well-known company',
                'anonymous company', 'confidential employer', 'global firm',
                'dynamic team', 'exciting opportunity'],
  },
  {
    id: 'salary_mismatch',
    label: 'Unusually High Salary',
    category: 'warning',
    weight: 18,
    keywords: ['$5000 per week', '$10000 per week', '$500 per hour',
                '$1000 per day', 'top pay', 'exceptional pay',
                'high compensation', 'above industry standard'],
  },
  {
    id: 'grammar_issues',
    label: 'Spelling / Grammar Issues',
    category: 'warning',
    weight: 12,
    // handled separately via heuristic
    keywords: [],
  },
  {
    id: 'reshipping',
    label: 'Reshipping / Package Handling',
    category: 'warning',
    weight: 25,
    keywords: ['reship', 're-ship', 'package handler', 'parcel forwarder',
                'receive packages', 'forward packages', 'shipping manager',
                'package inspector', 'mystery shopper', 'secret shopper'],
  },
  {
    id: 'crypto_payment',
    label: 'Cryptocurrency Payment',
    category: 'warning',
    weight: 20,
    keywords: ['bitcoin', 'ethereum', 'crypto payment', 'paid in crypto',
                'usdt', 'cryptocurrency', 'blockchain payment', 'nft'],
  },
  {
    id: 'check_cashing',
    label: 'Check Cashing Scheme',
    category: 'red_flag',
    weight: 38,
    keywords: ['cash this check', 'deposit check', 'overpayment check',
                'cashier\'s check', 'certified check', 'money order deposit',
                'keep a portion', 'send the remainder'],
  },

  // --- Ghost Job signals ---
  {
    id: 'evergreen_posting',
    label: 'Perpetually Open Role',
    category: 'ghost_job',
    weight: 10,
    keywords: ['always hiring', 'continuously hiring', 'ongoing opportunity',
                'rolling applications', 'open until filled', 'pool of candidates'],
  },
  {
    id: 'no_contact_info',
    label: 'No Direct Contact',
    category: 'ghost_job',
    weight: 8,
    keywords: ['do not call', 'no phone calls', 'no emails please',
                'apply through portal only', 'no inquiries'],
  },

  // --- Structural signals ---
  {
    id: 'no_qualifications',
    label: 'No Qualifications Listed',
    category: 'structural',
    weight: 12,
    keywords: ['no qualifications needed', 'no degree required', 'no resume',
                'no references', 'all backgrounds welcome', 'no background check'],
  },
  {
    id: 'mlm_pyramid',
    label: 'MLM / Pyramid Structure',
    category: 'red_flag',
    weight: 32,
    keywords: ['recruit others', 'build your team', 'downline', 'upline', 'network marketing',
                'multi-level', 'mlm', 'direct sales opportunity', 'residual income',
                'commission-only recruiting'],
  },

  // --- Positive signals ---
  {
    id: 'company_details',
    label: 'Detailed Company Info',
    category: 'positive',
    weight: -15,
    keywords: ['founded in', 'our team of', 'series a', 'series b', 'series c',
                'publicly traded', 'nasdaq', 'nyse', 'fortune 500', 'inc 5000',
                'registered company', 'verified employer'],
  },
  {
    id: 'structured_interview',
    label: 'Structured Interview Process',
    category: 'positive',
    weight: -12,
    keywords: ['phone screen', 'technical interview', 'background check required',
                'reference check', 'offer letter', 'sign offer', 'onboarding'],
  },
  {
    id: 'benefits_listed',
    label: 'Legitimate Benefits',
    category: 'positive',
    weight: -10,
    keywords: ['health insurance', 'dental', 'vision', '401k', '401(k)', 'pto',
                'paid time off', 'parental leave', 'stock options', 'equity'],
  },
];

function detectGrammarIssues(text) {
  // Simple heuristic: count repeated punctuation, all-caps words, misspellings
  let score = 0;
  if ((text.match(/!!+/g) || []).length > 0) score += 3;
  if ((text.match(/\?\?+/g) || []).length > 0) score += 3;
  const words = text.split(/\s+/);
  const allCapsWords = words.filter(w => w.length > 3 && w === w.toUpperCase() && /[A-Z]/.test(w));
  if (allCapsWords.length > 3) score += 5;
  // Very long sentences (run-ons)
  const sentences = text.split(/[.!?]/);
  const runOns = sentences.filter(s => s.split(' ').length > 55);
  score += runOns.length * 3;
  return score >= 5;
}

function runDemo(text) {
  // Use the full 54-signal engine if loaded via analyze-bridge.js
  if (window.sentinelAnalyze) {
    const result = window.sentinelAnalyze(text);
    const triggered = result.signals.map(s => ({
      id: s.id, label: s.label, category: s.category,
      weight: s.weight, matched: s.evidence || s.detail,
    }));
    return { scamScore: result.scamScore, triggered };
  }

  // Fallback: simple keyword matching
  const lower = text.toLowerCase();
  const triggered = [];
  SIGNALS.forEach(signal => {
    if (signal.id === 'grammar_issues') {
      if (detectGrammarIssues(text)) {
        triggered.push({ ...signal, matched: 'Excessive caps / punctuation / run-on sentences detected' });
      }
      return;
    }
    const hit = signal.keywords.find(kw => lower.includes(kw.toLowerCase()));
    if (hit) {
      triggered.push({ ...signal, matched: `"${hit}"` });
    }
  });
  let rawScore = 0;
  triggered.forEach(s => { rawScore += s.weight; });
  const scamScore = Math.max(0, Math.min(100, rawScore));
  return { scamScore, triggered };
}

function scoreClass(score) {
  if (score >= 65) return 'score-red';
  if (score >= 40) return 'score-orange';
  if (score >= 20) return 'score-yellow';
  return 'score-green';
}

function scoreVerdict(score) {
  if (score >= 65) return { label: 'HIGH RISK', desc: 'This posting contains multiple strong scam indicators. Do not proceed.' };
  if (score >= 40) return { label: 'SUSPICIOUS', desc: 'Several warning signs detected. Investigate this employer carefully.' };
  if (score >= 20) return { label: 'SOME WARNINGS', desc: 'Minor red flags present. Verify the company before applying.' };
  return { label: 'LIKELY SAFE', desc: 'No significant scam signals detected in this posting.' };
}

function categoryColor(cat) {
  return {
    red_flag:  'sig-red',
    warning:   'sig-orange',
    ghost_job: 'sig-yellow',
    structural:'sig-blue',
    positive:  'sig-green',
  }[cat] || 'sig-blue';
}

function weightBg(w) {
  if (w < 0)  return 'background:rgba(63,185,80,.15);color:var(--green)';
  if (w >= 30) return 'background:rgba(248,81,73,.15);color:var(--red)';
  if (w >= 18) return 'background:rgba(219,109,40,.15);color:var(--orange)';
  return 'background:rgba(210,153,34,.12);color:var(--yellow)';
}

function buildResults({ scamScore, triggered }) {
  const verdict = scoreVerdict(scamScore);
  const cls = scoreClass(scamScore);

  const positives = triggered.filter(s => s.weight < 0);
  const negatives = triggered.filter(s => s.weight >= 0);

  const signalRows = triggered.length === 0
    ? '<p style="color:var(--muted);font-size:.875rem;text-align:center;padding:16px 0">No signals matched.</p>'
    : [...negatives, ...positives].map(s => `
        <div class="signal-item">
          <span class="signal-dot ${categoryColor(s.category)}"></span>
          <span class="signal-name">${s.label}</span>
          <span class="signal-desc">Matched: ${s.matched}</span>
          <span class="signal-weight" style="${weightBg(s.weight)}">${s.weight > 0 ? '+' : ''}${s.weight}</span>
        </div>`).join('');

  return `
    <div class="score-header">
      <div class="score-circle ${cls}">
        <span class="score-num">${scamScore}</span>
        <span class="score-label">/ 100</span>
      </div>
      <div class="score-meta">
        <h3>${verdict.label}</h3>
        <p>${verdict.desc}</p>
        <p style="margin-top:8px;font-size:.8rem;color:var(--muted)">${triggered.length} signal${triggered.length !== 1 ? 's' : ''} matched &middot; ${negatives.length} red flags &middot; ${positives.length} positive indicators</p>
      </div>
    </div>
    <div class="signals-list">${signalRows}</div>
  `;
}

/* Sample postings */
const SAMPLES = {
  scam: `URGENT HIRING - Work From Home - No Experience Needed!

We are a fast-growing company looking for motivated individuals to join our team immediately!
Earn up to $1,000 per day working from home — guaranteed income every week!

• No experience required — anyone can do this!
• Set your own hours, unlimited earning potential
• Be your own boss and achieve financial freedom

To get started, you'll need to purchase your starter kit ($299) to cover training materials and equipment.
Once hired, you'll receive a certified check — simply deposit it and send us $150 for processing fees.

We also handle package shipments — you'll receive packages at your home and reship them to our clients.

APPLY NOW — limited spots available! Offer expires TODAY.

For faster processing, please send a copy of your ID and social security number along with your bank routing number to begin payroll setup.

Don't miss this amazing opportunity!!!`,

  safe: `Senior Software Engineer — Platform Infrastructure
Acme Corp | Remote (US) | Full-time

About Acme Corp:
Founded in 2015, Acme Corp (NASDAQ: ACME) is a cloud infrastructure company trusted by 10,000+ businesses. Our team of 450 engineers builds the tools that power modern software delivery.

Role Overview:
We're looking for a Senior Software Engineer to join our Platform team. You'll own the design and implementation of our internal developer tooling, working closely with our SRE and product teams.

Qualifications:
• 5+ years of software engineering experience
• Strong proficiency in Go or Rust
• Experience with Kubernetes, Terraform
• BS/MS in Computer Science or equivalent experience

Interview Process:
1. Initial phone screen (30 min) with recruiter
2. Technical interview with engineering panel
3. System design interview
4. Reference check and background check

Compensation: $160,000–$200,000 base salary
Benefits: Health, dental, vision insurance; 401(k) with 4% match; 20 days PTO; parental leave; equity

Apply at careers.acmecorp.com — no recruiters please.`
};

function loadSample(type) {
  const ta = document.getElementById('jobInput');
  if (ta) {
    ta.value = SAMPLES[type] || '';
    ta.focus();
  }
}

function analyzeJob() {
  const ta = document.getElementById('jobInput');
  const result = document.getElementById('demoResult');
  if (!ta || !result) return;

  const text = ta.value.trim();
  if (text.length < 30) {
    ta.style.borderBottom = '2px solid var(--red)';
    ta.placeholder = 'Please paste a job description (at least 30 characters)...';
    setTimeout(() => { ta.style.borderBottom = ''; }, 1500);
    return;
  }

  const analysis = runDemo(text);
  result.innerHTML = buildResults(analysis);
  result.style.display = 'block';
  result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function clearDemo() {
  const ta = document.getElementById('jobInput');
  const result = document.getElementById('demoResult');
  if (ta) ta.value = '';
  if (result) { result.innerHTML = ''; result.style.display = 'none'; }
}

// Expose to HTML onclick attributes
window.analyzeJob = analyzeJob;
window.clearDemo  = clearDemo;
window.loadSample = loadSample;
