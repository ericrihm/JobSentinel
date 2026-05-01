"""Multi-tier AI analysis for job postings."""

import logging
import time
from typing import Optional

from sentinel.models import JobPosting, ScamSignal, SignalCategory, ValidationResult, RiskLevel
from sentinel.signals import extract_signals
from sentinel.scorer import build_result
from sentinel.config import get_config

logger = logging.getLogger(__name__)

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

_AMBIGUOUS_LOW = 0.3
_AMBIGUOUS_HIGH = 0.7


def _haiku_model() -> str:
    return get_config().ai_model


def _sonnet_model() -> str:
    return get_config().ai_model_deep

_SYSTEM_PROMPT = (
    "You are an expert at detecting fraudulent and scam job postings on LinkedIn. "
    "Analyze the provided job posting and assess whether it is a scam. "
    "Focus on: payment requests, unrealistic compensation, vague descriptions, "
    "suspicious contact methods, fake company indicators, and high-pressure tactics. "
    "Be concise and structured."
)


def analyze_job(job: JobPosting, use_ai: bool = True) -> ValidationResult:
    """Full analysis pipeline:
    1. Extract signals (fast regex/heuristic, <10ms)
    2. Score and classify
    3. If score is ambiguous (0.3-0.7) and use_ai=True, escalate to AI tier
    4. Return complete ValidationResult
    """
    label = job.title or job.url or "(unknown)"
    logger.info("Starting analysis: %s", label)
    start_ms = time.monotonic() * 1000

    signals = extract_signals(job)
    result = build_result(job, signals)

    if use_ai and _AMBIGUOUS_LOW <= result.scam_score <= _AMBIGUOUS_HIGH:
        ai_text, tier = _escalate_to_ai(job, signals, result.scam_score)
        result.ai_analysis = ai_text
        result.ai_tier_used = tier

    result.analysis_time_ms = (time.monotonic() * 1000) - start_ms
    logger.info(
        "Analysis complete: %s — score=%.2f risk=%s time=%.0fms",
        label,
        result.scam_score,
        result.risk_level.value,
        result.analysis_time_ms,
    )
    return result


def analyze_text(text: str, title: str = "", company: str = "", use_ai: bool = True) -> ValidationResult:
    """Convenience: analyze from raw text (creates JobPosting internally)."""
    job = JobPosting(
        description=text,
        title=title,
        company=company,
        source="text",
    )
    return analyze_job(job, use_ai=use_ai)


def analyze_url(url: str, use_ai: bool = True) -> ValidationResult:
    """Analyze a LinkedIn job URL (requires httpx + scanner module for parsing)."""
    from sentinel.scanner import parse_job_url
    job = parse_job_url(url)
    return analyze_job(job, use_ai=use_ai)


def _escalate_to_ai(
    job: JobPosting,
    signals: list[ScamSignal],
    current_score: float,
) -> tuple[str, str]:
    """Call Claude API for deeper analysis on ambiguous postings.

    Returns (ai_analysis_text, tier_used).
    Tier selection:
    - Try haiku first (fast, cheap)
    - If still ambiguous, escalate to sonnet
    Falls back gracefully if anthropic not installed.
    """
    if not get_config().ai_enabled:
        return ("", "disabled")

    if not _ANTHROPIC_AVAILABLE:
        return ("", "none")

    signal_summary = "; ".join(
        f"{s.name} ({s.category.value})" for s in signals[:10]
    )

    posting_text = (
        f"Title: {job.title}\n"
        f"Company: {job.company}\n"
        f"Location: {job.location}\n"
        f"Salary: {job.salary_min}-{job.salary_max} {job.salary_currency}\n"
        f"Employment type: {job.employment_type}\n"
        f"Experience level: {job.experience_level}\n"
        f"Description:\n{job.description[:3000]}\n\n"
        f"Detected signals: {signal_summary}\n"
        f"Current heuristic score: {current_score:.2f} (0=safe, 1=scam)"
    )

    user_message = (
        f"Analyze this job posting and determine if it is a scam or legitimate.\n\n"
        f"{posting_text}\n\n"
        f"Provide: (1) scam likelihood assessment, (2) key red flags or legitimacy "
        f"indicators you observe, (3) overall recommendation (safe/suspicious/scam)."
    )

    client = _anthropic.Anthropic()

    haiku = _haiku_model()
    sonnet = _sonnet_model()

    # Tier 1: Haiku (fast, cheap)
    try:
        response = client.messages.create(
            model=haiku,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        analysis = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        if analysis:
            return (analysis, haiku)
    except Exception:
        logger.warning("Haiku AI tier failed; falling back to Sonnet", exc_info=True)

    # Tier 2: Sonnet (deeper analysis for persistent ambiguity)
    try:
        response = client.messages.create(
            model=sonnet,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        analysis = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        if analysis:
            return (analysis, sonnet)
    except Exception:
        logger.warning("Sonnet AI tier failed; returning empty analysis", exc_info=True)

    return ("", "failed")


def batch_analyze(
    jobs: list[JobPosting],
    use_ai: bool = False,
) -> list[ValidationResult]:
    """Analyze multiple postings. AI disabled by default for batch to save cost."""
    return [analyze_job(job, use_ai=use_ai) for job in jobs]


def format_result_text(result: ValidationResult) -> str:
    """Human-readable analysis output with risk indicators."""
    lines: list[str] = []

    risk_icons = {
        RiskLevel.SAFE: "[SAFE]",
        RiskLevel.LOW: "[LOW]",
        RiskLevel.SUSPICIOUS: "[SUSPICIOUS]",
        RiskLevel.HIGH: "[HIGH RISK]",
        RiskLevel.SCAM: "[SCAM]",
    }
    icon = risk_icons.get(result.risk_level, "[UNKNOWN]")

    lines.append(f"{icon} {result.risk_label()}")
    lines.append(f"Scam score: {result.scam_score:.0%}  |  Confidence: {result.confidence:.0%}")
    lines.append(f"Analysis time: {result.analysis_time_ms:.1f}ms")

    job = result.job
    if job.title or job.company:
        lines.append("")
        if job.title:
            lines.append(f"Job:     {job.title}")
        if job.company:
            lines.append(f"Company: {job.company}")
        if job.url:
            lines.append(f"URL:     {job.url}")

    if result.red_flags:
        lines.append("")
        lines.append(f"Red flags ({len(result.red_flags)}):")
        for s in result.red_flags:
            detail = f" — {s.detail}" if s.detail else ""
            lines.append(f"  ! {s.name}{detail}")

    if result.warnings:
        lines.append("")
        lines.append(f"Warnings ({len(result.warnings)}):")
        for s in result.warnings:
            detail = f" — {s.detail}" if s.detail else ""
            lines.append(f"  ~ {s.name}{detail}")

    if result.ghost_indicators:
        lines.append("")
        lines.append(f"Ghost job indicators ({len(result.ghost_indicators)}):")
        for s in result.ghost_indicators:
            detail = f" — {s.detail}" if s.detail else ""
            lines.append(f"  ? {s.name}{detail}")

    if result.positive_signals:
        lines.append("")
        lines.append(f"Positive signals ({len(result.positive_signals)}):")
        for s in result.positive_signals:
            detail = f" — {s.detail}" if s.detail else ""
            lines.append(f"  + {s.name}{detail}")

    if result.ai_analysis:
        lines.append("")
        lines.append(f"AI analysis ({result.ai_tier_used}):")
        for line in result.ai_analysis.strip().splitlines():
            lines.append(f"  {line}")

    return "\n".join(lines)
