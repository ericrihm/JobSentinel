"""Command-line interface for Sentinel LinkedIn job scam detection."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

# ---------------------------------------------------------------------------
# Risk-level styling helpers
# ---------------------------------------------------------------------------

_RISK_STYLES: dict[str, dict[str, Any]] = {
    "safe":       {"fg": "green",  "bold": False},
    "low":        {"fg": "green",  "bold": False},
    "suspicious": {"fg": "yellow", "bold": False},
    "high":       {"fg": "red",    "bold": False},
    "scam":       {"fg": "bright_red", "bold": True},
}

_RISK_LABELS: dict[str, str] = {
    "safe":       "SAFE",
    "low":        "LOW RISK",
    "suspicious": "SUSPICIOUS",
    "high":       "HIGH RISK",
    "scam":       "SCAM",
}


def _style_risk(level: str) -> str:
    """Return a colored, styled risk-level badge string."""
    label = _RISK_LABELS.get(level.lower(), level.upper())
    style = _RISK_STYLES.get(level.lower(), {"fg": "white", "bold": False})
    return click.style(f"[{label}]", **style)


def _output(ctx: click.Context, data: Any, text: str) -> None:
    """Print either JSON or formatted text depending on ctx.obj['json']."""
    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        click.echo(text)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="sentinel")
@click.option("--json-output", "use_json", is_flag=True,
              help="Output results as JSON (machine-readable).")
@click.pass_context
def main(ctx: click.Context, use_json: bool) -> None:
    """Sentinel — LinkedIn job scam detection and validation platform."""
    from sentinel.config import setup_logging
    setup_logging()
    ctx.ensure_object(dict)
    ctx.obj["json"] = use_json


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_text", required=False, default=None)
@click.option("--title", default="", help="Job title (for raw-text input).")
@click.option("--company", default="", help="Company name (for raw-text input).")
@click.option("--no-ai", is_flag=True, default=False,
              help="Disable AI escalation (heuristics only, faster).")
@click.option("--file", "input_file", type=click.Path(exists=True), default=None,
              help="Batch analyze from JSON file")
@click.pass_context
def analyze(  # noqa: E501
    ctx: click.Context, input_text: str | None, title: str, company: str,
    no_ai: bool, input_file: str | None,
) -> None:
    """Analyze a job posting for scam signals.

    INPUT_TEXT can be a LinkedIn URL or raw job description text.
    Use --file to batch-analyze from a JSON file.
    """
    from sentinel.analyzer import analyze_text, analyze_url, batch_analyze
    from sentinel.db import SentinelDB

    use_ai = not no_ai

    # --- Batch mode ---
    if input_file is not None:
        import os

        from sentinel.scanner import load_jobs_from_file
        try:
            jobs = load_jobs_from_file(input_file)
        except Exception as exc:
            if ctx.obj.get("json"):
                click.echo(json.dumps({"error": str(exc)}, indent=2))
            else:
                click.echo(click.style(f"Error loading file: {exc}", fg="red"), err=True)
            sys.exit(1)

        results = batch_analyze(jobs, use_ai=use_ai)
        results.sort(key=lambda r: r.scam_score, reverse=True)

        # Risk-level counts
        _RISK_ORDER = ["scam", "high", "suspicious", "low", "safe"]
        counts: dict[str, int] = {level: 0 for level in _RISK_ORDER}
        for r in results:
            level = r.risk_level.value
            if level in counts:
                counts[level] += 1

        file_basename = os.path.basename(input_file)

        if ctx.obj.get("json"):
            click.echo(json.dumps(
                {
                    "file": input_file,
                    "total": len(results),
                    "results": [r.to_dict() for r in results],
                    "summary": counts,
                },
                indent=2,
            ))
            return

        # Formatted table
        click.echo("")
        click.echo(click.style(
            f"  Batch Analysis: {len(results)} jobs from {file_basename}", bold=True
        ))
        click.echo("  " + "═" * 55)

        _COL_SCORE = 6
        _COL_RISK = 14
        _COL_TITLE = 26
        _COL_COMPANY = 20

        header = (
            f"  {'Score':<{_COL_SCORE}}  {'Risk':<{_COL_RISK}}  "
            f"{'Title':<{_COL_TITLE}}  {'Company':<{_COL_COMPANY}}"
        )
        click.echo(click.style(header, bold=True))
        click.echo("  " + "─" * 55)

        for r in results:
            score_str = f"{r.scam_score:.2f}"
            risk_badge = _style_risk(r.risk_level.value)
            title_str = (r.job.title or "(no title)")[:_COL_TITLE]
            company_str = (r.job.company or "(unknown)")[:_COL_COMPANY]
            click.echo(
                f"  {score_str:<{_COL_SCORE}}  {risk_badge:<{_COL_RISK}}  "
                f"{title_str:<{_COL_TITLE}}  {company_str:<{_COL_COMPANY}}"
            )

        summary_parts = []
        for level in _RISK_ORDER:
            if counts[level] > 0:
                label = level.upper()
                summary_parts.append(f"{counts[level]} {label}")
        click.echo("")
        click.echo(f"  Summary: {', '.join(summary_parts) if summary_parts else 'no results'}")
        click.echo("")
        return

    # --- Single-job mode ---
    if not input_text:
        click.echo(click.style(
            "Error: provide INPUT_TEXT or --file for batch mode.", fg="red"
        ), err=True)
        sys.exit(1)

    is_url = input_text.startswith("http://") or input_text.startswith("https://")

    try:
        if is_url:
            result = analyze_url(input_text, use_ai=use_ai)
        else:
            result = analyze_text(input_text, title=title, company=company, use_ai=use_ai)
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"Error during analysis: {exc}", fg="red"), err=True)
        sys.exit(1)

    # Persist to DB (best-effort)
    try:
        db = SentinelDB()
        db.save_job({
            "url": result.job.url,
            "title": result.job.title,
            "company": result.job.company,
            "location": result.job.location,
            "description": result.job.description,
            "salary_min": result.job.salary_min,
            "salary_max": result.job.salary_max,
            "scam_score": result.scam_score,
            "risk_level": result.risk_level.value,
            "signal_count": len(result.signals),
            "signals_json": [
                {"name": s.name, "category": s.category.value, "detail": s.detail}
                for s in result.signals
            ],
        })
        db.close()
    except Exception:
        pass

    if ctx.obj.get("json"):
        click.echo(json.dumps(result.to_dict(), indent=2))
        return

    # --- Formatted output ---
    risk_badge = _style_risk(result.risk_level.value)
    click.echo("")
    click.echo(f"  {risk_badge}  {result.risk_label()}")
    click.echo(
        f"  Score: {click.style(f'{result.scam_score:.0%}', bold=True)}  "
        f"| Confidence: {result.confidence:.0%}  "
        f"| Analysis: {result.analysis_time_ms:.0f}ms"
    )

    job = result.job
    if job.title or job.company:
        click.echo("")
        if job.title:
            click.echo(f"  Job:     {job.title}")
        if job.company:
            click.echo(f"  Company: {job.company}")
        if job.location:
            click.echo(f"  Location:{job.location}")
        if job.url:
            click.echo(f"  URL:     {job.url}")

    if result.red_flags:
        click.echo("")
        click.echo(click.style(f"  Red Flags ({len(result.red_flags)}):", fg="red", bold=True))
        for s in result.red_flags:
            detail = f" — {s.detail}" if s.detail else ""
            click.echo(click.style(f"    ! {s.name}{detail}", fg="red"))

    if result.warnings:
        click.echo("")
        click.echo(click.style(f"  Warnings ({len(result.warnings)}):", fg="yellow", bold=True))
        for s in result.warnings:
            detail = f" — {s.detail}" if s.detail else ""
            click.echo(click.style(f"    ~ {s.name}{detail}", fg="yellow"))

    if result.ghost_indicators:
        click.echo("")
        click.echo(click.style(f"  Ghost Job Indicators ({len(result.ghost_indicators)}):", fg="cyan", bold=True))
        for s in result.ghost_indicators:
            detail = f" — {s.detail}" if s.detail else ""
            click.echo(click.style(f"    ? {s.name}{detail}", fg="cyan"))

    if result.positive_signals:
        click.echo("")
        click.echo(click.style(f"  Positive Signals ({len(result.positive_signals)}):", fg="green", bold=True))
        for s in result.positive_signals:
            detail = f" — {s.detail}" if s.detail else ""
            click.echo(click.style(f"    + {s.name}{detail}", fg="green"))

    if result.ai_analysis:
        click.echo("")
        tier_label = click.style(f"({result.ai_tier_used})", fg="blue")
        click.echo(f"  AI Analysis {tier_label}:")
        for line in result.ai_analysis.strip().splitlines():
            click.echo(f"    {line}")

    click.echo("")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@main.command()
@click.argument("company_name")
@click.option("--domain", default="", help="Company domain for WHOIS check (e.g. example.com).")
@click.option("--refresh", is_flag=True, default=False,
              help="Bypass cache and force fresh validation.")
@click.pass_context
def validate(ctx: click.Context, company_name: str, domain: str, refresh: bool) -> None:
    """Validate a company's legitimacy."""
    from sentinel.validator import validate_company

    try:
        profile = validate_company(company_name, domain=domain, refresh=refresh)
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    data: dict[str, Any] = {
        "company": profile.name,
        "is_verified": profile.is_verified,
        "verification_source": profile.verification_source,
        "has_linkedin_page": profile.has_linkedin_page,
        "linkedin_url": (
            profile.linkedin_url if hasattr(profile, "linkedin_url")
            else profile.company_linkedin_url if hasattr(profile, "company_linkedin_url")
            else ""
        ),
        "linkedin_followers": profile.linkedin_followers,
        "employee_count": profile.employee_count,
        "industry": profile.industry,
        "whois_age_days": profile.whois_age_days,
        "domain": profile.domain,
    }

    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2))
        return

    if profile.is_verified:
        badge = click.style("[VERIFIED]", fg="green", bold=True)
    elif profile.has_linkedin_page:
        badge = click.style("[LINKEDIN FOUND]", fg="yellow")
    else:
        badge = click.style("[UNVERIFIED]", fg="red")

    click.echo("")
    click.echo(f"  {badge}  {profile.name}")
    if profile.verification_source:
        click.echo(f"  Source:     {profile.verification_source}")
    if profile.domain:
        click.echo(f"  Domain:     {profile.domain}")
    if profile.whois_age_days:
        age_years = profile.whois_age_days // 365
        age_label = f"{profile.whois_age_days} days" + (f" (~{age_years} yr)" if age_years else "")
        click.echo(f"  WHOIS age:  {age_label}")
    if profile.employee_count:
        click.echo(f"  Employees:  {profile.employee_count:,}")
    if profile.industry:
        click.echo(f"  Industry:   {profile.industry}")
    if getattr(profile, "linkedin_url", "") or getattr(profile, "company_linkedin_url", ""):
        li_url = getattr(profile, "linkedin_url", "") or getattr(profile, "company_linkedin_url", "")
        click.echo(f"  LinkedIn:   {li_url}")
    click.echo("")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@main.command()
@click.argument("url")
@click.option("--reason", default="", help="Why you believe this is a scam.")
@click.option("--legitimate", "is_scam", flag_value=False, default=True,
              help="Mark as legitimate instead (correct a false positive).")
@click.pass_context
def report(ctx: click.Context, url: str, reason: str, is_scam: bool) -> None:
    """Report a job posting as a scam (or mark it as legitimate)."""
    from sentinel.db import SentinelDB
    from sentinel.knowledge import KnowledgeBase

    # Try to get our prior prediction from DB
    our_prediction = 0.0
    try:
        db = SentinelDB()
        existing = db.get_job(url)
        if existing:
            our_prediction = existing.get("scam_score", 0.0)
        db.close()
    except Exception:
        pass

    try:
        kb = KnowledgeBase()
        kb.report_scam(url, is_scam=is_scam, reason=reason, our_prediction=our_prediction)
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    verdict = "SCAM" if is_scam else "LEGITIMATE"
    data = {
        "url": url,
        "verdict": verdict.lower(),
        "reason": reason,
        "recorded": True,
    }

    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2))
        return

    icon = click.style(f"[{verdict}]", fg="red" if is_scam else "green", bold=True)
    click.echo("")
    click.echo(f"  {icon}  Report recorded.")
    click.echo(f"  URL: {url}")
    if reason:
        click.echo(f"  Reason: {reason}")
    click.echo(click.style("  Thank you — your report improves detection accuracy.", fg="cyan"))
    click.echo("")


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--type", "pattern_type",
    type=click.Choice(["red-flag", "warning", "ghost", "all"], case_sensitive=False),
    default="all",
    help="Filter patterns by type.",
)
@click.option("--status",
              type=click.Choice(["active", "candidate", "deprecated", "all"], case_sensitive=False),
              default="active",
              help="Pattern lifecycle status to show.")
@click.pass_context
def patterns(ctx: click.Context, pattern_type: str, status: str) -> None:
    """Show known scam patterns and their detection effectiveness."""
    from sentinel.db import SentinelDB

    _CATEGORY_MAP = {
        "red-flag": "red_flag",
        "warning":  "warning",
        "ghost":    "ghost_job",
    }
    category_filter = _CATEGORY_MAP.get(pattern_type)

    db = SentinelDB()
    try:
        if status == "all":
            rows = (
                db.get_patterns("active")
                + db.get_patterns("candidate")
                + db.get_patterns("deprecated")
            )
        else:
            rows = db.get_patterns(status)
    finally:
        db.close()

    if category_filter:
        rows = [r for r in rows if r.get("category") == category_filter]

    if not rows:
        msg = f"No {pattern_type} patterns found."
        if ctx.obj.get("json"):
            click.echo(json.dumps({"patterns": [], "count": 0, "message": msg}))
        else:
            click.echo(click.style(f"  {msg}", fg="yellow"))
        return

    # Sort by Bayesian score descending
    rows.sort(key=lambda r: r.get("alpha", 1.0) / (r.get("alpha", 1.0) + r.get("beta", 1.0)), reverse=True)

    if ctx.obj.get("json"):
        output = []
        for r in rows:
            total = r.get("true_positives", 0) + r.get("false_positives", 0)
            output.append({
                "pattern_id": r["pattern_id"],
                "name": r["name"],
                "description": r.get("description", ""),
                "category": r.get("category", ""),
                "status": r.get("status", "active"),
                "observations": r.get("observations", 0),
                "precision": round(r.get("true_positives", 0) / total, 3) if total else None,
                "bayesian_score": round(r.get("alpha", 1.0) / (r.get("alpha", 1.0) + r.get("beta", 1.0)), 3),
                "keywords": r.get("keywords", []),
            })
        click.echo(json.dumps({"patterns": output, "count": len(output)}, indent=2))
        return

    _CATEGORY_COLORS = {
        "red_flag":  "red",
        "warning":   "yellow",
        "ghost_job": "cyan",
        "structural": "white",
        "positive":  "green",
    }
    _STATUS_COLORS = {
        "active":     "green",
        "candidate":  "yellow",
        "deprecated": "bright_black",
    }

    click.echo("")
    click.echo(click.style(f"  Scam Patterns ({len(rows)} shown)", bold=True))
    click.echo("  " + "─" * 70)

    for r in rows:
        cat = r.get("category", "")
        cat_color = _CATEGORY_COLORS.get(cat, "white")
        cat_label = cat.replace("_", "-").upper()
        status_label = r.get("status", "active").upper()
        status_color = _STATUS_COLORS.get(r.get("status", "active"), "white")

        obs = r.get("observations", 0)
        tp = r.get("true_positives", 0)
        fp = r.get("false_positives", 0)
        total = tp + fp
        precision_str = f"{tp / total:.0%} precision" if total else "no observations"
        score = r.get("alpha", 1.0) / (r.get("alpha", 1.0) + r.get("beta", 1.0))

        click.echo(
            f"  {click.style(cat_label, fg=cat_color):<22} "
            f"{click.style(status_label, fg=status_color):<12} "
            f"{r['name']}"
        )
        click.echo(
            f"  {'':22} {'':12} "
            f"Score: {score:.2f}  |  Obs: {obs}  |  {precision_str}"
        )
        if r.get("description"):
            desc = r["description"]
            if len(desc) > 90:
                desc = desc[:87] + "..."
            click.echo(f"  {'':22} {'':12} {click.style(desc, fg='bright_black')}")
        click.echo("")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show detection statistics and accuracy metrics."""
    from sentinel.db import SentinelDB
    from sentinel.flywheel import DetectionFlywheel

    db = SentinelDB()
    try:
        raw_stats = db.get_stats()
    finally:
        db.close()

    # Accuracy details from flywheel
    try:
        fw = DetectionFlywheel()
        accuracy = fw.compute_accuracy()
        regression = fw.detect_regression()
    except Exception:
        accuracy = {}
        regression = {}

    data = {**raw_stats, "accuracy_detail": accuracy, "regression": regression}

    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2))
        return

    click.echo("")
    click.echo(click.style("  Sentinel Detection Statistics", bold=True))
    click.echo("  " + "─" * 50)

    click.echo(f"  Jobs analyzed:       {raw_stats.get('total_jobs_analyzed', 0):>8,}")
    click.echo(f"  Scams detected:      {raw_stats.get('scam_jobs_detected', 0):>8,}")
    click.echo(f"  Avg scam score:      {raw_stats.get('avg_scam_score', 0.0):>8.1%}")
    click.echo(f"  Active patterns:     {raw_stats.get('active_patterns', 0):>8,}")
    click.echo(f"  Known companies:     {raw_stats.get('total_companies', 0):>8,}")
    click.echo(f"  Verified companies:  {raw_stats.get('verified_companies', 0):>8,}")

    click.echo("")
    click.echo(click.style("  User Feedback (Learning Flywheel)", bold=True))
    click.echo("  " + "─" * 50)
    total_reports = raw_stats.get("total_user_reports", 0)
    click.echo(f"  Total reports:   {total_reports:>8,}")
    click.echo(f"  Scam reports:    {raw_stats.get('scam_reports', 0):>8,}")
    pred_acc = raw_stats.get("prediction_accuracy", 0.0)
    acc_color = "green" if pred_acc >= 0.80 else ("yellow" if pred_acc >= 0.60 else "red")
    click.echo(
        f"  Prediction acc:  {click.style(f'{pred_acc:>8.1%}', fg=acc_color)}"
    )

    if accuracy:
        click.echo(f"  Precision:       {accuracy.get('precision', 0.0):>8.1%}")
        click.echo(f"  Recall:          {accuracy.get('recall', 0.0):>8.1%}")
        click.echo(f"  F1 score:        {accuracy.get('f1', 0.0):>8.3f}")

    if regression:
        alarm = regression.get("alarm", False)
        alarm_str = click.style("YES — retraining recommended", fg="red", bold=True) if alarm \
            else click.style("No", fg="green")
        click.echo("")
        click.echo(click.style("  Regression Detector (CUSUM)", bold=True))
        click.echo("  " + "─" * 50)
        click.echo(f"  Regression alarm:  {alarm_str}")
        click.echo(f"  CUSUM statistic:   {regression.get('cusum_statistic', 0.0):>8.3f}")
        if regression.get("rolling_precision") is not None:
            click.echo(f"  Rolling precision: {regression.get('rolling_precision', 0.0):>8.1%}")

    last_cycle = raw_stats.get("last_flywheel_cycle", {})
    if last_cycle:
        click.echo("")
        click.echo(f"  Last flywheel run: {last_cycle.get('cycle_ts', 'never')}")
        click.echo(f"  Last precision:    {last_cycle.get('precision', 0.0):.1%}")

    click.echo("")


# ---------------------------------------------------------------------------
# evolve
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def evolve(ctx: click.Context) -> None:
    """Run the self-improvement cycle on detection patterns.

    This promotes high-precision patterns, deprecates low-performers,
    and runs CUSUM regression detection.
    """
    from sentinel.flywheel import DetectionFlywheel

    if not ctx.obj.get("json"):
        click.echo(click.style("  Running flywheel evolution cycle...", fg="cyan"))

    try:
        fw = DetectionFlywheel()
        result = fw.run_cycle()
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    if ctx.obj.get("json"):
        click.echo(json.dumps(result, indent=2, default=str))
        return

    click.echo("")
    click.echo(click.style("  Flywheel Cycle Complete", bold=True))
    click.echo("  " + "─" * 50)
    click.echo(f"  Cycle number:       {result.get('cycle_number', '?')}")
    click.echo(f"  Timestamp:          {result.get('cycle_ts', '')}")
    click.echo(f"  Total analyzed:     {result.get('total_analyzed', 0):,}")
    click.echo(f"  Precision:          {result.get('precision', 0.0):.1%}")
    click.echo(f"  Recall:             {result.get('recall', 0.0):.1%}")
    click.echo(f"  F1:                 {result.get('f1', 0.0):.3f}")

    promoted = result.get("patterns_promoted", [])
    deprecated = result.get("patterns_deprecated", [])

    if promoted:
        click.echo("")
        click.echo(click.style(f"  Promoted ({len(promoted)}):", fg="green", bold=True))
        for pid in promoted:
            click.echo(click.style(f"    + {pid}", fg="green"))

    if deprecated:
        click.echo("")
        click.echo(click.style(f"  Deprecated ({len(deprecated)}):", fg="yellow", bold=True))
        for pid in deprecated:
            click.echo(click.style(f"    - {pid}", fg="yellow"))

    alarm = result.get("regression_alarm", False)
    if alarm:
        click.echo("")
        click.echo(click.style(
            "  WARNING: Regression alarm triggered — precision is declining.",
            fg="red", bold=True
        ))
        click.echo(click.style(
            "  Consider reviewing pattern weights or reseeding the knowledge base.",
            fg="yellow"
        ))
    else:
        click.echo(click.style("  No regression detected.", fg="green"))

    click.echo("")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@main.command()
@click.option("--port", default=8080, show_default=True, help="HTTP port to listen on.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload (dev mode).")
def serve(port: int, host: str, reload: bool) -> None:
    """Start the Sentinel REST API server."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        click.echo(
            click.style(
                "uvicorn is not installed. Run: pip install fastapi uvicorn",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    click.echo(click.style(f"  Starting Sentinel API on http://{host}:{port}", fg="cyan", bold=True))
    click.echo(click.style("  Press CTRL+C to stop.", fg="bright_black"))
    click.echo("")

    import uvicorn
    uvicorn.run(
        "sentinel.api:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
    )


@main.command()
@click.option("--strategies", default=3, help="Max strategies per cycle")
@click.pass_context
def innovate(ctx, strategies):
    """Run the innovation flywheel — autonomous self-improvement cycle."""
    from sentinel.innovation import InnovationEngine

    engine = InnovationEngine()
    results = engine.run_cycle(max_strategies=strategies)

    if ctx.obj.get("json"):
        click.echo(json.dumps([{
            "strategy": r.strategy,
            "success": r.success,
            "detail": r.detail,
        } for r in results], indent=2))
    else:
        click.echo(click.style("Innovation Cycle Complete\n", bold=True))
        for r in results:
            icon = click.style("++", fg="green") if r.success else click.style("--", fg="red")
            click.echo(f"  {icon} {r.strategy}: {r.detail}")

        # Show strategy rankings
        rankings = engine.get_strategy_rankings()
        click.echo(click.style("\nStrategy Rankings (Thompson Sampling):", bold=True))
        for s in rankings[:5]:
            bar = "#" * int(s["mean"] * 20)
            click.echo(f"  {s['mean']:.2f} [{bar:20s}] {s['name']} ({s['attempts']} runs)")


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

@main.command()
@click.option("--query", "-q", required=True, help="Job search keywords.")
@click.option("--location", "-l", default="", help="Location filter (city, country, or 'remote').")
@click.option("--limit", default=10, show_default=True, help="Max results to fetch and analyze.")
@click.option("--no-ai", is_flag=True, default=False, help="Disable AI escalation (faster).")
@click.pass_context
def scan(ctx: click.Context, query: str, location: str, limit: int, no_ai: bool) -> None:
    """Scan LinkedIn search results and rank by scam score.

    Fetches public LinkedIn search results for the given query, analyzes each
    job posting for scam signals, and prints a table ranked by scam score
    (highest risk first).
    """
    from sentinel.analyzer import analyze_job
    from sentinel.scanner import scrape_search_results

    use_ai = not no_ai

    if not ctx.obj.get("json"):
        click.echo(click.style(
            f"  Scanning LinkedIn for: {query!r}"
            + (f" in {location!r}" if location else ""),
            fg="cyan",
        ))

    try:
        jobs = scrape_search_results(query, location=location, limit=limit)
    except ImportError as exc:
        click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    if not jobs:
        msg = "No results found (LinkedIn may have blocked the request or returned no jobs)."
        if ctx.obj.get("json"):
            click.echo(json.dumps({"query": query, "location": location, "results": [], "message": msg}))
        else:
            click.echo(click.style(f"  {msg}", fg="yellow"))
        return

    # Analyze each job
    results = []
    for job in jobs:
        try:
            result = analyze_job(job, use_ai=use_ai)
        except Exception:
            continue
        results.append(result)

    # Sort by scam score descending (highest risk first)
    results.sort(key=lambda r: r.scam_score, reverse=True)

    if ctx.obj.get("json"):
        click.echo(json.dumps(
            {
                "query": query,
                "location": location,
                "total": len(results),
                "results": [r.to_dict() for r in results],
            },
            indent=2,
        ))
        return

    # --- Formatted table ---
    click.echo("")
    click.echo(click.style(
        f"  Scan Results: {len(results)} jobs for {query!r}"
        + (f" in {location!r}" if location else ""),
        bold=True,
    ))
    click.echo("  " + "═" * 70)

    _COL_SCORE = 6
    _COL_RISK = 14
    _COL_TITLE = 28
    _COL_COMPANY = 22

    header = (
        f"  {'Score':<{_COL_SCORE}}  {'Risk':<{_COL_RISK}}  "
        f"{'Title':<{_COL_TITLE}}  {'Company':<{_COL_COMPANY}}"
    )
    click.echo(click.style(header, bold=True))
    click.echo("  " + "─" * 70)

    for r in results:
        score_str = f"{r.scam_score:.2f}"
        risk_badge = _style_risk(r.risk_level.value)
        title_str = (r.job.title or "(no title)")[:_COL_TITLE]
        company_str = (r.job.company or "(unknown)")[:_COL_COMPANY]
        click.echo(
            f"  {score_str:<{_COL_SCORE}}  {risk_badge:<{_COL_RISK}}  "
            f"{title_str:<{_COL_TITLE}}  {company_str:<{_COL_COMPANY}}"
        )

    click.echo("")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@main.command()
@click.option("--query", "-q", multiple=True, required=True, help="Search query (can specify multiple)")
@click.option("--location", "-l", default="", help="Job location filter")
@click.option("--sources", "-s", default="all", help="Comma-separated source names or 'all'")
@click.option("--limit", default=25, help="Max jobs per source")
@click.option("--no-ai", is_flag=True, help="Disable AI escalation")
@click.option("--throttle", default=1.0, help="Seconds between source fetches")
@click.pass_context
def ingest(ctx, query, location, sources, limit, no_ai, throttle):
    """Ingest jobs from external sources for flywheel training."""
    try:
        from sentinel.ingest import IngestionPipeline
    except ImportError as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    source_list = None if sources.strip().lower() == "all" else [
        s.strip() for s in sources.split(",") if s.strip()
    ]

    pipeline = IngestionPipeline()

    all_runs = []
    for q in query:
        if not ctx.obj.get("json"):
            click.echo(click.style(
                f"  Ingesting: {q!r}" + (f" in {location!r}" if location else ""),
                fg="cyan",
            ))
        try:
            run = pipeline.run(
                query=q,
                location=location,
                sources=source_list,
                limit_per_source=limit,
                use_ai=not no_ai,
                throttle_seconds=throttle,
            )
        except Exception as exc:
            if ctx.obj.get("json"):
                click.echo(json.dumps({"error": str(exc)}, indent=2))
            else:
                click.echo(click.style(f"  Error during ingest: {exc}", fg="red"), err=True)
            sys.exit(1)
        all_runs.append(run)

    if ctx.obj.get("json"):
        click.echo(json.dumps([r.to_dict() for r in all_runs], indent=2, default=str))
        return

    # Print summary table
    click.echo("")
    click.echo(click.style("  Ingestion Summary", bold=True))
    click.echo("  " + "─" * 60)
    hdr = f"  {'Query':<22}  {'Sources':<18}  {'Fetched':>7}  {'New':>5}  {'High Risk':>9}"
    click.echo(click.style(hdr, bold=True))
    click.echo("  " + "─" * 60)

    for run in all_runs:
        q_trunc = run.query[:22]
        sources_str = ", ".join(run.sources_queried)[:18] if run.sources_queried else "none"
        click.echo(
            f"  {q_trunc:<22}  {sources_str:<18}  {run.jobs_fetched:>7}  "
            f"{run.jobs_new:>5}  {run.high_risk_count:>9}"
        )

    total_fetched = sum(r.jobs_fetched for r in all_runs)
    total_new = sum(r.jobs_new for r in all_runs)
    total_high = sum(r.high_risk_count for r in all_runs)
    click.echo("  " + "─" * 60)
    click.echo(
        f"  {'TOTAL':<22}  {'':18}  {total_fetched:>7}  {total_new:>5}  {total_high:>9}"
    )
    click.echo("")


# ---------------------------------------------------------------------------
# ingest-history
# ---------------------------------------------------------------------------

@main.command("ingest-history")
@click.option("--limit", default=10, help="Number of recent runs to show")
@click.pass_context
def ingest_history(ctx, limit):
    """Show recent ingestion run history."""
    from sentinel.db import SentinelDB
    db = SentinelDB()
    try:
        rows = db.get_ingestion_history(limit=limit)
    finally:
        db.close()

    if ctx.obj.get("json"):
        click.echo(json.dumps({"runs": rows, "count": len(rows)}, indent=2, default=str))
        return

    if not rows:
        click.echo(click.style("  No ingestion runs found.", fg="yellow"))
        return

    click.echo("")
    click.echo(click.style(f"  Ingestion History (last {len(rows)} runs)", bold=True))
    click.echo("  " + "─" * 70)
    hdr = f"  {'Started':<26}  {'Query':<22}  {'Fetched':>7}  {'New':>5}  {'High Risk':>9}"
    click.echo(click.style(hdr, bold=True))
    click.echo("  " + "─" * 70)

    for row in rows:
        started = (row.get("started_at") or "")[:26]
        q_trunc = (row.get("query") or "")[:22]
        click.echo(
            f"  {started:<26}  {q_trunc:<22}  {row.get('jobs_fetched', 0):>7}  "
            f"{row.get('jobs_new', 0):>5}  {row.get('high_risk_count', 0):>9}"
        )
    click.echo("")


# ---------------------------------------------------------------------------
# auto
# ---------------------------------------------------------------------------

@main.command()
@click.option("--queries", "-q", multiple=True,
              default=("software engineer", "data analyst", "remote work from home"),
              help="Queries to run")
@click.option("--location", "-l", default="", help="Location filter")
@click.option("--flywheel/--no-flywheel", default=True, help="Run flywheel after ingestion")
@click.option("--no-ai", is_flag=True, help="Disable AI")
@click.option("--loop", "loop_count", default=1, help="Number of cycles (0=unlimited)")
@click.option("--interval", default=3600, help="Seconds between cycles")
@click.pass_context
def auto(ctx, queries, location, flywheel, no_ai, loop_count, interval):
    """Run full automated cycle: ingest from all sources -> score -> learn -> evolve."""
    # Multi-cycle mode: delegate to SentinelDaemon
    if loop_count != 1:
        from sentinel.daemon import SentinelDaemon

        if not ctx.obj.get("json"):
            click.echo(click.style(
                f"  Starting daemon loop: {loop_count or 'unlimited'} cycles, {interval}s interval",
                fg="cyan",
            ))

        daemon = SentinelDaemon(
            queries=list(queries),
            location=location,
            interval_seconds=interval,
            use_ai=not no_ai,
            max_cycles=loop_count,
        )
        results = daemon.run()

        for r in results:
            if ctx.obj.get("json"):
                import dataclasses
                click.echo(json.dumps(dataclasses.asdict(r), indent=2, default=str))
            else:
                click.echo(f"  Cycle {r.cycle_number}: "
                           f"{r.jobs_new} new jobs, "
                           f"{r.high_risk_count} high-risk, "
                           f"regression={r.regression_detected}, "
                           f"{r.duration_seconds:.1f}s")
        return

    # Single-cycle mode: original behavior
    try:
        from sentinel.ingest import IngestionPipeline
    except ImportError as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    if not ctx.obj.get("json"):
        click.echo(click.style("  Running automated ingest cycle...", fg="cyan"))

    pipeline = IngestionPipeline()

    try:
        runs = pipeline.auto_ingest(
            queries=list(queries),
            location=location,
            run_flywheel=flywheel,
        )
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error during auto ingest: {exc}", fg="red"), err=True)
        sys.exit(1)

    total_fetched = sum(r.jobs_fetched for r in runs)
    total_new = sum(r.jobs_new for r in runs)
    total_high = sum(r.high_risk_count for r in runs)

    if ctx.obj.get("json"):
        click.echo(json.dumps({
            "queries": list(queries),
            "location": location,
            "runs": [r.to_dict() for r in runs],
            "total_fetched": total_fetched,
            "total_new": total_new,
            "total_high_risk": total_high,
            "flywheel_ran": flywheel,
        }, indent=2, default=str))
        return

    click.echo("")
    click.echo(click.style("  Auto Cycle Complete", bold=True))
    click.echo("  " + "─" * 50)
    click.echo(f"  Queries run:         {len(runs):>8,}")
    click.echo(f"  Total jobs fetched:  {total_fetched:>8,}")
    click.echo(f"  New jobs added:      {total_new:>8,}")
    high_color = "red" if total_high > 0 else "green"
    click.echo(
        f"  High-risk detected:  {click.style(str(total_high).rjust(8), fg=high_color)}"
    )
    if flywheel:
        click.echo(click.style("  Flywheel cycle ran after ingestion.", fg="cyan"))

    click.echo("")


# ---------------------------------------------------------------------------
# daemon
# ---------------------------------------------------------------------------

@main.command()
@click.option("--queries", "-q", multiple=True,
              default=("software engineer", "data analyst", "remote work from home"),
              help="Search queries to run each cycle")
@click.option("--location", "-l", default="", help="Location filter")
@click.option("--interval", default=3600, help="Seconds between cycles")
@click.option("--max-cycles", default=0, help="Max cycles (0=unlimited)")
@click.option("--no-ai", is_flag=True, help="Disable AI escalation")
@click.pass_context
def daemon(ctx, queries, location, interval, max_cycles, no_ai):
    """Run Sentinel as a continuous autonomous daemon."""
    from sentinel.daemon import SentinelDaemon

    if not ctx.obj.get("json"):
        click.echo(click.style(
            f"  Sentinel daemon starting: {max_cycles or 'unlimited'} cycles, {interval}s interval",
            fg="cyan",
            bold=True,
        ))
        click.echo(click.style("  Press CTRL+C to stop gracefully.", fg="bright_black"))
        click.echo("")

    d = SentinelDaemon(
        queries=list(queries),
        location=location,
        interval_seconds=interval,
        use_ai=not no_ai,
        max_cycles=max_cycles,
    )
    results = d.run()

    if ctx.obj.get("json"):
        import dataclasses
        click.echo(json.dumps(
            [dataclasses.asdict(r) for r in results],
            indent=2,
            default=str,
        ))
    else:
        click.echo("")
        click.echo(click.style("  Daemon Summary", bold=True))
        click.echo("  " + "─" * 60)
        for r in results:
            status = click.style("REGRESSION", fg="red") if r.regression_detected \
                else click.style("OK", fg="green")
            click.echo(
                f"  Cycle {r.cycle_number:>3}: "
                f"fetched={r.jobs_fetched:>4}  new={r.jobs_new:>4}  "
                f"high_risk={r.high_risk_count:>3}  "
                f"flywheel={'Y' if r.flywheel_ran else 'N'}  "
                f"innovation={'Y' if r.innovation_ran else 'N'}  "
                f"[{status}]  {r.duration_seconds:.1f}s"
            )
        total_new = sum(r.jobs_new for r in results)
        total_high = sum(r.high_risk_count for r in results)
        click.echo("  " + "─" * 60)
        click.echo(f"  Total: {len(results)} cycles, {total_new} new jobs, {total_high} high-risk")
        click.echo("")


@main.command("innovation-report")
@click.pass_context
def innovation_report(ctx: click.Context) -> None:
    """Show per-arm meta-learning stats: precision delta, exploration vs exploitation.

    Displays how much each Thompson Sampling strategy arm has actually
    improved precision, so you can see which arms are earning their keep.
    """
    from sentinel.innovation import InnovationEngine

    engine = InnovationEngine()
    report = engine.get_meta_learning_report()

    if ctx.obj.get("json"):
        click.echo(json.dumps(report, indent=2))
        return

    total_runs = report["total_strategy_runs"]
    total_prec_runs = report["total_precision_runs"]
    total_delta = report["total_cumulative_delta"]

    click.echo("")
    click.echo(click.style("  Innovation Meta-Learning Report", bold=True))
    click.echo("  " + "═" * 70)
    click.echo(f"  Total strategy runs:     {total_runs:>6,}")
    click.echo(f"  Total precision samples: {total_prec_runs:>6,}")
    delta_color = "green" if total_delta >= 0 else "red"
    click.echo(
        f"  Cumulative Δprecision:   "
        + click.style(f"{total_delta:>+.4f}", fg=delta_color)
    )

    if report["most_effective_arm"]:
        click.echo(
            f"  Most effective arm:      "
            + click.style(report["most_effective_arm"], fg="green", bold=True)
        )
    if report["least_effective_arm"]:
        click.echo(
            f"  Least effective arm:     "
            + click.style(report["least_effective_arm"], fg="yellow")
        )
    click.echo(
        f"  Most under-explored:     "
        + click.style(report["most_under_explored"], fg="cyan")
    )

    click.echo("")
    click.echo(click.style("  Per-Arm Statistics", bold=True))
    click.echo("  " + "─" * 70)

    _HDR = (
        f"  {'Arm':<28}  {'Runs':>5}  {'Avg Δprec':>10}  "
        f"{'Cum Δprec':>10}  {'Best':>7}  {'Explore':>7}"
    )
    click.echo(click.style(_HDR, bold=True))
    click.echo("  " + "─" * 70)

    for arm in report["arms"]:
        avg = arm["avg_precision_delta"]
        cum = arm["cumulative_precision_delta"]
        best = arm["best_improvement"]
        bonus = arm["exploration_bonus"]
        runs = arm["runs"]

        avg_color = "green" if avg > 0 else ("red" if avg < 0 else "white")
        cum_color = "green" if cum > 0 else ("red" if cum < 0 else "white")

        # Highlight most/least effective
        name = arm["name"]
        if name == report["most_effective_arm"]:
            name_str = click.style(f"* {name}", fg="green", bold=True)
            name_pad = 26
        elif name == report["least_effective_arm"]:
            name_str = click.style(f"~ {name}", fg="yellow")
            name_pad = 26
        elif name == report["most_under_explored"]:
            name_str = click.style(f"? {name}", fg="cyan")
            name_pad = 26
        else:
            name_str = f"  {name}"
            name_pad = 26

        avg_str = click.style(f"{avg:>+.4f}", fg=avg_color)
        cum_str = click.style(f"{cum:>+.4f}", fg=cum_color)

        click.echo(
            f"  {name_str:<{name_pad}}  {runs:>5}  {avg_str}      "
            f"{cum_str}    {best:>+.4f}  {bonus:>7.4f}"
        )

    click.echo("")
    click.echo(click.style("  Legend:", fg="bright_black"))
    click.echo(click.style("    * most effective  ~ least effective  ? most under-explored", fg="bright_black"))
    click.echo("")


# ---------------------------------------------------------------------------
# needs-review
# ---------------------------------------------------------------------------

@main.command("needs-review")
@click.option(
    "--score-threshold", default=0.5, show_default=True, type=float,
    help="Minimum scam score to surface (jobs with score > this value).",
)
@click.option(
    "--confidence-threshold", default=0.4, show_default=True, type=float,
    help="Maximum confidence to surface (jobs with confidence < this value).",
)
@click.option("--limit", default=50, show_default=True, help="Max rows to return.")
@click.pass_context
def needs_review(ctx: click.Context, score_threshold: float, confidence_threshold: float, limit: int) -> None:
    """Surface high-score, low-confidence jobs that need human review.

    These are jobs where the model's scam score is elevated but its confidence
    is low — the signal is ambiguous and human judgment would improve accuracy.
    """
    from sentinel.db import SentinelDB

    db = SentinelDB()
    try:
        jobs = db.get_jobs_for_review(
            score_threshold=score_threshold,
            confidence_threshold=confidence_threshold,
            limit=limit,
        )
    finally:
        db.close()

    if ctx.obj.get("json"):
        click.echo(json.dumps({
            "count": len(jobs),
            "score_threshold": score_threshold,
            "confidence_threshold": confidence_threshold,
            "jobs": [
                {
                    "url": j.get("url", ""),
                    "title": j.get("title", ""),
                    "company": j.get("company", ""),
                    "scam_score": j.get("scam_score"),
                    "confidence": j.get("confidence"),
                    "risk_level": j.get("risk_level", ""),
                    "analyzed_at": j.get("analyzed_at", ""),
                }
                for j in jobs
            ],
        }, indent=2))
        return

    if not jobs:
        click.echo(click.style("  No jobs need review.", fg="green"))
        return

    click.echo("")
    click.echo(click.style(
        f"  Jobs Needing Review ({len(jobs)} found — score > {score_threshold:.0%}, confidence < {confidence_threshold:.0%})",
        bold=True,
    ))
    click.echo("  " + "═" * 72)

    _COL_SCORE = 6
    _COL_CONF = 6
    _COL_RISK = 14
    _COL_TITLE = 24
    _COL_COMPANY = 20

    header = (
        f"  {'Score':<{_COL_SCORE}}  {'Conf':<{_COL_CONF}}  {'Risk':<{_COL_RISK}}  "
        f"{'Title':<{_COL_TITLE}}  {'Company':<{_COL_COMPANY}}"
    )
    click.echo(click.style(header, bold=True))
    click.echo("  " + "─" * 72)

    for j in jobs:
        score = j.get("scam_score") or 0.0
        conf = j.get("confidence")
        conf_str = f"{conf:.2f}" if conf is not None else "N/A"
        score_str = f"{score:.2f}"
        risk_badge = _style_risk(j.get("risk_level", "suspicious"))
        title_str = (j.get("title") or "(no title)")[:_COL_TITLE]
        company_str = (j.get("company") or "(unknown)")[:_COL_COMPANY]
        click.echo(
            f"  {score_str:<{_COL_SCORE}}  {conf_str:<{_COL_CONF}}  {risk_badge:<{_COL_RISK}}  "
            f"{title_str:<{_COL_TITLE}}  {company_str:<{_COL_COMPANY}}"
        )

    click.echo("")
    click.echo(click.style(
        "  Tip: run 'sentinel report <url>' to confirm or deny these listings.",
        fg="cyan",
    ))
    click.echo("")


@main.command()
@click.pass_context
def mesh(ctx: click.Context) -> None:
    """Show the flywheel dependency graph as ASCII."""
    from sentinel.mesh import build_default_mesh

    m = build_default_mesh()

    if ctx.obj.get("json"):
        click.echo(json.dumps(m.get_dependency_graph(), indent=2))
        return

    click.echo("")
    click.echo(m.render_ascii())
    click.echo("")


@main.command()
@click.option("--history", "show_history", is_flag=True, default=False,
              help="Show recent cascade events.")
@click.option("--preview", "run_preview", is_flag=True, default=False,
              help="Run impact preview on current pending weight changes.")
@click.option("--limit", default=20, show_default=True, help="Rows to show in history.")
@click.pass_context
def cascade(ctx: click.Context, show_history: bool, run_preview: bool, limit: int) -> None:
    """Cascade impact detection — preview weight changes or review history."""
    from sentinel.db import SentinelDB
    from sentinel.mesh import CascadeDetector, build_default_mesh

    if show_history:
        db = SentinelDB()
        try:
            events = db.get_cascade_history(limit=limit)
        finally:
            db.close()

        if ctx.obj.get("json"):
            click.echo(json.dumps({"events": events, "count": len(events)}, indent=2, default=str))
            return

        if not events:
            click.echo(click.style("  No cascade events recorded yet.", fg="yellow"))
            return

        click.echo("")
        click.echo(click.style(f"  Cascade Event History ({len(events)} events)", bold=True))
        click.echo("  " + "=" * 70)
        hdr = f"  {'Timestamp':<28}  {'Trigger':<22}  {'Type':<18}"
        click.echo(click.style(hdr, bold=True))
        click.echo("  " + "-" * 70)
        for ev in events:
            ts = (ev.get("timestamp") or "")[:28]
            trigger = (ev.get("trigger") or "")[:22]
            change_type = (ev.get("change_type") or "")[:18]
            click.echo(f"  {ts:<28}  {trigger:<22}  {change_type:<18}")
        click.echo("")
        return

    if run_preview:
        from sentinel import scorer

        db = SentinelDB()
        try:
            detector = CascadeDetector(mesh=build_default_mesh())
            current_weights = scorer._load_learned_weights()
            # For preview, compare current weights against a ±0 delta (identity check)
            # In practice the caller would provide old vs new weights; here we demonstrate
            # the mechanism against the live DB sample.
            report = detector.preview_impact(db, current_weights, current_weights, sample_size=100)
        finally:
            db.close()

        if ctx.obj.get("json"):
            click.echo(json.dumps(report.to_dict(), indent=2))
            return

        risk_colors = {"SAFE": "green", "MODERATE": "yellow", "HIGH": "red"}
        risk_color = risk_colors.get(report.risk_level, "white")

        click.echo("")
        click.echo(click.style("  Cascade Impact Preview", bold=True))
        click.echo("  " + "=" * 50)
        click.echo(f"  Impact level:    {click.style(report.risk_level, fg=risk_color, bold=True)}")
        click.echo(f"  Jobs sampled:    {report.jobs_sampled}")
        click.echo(f"  Would change:    {report.classifications_changed} "
                   f"({report.change_rate:.1%} of sample)")
        click.echo(f"  Promoted:        {report.promoted_count} jobs (to higher risk)")
        click.echo(f"  Demoted:         {report.demoted_count} jobs (to lower risk)")
        click.echo(f"  Score delta:     mean={report.score_delta_mean:+.4f}  "
                   f"std={report.score_delta_std:.4f}")
        click.echo("")
        return

    # Default: show both ASCII graph + short stats
    db = SentinelDB()
    try:
        events = db.get_cascade_history(limit=5)
    finally:
        db.close()

    m = build_default_mesh()
    click.echo("")
    click.echo(click.style("  Flywheel Mesh", bold=True))
    click.echo(m.render_ascii())
    if events:
        click.echo("")
        click.echo(click.style(f"  Recent Cascade Events (last {len(events)}):", bold=True))
        for ev in events:
            ts = (ev.get("timestamp") or "")[:22]
            click.echo(f"    {ts}  {ev.get('trigger', '')}  [{ev.get('change_type', '')}]")
    click.echo("")


@main.command()
@click.pass_context
def plugins(ctx):
    """Show plugin integration status."""
    status = {"plugins": []}
    try:
        from sentinel.ecosystem import read_ecosystem_context
        context = read_ecosystem_context()
        status["plugins"].append({"name": "ecosystem", "status": "connected", "data": context})
    except ImportError:
        status["plugins"].append({"name": "ecosystem", "status": "not installed"})
    if ctx.obj.get("json"):
        click.echo(json.dumps(status, indent=2, default=str))
    else:
        click.echo(click.style("Plugin Status\n", bold=True))
        for p in status["plugins"]:
            color = "green" if p["status"] == "connected" else "yellow"
            click.echo(f"  {p['name']}: {click.style(p['status'], fg=color)}")


# ---------------------------------------------------------------------------
# Sparkline helper (exported so tests can import it)
# ---------------------------------------------------------------------------

_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 20) -> str:
    """Return a fixed-width ASCII sparkline string for *values*.

    - Empty list  -> a string of *width* spaces.
    - All-equal   -> a string of *width* lowest-block chars (index 0 = ' ').
    """
    if not values:
        return " " * width

    # Resample to exactly *width* buckets
    n = len(values)
    buckets: list[float] = []
    for i in range(width):
        idx = int(i * n / width)
        buckets.append(values[min(idx, n - 1)])

    lo = min(buckets)
    hi = max(buckets)
    span = hi - lo

    chars = []
    for v in buckets:
        if span == 0:
            char_idx = 0
        else:
            char_idx = int((v - lo) / span * (len(_SPARK_BLOCKS) - 1))
        chars.append(_SPARK_BLOCKS[char_idx])

    return "".join(chars)


# ---------------------------------------------------------------------------
# trends
# ---------------------------------------------------------------------------

@main.command()
@click.option("--days", default=30, show_default=True,
              help="Look-back window in days (7, 30, or 90).")
@click.pass_context
def trends(ctx: click.Context, days: int) -> None:
    """Show precision/recall/F1 trends, signal firing rate changes, and source quality.

    Queries flywheel_metrics over time and renders a trend report with
    sparklines for the last --days days.
    """
    from sentinel.db import SentinelDB
    from sentinel.flywheel import DetectionFlywheel

    db = SentinelDB()
    try:
        metrics_history = db.get_flywheel_metrics_history(days=days)
        source_stats = db.get_source_stats()
    finally:
        db.close()

    fw = DetectionFlywheel()
    drift = fw.detect_input_drift(window_days=7)

    if ctx.obj.get("json"):
        click.echo(json.dumps({
            "metrics_history": metrics_history,
            "source_stats": source_stats,
            "drift": drift,
            "window_days": days,
        }, indent=2, default=str))
        return

    click.echo("")
    click.echo(click.style(f"  Sentinel Trends (last {days} days)", bold=True))
    click.echo("  " + "=" * 60)

    # --- Precision / Recall / F1 ---
    if metrics_history:
        rows_asc = list(reversed(metrics_history))  # chronological order

        precision_vals = [r.get("precision", 0.0) for r in rows_asc]
        recall_vals = [r.get("recall", 0.0) for r in rows_asc]
        f1_vals = [r.get("f1", 0.0) for r in rows_asc]

        latest = rows_asc[-1]
        first = rows_asc[0]
        prec_delta = latest.get("precision", 0.0) - first.get("precision", 0.0)
        recall_delta = latest.get("recall", 0.0) - first.get("recall", 0.0)
        f1_delta = latest.get("f1", 0.0) - first.get("f1", 0.0)

        def _delta_style(d: float) -> str:
            arrow = "+" if d >= 0 else ""
            color = "green" if d > 0.005 else ("red" if d < -0.005 else "yellow")
            return click.style(f"{arrow}{d:+.3f}", fg=color)

        click.echo("")
        click.echo(click.style("  Model Metrics", bold=True))
        click.echo("  " + "-" * 60)
        click.echo(
            f"  {'Metric':<12}  {'Current':>8}  {'Change':>8}  Sparkline"
        )
        click.echo("  " + "-" * 60)
        click.echo(
            f"  {'Precision':<12}  {latest.get('precision', 0.0):>8.1%}  "
            f"{_delta_style(prec_delta):>8}  {_sparkline(precision_vals, 20)}"
        )
        click.echo(
            f"  {'Recall':<12}  {latest.get('recall', 0.0):>8.1%}  "
            f"{_delta_style(recall_delta):>8}  {_sparkline(recall_vals, 20)}"
        )
        click.echo(
            f"  {'F1':<12}  {latest.get('f1', 0.0):>8.3f}  "
            f"{_delta_style(f1_delta):>8}  {_sparkline(f1_vals, 20)}"
        )
        click.echo(f"\n  Cycles in window: {len(metrics_history)}")
    else:
        click.echo("")
        click.echo(click.style("  No flywheel cycle data in this window.", fg="yellow"))

    # --- Signal firing rate changes from drift detector ---
    click.echo("")
    click.echo(click.style("  Signal Firing Rate Changes (last 7 days)", bold=True))
    click.echo("  " + "-" * 60)

    drift_signals = drift.get("changed_signals", [])
    if drift_signals:
        click.echo(f"  {'Signal':<30}  {'Baseline':>9}  {'Recent':>9}  {'Delta':>8}")
        click.echo("  " + "-" * 60)
        for sig in drift_signals[:10]:
            delta = sig["delta"]
            delta_color = "red" if delta > 0.02 else ("green" if delta < -0.02 else "white")
            arrow = "+" if delta >= 0 else ""
            click.echo(
                f"  {sig['signal']:<30}  {sig['baseline_rate']:>9.4f}  "
                f"{sig['recent_rate']:>9.4f}  "
                + click.style(f"{arrow}{delta:>+.4f}", fg=delta_color)
            )
    else:
        click.echo(click.style("  No signal rate history available yet.", fg="yellow"))

    # --- Drift status ---
    click.echo("")
    click.echo(click.style("  Input Drift Status", bold=True))
    click.echo("  " + "-" * 60)
    alarm = drift.get("alarm", False)
    score = drift.get("drift_score", 0.0)
    drift_color = "red" if alarm else "green"
    alarm_str = click.style("DRIFT DETECTED", fg="red", bold=True) if alarm \
        else click.style("Stable", fg="green")
    click.echo(f"  Status:    {alarm_str}")
    click.echo(f"  JSD Score: {click.style(f'{score:.4f}', fg=drift_color)}")
    click.echo(f"  chi2:      {drift.get('chi2_statistic', 0.0):.4f}")

    # --- Source quality ---
    if source_stats:
        click.echo("")
        click.echo(click.style("  Source Quality", bold=True))
        click.echo("  " + "-" * 60)
        click.echo(f"  {'Source':<18}  {'Ingested':>9}  {'Scams':>7}  {'Avg Score':>10}")
        click.echo("  " + "-" * 60)
        for s in source_stats[:8]:
            click.echo(
                f"  {(s.get('source') or 'unknown')[:18]:<18}  "
                f"{s.get('jobs_ingested', 0):>9}  "
                f"{s.get('scams_detected', 0):>7}  "
                f"{s.get('avg_score', 0.0):>10.1%}"
            )

    click.echo("")


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------

@main.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """One-stop health dashboard: precision, drift, shadow tests, daemon status.

    Shows current model performance, regression alarm state, input drift
    detection, active shadow A/B tests, innovation arm stats, and source
    circuit breaker states.
    """
    from sentinel.db import SentinelDB
    from sentinel.flywheel import DetectionFlywheel

    db = SentinelDB()
    try:
        source_stats = db.get_source_stats()
        active_shadow = db.get_active_shadow_run()
    finally:
        db.close()

    fw = DetectionFlywheel()
    health_data = fw.get_health()
    drift = fw.detect_input_drift(window_days=7)

    # Innovation arm stats (best-effort)
    innovation_stats: list[dict] = []
    try:
        from sentinel.innovation import InnovationEngine
        engine = InnovationEngine()
        innovation_stats = engine.get_strategy_rankings()
    except Exception:
        pass

    if ctx.obj.get("json"):
        click.echo(json.dumps({
            "health": health_data,
            "drift": drift,
            "active_shadow": active_shadow,
            "source_stats": source_stats,
            "innovation_stats": innovation_stats,
        }, indent=2, default=str))
        return

    # ---- Formatted dashboard ----
    grade = health_data.get("grade", "?")
    grade_colors = {"A": "green", "B": "green", "C": "yellow", "D": "red", "F": "bright_red"}
    grade_color = grade_colors.get(grade, "white")

    click.echo("")
    click.echo(click.style("  Sentinel Health Dashboard", bold=True))
    click.echo("  " + "=" * 60)

    # -- Model Metrics --
    precision = health_data.get("precision", 0.0)
    recall = health_data.get("recall", 0.0)
    f1 = health_data.get("f1", 0.0)
    prec_color = "green" if precision >= 0.80 else ("yellow" if precision >= 0.60 else "red")

    click.echo("")
    click.echo(click.style("  Model Metrics", bold=True))
    click.echo("  " + "-" * 60)
    click.echo(
        f"  Grade:       {click.style(f'[{grade}]', fg=grade_color, bold=True)}"
    )
    click.echo(
        f"  Precision:   {click.style(f'{precision:.1%}', fg=prec_color, bold=True)}"
        f"  |  Recall: {recall:.1%}  |  F1: {f1:.3f}"
    )
    click.echo(
        f"  Jobs:        {health_data.get('total_jobs_analyzed', 0):,} analyzed  |  "
        f"{health_data.get('total_user_reports', 0):,} user reports"
    )

    # -- Regression --
    click.echo("")
    click.echo(click.style("  Regression (CUSUM)", bold=True))
    click.echo("  " + "-" * 60)
    reg_alarm = health_data.get("regression_alarm", False)
    reg_str = click.style("ALARM — retrain recommended", fg="red", bold=True) if reg_alarm \
        else click.style("OK", fg="green")
    click.echo(f"  Status:    {reg_str}")
    click.echo(f"  CUSUM:     {health_data.get('cusum_statistic', 0.0):.4f}")

    # -- Drift --
    click.echo("")
    click.echo(click.style("  Drift Detection", bold=True))
    click.echo("  " + "-" * 60)
    drift_alarm = drift.get("alarm", False)
    drift_str = click.style("DRIFT DETECTED", fg="red", bold=True) if drift_alarm \
        else click.style("Stable", fg="green")
    click.echo(f"  Input Drift: {drift_str}  (JSD={drift.get('drift_score', 0.0):.4f})")
    if drift_alarm and drift.get("changed_signals"):
        top_sig = drift["changed_signals"][0]
        click.echo(
            f"  Top change:  {top_sig['signal']}  "
            f"({top_sig['baseline_rate']:.3f} -> {top_sig['recent_rate']:.3f})"
        )

    # -- Patterns --
    click.echo("")
    click.echo(click.style("  Patterns", bold=True))
    click.echo("  " + "-" * 60)
    click.echo(
        f"  Active:     {health_data.get('active_patterns', 0):>5}  |  "
        f"Candidate: {health_data.get('candidate_patterns', 0):>5}  |  "
        f"Deprecated: {health_data.get('deprecated_patterns', 0):>5}"
    )

    # -- Shadow Tests --
    click.echo("")
    click.echo(click.style("  Shadow Tests (A/B Weight Testing)", bold=True))
    click.echo("  " + "-" * 60)
    if active_shadow:
        run_id = active_shadow.get("id", "?")
        base_p = active_shadow.get("baseline_precision", 0.0)
        shad_p = active_shadow.get("shadow_precision", 0.0)
        jobs_eval = active_shadow.get("jobs_evaluated", 0)
        delta = shad_p - base_p
        delta_color = "green" if delta > 0 else ("red" if delta < 0 else "yellow")
        click.echo(click.style(f"  Active shadow run #{run_id}", fg="cyan", bold=True))
        click.echo(
            f"  Baseline: {base_p:.1%}  |  Shadow: {shad_p:.1%}  |  "
            + click.style(f"Delta: {delta:+.1%}", fg=delta_color)
        )
        click.echo(f"  Jobs evaluated: {jobs_eval}")
    else:
        click.echo(click.style("  No active shadow run.", fg="bright_black"))

    # -- Innovation Arms --
    if innovation_stats:
        click.echo("")
        click.echo(click.style("  Innovation Arms (Top 3)", bold=True))
        click.echo("  " + "-" * 60)
        for s in innovation_stats[:3]:
            bar = "#" * int(s.get("mean", 0.0) * 15)
            click.echo(
                f"  {s.get('mean', 0.0):.2f} [{bar:<15s}] "
                f"{s.get('name', '?')} ({s.get('attempts', 0)} runs)"
            )

    # -- Source quality --
    if source_stats:
        click.echo("")
        click.echo(click.style("  Source Quality", bold=True))
        click.echo("  " + "-" * 60)
        for s in source_stats[:5]:
            src = (s.get("source") or "unknown")[:18]
            ingested = s.get("jobs_ingested", 0)
            scams = s.get("scams_detected", 0)
            rate = scams / ingested if ingested > 0 else 0.0
            rate_color = "red" if rate > 0.3 else "yellow" if rate > 0.1 else "green"
            click.echo(
                f"  {src:<20}  {ingested:>6} jobs  "
                + click.style(f"{rate:.1%} scam rate", fg=rate_color)
            )

    click.echo("")
    click.echo(click.style(
        f"  Checked at: {health_data.get('checked_at', 'N/A')}",
        fg="bright_black",
    ))
    click.echo("")


# ---------------------------------------------------------------------------
# research
# ---------------------------------------------------------------------------

@main.command()
@click.option("--topic", "-t", default="", help="Run targeted research on a specific topic.")
@click.option("--auto", "auto_mode", is_flag=True, default=False,
              help="Let the engine automatically pick topics based on detection gaps.")
@click.option("--history", "show_history", is_flag=True, default=False,
              help="Show past research runs and their impact.")
@click.option("--topics", "show_topics", is_flag=True, default=False,
              help="Show prioritized research topic list.")
@click.option("--budget", default=2, show_default=True,
              help="Number of research prompts to run per cycle.")
@click.pass_context
def research(
    ctx: click.Context,
    topic: str,
    auto_mode: bool,
    show_history: bool,
    show_topics: bool,
    budget: int,
) -> None:
    """Run the research flywheel to discover new fraud detection knowledge.

    Use --auto to let the engine pick the most impactful topics, or --topic
    to target a specific area. Use --history or --topics to inspect past runs.
    """
    from sentinel.db import SentinelDB
    from sentinel.research import ResearchEngine, ResearchTopic

    db = SentinelDB()

    # --- Show history ---
    if show_history:
        rows = db.get_research_history(limit=20)
        db.close()

        if ctx.obj.get("json"):
            click.echo(json.dumps({"runs": rows, "count": len(rows)}, indent=2, default=str))
            return

        if not rows:
            click.echo(click.style("  No research history found.", fg="yellow"))
            return

        click.echo("")
        click.echo(click.style(f"  Research History ({len(rows)} runs)", bold=True))
        click.echo("  " + "=" * 70)
        click.echo(click.style(
            f"  {'Timestamp':<22}  {'Topic':<22}  {'Extracted':>9}  {'Adopted':>7}  {'Delta':>8}",
            bold=True,
        ))
        click.echo("  " + "-" * 70)

        for r in rows:
            ts = (r.get("timestamp") or "")[:22]
            tp = (r.get("topic") or "unknown")[:22]
            extracted = r.get("patterns_extracted", 0)
            adopted = r.get("patterns_adopted", 0)
            delta = r.get("precision_delta", 0.0)
            delta_color = "green" if delta > 0 else ("red" if delta < 0 else "white")
            click.echo(
                f"  {ts:<22}  {tp:<22}  {extracted:>9}  {adopted:>7}  "
                + click.style(f"{delta:>+.4f}", fg=delta_color)
            )

        click.echo("")
        return

    # --- Show topics ---
    if show_topics:
        engine = ResearchEngine(db=db, research_budget=budget)
        topics_list = engine.prioritize_next_research()
        db.close()

        if ctx.obj.get("json"):
            click.echo(json.dumps({
                "topics": [
                    {
                        "area": t.area,
                        "priority": t.priority,
                        "reason": t.reason,
                        "last_researched": t.last_researched,
                    }
                    for t in topics_list
                ],
                "count": len(topics_list),
            }, indent=2, default=str))
            return

        if not topics_list:
            click.echo(click.style("  No research topics identified.", fg="green"))
            return

        click.echo("")
        click.echo(click.style(f"  Research Topics ({len(topics_list)} identified)", bold=True))
        click.echo("  " + "=" * 70)
        click.echo(click.style(
            f"  {'Priority':>8}  {'Area':<28}  {'Reason':<32}",
            bold=True,
        ))
        click.echo("  " + "-" * 70)

        for t in topics_list:
            prio = t.priority
            prio_color = "red" if prio >= 0.7 else ("yellow" if prio >= 0.4 else "green")
            reason_trunc = t.reason[:32]
            click.echo(
                f"  {click.style(f'{prio:>8.2f}', fg=prio_color)}  "
                f"{t.area:<28}  {reason_trunc:<32}"
            )

        click.echo("")
        return

    # --- Run targeted research ---
    if topic:
        if not ctx.obj.get("json"):
            click.echo(click.style(f"  Researching topic: {topic!r}...", fg="cyan"))

        engine = ResearchEngine(db=db, research_budget=1)
        research_topic = ResearchTopic(
            area=topic.replace(" ", "_"),
            priority=0.9,
            reason=f"User-requested research on '{topic}'",
        )
        prompts = engine.generate_research_prompts([research_topic])
        results = []
        for prompt in prompts:
            result = engine.execute_research(prompt)
            if result.extracted_patterns:
                engine.integrate_findings(result.extracted_patterns)
                engine.db.insert_research_run({
                    "topic": research_topic.area,
                    "prompt": prompt.prompt_text[:500],
                    "response_summary": result.raw_response[:500],
                    "patterns_extracted": len(result.extracted_patterns),
                    "patterns_adopted": len([
                        p for p in result.extracted_patterns
                        if p.get("type") in ("keyword", "evasion_tactic", "new_category")
                    ]),
                })
            results.append(result)

        db.close()

        if ctx.obj.get("json"):
            click.echo(json.dumps({
                "topic": topic,
                "results": [
                    {
                        "patterns_extracted": len(r.extracted_patterns),
                        "confidence": r.confidence,
                        "raw_response_length": len(r.raw_response),
                    }
                    for r in results
                ],
            }, indent=2, default=str))
            return

        for r in results:
            click.echo("")
            click.echo(click.style("  Research Complete", bold=True))
            click.echo("  " + "-" * 50)
            click.echo(f"  Topic:      {topic}")
            click.echo(f"  Patterns:   {len(r.extracted_patterns)}")
            click.echo(f"  Confidence: {r.confidence:.0%}")
            if r.extracted_patterns:
                click.echo("")
                click.echo(click.style("  Extracted Patterns:", bold=True))
                for p in r.extracted_patterns[:10]:
                    ptype = p.get("type", "unknown")
                    keyword = p.get("keyword", p.get("signal_name", ""))
                    icon = "+" if ptype == "keyword" else ("!" if ptype == "evasion_tactic" else "~")
                    click.echo(f"    {icon} [{ptype}] {keyword}")
            elif not r.raw_response:
                click.echo(click.style(
                    "  No AI response received (AI may be disabled or unavailable).",
                    fg="yellow",
                ))

        click.echo("")
        return

    # --- Auto mode ---
    if auto_mode:
        if not ctx.obj.get("json"):
            click.echo(click.style("  Running autonomous research cycle...", fg="cyan"))

        engine = ResearchEngine(db=db, research_budget=budget)
        results = engine.run_cycle()
        report = engine.get_report()
        db.close()

        if ctx.obj.get("json"):
            click.echo(json.dumps({
                "results": [
                    {
                        "topic": r.topic.area,
                        "patterns_extracted": len(r.extracted_patterns),
                        "confidence": r.confidence,
                    }
                    for r in results
                ],
                "report": report,
            }, indent=2, default=str))
            return

        click.echo("")
        click.echo(click.style("  Research Cycle Complete", bold=True))
        click.echo("  " + "=" * 50)

        if not results:
            click.echo(click.style("  No topics above research threshold.", fg="green"))
        else:
            for r in results:
                extracted = len(r.extracted_patterns)
                topic_label = r.topic.area[:30]
                icon = click.style("++", fg="green") if extracted > 0 else click.style("--", fg="yellow")
                click.echo(
                    f"  {icon} {topic_label}: {extracted} patterns, "
                    f"confidence={r.confidence:.0%}"
                )

        click.echo("")
        click.echo(click.style("  Cumulative Stats:", bold=True))
        click.echo(f"  Total runs:       {report.get('total_research_runs', 0)}")
        click.echo(f"  Patterns found:   {report.get('total_patterns_extracted', 0)}")
        click.echo(f"  Patterns adopted: {report.get('total_patterns_adopted', 0)}")
        click.echo(f"  Adoption rate:    {report.get('adoption_rate', 0.0):.0%}")

        # Show top prompt templates
        rankings = report.get("prompt_template_rankings", [])
        if rankings:
            click.echo("")
            click.echo(click.style("  Prompt Template Rankings:", bold=True))
            for t in rankings[:4]:
                bar = "#" * int(t.get("mean", 0.5) * 10)
                click.echo(
                    f"    {t.get('mean', 0.5):.2f} [{bar:<10s}] "
                    f"{t.get('template_id', '?')} ({t.get('uses', 0)} uses)"
                )

        click.echo("")
        return

    # No action specified
    click.echo(click.style(
        "  Specify --auto, --topic <name>, --history, or --topics. "
        "Run 'sentinel research --help' for details.",
        fg="yellow",
    ))


# ---------------------------------------------------------------------------
# cortex
# ---------------------------------------------------------------------------

@main.command()
@click.option("--investigations", "show_investigations", is_flag=True, default=False,
              help="Show open investigations.")
@click.option("--signals", "show_signals", is_flag=True, default=False,
              help="Show recent cross-system signals.")
@click.option("--velocity", "show_velocity", is_flag=True, default=False,
              help="Show learning velocity chart.")
@click.pass_context
def cortex(
    ctx: click.Context,
    show_investigations: bool,
    show_signals: bool,
    show_velocity: bool,
) -> None:
    """Show the Cortex meta-cognitive intelligence layer status.

    Displays overall system health, strategic mode, learning velocity,
    active investigations, and cross-system signal routing.
    """
    from sentinel.cortex import Cortex
    from sentinel.db import SentinelDB

    db = SentinelDB()
    try:
        cx = Cortex(db=db)
        report = cx.generate_report()
    finally:
        db.close()

    if ctx.obj.get("json"):
        click.echo(json.dumps(report, indent=2, default=str))
        return

    # --- Investigations sub-view ---
    if show_investigations:
        investigations = report.get("active_investigations", [])
        if not investigations:
            click.echo(click.style("  No open investigations.", fg="green"))
            return

        click.echo("")
        click.echo(click.style(
            f"  Cortex Investigations ({len(investigations)} active)", bold=True,
        ))
        click.echo("  " + "=" * 70)

        for inv in investigations:
            status = inv.get("status", "open")
            status_color = (
                "yellow" if status == "open"
                else ("red" if status == "stale" else "green")
            )
            click.echo("")
            click.echo(
                f"  {click.style(f'[{status.upper()}]', fg=status_color)}  "
                f"{inv.get('id', '?')}"
            )
            click.echo(f"  Trigger:    {inv.get('trigger', 'unknown')}")
            click.echo(f"  Hypothesis: {inv.get('hypothesis', '')[:80]}")
            click.echo(f"  Open for:   {inv.get('cycles_open', 0)} cycle(s)")
            if inv.get("resolution"):
                click.echo(f"  Resolution: {inv['resolution'][:80]}")

        click.echo("")
        return

    # --- Signals sub-view ---
    if show_signals:
        signals = report.get("recent_signals", [])
        if not signals:
            click.echo(click.style("  No recent cortex signals.", fg="green"))
            return

        click.echo("")
        click.echo(click.style(f"  Cortex Signals ({len(signals)} recent)", bold=True))
        click.echo("  " + "=" * 70)
        click.echo(click.style(
            f"  {'Timestamp':<22}  {'Source':<12}  {'Target':<12}  "
            f"{'Type':<24}  {'Prio':>4}",
            bold=True,
        ))
        click.echo("  " + "-" * 70)

        for sig in signals:
            ts = (sig.get("timestamp") or "")[:22]
            source = (sig.get("source") or "?")[:12]
            target = (sig.get("target") or "?")[:12]
            sig_type = (sig.get("signal_type") or "?")[:24]
            prio = sig.get("priority", 0.0)
            prio_color = (
                "red" if prio >= 0.8
                else ("yellow" if prio >= 0.5 else "green")
            )
            click.echo(
                f"  {ts:<22}  {source:<12}  {target:<12}  {sig_type:<24}  "
                + click.style(f"{prio:>4.1f}", fg=prio_color)
            )

        click.echo("")
        return

    # --- Velocity sub-view ---
    if show_velocity:
        precision_history = report.get("precision_history", [])
        velocity = report.get("learning_velocity", 0.0)
        trend = report.get("velocity_trend", "stable")

        click.echo("")
        click.echo(click.style("  Learning Velocity", bold=True))
        click.echo("  " + "=" * 50)

        vel_color = (
            "green" if velocity > 0.005
            else ("red" if velocity < -0.005 else "yellow")
        )
        trend_arrows = {"improving": "+", "declining": "-", "stable": "="}
        arrow = trend_arrows.get(trend, "?")

        click.echo(
            f"  Velocity:  {click.style(f'{velocity:+.4f}/cycle', fg=vel_color)}"
            f"  [{arrow}] {trend}"
        )

        if precision_history:
            spark = _sparkline(precision_history, 30)
            click.echo(f"  Precision: {spark}")
            click.echo(
                f"  Range:     {min(precision_history):.3f} - "
                f"{max(precision_history):.3f}"
            )
            click.echo(f"  Samples:   {len(precision_history)}")

        click.echo("")
        return

    # --- Default: full cortex dashboard ---
    grade = report.get("health_grade", "?")
    grade_colors = {
        "A": "green", "B": "green", "C": "yellow",
        "D": "red", "F": "bright_red",
    }
    grade_color = grade_colors.get(grade, "white")

    click.echo("")
    click.echo(click.style("  Cortex Intelligence Layer", bold=True))
    click.echo("  " + "=" * 60)

    # Health grade
    click.echo("")
    click.echo(click.style("  System Health", bold=True))
    click.echo("  " + "-" * 60)
    click.echo(
        f"  Overall Grade:  "
        f"{click.style(f'[{grade}]', fg=grade_color, bold=True)}"
        f"  ({report.get('avg_health_score', 0.0):.0%})"
    )

    # Per-subsystem health
    subsystem_health = report.get("subsystem_health", {})
    for name, score in sorted(subsystem_health.items(), key=lambda x: x[1]):
        bar_len = int(score * 20)
        bar = "#" * bar_len
        bar_color = (
            "green" if score >= 0.7
            else ("yellow" if score >= 0.4 else "red")
        )
        click.echo(
            f"  {name:<14}  "
            f"{click.style(f'{score:.0%}', fg=bar_color):>6}  "
            f"[{click.style(f'{bar:<20s}', fg=bar_color)}]"
        )

    # Strategic mode
    click.echo("")
    click.echo(click.style("  Strategic Mode", bold=True))
    click.echo("  " + "-" * 60)
    mode = report.get("strategic_mode", "OBSERVE")
    mode_colors = {
        "EXPAND": "green", "STABILIZE": "red",
        "INVESTIGATE": "yellow", "OPTIMIZE": "cyan",
        "OBSERVE": "white",
    }
    click.echo(
        f"  Mode:       "
        f"{click.style(mode, fg=mode_colors.get(mode, 'white'), bold=True)}"
    )
    priorities = report.get("strategic_priorities", [])
    if len(priorities) > 1:
        click.echo(f"  Priorities: {' > '.join(priorities)}")

    # Learning velocity
    velocity = report.get("learning_velocity", 0.0)
    trend = report.get("velocity_trend", "stable")
    vel_color = (
        "green" if velocity > 0.005
        else ("red" if velocity < -0.005 else "yellow")
    )
    click.echo("")
    click.echo(click.style("  Learning Velocity", bold=True))
    click.echo("  " + "-" * 60)
    click.echo(
        f"  Velocity:   "
        f"{click.style(f'{velocity:+.4f}/cycle', fg=vel_color)}"
        f"  [{trend}]"
    )

    # Investigations summary
    investigations = report.get("active_investigations", [])
    click.echo("")
    click.echo(click.style("  Investigations", bold=True))
    click.echo("  " + "-" * 60)
    open_count = sum(
        1 for i in investigations if i.get("status") == "open"
    )
    stale_count = sum(
        1 for i in investigations if i.get("status") == "stale"
    )
    if investigations:
        click.echo(f"  Open: {open_count}  |  Stale: {stale_count}")
        for inv in investigations[:3]:
            status = inv.get("status", "open")
            status_color = "yellow" if status == "open" else "red"
            click.echo(
                f"    {click.style(f'[{status.upper()}]', fg=status_color)} "
                f"{inv.get('trigger', '?')} — "
                f"{inv.get('hypothesis', '')[:50]}"
            )
    else:
        click.echo(click.style("  No active investigations.", fg="green"))

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        click.echo("")
        click.echo(click.style("  Recommendations", bold=True))
        click.echo("  " + "-" * 60)
        for rec in recs[:5]:
            if rec.startswith("URGENT"):
                click.echo(
                    click.style(f"  ! {rec}", fg="red", bold=True)
                )
            elif rec.startswith("WARNING"):
                click.echo(click.style(f"  ~ {rec}", fg="yellow"))
            else:
                click.echo(f"    {rec}")

    click.echo("")
    click.echo(click.style(
        f"  Cycle: {report.get('cycle_number', 0)}",
        fg="bright_black",
    ))
    click.echo("")


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------

@main.command()
@click.option("--stats", "show_stats", is_flag=True, default=False,
              help="Show feedback pipeline statistics.")
@click.option("--import", "import_file", default=None, type=click.Path(),
              help="Import labeled data from a CSV or JSON file.")
@click.option("--rescan", "do_rescan", is_flag=True, default=False,
              help="Re-analyze recently scored jobs and detect score drift.")
@click.option("--synthetic", "do_synthetic", is_flag=True, default=False,
              help="Generate synthetic feedback for high-confidence predictions.")
@click.option("--days", default=7, show_default=True,
              help="Look-back window in days for --rescan.")
@click.option("--sample-size", default=50, show_default=True,
              help="Max jobs to rescan per run.")
@click.option("--n", "synthetic_n", default=20, show_default=True,
              help="Max synthetic reports to generate.")
@click.pass_context
def feedback(
    ctx: click.Context,
    show_stats: bool,
    import_file: str | None,
    do_rescan: bool,
    do_synthetic: bool,
    days: int,
    sample_size: int,
    synthetic_n: int,
) -> None:
    """Active learning feedback pipeline.

    Use --stats to inspect coverage, --rescan to detect score drift, or
    --import <file> to bulk-import labeled training data.
    """
    from sentinel.db import SentinelDB
    from sentinel.feedback import FeedbackPipeline

    db = SentinelDB()
    pipeline = FeedbackPipeline(db=db)

    # --- Stats ---
    if show_stats:
        stats = pipeline.get_feedback_stats()
        if ctx.obj.get("json"):
            click.echo(json.dumps(stats, indent=2))
            return

        click.echo("")
        click.echo(click.style("  Feedback Pipeline Statistics", bold=True))
        click.echo("  " + "─" * 50)
        click.echo(f"  Total reports:       {stats.get('total_reports', 0):>8,}")
        click.echo(f"    Manual:            {stats.get('manual_reports', 0):>8,}")
        click.echo(f"    Synthetic:         {stats.get('synthetic_reports', 0):>8,}")
        click.echo(f"    Imported:          {stats.get('imported_reports', 0):>8,}")
        click.echo("")
        scored = stats.get("total_scored_jobs", 0)
        reported = stats.get("reported_jobs", 0)
        coverage = stats.get("feedback_coverage", 0.0)
        cov_color = "green" if coverage >= 0.1 else ("yellow" if coverage >= 0.01 else "red")
        click.echo(f"  Scored jobs:         {scored:>8,}")
        click.echo(f"  Reported jobs:       {reported:>8,}")
        click.echo(
            f"  Coverage:            "
            + click.style(f"{coverage:>7.1%}", fg=cov_color)
        )

        trend = stats.get("reports_per_day", [])
        if trend:
            click.echo("")
            click.echo(click.style("  Reports/day (last 7 days):", bold=True))
            for i, count in enumerate(trend):
                bar = "#" * min(count, 40)
                click.echo(f"    Day -{len(trend)-i-1}: {bar} ({count})")

        click.echo("")
        return

    # --- Import labeled data ---
    if import_file:
        if not ctx.obj.get("json"):
            click.echo(click.style(f"  Importing labeled data from: {import_file}", fg="cyan"))
        result = pipeline.import_labeled_data(import_file)

        if ctx.obj.get("json"):
            click.echo(json.dumps({
                "filepath": result.filepath,
                "rows_read": result.rows_read,
                "rows_imported": result.rows_imported,
                "rows_skipped": result.rows_skipped,
                "errors": result.errors,
            }, indent=2))
            return

        click.echo("")
        click.echo(click.style("  Import Complete", bold=True))
        click.echo("  " + "─" * 50)
        click.echo(f"  File:      {result.filepath}")
        click.echo(f"  Read:      {result.rows_read:,}")
        imported_color = "green" if result.rows_imported > 0 else "yellow"
        click.echo(f"  Imported:  {click.style(str(result.rows_imported), fg=imported_color)}")
        click.echo(f"  Skipped:   {result.rows_skipped:,}")
        if result.errors:
            click.echo("")
            click.echo(click.style(f"  Errors ({len(result.errors)}):", fg="yellow"))
            for err in result.errors[:5]:
                click.echo(f"    {err}")
            if len(result.errors) > 5:
                click.echo(f"    ... and {len(result.errors) - 5} more")
        click.echo("")
        return

    # --- Rescan ---
    if do_rescan:
        if not ctx.obj.get("json"):
            click.echo(click.style(
                f"  Rescanning up to {sample_size} jobs from the last {days} days...",
                fg="cyan",
            ))
        result = pipeline.rescan_and_compare(days=days, sample_size=sample_size)

        if ctx.obj.get("json"):
            click.echo(json.dumps({
                "jobs_rescanned": result.jobs_rescanned,
                "jobs_drifted": result.jobs_drifted,
                "drift_threshold": result.drift_threshold,
                "avg_delta": result.avg_delta,
                "max_delta": result.max_delta,
                "drifted_urls": result.drifted_urls,
            }, indent=2))
            return

        click.echo("")
        click.echo(click.style("  Rescan Complete", bold=True))
        click.echo("  " + "─" * 50)
        click.echo(f"  Jobs rescanned:   {result.jobs_rescanned:,}")
        drifted_color = "red" if result.jobs_drifted > 0 else "green"
        click.echo(
            f"  Score drift (>{result.drift_threshold:.0%}):  "
            + click.style(str(result.jobs_drifted), fg=drifted_color)
        )
        if result.jobs_rescanned > 0:
            click.echo(f"  Avg delta:        {result.avg_delta:.3f}")
            click.echo(f"  Max delta:        {result.max_delta:.3f}")
        if result.drifted_urls:
            click.echo("")
            click.echo(click.style("  Drifted jobs:", bold=True))
            for url in result.drifted_urls[:5]:
                click.echo(f"    {url}")
            if len(result.drifted_urls) > 5:
                click.echo(f"    ... and {len(result.drifted_urls) - 5} more")
        click.echo("")
        return

    # --- Synthetic feedback ---
    if do_synthetic:
        if not ctx.obj.get("json"):
            click.echo(click.style(
                f"  Generating up to {synthetic_n} synthetic feedback reports...",
                fg="cyan",
            ))
        reports = pipeline.generate_synthetic_feedback(n=synthetic_n)

        if ctx.obj.get("json"):
            click.echo(json.dumps([{
                "url": r.url,
                "is_scam": r.is_scam,
                "our_prediction": r.our_prediction,
                "confidence": r.confidence,
                "reason": r.reason,
            } for r in reports], indent=2))
            return

        click.echo("")
        click.echo(click.style("  Synthetic Feedback Generated", bold=True))
        click.echo("  " + "─" * 50)
        scam_count = sum(1 for r in reports if r.is_scam)
        legit_count = len(reports) - scam_count
        click.echo(f"  Total generated: {len(reports):,}")
        click.echo(click.style(f"  Scam reports:    {scam_count}", fg="red" if scam_count else "white"))
        click.echo(click.style(f"  Legit reports:   {legit_count}", fg="green" if legit_count else "white"))
        click.echo("")
        return

    # No action
    click.echo(click.style(
        "  Specify --stats, --rescan, --synthetic, or --import <file>. "
        "Run 'sentinel feedback --help' for details.",
        fg="yellow",
    ))


# ---------------------------------------------------------------------------
# deep-analyze
# ---------------------------------------------------------------------------

@main.command("deep-analyze")
@click.argument("input_text")
@click.pass_context
def deep_analyze(ctx: click.Context, input_text: str) -> None:
    """Run a deep analysis using Nexus if available, else regular analyze."""
    try:
        from sentinel.nexus import Nexus  # type: ignore[import]
        nexus = Nexus()
        result = nexus.deep_analyze(input_text)
        if ctx.obj.get("json"):
            click.echo(json.dumps(result if isinstance(result, dict) else vars(result),
                                  indent=2, default=str))
        else:
            click.echo(click.style("  Deep Analysis (Nexus)", bold=True))
            click.echo(f"  {result}")
    except (ImportError, AttributeError):
        # Nexus not available — fall back to regular analyze
        from sentinel.analyzer import analyze_text, analyze_url
        is_url = input_text.startswith("http://") or input_text.startswith("https://")
        try:
            result = analyze_url(input_text) if is_url else analyze_text(input_text)
        except Exception as exc:
            if ctx.obj.get("json"):
                click.echo(json.dumps({"error": str(exc)}, indent=2))
            else:
                click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
            sys.exit(1)
        if ctx.obj.get("json"):
            click.echo(json.dumps(result.to_dict(), indent=2))
            return
        risk_badge = _style_risk(result.risk_level.value)
        click.echo("")
        click.echo(click.style("  Note: Nexus not available, using standard analysis.", fg="yellow"))
        click.echo(f"  {risk_badge}  Score: {result.scam_score:.0%}  |  {result.risk_label()}")
        click.echo("")


# ---------------------------------------------------------------------------
# autonomic-health
# ---------------------------------------------------------------------------

@main.command("autonomic-health")
@click.pass_context
def autonomic_health(ctx: click.Context) -> None:
    """Show the autonomic self-healing dashboard."""
    try:
        from sentinel.autonomic import AutonomicController  # type: ignore[import]
    except ImportError as exc:
        click.echo(click.style(f"  autonomic module unavailable: {exc}", fg="red"), err=True)
        sys.exit(1)

    try:
        controller = AutonomicController()
        status = controller.get_status()
        dashboard = controller.dashboard.snapshot()
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    data = {"status": status, "dashboard": dashboard}

    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2, default=str))
        return

    click.echo("")
    click.echo(click.style("  Sentinel Health Dashboard", bold=True))
    click.echo("  " + "─" * 50)
    click.echo(f"  Cycles run:           {status.get('cycle_count', 0):>8,}")
    click.echo(f"  Consecutive failures: {status.get('consecutive_failures', 0):>8,}")
    click.echo(f"  Backoff (s):          {status.get('current_backoff_seconds', 0):>8.1f}")
    locked = status.get("regression_guard_locked", False)
    lock_str = click.style("LOCKED", fg="red") if locked else click.style("ok", fg="green")
    click.echo(f"  Regression guard:     {lock_str}")
    click.echo(f"  Budget remaining:     {status.get('regression_budget_remaining', 0):>8,}")
    click.echo(f"  Iterator temp:        {status.get('iterator_temperature', 0.0):>8.3f}")
    click.echo(f"  Best precision:       {status.get('iterator_best_precision', 0.0):>8.1%}")
    last = status.get("last_cycle", {})
    if last.get("status"):
        overall = last["status"]
        color = "green" if overall == "healthy" else ("yellow" if overall == "degraded" else "red")
        click.echo(f"  Last cycle:           {click.style(overall, fg=color)}")
    click.echo("")


# ---------------------------------------------------------------------------
# temporal
# ---------------------------------------------------------------------------

@main.command("temporal")
@click.option("--trends", "mode", flag_value="trends", default=True,
              help="Show scam evolution trends (default).")
@click.option("--predict", "mode", flag_value="predict",
              help="Show next-week volume predictions.")
@click.option("--drift", "mode", flag_value="drift",
              help="Show pattern drift report.")
@click.pass_context
def temporal(ctx: click.Context, mode: str) -> None:
    """Show temporal scam analysis: trends, predictions, or drift."""
    try:
        from sentinel.temporal import (  # type: ignore[import]
            PatternDrift, PredictiveModel, ScamEvolutionTracker,
        )
    except ImportError as exc:
        click.echo(click.style(f"  temporal module unavailable: {exc}", fg="red"), err=True)
        sys.exit(1)

    try:
        if mode == "trends":
            from sentinel.db import SentinelDB
            db = SentinelDB()
            tracker = ScamEvolutionTracker(db=db)
            lifecycles = tracker.all_lifecycles()
            db.close()
            data = {
                name: {
                    "status": lc.status,
                    "ema_rate": round(lc.ema_rate, 4),
                    "trend": round(lc.trend, 4),
                    "peak_rate": round(lc.peak_rate, 4),
                    "first_seen": lc.first_seen,
                    "last_seen": lc.last_seen,
                }
                for name, lc in lifecycles.items()
            }
            if ctx.obj.get("json"):
                click.echo(json.dumps({"trends": data, "count": len(data)}, indent=2))
                return
            click.echo("")
            click.echo(click.style(f"  Scam Evolution Trends ({len(data)} patterns)", bold=True))
            click.echo("  " + "─" * 60)
            if not data:
                click.echo(click.style("  No trend data yet.", fg="yellow"))
            for name, lc in sorted(data.items(), key=lambda kv: kv[1]["ema_rate"], reverse=True):
                status = lc["status"]
                color = "red" if status == "growing" else ("green" if status == "dying" else "yellow")
                click.echo(
                    f"  {click.style(status.upper()[:10], fg=color):<14}  "
                    f"rate={lc['ema_rate']:.3f}  trend={lc['trend']:+.4f}  {name}"
                )
            click.echo("")

        elif mode == "predict":
            model = PredictiveModel()
            prediction = model.predict_next_week()
            data = {
                "week": prediction.week_key,
                "predicted_volume": prediction.predicted_volume,
                "lower": prediction.lower_bound,
                "upper": prediction.upper_bound,
                "confidence": prediction.confidence,
                "method": prediction.method,
            }
            if ctx.obj.get("json"):
                click.echo(json.dumps(data, indent=2, default=str))
                return
            click.echo("")
            click.echo(click.style("  Next-Week Scam Volume Prediction", bold=True))
            click.echo("  " + "─" * 50)
            click.echo(f"  Week:       {prediction.week_key}")
            click.echo(f"  Predicted:  {prediction.predicted_volume:.1f}")
            click.echo(f"  Range:      {prediction.lower_bound:.1f} – {prediction.upper_bound:.1f}")
            click.echo(f"  Confidence: {prediction.confidence:.1%}")
            click.echo(f"  Method:     {prediction.method}")
            click.echo("")

        else:  # drift
            from sentinel.db import SentinelDB
            db = SentinelDB()
            drift = PatternDrift()
            report = drift.compare_from_db(db)
            db.close()
            data = {
                "kl_divergence": round(report.kl_divergence, 4),
                "is_significant": report.is_significant,
                "top_shifted": report.top_shifted,
                "summary": report.summary,
            }
            if ctx.obj.get("json"):
                click.echo(json.dumps(data, indent=2, default=str))
                return
            click.echo("")
            click.echo(click.style("  Pattern Drift Report", bold=True))
            click.echo("  " + "─" * 50)
            color = "red" if report.is_significant else "green"
            sig_str = click.style("YES", fg="red") if report.is_significant else click.style("No", fg="green")
            click.echo(f"  Significant drift: {sig_str}")
            click.echo(f"  KL divergence:     {report.kl_divergence:.4f}")
            if report.top_shifted:
                click.echo("")
                click.echo(click.style("  Top shifted signals:", bold=True))
                for sig in report.top_shifted[:5]:
                    click.echo(f"    {sig}")
            if report.summary:
                click.echo(f"\n  {report.summary}")
            click.echo("")

    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

@main.command("network")
@click.option("--clusters", "mode", flag_value="clusters", default=True,
              help="Show scam network clusters (default).")
@click.option("--sybils", "mode", flag_value="sybils",
              help="Detect sybil recruiter accounts.")
@click.pass_context
def network(ctx: click.Context, mode: str) -> None:
    """Analyze the scam network graph: clusters or sybil recruiters."""
    try:
        from sentinel.graph import RecruiterProfiler, ScamNetworkGraph  # type: ignore[import]
    except ImportError as exc:
        click.echo(click.style(f"  graph module unavailable: {exc}", fg="red"), err=True)
        sys.exit(1)

    try:
        if mode == "clusters":
            graph = ScamNetworkGraph()
            clusters = graph.get_clusters()
            data = [
                {
                    "cluster_id": c.cluster_id,
                    "size": c.size,
                    "avg_scam_score": round(c.avg_scam_score, 3),
                    "common_signals": c.common_signals,
                }
                for c in clusters
            ]
            if ctx.obj.get("json"):
                click.echo(json.dumps({"clusters": data, "count": len(data)}, indent=2))
                return
            click.echo("")
            click.echo(click.style(f"  Scam Network Clusters ({len(data)} found)", bold=True))
            click.echo("  " + "─" * 60)
            if not data:
                click.echo(click.style("  No clusters detected.", fg="yellow"))
            for c in data:
                score = c["avg_scam_score"]
                color = "red" if score >= 0.7 else ("yellow" if score >= 0.4 else "green")
                score_str = click.style(f"{score:.2f}", fg=color)
                click.echo(
                    f"  Cluster {c['cluster_id'][:8]}  "
                    f"size={c['size']:>4}  "
                    f"avg_score={score_str}  "
                    f"signals={c['common_signals'][:3]}"
                )
            click.echo("")

        else:  # sybils
            profiler = RecruiterProfiler()
            sybil_groups = profiler.detect_sybils()
            data = [list(group) for group in sybil_groups]
            if ctx.obj.get("json"):
                click.echo(json.dumps({"sybil_groups": data, "count": len(data)}, indent=2))
                return
            click.echo("")
            click.echo(click.style(f"  Sybil Recruiter Groups ({len(data)} detected)", bold=True))
            click.echo("  " + "─" * 50)
            if not data:
                click.echo(click.style("  No sybil accounts detected.", fg="green"))
            for i, group in enumerate(data, 1):
                click.echo(click.style(f"  Group {i} ({len(group)} accounts):", fg="red"))
                for rid in group:
                    click.echo(f"    - {rid}")
            click.echo("")

    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# counterfactual
# ---------------------------------------------------------------------------

@main.command("counterfactual")
@click.option("--recent", "recent_n", type=int, default=None,
              help="Run counterfactual analysis on the last N missed detections.")
@click.option("--gaps", is_flag=True, default=False,
              help="Show signal gap analysis.")
@click.pass_context
def counterfactual(ctx: click.Context, recent_n: int | None, gaps: bool) -> None:
    """Run counterfactual analysis on recent misses or show signal gaps."""
    try:
        from sentinel.counterfactual import FailureAnalyzer, SignalGapFinder  # type: ignore[import]
    except ImportError as exc:
        click.echo(click.style(f"  counterfactual module unavailable: {exc}", fg="red"), err=True)
        sys.exit(1)

    try:
        if gaps:
            finder = SignalGapFinder()
            gap_results = finder.find_gaps()
            data = {"gaps": gap_results, "count": len(gap_results)}
            if ctx.obj.get("json"):
                click.echo(json.dumps(data, indent=2, default=str))
                return
            click.echo("")
            click.echo(click.style(f"  Signal Gap Analysis ({len(gap_results)} gaps)", bold=True))
            click.echo("  " + "─" * 60)
            if not gap_results:
                click.echo(click.style("  No signal gaps detected.", fg="green"))
            for gap in gap_results[:10]:
                click.echo(f"  {gap}")
            click.echo("")
        elif recent_n is not None:
            analyzer = FailureAnalyzer()
            summary = analyzer.summary()
            data = {"recent_n": recent_n, "summary": summary}
            if ctx.obj.get("json"):
                click.echo(json.dumps(data, indent=2, default=str))
                return
            click.echo("")
            click.echo(click.style(f"  Counterfactual: Last {recent_n} misses", bold=True))
            click.echo("  " + "─" * 60)
            click.echo(f"  False negatives:  {summary.get('false_negatives', 0):>6,}")
            click.echo(f"  False positives:  {summary.get('false_positives', 0):>6,}")
            click.echo(f"  Total failures:   {summary.get('total', 0):>6,}")
            dist = summary.get("failure_mode_distribution", {})
            if dist:
                click.echo("")
                click.echo(click.style("  Failure modes:", bold=True))
                for mode_name, count in sorted(dist.items(), key=lambda kv: kv[1], reverse=True):
                    click.echo(f"    {mode_name:<30} {count:>4}")
            click.echo("")
        else:
            click.echo(click.style(
                "  Specify --recent N or --gaps. Run 'sentinel counterfactual --help'.",
                fg="yellow",
            ))

    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# llm-check
# ---------------------------------------------------------------------------

@main.command("llm-check")
@click.argument("text")
@click.pass_context
def llm_check(ctx: click.Context, text: str) -> None:
    """Check if a piece of text appears to be LLM-generated."""
    try:
        from sentinel.llm_detect import LLMDetector  # type: ignore[import]
    except ImportError as exc:
        click.echo(click.style(f"  llm_detect module unavailable: {exc}", fg="red"), err=True)
        sys.exit(1)

    try:
        detector = LLMDetector()
        result = detector.detect(text)
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    data = {
        "is_llm_generated": result.is_llm_generated,
        "score": round(result.score, 4),
        "confidence": round(result.confidence, 4),
        "signals": result.signals,
    }

    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2))
        return

    verdict = click.style("LLM-GENERATED", fg="red", bold=True) if result.is_llm_generated \
        else click.style("Human-written", fg="green")
    click.echo("")
    click.echo(click.style("  LLM Detection Result", bold=True))
    click.echo("  " + "─" * 50)
    click.echo(f"  Verdict:    {verdict}")
    click.echo(f"  Score:      {result.score:.1%}")
    click.echo(f"  Confidence: {result.confidence:.1%}")
    if result.signals:
        click.echo("")
        click.echo(click.style("  Signals detected:", bold=True))
        for sig in result.signals:
            click.echo(f"    - {sig}")
    click.echo("")


# ---------------------------------------------------------------------------
# econ-check
# ---------------------------------------------------------------------------

@main.command("econ-check")
@click.argument("text")
@click.option("--title", default="", help="Job title.")
@click.option("--company", default="", help="Company name.")
@click.option("--location", default="", help="Job location.")
@click.pass_context
def econ_check(ctx: click.Context, text: str, title: str, company: str, location: str) -> None:
    """Run economic validation on a job posting."""
    try:
        from sentinel.economics import validate_economics  # type: ignore[import]
        from sentinel.models import JobPosting
    except ImportError as exc:
        click.echo(click.style(f"  economics module unavailable: {exc}", fg="red"), err=True)
        sys.exit(1)

    try:
        job = JobPosting(
            title=title,
            company=company,
            location=location,
            description=text,
        )
        result = validate_economics(job)
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    data = {
        "overall_suspicious": result.is_suspicious,
        "overall_score": round(result.suspicion_score, 4),
        "flags": result.flags,
    }
    if result.salary:
        data["salary"] = {"suspicious": result.salary.is_suspicious, "flags": result.salary.flags}
    if result.geography:
        data["geography"] = {"suspicious": result.geography.is_suspicious, "flags": result.geography.flags}
    if result.benefits:
        data["benefits"] = {"suspicious": result.benefits.is_suspicious, "flags": result.benefits.flags}

    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2, default=str))
        return

    verdict = click.style("[SUSPICIOUS]", fg="red", bold=True) if result.is_suspicious \
        else click.style("[OK]", fg="green")
    click.echo("")
    click.echo(click.style("  Economic Validation", bold=True))
    click.echo("  " + "─" * 50)
    click.echo(f"  Result:  {verdict}")
    click.echo(f"  Score:   {result.suspicion_score:.1%}")
    if result.flags:
        click.echo("")
        click.echo(click.style("  Flags:", bold=True))
        for flag in result.flags:
            click.echo(click.style(f"    ! {flag}", fg="yellow"))
    click.echo("")


# ---------------------------------------------------------------------------
# brain
# ---------------------------------------------------------------------------

@main.command("brain")
@click.pass_context
def brain(ctx: click.Context) -> None:
    """Show full Nexus dashboard (falls back to health + stats if Nexus unavailable)."""
    nexus_data: dict = {}
    try:
        from sentinel.nexus import Nexus  # type: ignore[import]
        nexus = Nexus()
        nexus_data = nexus.get_dashboard() if hasattr(nexus, "get_dashboard") else {}
    except (ImportError, AttributeError, Exception):
        pass

    # Always include health and stats
    health_data: dict = {}
    stats_data: dict = {}
    try:
        from sentinel.flywheel import DetectionFlywheel
        fw = DetectionFlywheel()
        health_data = fw.get_health()
    except Exception:
        pass
    try:
        from sentinel.db import SentinelDB
        db = SentinelDB()
        stats_data = db.get_stats()
        db.close()
    except Exception:
        pass

    data = {"nexus": nexus_data, "health": health_data, "stats": stats_data}

    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2, default=str))
        return

    click.echo("")
    click.echo(click.style("  Sentinel Brain / Nexus Dashboard", bold=True))
    click.echo("  " + "═" * 60)

    if nexus_data:
        click.echo(click.style("  Nexus State:", bold=True))
        for k, v in nexus_data.items():
            click.echo(f"    {k}: {v}")
        click.echo("")

    if health_data:
        grade = health_data.get("grade", "?")
        grade_color = "green" if grade in ("A", "B") else ("yellow" if grade == "C" else "red")
        click.echo(click.style("  System Health:", bold=True))
        click.echo(f"    Grade:     {click.style(grade, fg=grade_color, bold=True)}")
        click.echo(f"    Precision: {health_data.get('precision', 0.0):.1%}")
        click.echo(f"    Recall:    {health_data.get('recall', 0.0):.1%}")
        click.echo(f"    F1:        {health_data.get('f1', 0.0):.3f}")
        click.echo("")

    if stats_data:
        click.echo(click.style("  Detection Stats:", bold=True))
        click.echo(f"    Jobs analyzed: {stats_data.get('total_jobs_analyzed', 0):,}")
        click.echo(f"    Scams found:   {stats_data.get('scam_jobs_detected', 0):,}")
        click.echo(f"    Active patterns: {stats_data.get('active_patterns', 0):,}")
        click.echo("")


# ---------------------------------------------------------------------------
# self-heal
# ---------------------------------------------------------------------------

@main.command("self-heal")
@click.pass_context
def self_heal(ctx: click.Context) -> None:
    """Run one autonomic healing cycle."""
    try:
        from sentinel.autonomic import AutonomicController  # type: ignore[import]
    except ImportError as exc:
        click.echo(click.style(f"  autonomic module unavailable: {exc}", fg="red"), err=True)
        sys.exit(1)

    if not ctx.obj.get("json"):
        click.echo(click.style("  Running autonomic healing cycle...", fg="cyan"))

    try:
        controller = AutonomicController()
        cycle = controller.run_cycle()
    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    data = {
        "cycle_number": cycle.cycle_number,
        "overall_status": cycle.overall_status,
        "healed": cycle.healed,
        "improvement_ran": cycle.improvement_ran,
        "backoff_seconds": cycle.backoff_seconds,
        "timestamp": cycle.timestamp,
    }

    if ctx.obj.get("json"):
        click.echo(json.dumps(data, indent=2, default=str))
        return

    status_color = "green" if cycle.overall_status == "healthy" \
        else ("yellow" if cycle.overall_status == "degraded" else "red")
    click.echo("")
    click.echo(click.style("  Autonomic Heal Cycle Complete", bold=True))
    click.echo("  " + "─" * 50)
    click.echo(f"  Cycle:     #{cycle.cycle_number}")
    click.echo(f"  Status:    {click.style(cycle.overall_status, fg=status_color)}")
    click.echo(f"  Healed:    {'Yes' if cycle.healed else 'No'}")
    click.echo(f"  Improved:  {'Yes' if cycle.improvement_ran else 'No'}")
    if cycle.backoff_seconds:
        click.echo(f"  Backoff:   {cycle.backoff_seconds:.1f}s")
    click.echo("")


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

@main.command("checkpoint")
@click.argument("action", type=click.Choice(["save", "list", "rollback"]))
@click.argument("tag", required=False, default=None)
@click.pass_context
def checkpoint(ctx: click.Context, action: str, tag: str | None) -> None:
    """Manage system checkpoints.

    \b
    Actions:
      save <tag>       Save a checkpoint with the given tag.
      list             List all saved checkpoints.
      rollback <tag>   Roll back to a checkpoint by tag.
    """
    try:
        from sentinel.autonomic import CheckpointManager  # type: ignore[import]
        from sentinel.db import SentinelDB
    except ImportError as exc:
        click.echo(click.style(f"  autonomic module unavailable: {exc}", fg="red"), err=True)
        sys.exit(1)

    try:
        db = SentinelDB()
        manager = CheckpointManager(db)

        if action == "save":
            if not tag:
                click.echo(click.style("  Error: provide a tag for 'save'.", fg="red"), err=True)
                db.close()
                sys.exit(1)
            cp = manager.save(tag=tag)
            db.close()
            data = {"action": "save", "tag": cp.tag, "created_at": cp.created_at}
            if ctx.obj.get("json"):
                click.echo(json.dumps(data, indent=2, default=str))
                return
            click.echo("")
            click.echo(click.style(f"  Checkpoint saved: {cp.tag}", fg="green"))
            click.echo(f"  At: {cp.created_at}")
            click.echo("")

        elif action == "list":
            checkpoints = manager.list_checkpoints()
            db.close()
            if ctx.obj.get("json"):
                click.echo(json.dumps({"checkpoints": checkpoints, "count": len(checkpoints)},
                                      indent=2, default=str))
                return
            click.echo("")
            click.echo(click.style(f"  Checkpoints ({len(checkpoints)} total)", bold=True))
            click.echo("  " + "─" * 50)
            if not checkpoints:
                click.echo(click.style("  No checkpoints saved.", fg="yellow"))
            for cp in checkpoints:
                click.echo(f"  {cp.get('tag', '?'):<30}  {cp.get('created_at', '')}")
            click.echo("")

        else:  # rollback
            if not tag:
                click.echo(click.style("  Error: provide a tag for 'rollback'.", fg="red"), err=True)
                db.close()
                sys.exit(1)
            cp = manager.rollback(tag=tag)
            db.close()
            data = {"action": "rollback", "tag": cp.tag, "created_at": cp.created_at}
            if ctx.obj.get("json"):
                click.echo(json.dumps(data, indent=2, default=str))
                return
            click.echo("")
            click.echo(click.style(f"  Rolled back to checkpoint: {cp.tag}", fg="yellow", bold=True))
            click.echo(f"  Originally saved: {cp.created_at}")
            click.echo("")

    except Exception as exc:
        if ctx.obj.get("json"):
            click.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            click.echo(click.style(f"  Error: {exc}", fg="red"), err=True)
        sys.exit(1)
