"""Command-line interface for Sentinel LinkedIn job scam detection."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from sentinel.models import RiskLevel


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
    ctx.ensure_object(dict)
    ctx.obj["json"] = use_json


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

@main.command()
@click.argument("input_text")
@click.option("--title", default="", help="Job title (for raw-text input).")
@click.option("--company", default="", help="Company name (for raw-text input).")
@click.option("--no-ai", is_flag=True, default=False,
              help="Disable AI escalation (heuristics only, faster).")
@click.pass_context
def analyze(ctx: click.Context, input_text: str, title: str, company: str, no_ai: bool) -> None:
    """Analyze a job posting for scam signals.

    INPUT_TEXT can be a LinkedIn URL or raw job description text.
    """
    from sentinel.analyzer import analyze_job, analyze_text, analyze_url, format_result_text
    from sentinel.scanner import parse_job_text
    from sentinel.db import SentinelDB

    use_ai = not no_ai
    is_url = input_text.startswith("http://") or input_text.startswith("https://")

    try:
        if is_url:
            result = analyze_url(input_text)
        else:
            result = analyze_text(input_text, title=title, company=company)
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
@click.pass_context
def validate(ctx: click.Context, company_name: str, domain: str) -> None:
    """Validate a company's legitimacy."""
    from sentinel.validator import validate_company

    try:
        profile = validate_company(company_name, domain=domain)
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
        "linkedin_url": profile.linkedin_url if hasattr(profile, "linkedin_url") else profile.company_linkedin_url if hasattr(profile, "company_linkedin_url") else "",
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
    from sentinel.knowledge import KnowledgeBase
    from sentinel.db import SentinelDB

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


@main.command()
@click.pass_context
def ecosystem(ctx):
    """Show ecosystem integration status."""
    from sentinel.ecosystem import read_ecosystem_context

    context = read_ecosystem_context()
    if ctx.obj.get("json"):
        click.echo(json.dumps(context, indent=2, default=str))
    else:
        click.echo(click.style("Ecosystem Integration\n", bold=True))
        if context.get("session_briefing"):
            click.echo(f"  Session Bridge: {click.style('connected', fg='green')}")
        else:
            click.echo(f"  Session Bridge: {click.style('no briefing', fg='yellow')}")
        patterns = context.get("engram_patterns", [])
        if patterns:
            click.echo(f"  Engram: {click.style(f'{len(patterns)} patterns', fg='green')}")
        else:
            click.echo(f"  Engram: {click.style('not connected', fg='yellow')}")
