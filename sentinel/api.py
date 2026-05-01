"""FastAPI REST API for Sentinel scam detection.

Endpoints:
    POST /api/analyze   — analyze a job posting (text, URL, or JSON dict)
    POST /api/report    — submit a scam/legitimate verdict
    GET  /api/patterns  — list known scam patterns
    GET  /api/stats     — detection statistics
    GET  /api/health    — service health check

Optional dependency: fastapi + uvicorn.
Returns a clear ImportError if not installed.
"""

import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter — in-memory sliding window, no external dependencies
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window rate limiter keyed by client IP.

    Tracks request timestamps per IP in a 60-second window. Thread-safety is
    not required because FastAPI runs handlers in a single event loop thread
    for sync endpoints (which these are).
    """

    def __init__(self, rpm: int = 60) -> None:
        self.rpm = rpm
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Return True if the IP is under the rate limit, False if exceeded.

        Side-effect: records the current request timestamp when allowed.
        """
        now = time.monotonic()
        window = self.requests[client_ip]
        # Evict entries older than the 60-second window
        self.requests[client_ip] = [t for t in window if now - t < 60]
        if len(self.requests[client_ip]) >= self.rpm:
            return False
        self.requests[client_ip].append(now)
        return True

# ---------------------------------------------------------------------------
# Request / response models — defined at module scope so Pydantic can resolve
# forward references correctly with FastAPI 0.136+ / Pydantic v2.
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field, field_validator

    class AnalyzeRequest(BaseModel):
        """Payload for POST /api/analyze.  Provide one of: text, url, or job_data."""
        text: Optional[str] = Field(None, max_length=50000, description="Raw job description text.")
        url: Optional[str] = Field(None, max_length=2048, description="LinkedIn job posting URL.")
        job_data: Optional[dict] = Field(None, description="Structured job dict (JSON).")
        title: str = Field("", max_length=500, description="Job title (used with text input).")
        company: str = Field("", max_length=500, description="Company name (used with text input).")
        use_ai: bool = Field(True, description="Enable AI escalation for ambiguous cases.")

        @field_validator("url")
        @classmethod
        def url_must_have_http_scheme(cls, v: Optional[str]) -> Optional[str]:
            if v is not None and not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("url must start with http:// or https://")
            return v

        @field_validator("text", "title", "company")
        @classmethod
        def no_script_tags(cls, v: Optional[str]) -> Optional[str]:
            if v is not None and re.search(r"<script", v, re.IGNORECASE):
                raise ValueError("Input may not contain script tags")
            return v

    class ReportRequest(BaseModel):
        url: str = Field(..., max_length=2048, description="Job posting URL being reported.")
        is_scam: bool = Field(..., description="True if this is a scam, False if legitimate.")
        reason: str = Field("", max_length=5000, description="Optional explanation.")

        @field_validator("reason")
        @classmethod
        def no_script_in_reason(cls, v: str) -> str:
            if v and re.search(r"<script", v, re.IGNORECASE):
                raise ValueError("reason may not contain script tags")
            return v

    class ReportResponse(BaseModel):
        url: str
        verdict: str
        reason: str
        recorded: bool
        message: str

    class PatternOut(BaseModel):
        pattern_id: str
        name: str
        description: str
        category: str
        status: str
        observations: int
        precision: Optional[float]
        bayesian_score: float
        keywords: list[str]

    class PatternsResponse(BaseModel):
        patterns: list[PatternOut]
        count: int

    class StatsResponse(BaseModel):
        total_jobs_analyzed: int
        scam_jobs_detected: int
        avg_scam_score: float
        total_user_reports: int
        scam_reports: int
        prediction_accuracy: float
        active_patterns: int
        total_companies: int
        verified_companies: int
        last_flywheel_cycle: dict

    class HealthResponse(BaseModel):
        status: str
        healthy: bool
        grade: str
        precision: float
        recall: float
        f1: float
        active_patterns: int
        total_jobs_analyzed: int
        total_user_reports: int
        regression_alarm: bool
        cusum_statistic: float
        cycle_count: int
        checked_at: str

except ImportError:
    # FastAPI/Pydantic not installed — models won't be available at import time.
    # create_app() will raise a clear ImportError when called.
    pass


def create_app():  # noqa: C901
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException, Query, Request, Response
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError(
            "FastAPI and Uvicorn are required for the API server.\n"
            "Install them with:  pip install fastapi uvicorn"
        )

    from sentinel.config import get_config

    # -----------------------------------------------------------------------
    # App setup
    # -----------------------------------------------------------------------

    app = FastAPI(
        title="Sentinel",
        description=(
            "LinkedIn Job Scam Detection API — "
            "analyze job postings, report scams, and explore detection patterns."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Rate-limit + API-key middleware
    # -----------------------------------------------------------------------

    cfg = get_config()
    _rate_limiter = RateLimiter(rpm=cfg.rate_limit_rpm)
    _api_key = os.environ.get("SENTINEL_API_KEY", "")

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        # --- Optional API key authentication ---
        if _api_key:
            provided = request.headers.get("X-API-Key", "")
            if provided != _api_key:
                return Response(
                    content='{"detail":"Invalid or missing API key"}',
                    status_code=401,
                    media_type="application/json",
                    headers={"WWW-Authenticate": "ApiKey"},
                )

        # --- Sliding-window rate limit ---
        client_host = (request.client.host if request.client else "unknown")
        if not _rate_limiter.is_allowed(client_host):
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        return await call_next(request)

    # -----------------------------------------------------------------------
    # POST /api/analyze
    # -----------------------------------------------------------------------

    @app.post("/api/analyze", summary="Analyze a job posting")
    def analyze_endpoint(req: AnalyzeRequest) -> Any:
        """Analyze a job posting for scam signals.

        Supply one of:
        - **text** — raw job description (optionally with title and company)
        - **url** — LinkedIn job posting URL
        - **job_data** — pre-structured JSON dict
        """
        from sentinel.analyzer import analyze_job, analyze_text, analyze_url
        from sentinel.scanner import parse_job_json
        from sentinel.db import SentinelDB

        if not req.text and not req.url and not req.job_data:
            raise HTTPException(
                status_code=422,
                detail="Provide at least one of: text, url, or job_data.",
            )

        try:
            if req.url:
                result = analyze_url(req.url)
            elif req.job_data:
                job = parse_job_json(req.job_data)
                result = analyze_job(job, use_ai=req.use_ai)
            else:
                result = analyze_text(
                    req.text or "",
                    title=req.title,
                    company=req.company,
                )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

        # Persist to DB (best-effort — never fail the request)
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
            logger.warning("Failed to persist analysis result to DB", exc_info=True)

        logger.info(
            "POST /api/analyze: score=%.2f risk=%s signals=%d",
            result.scam_score,
            result.risk_level.value,
            len(result.signals),
        )
        return result.to_dict()

    # -----------------------------------------------------------------------
    # POST /api/report
    # -----------------------------------------------------------------------

    @app.post("/api/report", response_model=ReportResponse, summary="Submit a scam report")
    def report_endpoint(req: ReportRequest) -> Any:
        """Submit a user verdict on a job posting.

        Reports feed the learning flywheel — every confirmed scam or legitimate
        marking updates Bayesian signal weights via Thompson Sampling.
        """
        from sentinel.knowledge import KnowledgeBase
        from sentinel.db import SentinelDB

        our_prediction = 0.0
        try:
            db = SentinelDB()
            existing = db.get_job(req.url)
            if existing:
                our_prediction = existing.get("scam_score", 0.0)
            db.close()
        except Exception:
            logger.warning("Could not retrieve prior prediction for %s", req.url, exc_info=True)

        try:
            kb = KnowledgeBase()
            kb.report_scam(
                req.url,
                is_scam=req.is_scam,
                reason=req.reason,
                our_prediction=our_prediction,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to record report: {exc}"
            ) from exc

        logger.info("POST /api/report: url=%s verdict=%s", req.url, "scam" if req.is_scam else "legitimate")
        return {
            "url": req.url,
            "verdict": "scam" if req.is_scam else "legitimate",
            "reason": req.reason,
            "recorded": True,
            "message": "Report recorded. Thank you for improving Sentinel's accuracy.",
        }

    # -----------------------------------------------------------------------
    # GET /api/patterns
    # -----------------------------------------------------------------------

    @app.get("/api/patterns", response_model=PatternsResponse, summary="List scam patterns")
    def patterns_endpoint(
        category: Optional[str] = Query(
            None,
            description=(
                "Filter by category: red_flag, warning, ghost_job, structural, positive"
            ),
        ),
        status: str = Query(
            "active",
            description="Pattern lifecycle status: active, candidate, deprecated, all",
        ),
    ) -> Any:
        """Return known scam detection patterns with their Bayesian effectiveness scores."""
        from sentinel.db import SentinelDB

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

        if category:
            rows = [r for r in rows if r.get("category") == category]

        rows.sort(
            key=lambda r: r.get("alpha", 1.0) / (r.get("alpha", 1.0) + r.get("beta", 1.0)),
            reverse=True,
        )

        out: list[dict] = []
        for r in rows:
            tp = r.get("true_positives", 0)
            fp = r.get("false_positives", 0)
            total = tp + fp
            alpha = r.get("alpha", 1.0)
            beta = r.get("beta", 1.0)
            out.append({
                "pattern_id": r["pattern_id"],
                "name": r["name"],
                "description": r.get("description", ""),
                "category": r.get("category", ""),
                "status": r.get("status", "active"),
                "observations": r.get("observations", 0),
                "precision": round(tp / total, 3) if total else None,
                "bayesian_score": round(alpha / (alpha + beta), 3),
                "keywords": r.get("keywords", []),
            })

        return {"patterns": out, "count": len(out)}

    # -----------------------------------------------------------------------
    # GET /api/stats
    # -----------------------------------------------------------------------

    @app.get("/api/stats", response_model=StatsResponse, summary="Detection statistics")
    def stats_endpoint() -> Any:
        """Return aggregate detection statistics including accuracy and flywheel metrics."""
        from sentinel.db import SentinelDB

        db = SentinelDB()
        try:
            raw = db.get_stats()
        finally:
            db.close()

        return {
            "total_jobs_analyzed": raw.get("total_jobs_analyzed", 0),
            "scam_jobs_detected": raw.get("scam_jobs_detected", 0),
            "avg_scam_score": raw.get("avg_scam_score", 0.0),
            "total_user_reports": raw.get("total_user_reports", 0),
            "scam_reports": raw.get("scam_reports", 0),
            "prediction_accuracy": raw.get("prediction_accuracy", 0.0),
            "active_patterns": raw.get("active_patterns", 0),
            "total_companies": raw.get("total_companies", 0),
            "verified_companies": raw.get("verified_companies", 0),
            "last_flywheel_cycle": raw.get("last_flywheel_cycle", {}),
        }

    # -----------------------------------------------------------------------
    # GET /api/health
    # -----------------------------------------------------------------------

    @app.get("/api/health", response_model=HealthResponse, summary="Service health check")
    def health_endpoint() -> Any:
        """Return a health summary including model accuracy grade and regression status."""
        from sentinel.flywheel import DetectionFlywheel

        try:
            fw = DetectionFlywheel()
            health = fw.get_health()
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Health check failed: {exc}"
            ) from exc

        return {
            "status": "ok" if health.get("healthy") else "degraded",
            "healthy": health.get("healthy", True),
            "grade": health.get("grade", "?"),
            "precision": health.get("precision", 0.0),
            "recall": health.get("recall", 0.0),
            "f1": health.get("f1", 0.0),
            "active_patterns": health.get("active_patterns", 0),
            "total_jobs_analyzed": health.get("total_jobs_analyzed", 0),
            "total_user_reports": health.get("total_user_reports", 0),
            "regression_alarm": health.get("regression_alarm", False),
            "cusum_statistic": health.get("cusum_statistic", 0.0),
            "cycle_count": health.get("cycle_count", 0),
            "checked_at": health.get("checked_at", ""),
        }

    return app
