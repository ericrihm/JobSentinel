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
from typing import Any

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
        text: str | None = Field(None, max_length=50000, description="Raw job description text.")
        url: str | None = Field(None, max_length=2048, description="LinkedIn job posting URL.")
        job_data: dict | None = Field(None, description="Structured job dict (JSON).")
        title: str = Field("", max_length=500, description="Job title (used with text input).")
        company: str = Field("", max_length=500, description="Company name (used with text input).")
        use_ai: bool = Field(True, description="Enable AI escalation for ambiguous cases.")

        @field_validator("url")
        @classmethod
        def url_must_have_http_scheme(cls, v: str | None) -> str | None:
            if v is not None and not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("url must start with http:// or https://")
            return v

        @field_validator("text", "title", "company")
        @classmethod
        def no_script_tags(cls, v: str | None) -> str | None:
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
        precision: float | None
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
        cold_start: bool = False
        checked_at: str

    class AnalyzeUrlRequest(BaseModel):
        """Payload for POST /api/v1/analyze-url."""
        url: str = Field(..., max_length=2048, description="URL to analyze for scam indicators.")

        @field_validator("url")
        @classmethod
        def url_must_have_http_scheme(cls, v: str) -> str:
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("url must start with http:// or https://")
            return v

    class VerifyCompanyRequest(BaseModel):
        """Payload for POST /api/v1/verify-company."""
        company_name: str = Field(..., max_length=500, description="Company name to verify.")
        company_url: str = Field("", max_length=2048, description="Company website URL (optional).")
        linkedin_url: str = Field("", max_length=2048, description="Company LinkedIn URL (optional).")

    class AnalyzeLinksRequest(BaseModel):
        """Payload for POST /api/v1/analyze-links."""
        text: str = Field(..., max_length=50000, description="Text containing URLs to extract and analyze.")

        @field_validator("text")
        @classmethod
        def no_script_tags(cls, v: str) -> str:
            if re.search(r"<script", v, re.IGNORECASE):
                raise ValueError("Input may not contain script tags")
            return v

except ImportError:
    # FastAPI/Pydantic not installed — models won't be available at import time.
    # create_app() will raise a clear ImportError when called.
    pass


def create_app():  # noqa: C901
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException, Query, Request, Response
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as err:
        raise ImportError(
            "FastAPI and Uvicorn are required for the API server.\n"
            "Install them with:  pip install fastapi uvicorn"
        ) from err

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
        from sentinel.db import SentinelDB
        from sentinel.scanner import parse_job_json

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
        from sentinel.db import SentinelDB
        from sentinel.knowledge import KnowledgeBase

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
        category: str | None = Query(
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
            "cold_start": health.get("cold_start", False),
            "checked_at": health.get("checked_at", ""),
        }

    # -----------------------------------------------------------------------
    # POST /api/deep-analyze
    # -----------------------------------------------------------------------

    @app.post("/api/deep-analyze", summary="Full nexus deep analysis")
    def deep_analyze_endpoint(req: AnalyzeRequest) -> Any:
        """Run a deep analysis using Nexus if available, otherwise falls back to standard analysis."""
        if not req.text and not req.url and not req.job_data:
            raise HTTPException(
                status_code=422,
                detail="Provide at least one of: text, url, or job_data.",
            )

        # Try Nexus first
        try:
            from sentinel.nexus import Nexus  # type: ignore[import]
            nexus = Nexus()
            input_str = req.url or req.text or str(req.job_data)
            result = nexus.deep_analyze(input_str)
            return result if isinstance(result, dict) else vars(result)
        except (ImportError, AttributeError):
            pass
        except Exception as exc:
            logger.warning("Nexus deep_analyze failed, falling back: %s", exc)

        # Fall back to standard analysis
        try:
            from sentinel.analyzer import analyze_job, analyze_text, analyze_url
            from sentinel.scanner import parse_job_json

            if req.url:
                result = analyze_url(req.url)
            elif req.job_data:
                job = parse_job_json(req.job_data)
                result = analyze_job(job, use_ai=req.use_ai)
            else:
                result = analyze_text(req.text or "", title=req.title, company=req.company)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

        return {**result.to_dict(), "nexus_available": False}

    # -----------------------------------------------------------------------
    # GET /api/health/autonomic
    # -----------------------------------------------------------------------

    @app.get("/api/health/autonomic", summary="Autonomic system health dashboard")
    def autonomic_health_endpoint() -> Any:
        """Return a full health dashboard from the autonomic controller."""
        try:
            from sentinel.autonomic import AutonomicController  # type: ignore[import]
        except ImportError:
            raise HTTPException(status_code=501, detail="autonomic module not available")

        try:
            controller = AutonomicController()
            status = controller.get_status()
            dashboard = controller.dashboard.snapshot()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Health check failed: {exc}") from exc

        return {"status": status, "dashboard": dashboard}

    # -----------------------------------------------------------------------
    # GET /api/temporal/trends
    # -----------------------------------------------------------------------

    @app.get("/api/temporal/trends", summary="Scam evolution trends")
    def temporal_trends_endpoint() -> Any:
        """Return scam pattern lifecycle and evolution trend data."""
        try:
            from sentinel.temporal import ScamEvolutionTracker  # type: ignore[import]
        except ImportError:
            raise HTTPException(status_code=501, detail="temporal module not available")

        try:
            from sentinel.db import SentinelDB
            db = SentinelDB()
            tracker = ScamEvolutionTracker(db=db)
            lifecycles = tracker.all_lifecycles()
            db.close()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Temporal trends failed: {exc}") from exc

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
        return {"trends": data, "count": len(data)}

    # -----------------------------------------------------------------------
    # GET /api/temporal/predict
    # -----------------------------------------------------------------------

    @app.get("/api/temporal/predict", summary="Next-week scam volume predictions")
    def temporal_predict_endpoint() -> Any:
        """Return next-week scam volume predictions from the PredictiveModel."""
        try:
            from sentinel.temporal import PredictiveModel  # type: ignore[import]
        except ImportError:
            raise HTTPException(status_code=501, detail="temporal module not available")

        try:
            model = PredictiveModel()
            prediction = model.predict_next_week()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

        return {
            "week": prediction.week_key,
            "predicted_volume": prediction.predicted_volume,
            "lower_bound": prediction.lower_bound,
            "upper_bound": prediction.upper_bound,
            "confidence": prediction.confidence,
            "method": prediction.method,
        }

    # -----------------------------------------------------------------------
    # POST /api/llm-check
    # -----------------------------------------------------------------------

    @app.post("/api/llm-check", summary="Check if text is LLM-generated")
    def llm_check_endpoint(req: AnalyzeRequest) -> Any:
        """Detect whether the provided text was generated by an LLM."""
        try:
            from sentinel.llm_detect import LLMDetector  # type: ignore[import]
        except ImportError:
            raise HTTPException(status_code=501, detail="llm_detect module not available")

        text = req.text or ""
        if not text:
            raise HTTPException(status_code=422, detail="Provide text for LLM detection.")

        try:
            detector = LLMDetector()
            result = detector.detect(text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM detection failed: {exc}") from exc

        return {
            "is_llm_generated": result.is_llm_generated,
            "score": round(result.score, 4),
            "confidence": round(result.confidence, 4),
            "signals": result.signals,
        }

    # -----------------------------------------------------------------------
    # POST /api/econ-check
    # -----------------------------------------------------------------------

    @app.post("/api/econ-check", summary="Run economic validation on a job posting")
    def econ_check_endpoint(req: AnalyzeRequest) -> Any:
        """Validate economic plausibility of salary, benefits, and geography."""
        try:
            from sentinel.economics import validate_economics  # type: ignore[import]
            from sentinel.models import JobPosting
        except ImportError:
            raise HTTPException(status_code=501, detail="economics module not available")

        text = req.text or ""
        try:
            job = JobPosting(
                title=req.title,
                company=req.company,
                description=text,
                url=req.url or "",
            )
            result = validate_economics(job)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Economic validation failed: {exc}") from exc

        out: dict[str, Any] = {
            "overall_suspicious": result.is_suspicious,
            "overall_score": round(result.suspicion_score, 4),
            "flags": result.flags,
        }
        if result.salary:
            out["salary"] = {"suspicious": result.salary.is_suspicious, "flags": result.salary.flags}
        if result.geography:
            out["geography"] = {"suspicious": result.geography.is_suspicious, "flags": result.geography.flags}
        if result.benefits:
            out["benefits"] = {"suspicious": result.benefits.is_suspicious, "flags": result.benefits.flags}
        return out

    # -----------------------------------------------------------------------
    # GET /api/network/clusters
    # -----------------------------------------------------------------------

    @app.get("/api/network/clusters", summary="Scam network clusters")
    def network_clusters_endpoint() -> Any:
        """Return detected scam network clusters from the graph module."""
        try:
            from sentinel.graph import ScamNetworkGraph  # type: ignore[import]
        except ImportError:
            raise HTTPException(status_code=501, detail="graph module not available")

        try:
            graph = ScamNetworkGraph()
            clusters = graph.get_clusters()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Network analysis failed: {exc}") from exc

        data = [
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "avg_scam_score": round(c.avg_scam_score, 3),
                "common_signals": c.common_signals,
            }
            for c in clusters
        ]
        return {"clusters": data, "count": len(data)}

    # -----------------------------------------------------------------------
    # GET /api/brain
    # -----------------------------------------------------------------------

    @app.get("/api/brain", summary="Full nexus/brain dashboard")
    def brain_endpoint() -> Any:
        """Return the full Nexus dashboard state plus health and stats."""
        nexus_data: dict = {}
        try:
            from sentinel.nexus import Nexus  # type: ignore[import]
            nexus = Nexus()
            if hasattr(nexus, "get_dashboard"):
                nexus_data = nexus.get_dashboard()
        except (ImportError, AttributeError, Exception) as exc:
            logger.debug("Nexus unavailable for /api/brain: %s", exc)

        health_data: dict = {}
        try:
            from sentinel.flywheel import DetectionFlywheel
            fw = DetectionFlywheel()
            health_data = fw.get_health()
        except Exception as exc:
            logger.warning("Health unavailable for /api/brain: %s", exc)

        stats_data: dict = {}
        try:
            from sentinel.db import SentinelDB
            db = SentinelDB()
            stats_data = db.get_stats()
            db.close()
        except Exception as exc:
            logger.warning("Stats unavailable for /api/brain: %s", exc)

        return {
            "nexus": nexus_data,
            "health": health_data,
            "stats": stats_data,
        }

    # -----------------------------------------------------------------------
    # POST /api/v1/analyze-url
    # -----------------------------------------------------------------------

    @app.post("/api/v1/analyze-url", summary="Analyze a single URL for scam indicators")
    def analyze_url_endpoint(req: AnalyzeUrlRequest) -> Any:
        """Analyze a URL using LinkAnalyzer and return domain/reputation/redirect results.

        Calls ``LinkAnalyzer().analyze_all(url)`` on the provided URL.
        Returns the consolidated analysis dict including domain_analysis,
        reputation, redirect_chain, and domain_age.
        """
        from sentinel.link_analyzer import LinkAnalyzer

        try:
            analyzer = LinkAnalyzer()
            results = analyzer.analyze_all(req.url)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"URL analysis failed: {exc}"
            ) from exc

        # analyze_all returns a list (one entry per extracted URL).  Since we
        # passed a single well-formed URL, return the first result directly; if
        # nothing was extracted return an empty analysis.
        if results:
            return results[0]

        # Fallback: run domain analysis directly for URLs that weren't caught
        # by the text extractor (e.g., bare https:// without a trailing path).
        try:
            domain_analysis = analyzer.analyze_domain(req.url)
            reputation = analyzer.check_url_reputation(req.url)
            return {
                "url": req.url,
                "domain_analysis": domain_analysis,
                "reputation": reputation,
                "redirect_chain": None,
                "domain_age": None,
            }
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"URL domain analysis failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # POST /api/v1/verify-company
    # -----------------------------------------------------------------------

    @app.post("/api/v1/verify-company", summary="Verify company legitimacy")
    def verify_company_endpoint(req: VerifyCompanyRequest) -> Any:
        """Verify whether a company appears legitimate.

        Calls CompanyVerifier to check domain resolution, HTTPS, name heuristics,
        and LinkedIn presence.  Returns a dict with verification sub-results.
        """
        from sentinel.company_verifier import CompanyVerifier

        try:
            verifier = CompanyVerifier()
            result: dict[str, Any] = {}

            # Domain verification (if URL provided)
            domain_info = verifier.verify_domain(req.company_name, req.company_url)
            result["domain"] = domain_info

            # Existence check (heuristic, always runs)
            exists_info = verifier.check_company_exists(req.company_name)
            result["existence"] = exists_info

            # LinkedIn presence (if URL provided)
            if req.linkedin_url:
                linkedin_info = verifier.verify_linkedin_presence(req.linkedin_url)
                result["linkedin"] = linkedin_info
            else:
                result["linkedin"] = None

            # Address legitimacy (if company_url acts as location hint)
            result["company_name"] = req.company_name
            result["company_url"] = req.company_url
            result["linkedin_url"] = req.linkedin_url

        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Company verification failed: {exc}"
            ) from exc

        return result

    # -----------------------------------------------------------------------
    # POST /api/v1/analyze-links
    # -----------------------------------------------------------------------

    @app.post("/api/v1/analyze-links", summary="Extract and analyze all URLs in text")
    def analyze_links_endpoint(req: AnalyzeLinksRequest) -> Any:
        """Extract every URL from the provided text and analyze each one.

        Calls ``LinkAnalyzer().analyze_all(text)`` which runs domain analysis,
        reputation checks, redirect-chain following (for shorteners/suspicious
        domains), and domain-age lookups for each discovered URL.

        Returns: ``{"urls_found": N, "results": [...]}``.
        """
        from sentinel.link_analyzer import LinkAnalyzer

        try:
            analyzer = LinkAnalyzer()
            results = analyzer.analyze_all(req.text)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Link analysis failed: {exc}"
            ) from exc

        return {
            "urls_found": len(results),
            "results": results,
        }

    return app
