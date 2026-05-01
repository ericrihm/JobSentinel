"""Data models for job posting analysis and scam detection."""

from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(Enum):
    SAFE = "safe"
    LOW = "low"
    SUSPICIOUS = "suspicious"
    HIGH = "high"
    SCAM = "scam"


class SignalCategory(Enum):
    RED_FLAG = "red_flag"
    WARNING = "warning"
    GHOST_JOB = "ghost_job"
    STRUCTURAL = "structural"
    POSITIVE = "positive"


@dataclass
class JobPosting:
    """A LinkedIn job posting with extracted fields."""
    url: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    description: str = ""
    salary_min: float = 0.0
    salary_max: float = 0.0
    salary_currency: str = "USD"
    posted_date: str = ""
    applicant_count: int = 0
    experience_level: str = ""
    employment_type: str = ""
    industry: str = ""
    company_size: str = ""
    company_linkedin_url: str = ""
    recruiter_name: str = ""
    recruiter_connections: int = 0
    is_remote: bool = False
    is_repost: bool = False
    raw_html: str = ""
    source: str = "linkedin"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "salary_min": self.salary_min,
            "salary_max": self.salary_max,
            "posted_date": self.posted_date,
            "experience_level": self.experience_level,
            "employment_type": self.employment_type,
            "is_remote": self.is_remote,
        }


@dataclass
class ScamSignal:
    """A single scam indicator detected in a job posting."""
    name: str
    category: SignalCategory
    weight: float = 0.5
    confidence: float = 0.5
    detail: str = ""
    evidence: str = ""

    # Bayesian posterior (updated by flywheel)
    alpha: float = 1.0
    beta: float = 1.0

    @property
    def bayesian_weight(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class CompanyProfile:
    """Validation data for a company."""
    name: str
    linkedin_url: str = ""
    website: str = ""
    domain: str = ""
    employee_count: int = 0
    founded_year: int = 0
    industry: str = ""
    glassdoor_rating: float = 0.0
    is_verified: bool = False
    whois_age_days: int = 0
    has_linkedin_page: bool = False
    linkedin_followers: int = 0
    verification_source: str = ""


@dataclass
class ValidationResult:
    """Complete analysis result for a job posting."""
    job: JobPosting
    signals: list[ScamSignal] = field(default_factory=list)
    scam_score: float = 0.0
    confidence: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    company_profile: CompanyProfile | None = None
    ai_analysis: str = ""
    ai_tier_used: str = ""
    analysis_time_ms: float = 0.0

    @property
    def red_flags(self) -> list[ScamSignal]:
        return [s for s in self.signals if s.category == SignalCategory.RED_FLAG]

    @property
    def warnings(self) -> list[ScamSignal]:
        return [s for s in self.signals if s.category == SignalCategory.WARNING]

    @property
    def ghost_indicators(self) -> list[ScamSignal]:
        return [s for s in self.signals if s.category == SignalCategory.GHOST_JOB]

    @property
    def positive_signals(self) -> list[ScamSignal]:
        return [s for s in self.signals if s.category == SignalCategory.POSITIVE]

    def risk_label(self) -> str:
        if self.scam_score < 0.2:
            return "Verified Safe"
        elif self.scam_score < 0.4:
            return "Likely Legitimate"
        elif self.scam_score < 0.6:
            return "Suspicious"
        elif self.scam_score < 0.8:
            return "Likely Scam"
        return "Almost Certainly Scam"

    def to_dict(self) -> dict:
        return {
            "job": self.job.to_dict(),
            "scam_score": round(self.scam_score, 3),
            "confidence": round(self.confidence, 3),
            "risk_level": self.risk_level.value,
            "risk_label": self.risk_label(),
            "red_flags": [{"name": s.name, "detail": s.detail} for s in self.red_flags],
            "warnings": [{"name": s.name, "detail": s.detail} for s in self.warnings],
            "ghost_indicators": [{"name": s.name, "detail": s.detail} for s in self.ghost_indicators],
            "positive_signals": [{"name": s.name, "detail": s.detail} for s in self.positive_signals],
            "signal_count": len(self.signals),
            "ai_tier_used": self.ai_tier_used,
            "analysis_time_ms": round(self.analysis_time_ms, 1),
        }


@dataclass
class ScamPattern:
    """A known scam pattern stored in the knowledge base."""
    pattern_id: str
    name: str
    description: str
    category: SignalCategory
    regex: str = ""
    keywords: list[str] = field(default_factory=list)
    alpha: float = 1.0
    beta: float = 1.0
    observations: int = 0
    true_positives: int = 0
    false_positives: int = 0
    status: str = "active"  # active, deprecated, candidate
    created_at: str = ""
    updated_at: str = ""

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def bayesian_score(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class UserReport:
    """A user-submitted scam report that feeds the flywheel."""
    url: str
    is_scam: bool
    reason: str = ""
    reported_at: str = ""
    our_prediction: float = 0.0
    was_correct: bool = False
