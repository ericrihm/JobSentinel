"""Shared test fixtures."""

import pytest

from sentinel.models import JobPosting
from sentinel.db import SentinelDB
from sentinel.knowledge import KnowledgeBase
from sentinel.scorer import _RISK_THRESHOLDS


_DEFAULT_THRESHOLDS = dict(_RISK_THRESHOLDS)


@pytest.fixture(autouse=True)
def _reset_risk_thresholds():
    """Restore risk thresholds after each test to prevent cross-test pollution."""
    yield
    _RISK_THRESHOLDS.update(_DEFAULT_THRESHOLDS)


@pytest.fixture
def sample_job() -> JobPosting:
    """Realistic legitimate job posting — Google software engineer."""
    return JobPosting(
        url="https://www.linkedin.com/jobs/view/1234567890",
        title="Senior Software Engineer, Infrastructure",
        company="Google",
        location="Seattle, WA",
        description=(
            "As a Senior Software Engineer on Google's Infrastructure team, you will design, "
            "build, and maintain distributed systems that power our global compute platform. "
            "You will collaborate with cross-functional teams to define technical roadmaps, "
            "conduct code reviews, and mentor junior engineers.\n\n"
            "Minimum qualifications:\n"
            "- Bachelor's degree in Computer Science or equivalent practical experience\n"
            "- 5+ years of software development experience in Python, Go, or C++\n"
            "- Experience with large-scale distributed systems and microservices\n"
            "- Familiarity with Kubernetes, Borg, or similar container orchestration\n\n"
            "Preferred qualifications:\n"
            "- Experience with consensus protocols (Paxos, Raft)\n"
            "- Contributions to open-source infrastructure projects\n"
            "- Prior experience on-call for production systems at scale"
        ),
        salary_min=180_000.0,
        salary_max=280_000.0,
        salary_currency="USD",
        posted_date="2026-04-15",
        applicant_count=312,
        experience_level="Senior",
        employment_type="Full-time",
        industry="Technology",
        company_size="10001+",
        company_linkedin_url="https://www.linkedin.com/company/google",
        recruiter_name="Jane Smith",
        recruiter_connections=842,
        is_remote=False,
        is_repost=False,
        source="linkedin",
    )


@pytest.fixture
def scam_job() -> JobPosting:
    """Obvious scam job posting — guaranteed income, upfront payment, vague description."""
    return JobPosting(
        url="https://www.linkedin.com/jobs/view/9999999999",
        title="Work From Home — Earn $5,000/Week GUARANTEED",
        company="Global Opportunities LLC",
        location="Remote",
        description=(
            "We are hiring motivated individuals to join our team! "
            "Earn GUARANTEED $5,000 per week working from home. "
            "No experience required — anyone can qualify! "
            "You will be hired immediately after applying. No interview needed.\n\n"
            "Duties include: various tasks, assist with projects, general duties as assigned.\n\n"
            "To get started, you must pay a $99 registration fee and provide your Social Security "
            "Number and bank account number to process your direct deposit. "
            "Apply NOW — limited spots available! Offer expires soon. "
            "Contact us at globalopps@gmail.com for questions."
        ),
        salary_min=5_000.0,
        salary_max=25_000.0,
        salary_currency="USD",
        posted_date="2026-04-28",
        applicant_count=4801,
        experience_level="Entry level",
        employment_type="Contract",
        industry="",
        company_size="1-10",
        company_linkedin_url="",
        recruiter_name="Bob Johnson",
        recruiter_connections=12,
        is_remote=True,
        is_repost=False,
        source="linkedin",
    )


@pytest.fixture
def ghost_job() -> JobPosting:
    """Stale reposted ghost job — 60 days old, marked as repost."""
    return JobPosting(
        url="https://www.linkedin.com/jobs/view/5555555555",
        title="Marketing Coordinator",
        company="Acme Corp",
        location="Chicago, IL",
        description=(
            "Acme Corp is always looking for talented marketing coordinators to join our team. "
            "Responsibilities include assisting with campaigns, creating content, and other duties "
            "as assigned. We maintain an open pipeline of candidates for future opportunities."
        ),
        salary_min=45_000.0,
        salary_max=55_000.0,
        salary_currency="USD",
        posted_date="2026-03-01",
        applicant_count=203,
        experience_level="Entry level",
        employment_type="Full-time",
        industry="Marketing",
        company_size="51-200",
        company_linkedin_url="https://www.linkedin.com/company/acme-corp",
        recruiter_name="",
        recruiter_connections=0,
        is_remote=False,
        is_repost=True,
        source="linkedin",
    )


@pytest.fixture
def temp_db(tmp_path) -> SentinelDB:
    """SentinelDB backed by a temporary file; closed after the test."""
    db_path = str(tmp_path / "test_sentinel.db")
    db = SentinelDB(path=db_path)
    yield db
    db.close()


@pytest.fixture
def seeded_db(temp_db: SentinelDB) -> SentinelDB:
    """temp_db pre-populated with default scam patterns via KnowledgeBase."""
    kb = KnowledgeBase(db=temp_db)
    kb.seed_default_patterns()
    yield temp_db
