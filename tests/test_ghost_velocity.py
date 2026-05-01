"""Tests for enhanced ghost job signals and posting velocity / dedup signals.

Covers:
- talent_pool_language (positive + negative)
- high_applicant_count (positive + negative + edge cases)
- role_title_generic (positive + negative)
- high_posting_velocity (positive + negative, DB-backed)
- new_recruiter_account (positive + negative, convention-based)
- cross_posting_duplicate (positive + negative, DB-backed)
- DB methods: upsert/get_posting_velocity, record/get_duplicate_description_hash
- extract_signals_with_db integration
"""

import hashlib

import pytest

from sentinel.db import SentinelDB
from sentinel.models import JobPosting, SignalCategory
from sentinel.signals import (
    _description_hash,
    check_cross_posting_duplicate,
    check_high_applicant_count,
    check_high_posting_velocity,
    check_new_recruiter_account,
    check_role_title_generic,
    check_talent_pool_language,
    extract_signals,
    extract_signals_with_db,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path) -> SentinelDB:
    db = SentinelDB(path=str(tmp_path / "test_ghost.db"))
    yield db
    db.close()


def _job(**kwargs) -> JobPosting:
    """Minimal JobPosting factory — only sets supplied kwargs, rest are defaults."""
    return JobPosting(**kwargs)


# ---------------------------------------------------------------------------
# 1. talent_pool_language — positive cases
# ---------------------------------------------------------------------------


class TestTalentPoolLanguage:
    def test_talent_pipeline_keyword(self):
        job = _job(title="Engineer", description="Join our talent pipeline for future roles.")
        sig = check_talent_pool_language(job)
        assert sig is not None
        assert sig.name == "talent_pool_language"
        assert sig.category == SignalCategory.GHOST_JOB
        assert sig.weight == pytest.approx(0.55)

    def test_talent_community_keyword(self):
        job = _job(title="Various Positions", description="Become part of our talent community.")
        sig = check_talent_pool_language(job)
        assert sig is not None
        assert "talent_pool_language" == sig.name

    def test_future_openings_keyword(self):
        job = _job(title="Ops", description="Apply now for future openings at our company.")
        sig = check_talent_pool_language(job)
        assert sig is not None

    def test_expressions_of_interest(self):
        job = _job(title="Team Member", description="We accept expressions of interest year-round.")
        sig = check_talent_pool_language(job)
        assert sig is not None

    def test_no_talent_pool_language_returns_none(self):
        job = _job(
            title="Software Engineer",
            description="Build scalable Python services. Requires 3+ years experience.",
        )
        sig = check_talent_pool_language(job)
        assert sig is None

    def test_phrase_in_title_is_detected(self):
        """Keyword in title field should also be caught via _full_text."""
        job = _job(title="Join our talent community", description="We are hiring.")
        sig = check_talent_pool_language(job)
        assert sig is not None


# ---------------------------------------------------------------------------
# 2. high_applicant_count — positive / negative / edge cases
# ---------------------------------------------------------------------------


class TestHighApplicantCount:
    def test_fires_when_over_500_and_over_30_days(self):
        # 60 days ago
        job = _job(applicant_count=600, posted_date="2026-03-01")
        sig = check_high_applicant_count(job)
        assert sig is not None
        assert sig.name == "high_applicant_count"
        assert sig.category == SignalCategory.GHOST_JOB
        assert sig.weight == pytest.approx(0.48)

    def test_does_not_fire_when_under_500(self):
        job = _job(applicant_count=499, posted_date="2026-03-01")
        sig = check_high_applicant_count(job)
        assert sig is None

    def test_does_not_fire_when_recent_even_with_many_applicants(self):
        # posted_date ~today minus 5 days — well within the 30-day window
        job = _job(applicant_count=1000, posted_date="2026-04-25")
        sig = check_high_applicant_count(job)
        assert sig is None

    def test_does_not_fire_when_no_posted_date(self):
        job = _job(applicant_count=1000, posted_date="")
        sig = check_high_applicant_count(job)
        assert sig is None

    def test_boundary_exactly_500_does_not_fire(self):
        job = _job(applicant_count=500, posted_date="2026-03-01")
        sig = check_high_applicant_count(job)
        assert sig is None

    def test_boundary_501_and_31_days_does_fire(self):
        job = _job(applicant_count=501, posted_date="2026-03-30")
        sig = check_high_applicant_count(job)
        assert sig is not None


# ---------------------------------------------------------------------------
# 3. role_title_generic
# ---------------------------------------------------------------------------


class TestRoleTitleGeneric:
    def test_various_positions_fires(self):
        job = _job(title="Various Positions", description="We are hiring.")
        sig = check_role_title_generic(job)
        assert sig is not None
        assert sig.name == "role_title_generic"
        assert sig.category == SignalCategory.GHOST_JOB
        assert sig.weight == pytest.approx(0.42)

    def test_multiple_openings_fires(self):
        job = _job(title="Multiple Openings", description="Join us.")
        sig = check_role_title_generic(job)
        assert sig is not None

    def test_team_member_fires(self):
        job = _job(title="Team Member", description="General duties.")
        sig = check_role_title_generic(job)
        assert sig is not None

    def test_real_title_does_not_fire(self):
        job = _job(title="Senior Backend Engineer", description="Build APIs.")
        sig = check_role_title_generic(job)
        assert sig is None

    def test_partial_match_in_longer_title_does_not_fire(self):
        """'Team Member Relations Specialist' should NOT match the whole-title pattern."""
        job = _job(title="Team Member Relations Specialist", description="HR role.")
        sig = check_role_title_generic(job)
        assert sig is None


# ---------------------------------------------------------------------------
# 4. high_posting_velocity (DB-backed)
# ---------------------------------------------------------------------------


class TestHighPostingVelocity:
    def test_fires_when_over_20_in_24h(self, temp_db):
        temp_db.upsert_posting_velocity("ScamCo", postings_24h=25, postings_7d=100)
        job = _job(company="ScamCo", description="Some job.")
        sig = check_high_posting_velocity(job, db=temp_db)
        assert sig is not None
        assert sig.name == "high_posting_velocity"
        assert sig.category == SignalCategory.GHOST_JOB
        assert sig.weight == pytest.approx(0.65)

    def test_does_not_fire_when_under_threshold(self, temp_db):
        temp_db.upsert_posting_velocity("LegitCo", postings_24h=10, postings_7d=40)
        job = _job(company="LegitCo", description="Some job.")
        sig = check_high_posting_velocity(job, db=temp_db)
        assert sig is None

    def test_does_not_fire_when_no_db(self):
        job = _job(company="AnyCompany", description="Some job.")
        sig = check_high_posting_velocity(job, db=None)
        assert sig is None

    def test_does_not_fire_when_company_not_in_db(self, temp_db):
        job = _job(company="UnknownCorp", description="Some job.")
        sig = check_high_posting_velocity(job, db=temp_db)
        assert sig is None

    def test_boundary_exactly_20_does_not_fire(self, temp_db):
        temp_db.upsert_posting_velocity("BorderCo", postings_24h=20, postings_7d=60)
        job = _job(company="BorderCo", description="Some job.")
        sig = check_high_posting_velocity(job, db=temp_db)
        assert sig is None

    def test_boundary_21_fires(self, temp_db):
        temp_db.upsert_posting_velocity("BorderCo2", postings_24h=21, postings_7d=80)
        job = _job(company="BorderCo2", description="Some job.")
        sig = check_high_posting_velocity(job, db=temp_db)
        assert sig is not None


# ---------------------------------------------------------------------------
# 5. new_recruiter_account (convention: recruiter_connections == -1)
# ---------------------------------------------------------------------------


class TestNewRecruiterAccount:
    def test_fires_when_connections_is_negative_one(self):
        job = _job(recruiter_connections=-1)
        sig = check_new_recruiter_account(job)
        assert sig is not None
        assert sig.name == "new_recruiter_account"
        assert sig.category == SignalCategory.WARNING
        assert sig.weight == pytest.approx(0.55)

    def test_does_not_fire_for_zero_connections(self):
        """Zero connections is handled by check_low_recruiter_connections, not this signal."""
        job = _job(recruiter_connections=0)
        sig = check_new_recruiter_account(job)
        assert sig is None

    def test_does_not_fire_for_normal_connections(self):
        job = _job(recruiter_connections=500)
        sig = check_new_recruiter_account(job)
        assert sig is None


# ---------------------------------------------------------------------------
# 6. cross_posting_duplicate (DB-backed)
# ---------------------------------------------------------------------------


class TestCrossPostingDuplicate:
    def test_fires_when_same_hash_under_different_company(self, temp_db):
        desc = "We are a dynamic team looking for a go-getter to join us immediately!"
        h = _description_hash(desc)
        temp_db.record_description_hash(h, "ScamCorp A", "https://example.com/a")

        job = _job(company="ScamCorp B", description=desc)
        sig = check_cross_posting_duplicate(job, db=temp_db)
        assert sig is not None
        assert sig.name == "cross_posting_duplicate"
        assert sig.category == SignalCategory.RED_FLAG
        assert sig.weight == pytest.approx(0.72)
        assert "ScamCorp A" in sig.detail

    def test_does_not_fire_for_same_company(self, temp_db):
        desc = "Standard job description for a marketing role."
        h = _description_hash(desc)
        temp_db.record_description_hash(h, "Legit Inc", "https://example.com/job1")

        job = _job(company="Legit Inc", description=desc)
        sig = check_cross_posting_duplicate(job, db=temp_db)
        assert sig is None

    def test_does_not_fire_with_no_db(self):
        job = _job(company="Any Co", description="Some description.")
        sig = check_cross_posting_duplicate(job, db=None)
        assert sig is None

    def test_does_not_fire_for_empty_description(self, temp_db):
        job = _job(company="Co", description="")
        sig = check_cross_posting_duplicate(job, db=temp_db)
        assert sig is None

    def test_normalised_whitespace_matches(self, temp_db):
        """Descriptions that differ only in whitespace should hash to the same value."""
        desc_a = "We are hiring   a software engineer  ."
        desc_b = "We are hiring a software engineer ."
        h_a = _description_hash(desc_a)
        h_b = _description_hash(desc_b)
        assert h_a == h_b

        temp_db.record_description_hash(h_a, "Corp A", "https://example.com/1")
        job = _job(company="Corp B", description=desc_b)
        sig = check_cross_posting_duplicate(job, db=temp_db)
        assert sig is not None


# ---------------------------------------------------------------------------
# 7. DB methods: posting_velocity and description_hashes
# ---------------------------------------------------------------------------


class TestDBVelocityMethods:
    def test_upsert_and_get_posting_velocity(self, temp_db):
        temp_db.upsert_posting_velocity("MegaCorp", postings_24h=30, postings_7d=120)
        row = temp_db.get_posting_velocity("MegaCorp")
        assert row is not None
        assert row["postings_24h"] == 30
        assert row["postings_7d"] == 120

    def test_upsert_updates_existing(self, temp_db):
        temp_db.upsert_posting_velocity("UpdateCo", postings_24h=5, postings_7d=20)
        temp_db.upsert_posting_velocity("UpdateCo", postings_24h=50, postings_7d=200)
        row = temp_db.get_posting_velocity("UpdateCo")
        assert row["postings_24h"] == 50

    def test_get_posting_velocity_returns_none_for_unknown(self, temp_db):
        row = temp_db.get_posting_velocity("NobodyKnowsThis")
        assert row is None

    def test_get_posting_velocity_case_insensitive(self, temp_db):
        temp_db.upsert_posting_velocity("ACME Corp", postings_24h=15, postings_7d=60)
        row = temp_db.get_posting_velocity("acme corp")
        assert row is not None

    def test_record_and_get_description_hash(self, temp_db):
        h = "abc123deadbeef"
        temp_db.record_description_hash(h, "CompanyX", "https://x.com/job/1")
        matches = temp_db.get_duplicate_description(h, exclude_company="CompanyY")
        assert len(matches) == 1
        assert matches[0]["company_name"] == "CompanyX"

    def test_get_duplicate_description_excludes_correct_company(self, temp_db):
        h = "hashvalue999"
        temp_db.record_description_hash(h, "Alpha", "https://alpha.com/1")
        temp_db.record_description_hash(h, "Beta", "https://beta.com/1")

        matches = temp_db.get_duplicate_description(h, exclude_company="Alpha")
        assert all(m["company_name"] != "Alpha" for m in matches)
        assert any(m["company_name"] == "Beta" for m in matches)

    def test_duplicate_hash_record_same_company_is_idempotent(self, temp_db):
        h = "idempotent_hash"
        temp_db.record_description_hash(h, "UniCo", "https://uni.co/1")
        temp_db.record_description_hash(h, "UniCo", "https://uni.co/1")  # duplicate
        all_rows = temp_db.get_duplicate_description(h)
        assert len(all_rows) == 1  # INSERT OR IGNORE ensures idempotency


# ---------------------------------------------------------------------------
# 8. extract_signals_with_db integration
# ---------------------------------------------------------------------------


class TestExtractSignalsWithDB:
    def test_cross_posting_signal_appears_via_extract_signals_with_db(self, temp_db):
        desc = "Hire anyone who applies. No experience needed. Guaranteed income."
        h = _description_hash(desc)
        temp_db.record_description_hash(h, "Fraud LLC", "https://fraud.com/1")

        job = _job(company="Scam Inc", description=desc)
        signals = extract_signals_with_db(job, temp_db)
        names = [s.name for s in signals]
        assert "cross_posting_duplicate" in names

    def test_high_velocity_signal_appears_via_extract_signals_with_db(self, temp_db):
        temp_db.upsert_posting_velocity("RapidFire Co", postings_24h=50, postings_7d=300)
        job = _job(company="RapidFire Co", description="Some job description here.")
        signals = extract_signals_with_db(job, temp_db)
        names = [s.name for s in signals]
        assert "high_posting_velocity" in names

    def test_stateless_signals_still_run_in_extract_signals_with_db(self, temp_db):
        job = _job(
            title="Work from home",
            description="No experience required. Anyone can qualify.",
            is_remote=True,
        )
        signals = extract_signals_with_db(job, temp_db)
        names = [s.name for s in signals]
        assert "wfh_unrealistic_pay" in names

    def test_description_hash_helper_is_deterministic(self):
        desc = "  Hello   world  "
        assert _description_hash(desc) == _description_hash(desc)
        # Normalisation collapses multiple spaces
        assert _description_hash("hello world") == _description_hash("hello  world")
