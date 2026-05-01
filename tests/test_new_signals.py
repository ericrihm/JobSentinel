"""Tests for new signals: knowledge pattern matching, fraud types, positive signals."""

import os

import pytest

from sentinel.db import SentinelDB
from sentinel.knowledge import KnowledgeBase
from sentinel.models import JobPosting, SignalCategory
from sentinel.signals import (
    _reset_kb_cache,
    check_contact_channel_suspicious,
    check_evolved_mlm,
    check_fake_staffing_agency,
    check_government_impersonation,
    check_knowledge_patterns,
    check_pig_butchering_job,
    check_professional_application_process,
    check_survey_clickfarm,
    check_verified_company_website,
    check_visa_sponsorship_scam,
    check_established_company,
    extract_signals_with_kb,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(**overrides) -> JobPosting:
    """Create a minimal JobPosting with sensible defaults, overriding as needed."""
    defaults = {
        "url": "https://linkedin.com/jobs/view/123",
        "title": "Software Engineer",
        "company": "TestCorp",
        "location": "Remote",
        "description": "A normal job posting with nothing suspicious.",
        "company_linkedin_url": "https://linkedin.com/company/testcorp",
        "company_size": "500",
        "recruiter_connections": 200,
    }
    defaults.update(overrides)
    return JobPosting(**defaults)


@pytest.fixture(autouse=True)
def clear_kb_cache():
    """Ensure the knowledge-base cache is cleared between tests."""
    _reset_kb_cache()
    yield
    _reset_kb_cache()


# ===================================================================
# Knowledge pattern matching tests
# ===================================================================


class TestKnowledgePatternMatching:
    """Test check_knowledge_patterns wiring."""

    def test_regex_match_fake_check(self, tmp_path):
        """KB pattern with regex matches fake check scam language."""
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        kb = KnowledgeBase(db=db)
        kb.seed_default_patterns()
        db.close()

        job = _make_job(
            description="We will send check to your address. Please deposit check and wire back $500."
        )
        signals = check_knowledge_patterns(job, db_path=db_path)
        names = [s.name for s in signals]
        assert any("kb_fake_check_scam" in n for n in names)

    def test_keyword_match_mystery_shopper(self, tmp_path):
        """KB pattern matches via keyword fallback (mystery shopper)."""
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        kb = KnowledgeBase(db=db)
        kb.seed_default_patterns()
        db.close()

        # mystery_shopper pattern requires regex match or 2+ keywords
        job = _make_job(
            description="Become a mystery shopper! Evaluate stores and shop and get paid for your time."
        )
        signals = check_knowledge_patterns(job, db_path=db_path)
        names = [s.name for s in signals]
        assert any("mystery_shopper" in n for n in names)

    def test_no_match_on_clean_job(self, tmp_path):
        """Clean job posting should not match knowledge-base patterns."""
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        kb = KnowledgeBase(db=db)
        kb.seed_default_patterns()
        db.close()

        job = _make_job(
            description=(
                "We are looking for a senior Python developer with 5+ years experience "
                "in Django, PostgreSQL, and AWS. Must have a bachelor's degree."
            )
        )
        signals = check_knowledge_patterns(job, db_path=db_path)
        assert len(signals) == 0

    def test_kb_signal_has_correct_prefix(self, tmp_path):
        """KB signals should be named kb_<pattern_id>."""
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        kb = KnowledgeBase(db=db)
        kb.seed_default_patterns()
        db.close()

        job = _make_job(
            description="Join our network marketing team! Build your own team and earn unlimited earning potential."
        )
        signals = check_knowledge_patterns(job, db_path=db_path)
        for s in signals:
            assert s.name.startswith("kb_")

    def test_extract_signals_with_kb_includes_regular_and_kb(self, tmp_path):
        """extract_signals_with_kb includes both regular and KB signals."""
        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)
        kb = KnowledgeBase(db=db)
        kb.seed_default_patterns()
        db.close()

        job = _make_job(
            description="Stuff envelopes from home! Envelope stuffing pays $50/hr. No experience needed.",
            company_linkedin_url="",
        )
        signals = extract_signals_with_kb(job, db_path=db_path)
        names = [s.name for s in signals]
        # Should have both regular signals and kb_ prefixed signals
        regular = [n for n in names if not n.startswith("kb_")]
        kb_sigs = [n for n in names if n.startswith("kb_")]
        assert len(regular) > 0
        assert len(kb_sigs) > 0


# ===================================================================
# Pig butchering signal tests
# ===================================================================


class TestPigButcheringJob:
    def test_detects_cryptocurrency_trader(self):
        job = _make_job(
            title="Cryptocurrency Trader",
            description="Join our team as a cryptocurrency trader managing digital portfolios.",
        )
        sig = check_pig_butchering_job(job)
        assert sig is not None
        assert sig.name == "pig_butchering_job"
        assert sig.weight == 0.88

    def test_detects_defi_analyst(self):
        job = _make_job(
            description="We need a DeFi analyst to evaluate liquidity pools and yield farming strategies.",
        )
        sig = check_pig_butchering_job(job)
        assert sig is not None

    def test_detects_liquidity_provider(self):
        job = _make_job(description="Role: liquidity provider for our exchange platform.")
        sig = check_pig_butchering_job(job)
        assert sig is not None

    def test_no_false_positive_on_legit_finance(self):
        job = _make_job(
            description="Looking for a financial analyst to build Excel models and present to stakeholders."
        )
        sig = check_pig_butchering_job(job)
        assert sig is None


# ===================================================================
# Survey / click-farm signal tests
# ===================================================================


class TestSurveyClickfarm:
    def test_detects_online_survey_earn(self):
        job = _make_job(description="Take online surveys and earn money from home! Up to $500/week.")
        sig = check_survey_clickfarm(job)
        assert sig is not None
        assert sig.name == "survey_clickfarm"
        assert sig.weight == 0.75

    def test_detects_paid_per_click(self):
        job = _make_job(description="Get paid per click! Easy work from your phone.")
        sig = check_survey_clickfarm(job)
        assert sig is not None

    def test_detects_social_media_evaluator(self):
        job = _make_job(description="We need a social media evaluator to review ads on Facebook.")
        sig = check_survey_clickfarm(job)
        assert sig is not None

    def test_no_false_positive_on_research_role(self):
        job = _make_job(
            description="UX researcher needed to design and conduct user surveys and usability studies."
        )
        sig = check_survey_clickfarm(job)
        assert sig is None


# ===================================================================
# Visa sponsorship scam tests
# ===================================================================


class TestVisaSponsorshipScam:
    def test_detects_guaranteed_visa(self):
        job = _make_job(description="We offer guaranteed visa sponsorship for all applicants.")
        sig = check_visa_sponsorship_scam(job)
        assert sig is not None
        assert sig.name == "visa_sponsorship_scam"
        assert sig.weight == 0.82

    def test_detects_visa_processing_fee(self):
        job = _make_job(description="Visa processing fee of $500 is required to begin the application.")
        sig = check_visa_sponsorship_scam(job)
        assert sig is not None

    def test_detects_immigration_fee(self):
        job = _make_job(description="A small immigration fee is required to process your work authorization.")
        sig = check_visa_sponsorship_scam(job)
        assert sig is not None

    def test_no_false_positive_on_legit_visa(self):
        job = _make_job(description="This role offers visa sponsorship for qualified candidates. No fees.")
        sig = check_visa_sponsorship_scam(job)
        assert sig is None


# ===================================================================
# Government impersonation tests
# ===================================================================


class TestGovernmentImpersonation:
    def test_detects_fbi_with_personal_info(self):
        job = _make_job(
            description="The FBI is hiring analysts. Please provide your SSN and passport to apply.",
            company_linkedin_url="",
        )
        sig = check_government_impersonation(job)
        assert sig is not None
        assert sig.name == "government_impersonation"
        assert sig.weight == 0.88  # no company LinkedIn

    def test_detects_dod_with_info_request(self):
        job = _make_job(
            description=(
                "Department of Defense contractor position. "
                "Submit your personal information to get started."
            ),
            company_linkedin_url="https://linkedin.com/company/dod",
        )
        sig = check_government_impersonation(job)
        assert sig is not None
        assert sig.weight == 0.72  # has company LinkedIn

    def test_no_false_positive_without_info_request(self):
        job = _make_job(
            description="The FBI is hiring data scientists. Apply through usajobs.gov.",
        )
        sig = check_government_impersonation(job)
        assert sig is None


# ===================================================================
# Fake staffing agency tests
# ===================================================================


class TestFakeStaffingAgency:
    def test_detects_brand_mismatch(self):
        job = _make_job(
            title="Software Engineer at Google",
            company="Apex Recruiting",
            description="We are hiring for Google. Top pay and benefits.",
            company_linkedin_url="",
        )
        sig = check_fake_staffing_agency(job)
        assert sig is not None
        assert sig.name == "fake_staffing_agency"
        assert sig.weight == 0.68

    def test_no_flag_when_company_is_brand(self):
        job = _make_job(
            title="Software Engineer",
            company="Google",
            description="Google is hiring engineers for our cloud platform.",
            company_linkedin_url="https://linkedin.com/company/google",
        )
        sig = check_fake_staffing_agency(job)
        assert sig is None

    def test_no_flag_when_agency_has_linkedin(self):
        job = _make_job(
            title="Software Engineer at Amazon",
            company="Staffing Corp",
            description="We recruit for Amazon. Apply now.",
            company_linkedin_url="https://linkedin.com/company/staffingcorp",
        )
        sig = check_fake_staffing_agency(job)
        assert sig is None


# ===================================================================
# Evolved MLM tests
# ===================================================================


class TestEvolvedMLM:
    def test_detects_brand_ambassador(self):
        job = _make_job(
            description="Become a brand ambassador and earn unlimited income selling our products!"
        )
        sig = check_evolved_mlm(job)
        assert sig is not None
        assert sig.name == "evolved_mlm"
        assert sig.weight == 0.78

    def test_detects_independent_business_owner(self):
        job = _make_job(description="Be your own independent business owner with our opportunity.")
        sig = check_evolved_mlm(job)
        assert sig is not None

    def test_detects_starter_inventory(self):
        job = _make_job(description="Purchase your starter inventory to begin selling today!")
        sig = check_evolved_mlm(job)
        assert sig is not None

    def test_no_false_positive_on_legit_ambassador(self):
        job = _make_job(
            description=(
                "We are looking for a brand manager to oversee our product launches "
                "and coordinate with marketing teams."
            )
        )
        sig = check_evolved_mlm(job)
        assert sig is None


# ===================================================================
# Contact channel suspicious tests
# ===================================================================


class TestContactChannelSuspicious:
    def test_detects_telegram_apply(self):
        job = _make_job(description="Interested? Apply via Telegram @recruiter123")
        sig = check_contact_channel_suspicious(job)
        assert sig is not None
        assert sig.name == "contact_channel_suspicious"
        assert sig.weight == 0.65

    def test_detects_whatsapp_contact(self):
        job = _make_job(description="Contact us on WhatsApp for more details about this position.")
        sig = check_contact_channel_suspicious(job)
        assert sig is not None

    def test_detects_cell_only_no_linkedin(self):
        job = _make_job(
            description="Call me at 555-123-4567 for more information about this role.",
            company_linkedin_url="",
        )
        sig = check_contact_channel_suspicious(job)
        assert sig is not None
        assert sig.weight == 0.55

    def test_no_flag_with_linkedin_and_phone(self):
        job = _make_job(
            description="Our office number is 555-123-4567. Apply through our website.",
            company_linkedin_url="https://linkedin.com/company/testcorp",
        )
        # The cell_only_contact branch requires no company LinkedIn
        # and the telegram/whatsapp branch won't match
        sig = check_contact_channel_suspicious(job)
        assert sig is None


# ===================================================================
# Updated positive signal tests
# ===================================================================


class TestEstablishedCompanyThreshold:
    """Test that the threshold was lowered from 1000 to 100."""

    def test_100_employees_triggers(self):
        job = _make_job(company_size="100")
        sig = check_established_company(job)
        assert sig is not None
        assert sig.name == "established_company"

    def test_500_employees_triggers(self):
        job = _make_job(company_size="500")
        sig = check_established_company(job)
        assert sig is not None

    def test_50_employees_does_not_trigger(self):
        job = _make_job(company_size="50")
        sig = check_established_company(job)
        assert sig is None


class TestVerifiedCompanyWebsite:
    def test_detects_matching_website(self):
        job = _make_job(
            company="Acme",
            description="Apply at https://acme.com/careers for this role.",
        )
        sig = check_verified_company_website(job)
        assert sig is not None
        assert sig.name == "verified_company_website"
        assert sig.category == SignalCategory.POSITIVE

    def test_no_match_different_domain(self):
        job = _make_job(
            company="Acme",
            description="Apply at https://example.com/careers for this role.",
        )
        sig = check_verified_company_website(job)
        assert sig is None


class TestProfessionalApplicationProcess:
    def test_detects_greenhouse(self):
        job = _make_job(description="Apply through Greenhouse to submit your application.")
        sig = check_professional_application_process(job)
        assert sig is not None
        assert sig.name == "professional_application_process"
        assert sig.category == SignalCategory.POSITIVE

    def test_detects_careers_page(self):
        job = _make_job(description="Apply on our careers page at company.com/careers/")
        sig = check_professional_application_process(job)
        assert sig is not None

    def test_no_match_on_generic(self):
        job = _make_job(description="Send your resume to hr@company.com")
        sig = check_professional_application_process(job)
        assert sig is None


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_empty_description_no_crash(self):
        """All new signals should handle empty descriptions gracefully."""
        job = _make_job(description="", title="")
        assert check_pig_butchering_job(job) is None
        assert check_survey_clickfarm(job) is None
        assert check_visa_sponsorship_scam(job) is None
        assert check_government_impersonation(job) is None
        assert check_fake_staffing_agency(job) is None
        assert check_evolved_mlm(job) is None
        assert check_contact_channel_suspicious(job) is None
        assert check_verified_company_website(job) is None
        assert check_professional_application_process(job) is None

    def test_case_insensitivity(self):
        """Signals should be case-insensitive."""
        job = _make_job(description="APPLY VIA TELEGRAM for this amazing opportunity!")
        sig = check_contact_channel_suspicious(job)
        assert sig is not None

    def test_multiple_fraud_types_detected(self):
        """A posting with multiple fraud types should fire multiple signals."""
        job = _make_job(
            description=(
                "DeFi analyst needed! Apply via Telegram. "
                "Purchase your starter inventory to get started. "
                "Guaranteed visa sponsorship with a small immigration fee."
            ),
            company_linkedin_url="",
        )
        signals = []
        sig = check_pig_butchering_job(job)
        if sig:
            signals.append(sig)
        sig = check_contact_channel_suspicious(job)
        if sig:
            signals.append(sig)
        sig = check_evolved_mlm(job)
        if sig:
            signals.append(sig)
        sig = check_visa_sponsorship_scam(job)
        if sig:
            signals.append(sig)
        # At least 3 should fire
        assert len(signals) >= 3
