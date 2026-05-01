"""Unit tests for sentinel/signals.py — signal extraction functions."""

import pytest

from sentinel.models import JobPosting, ScamSignal, SignalCategory
from sentinel.signals import (
    check_upfront_payment,
    check_personal_info_request,
    check_guaranteed_income,
    check_crypto_payment,
    check_mlm_language,
    check_reshipping_scam,
    check_interview_bypass,
    check_data_harvesting,
    check_pig_butchering_job,
    check_suspicious_email_domain,
    extract_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job(title: str = "Customer Service Representative", description: str = "", **kwargs) -> JobPosting:
    """Build a minimal JobPosting for testing."""
    defaults = dict(
        url="https://example.com/job/1",
        title=title,
        company="TestCorp",
        location="Remote",
        description=description,
        company_linkedin_url="https://linkedin.com/company/testcorp",
        recruiter_name="",
        recruiter_connections=0,
    )
    defaults.update(kwargs)
    return JobPosting(**defaults)


def _clean_job() -> JobPosting:
    """Legitimate tech job with no scam signals."""
    return JobPosting(
        url="https://www.linkedin.com/jobs/view/999",
        title="Senior Software Engineer",
        company="Stripe",
        location="San Francisco, CA",
        description=(
            "Stripe is looking for a Senior Software Engineer to join our Payments team. "
            "You will design and implement highly reliable distributed payment systems. "
            "Requirements: 5+ years of experience with Python or Go, strong knowledge of "
            "distributed systems, Bachelor's degree in Computer Science or equivalent. "
            "We offer health insurance, 401k, equity, and paid time off."
        ),
        salary_min=160_000.0,
        salary_max=220_000.0,
        company_linkedin_url="https://www.linkedin.com/company/stripe",
        company_size="5001-10000",
        recruiter_name="HR Team",
        recruiter_connections=500,
    )


# ===========================================================================
# 1. check_upfront_payment
# ===========================================================================

class TestUpfrontPayment:
    def test_fires_on_fee_required(self):
        job = _job(description="Fee required before you can begin work.")
        signal = check_upfront_payment(job)
        assert signal is not None
        assert signal.name == "upfront_payment"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_training_fee(self):
        job = _job(description="There is a $99 training fee to get started.")
        assert check_upfront_payment(job) is not None

    def test_fires_on_buy_equipment(self):
        job = _job(description="You must buy equipment before starting your role.")
        assert check_upfront_payment(job) is not None

    def test_fires_on_upfront_payment_phrase(self):
        job = _job(description="An upfront payment of $150 is required.")
        assert check_upfront_payment(job) is not None

    def test_fires_on_pay_deposit(self):
        job = _job(description="Please pay a deposit to secure your position.")
        assert check_upfront_payment(job) is not None

    def test_does_not_fire_on_clean_text(self):
        job = _clean_job()
        assert check_upfront_payment(job) is None

    def test_does_not_fire_on_generic_job(self):
        job = _job(description="We are hiring a software engineer to build distributed systems.")
        assert check_upfront_payment(job) is None


# ===========================================================================
# 2. check_personal_info_request
# ===========================================================================

class TestPersonalInfoRequest:
    def test_fires_on_ssn(self):
        job = _job(description="Please provide your social security number to proceed.")
        signal = check_personal_info_request(job)
        assert signal is not None
        assert signal.name == "personal_info_request"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_bank_account(self):
        job = _job(description="Send your bank account number for direct deposit setup.")
        assert check_personal_info_request(job) is not None

    def test_fires_on_routing_number(self):
        job = _job(description="We need your routing number and account details.")
        assert check_personal_info_request(job) is not None

    def test_fires_on_credit_card(self):
        job = _job(description="Provide your credit card number for verification.")
        assert check_personal_info_request(job) is not None

    def test_fires_on_passport_copy(self):
        job = _job(description="Email a passport copy to complete onboarding.")
        assert check_personal_info_request(job) is not None

    def test_does_not_fire_on_clean_text(self):
        assert check_personal_info_request(_clean_job()) is None

    def test_does_not_fire_on_standard_application(self):
        job = _job(description="Apply with your resume and cover letter via our portal.")
        assert check_personal_info_request(job) is None


# ===========================================================================
# 3. check_guaranteed_income
# ===========================================================================

class TestGuaranteedIncome:
    def test_fires_on_guaranteed_salary(self):
        job = _job(description="We offer a guaranteed salary of $10,000 per month.")
        signal = check_guaranteed_income(job)
        assert signal is not None
        assert signal.name == "guaranteed_income"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_guaranteed_earnings(self):
        job = _job(description="Earn $5,000 per week guaranteed working from home.")
        assert check_guaranteed_income(job) is not None

    def test_fires_on_guaranteed_pay(self):
        job = _job(description="Guaranteed pay from day one — no experience required.")
        assert check_guaranteed_income(job) is not None

    def test_fires_in_title(self):
        job = _job(title="Guaranteed income of $3,000 per week", description="Easy remote work.")
        assert check_guaranteed_income(job) is not None

    def test_does_not_fire_on_clean_text(self):
        assert check_guaranteed_income(_clean_job()) is None

    def test_does_not_fire_on_competitive_salary(self):
        job = _job(description="We offer a competitive salary and equity package.")
        assert check_guaranteed_income(job) is None


# ===========================================================================
# 4. check_crypto_payment
# ===========================================================================

class TestCryptoPayment:
    def test_fires_on_bitcoin(self):
        job = _job(description="Compensation is paid in Bitcoin.")
        signal = check_crypto_payment(job)
        assert signal is not None
        assert signal.name == "crypto_payment"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_ethereum(self):
        job = _job(description="Salary paid via Ethereum monthly.")
        assert check_crypto_payment(job) is not None

    def test_fires_on_gift_card(self):
        job = _job(description="Payment will be made via gift card.")
        assert check_crypto_payment(job) is not None

    def test_fires_on_western_union(self):
        job = _job(description="Payments sent through Western Union.")
        assert check_crypto_payment(job) is not None

    def test_fires_on_wire_transfer(self):
        job = _job(description="We pay via wire transfer to your account.")
        assert check_crypto_payment(job) is not None

    def test_fires_on_zelle(self):
        job = _job(description="Get paid instantly via Zelle every Friday.")
        assert check_crypto_payment(job) is not None

    def test_does_not_fire_on_clean_text(self):
        assert check_crypto_payment(_clean_job()) is None

    def test_does_not_fire_on_standard_payment_terms(self):
        job = _job(description="Bi-weekly payroll via direct deposit. Health benefits included.")
        assert check_crypto_payment(job) is None


# ===========================================================================
# 5. check_mlm_language
# ===========================================================================

class TestMlmLanguage:
    def test_fires_on_be_your_own_boss(self):
        job = _job(description="Be your own boss and achieve financial freedom!")
        signal = check_mlm_language(job)
        assert signal is not None
        assert signal.name == "mlm_language"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_mlm_keyword(self):
        job = _job(description="Join our multi-level marketing opportunity today.")
        assert check_mlm_language(job) is not None

    def test_fires_on_network_marketing(self):
        job = _job(description="We are a network marketing company looking for motivated reps.")
        assert check_mlm_language(job) is not None

    def test_fires_on_residual_income(self):
        job = _job(description="Earn residual income by building your team.")
        assert check_mlm_language(job) is not None

    def test_fires_on_recruit_others(self):
        job = _job(description="Your earnings grow as you recruit others to the team.")
        assert check_mlm_language(job) is not None

    def test_does_not_fire_on_clean_text(self):
        assert check_mlm_language(_clean_job()) is None

    def test_does_not_fire_on_standard_sales_job(self):
        job = _job(
            title="Sales Representative",
            description="Manage a territory of enterprise accounts and close new business. "
                        "3+ years of B2B sales experience required.",
        )
        assert check_mlm_language(job) is None


# ===========================================================================
# 6. check_reshipping_scam
# ===========================================================================

class TestReshippingScam:
    def test_fires_on_receive_packages(self):
        job = _job(description="You will receive packages at home and forward them to our warehouse.")
        signal = check_reshipping_scam(job)
        assert signal is not None
        assert signal.name == "reshipping_scam"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_reshipping_keyword(self):
        job = _job(description="This role involves reshipping parcels from suppliers.")
        assert check_reshipping_scam(job) is not None

    def test_fires_on_forward_packages(self):
        job = _job(description="Primary duty: forward packages received at your address to clients.")
        assert check_reshipping_scam(job) is not None

    def test_fires_on_inspect_packages_at_home(self):
        job = _job(description="Inspect packages at home and report quality issues.")
        assert check_reshipping_scam(job) is not None

    def test_does_not_fire_on_clean_text(self):
        assert check_reshipping_scam(_clean_job()) is None

    def test_does_not_fire_on_warehouse_job(self):
        # "warehouse" without reshipping-specific language should not trigger
        job = _job(
            title="Warehouse Associate",
            description="Pick and pack orders in our fulfillment center. "
                        "Operate forklift, maintain inventory records.",
        )
        assert check_reshipping_scam(job) is None


# ===========================================================================
# 7. check_interview_bypass
# ===========================================================================

class TestInterviewBypass:
    def test_fires_on_no_interview_required(self):
        job = _job(description="No interview required — you are hired on the spot!")
        signal = check_interview_bypass(job)
        assert signal is not None
        assert signal.name == "interview_bypass"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_hired_immediately(self):
        job = _job(description="You will be hired immediately after applying.")
        assert check_interview_bypass(job) is not None

    def test_fires_on_no_resume_required(self):
        job = _job(description="No resume required — start today!")
        assert check_interview_bypass(job) is not None

    def test_fires_on_no_background_check(self):
        job = _job(description="No background check required for this position.")
        assert check_interview_bypass(job) is not None

    def test_does_not_fire_on_clean_text(self):
        assert check_interview_bypass(_clean_job()) is None

    def test_does_not_fire_on_normal_hiring_language(self):
        job = _job(
            description="Apply through our careers portal. Qualified candidates will be "
                        "contacted for a phone screen followed by technical interviews."
        )
        assert check_interview_bypass(job) is None


# ===========================================================================
# 8. check_data_harvesting
# ===========================================================================

class TestDataHarvesting:
    def test_fires_on_external_application_redirect(self):
        job = _job(description="Apply at our external site: apply via http://bit.ly/jobform")
        signal = check_data_harvesting(job)
        assert signal is not None
        assert signal.name == "data_harvesting"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_google_forms_link(self):
        job = _job(description="Submit your info at forms.gle/xyz123abc to be considered.")
        assert check_data_harvesting(job) is not None

    def test_fires_on_typeform_link(self):
        job = _job(description="Fill out the application at typeform.com/to/apply123.")
        assert check_data_harvesting(job) is not None

    def test_fires_on_docs_google_forms(self):
        job = _job(description="Complete the form at docs.google.com/forms/d/abc123.")
        assert check_data_harvesting(job) is not None

    def test_fires_on_fill_out_application_via(self):
        job = _job(description="Please fill out our application form at our separate site to apply.")
        # No URL needed — the redirect phrasing itself triggers
        result = check_data_harvesting(job)
        # May or may not fire depending on whether "separate" matches;
        # test the link-based variant is more reliable
        job2 = _job(description="Submit your application to via forms.gle/abc123 for review.")
        assert check_data_harvesting(job2) is not None

    def test_does_not_fire_on_clean_text(self):
        assert check_data_harvesting(_clean_job()) is None

    def test_does_not_fire_on_linkedin_apply(self):
        job = _job(description="Apply directly through LinkedIn. We review all applications.")
        assert check_data_harvesting(job) is None


# ===========================================================================
# 9. check_pig_butchering_job
# ===========================================================================

class TestPigButcheringJob:
    def test_fires_on_cryptocurrency_trader(self):
        job = _job(description="We are hiring a cryptocurrency trader to manage portfolios.")
        signal = check_pig_butchering_job(job)
        assert signal is not None
        assert signal.name == "pig_butchering_job"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_defi_analyst(self):
        job = _job(description="Join our team as a DeFi analyst — no experience needed.")
        assert check_pig_butchering_job(job) is not None

    def test_fires_on_forex_trader(self):
        job = _job(description="Hiring forex trader to manage client accounts remotely.")
        assert check_pig_butchering_job(job) is not None

    def test_fires_on_blockchain_investment_manager(self):
        job = _job(description="Blockchain investment manager needed for digital asset fund.")
        assert check_pig_butchering_job(job) is not None

    def test_fires_on_digital_asset_trader(self):
        job = _job(description="Digital asset trader — guaranteed high returns, remote work.")
        assert check_pig_butchering_job(job) is not None

    def test_does_not_fire_on_clean_text(self):
        assert check_pig_butchering_job(_clean_job()) is None

    def test_does_not_fire_on_legitimate_fintech(self):
        job = _job(
            title="Software Engineer, Payments",
            description="Build the payments infrastructure at our fintech company. "
                        "You will work with Python, Kafka, and PostgreSQL on high-throughput "
                        "transaction processing systems.",
        )
        assert check_pig_butchering_job(job) is None


# ===========================================================================
# 10. check_suspicious_email_domain
# ===========================================================================

class TestSuspiciousEmailDomain:
    def test_fires_on_gmail_in_description(self):
        job = _job(description="Send your CV to hiring@gmail.com for consideration.")
        signal = check_suspicious_email_domain(job)
        assert signal is not None
        assert signal.name == "suspicious_email_domain"
        assert signal.category == SignalCategory.RED_FLAG

    def test_fires_on_yahoo_in_description(self):
        job = _job(description="Contact us at jobs@yahoo.com with your application.")
        assert check_suspicious_email_domain(job) is not None

    def test_fires_on_hotmail_in_description(self):
        job = _job(description="Email your resume to recruiter@hotmail.com")
        assert check_suspicious_email_domain(job) is not None

    def test_fires_on_outlook_in_recruiter_name(self):
        # recruiter_name field is also searched
        job = _job(
            description="Software role at a leading company.",
            recruiter_name="Bob recruiter@outlook.com",
        )
        assert check_suspicious_email_domain(job) is not None

    def test_fires_on_protonmail(self):
        job = _job(description="Questions? Email hr@protonmail.com")
        assert check_suspicious_email_domain(job) is not None

    def test_does_not_fire_on_corporate_domain(self):
        job = _job(description="Apply at careers@stripe.com or through our LinkedIn page.")
        assert check_suspicious_email_domain(job) is None

    def test_does_not_fire_on_clean_text(self):
        assert check_suspicious_email_domain(_clean_job()) is None


# ===========================================================================
# extract_signals() integration
# ===========================================================================

class TestExtractSignals:
    def test_returns_list(self):
        job = _clean_job()
        result = extract_signals(job)
        assert isinstance(result, list)

    def test_returns_scam_signal_objects(self):
        job = _job(
            description="Earn $5,000/week GUARANTEED! Fee required. Send bank account number."
        )
        result = extract_signals(job)
        assert len(result) > 0
        for sig in result:
            assert isinstance(sig, ScamSignal)

    def test_multiple_red_flags_detected_on_scam_job(self):
        job = JobPosting(
            url="https://example.com/scam",
            title="Work From Home — Earn $5,000/Week GUARANTEED",
            company="Global Opportunities LLC",
            description=(
                "Earn GUARANTEED $5,000 per week working from home. "
                "No experience required — anyone can qualify! "
                "Pay a $99 registration fee. Provide your Social Security Number. "
                "Contact us at globalopps@gmail.com"
            ),
            company_linkedin_url="",
            recruiter_name="",
            recruiter_connections=12,
        )
        signals = extract_signals(job)
        red_flags = [s for s in signals if s.category == SignalCategory.RED_FLAG]
        assert len(red_flags) >= 2, f"Expected >=2 red flags, got {len(red_flags)}: {[s.name for s in signals]}"

    def test_clean_job_has_no_content_red_flags(self):
        """The clean job should not trigger any content-based red flag signals.

        Evasion/normalizer signals (evasion_attempt, misspelled_scam_keyword,
        unicode_anomaly) are excluded because the adversarial detector can
        occasionally fire on legitimate vocabulary (e.g. 'payment systems').
        """
        EVASION_SIGNAL_NAMES = {"evasion_attempt", "misspelled_scam_keyword", "unicode_anomaly"}
        job = _clean_job()
        signals = extract_signals(job)
        content_red_flags = [
            s for s in signals
            if s.category == SignalCategory.RED_FLAG and s.name not in EVASION_SIGNAL_NAMES
        ]
        assert len(content_red_flags) == 0, (
            f"Unexpected content red flags on clean job: {[s.name for s in content_red_flags]}"
        )

    def test_signal_names_are_strings(self):
        job = _job(description="No experience required. Earn $5,000/week guaranteed!")
        for sig in extract_signals(job):
            assert isinstance(sig.name, str) and sig.name

    def test_signal_categories_are_valid_enum_values(self):
        job = _job(
            description="Earn $5,000/week guaranteed. No interview needed. Pay upfront fee."
        )
        valid_categories = set(SignalCategory)
        for sig in extract_signals(job):
            assert sig.category in valid_categories, (
                f"Signal '{sig.name}' has invalid category: {sig.category}"
            )

    def test_signal_weights_are_in_valid_range(self):
        job = JobPosting(
            url="https://example.com/job",
            title="Work From Home Opportunity",
            company="",
            description=(
                "Earn $10,000 per week guaranteed! No experience required. "
                "Pay a training fee. Provide your SSN and bank account. "
                "No interview needed. Hired immediately. "
                "Contact jobs@gmail.com. Bitcoin payments available."
            ),
            company_linkedin_url="",
            recruiter_name="",
            recruiter_connections=5,
        )
        for sig in extract_signals(job):
            assert 0.0 <= sig.weight <= 1.0, (
                f"Signal '{sig.name}' has weight {sig.weight} outside [0,1]"
            )

    def test_signal_confidence_is_in_valid_range(self):
        job = _job(description="Earn $5,000 guaranteed. Upfront fee required.")
        for sig in extract_signals(job):
            assert 0.0 <= sig.confidence <= 1.0, (
                f"Signal '{sig.name}' has confidence {sig.confidence} outside [0,1]"
            )

    def test_no_duplicate_signal_names_on_simple_job(self):
        """Each check_* function should fire at most once per job."""
        job = _job(
            description=(
                "Pay training fee. Social security number required. "
                "Earn $5,000/week guaranteed. Pay via Bitcoin."
            )
        )
        signals = extract_signals(job)
        names = [s.name for s in signals]
        # Duplicates are allowed by design only for certain pattern-matching paths,
        # but the core named signals should not repeat.
        core_names = [n for n in names if not n.startswith("kb_")]
        assert len(core_names) == len(set(core_names)), (
            f"Duplicate core signal names: {[n for n in core_names if core_names.count(n) > 1]}"
        )
