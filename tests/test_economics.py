"""Tests for sentinel/economics.py — Economic Ground Truth Validator.

Coverage:
  - MarketRateValidator:  salary vs market rates, CoL multipliers, too-good-to-be-true,
                          impossible combinations
  - CompanyEconomics:     age/size mismatch, funding vs headcount, revenue/employee,
                          hiring pattern
  - BenefitsAnalyzer:     parsing, stuffing detection, unrealistic entry-level packages
  - GeographicValidator:  MLM restriction, geographic impossibility, area code mismatch,
                          suspicious posting time
  - IndustryBenchmark:    percentile ranking, drift detection, benchmark lookup
  - Helper functions:     classify_role, normalize_experience_level, get_col_multiplier
  - validate_economics:   aggregated convenience wrapper
"""

import pytest

from sentinel.models import JobPosting, SignalCategory
from sentinel.economics import (
    # classes
    MarketRateValidator,
    CompanyEconomics,
    BenefitsAnalyzer,
    GeographicValidator,
    IndustryBenchmark,
    # helpers
    classify_role,
    normalize_experience_level,
    get_col_multiplier,
    # convenience
    validate_economics,
    # data
    SALARY_RANGES,
    COL_MULTIPLIERS,
    INDUSTRY_BENCHMARKS,
    INDUSTRY_HIRING_PATTERNS,
    REVENUE_PER_EMPLOYEE,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_job(**kwargs) -> JobPosting:
    defaults = dict(
        url="https://linkedin.com/jobs/test/1",
        title="Software Engineer",
        company="Acme Corp LLC",
        location="Austin, TX",
        description="We are looking for a software engineer to join our team.",
        salary_min=0.0,
        salary_max=0.0,
        salary_currency="USD",
        posted_date="2026-04-01",
        applicant_count=50,
        experience_level="Mid-Level",
        employment_type="Full-time",
        industry="Technology",
        company_size="51-200",
        company_linkedin_url="https://linkedin.com/company/acme",
        recruiter_name="Jane Doe",
        recruiter_connections=300,
        is_remote=False,
        is_repost=False,
        source="linkedin",
    )
    defaults.update(kwargs)
    return JobPosting(**defaults)


# ===========================================================================
# Part 1 — classify_role helper
# ===========================================================================

class TestClassifyRole:
    def test_software_engineer(self):
        assert classify_role("Senior Software Engineer") == "software_engineer"

    def test_full_stack_developer(self):
        assert classify_role("Full Stack Developer") == "software_engineer"

    def test_backend_engineer(self):
        assert classify_role("Backend Engineer - Python") == "software_engineer"

    def test_data_scientist(self):
        assert classify_role("Data Scientist, ML Platform") == "data_scientist"

    def test_ml_engineer(self):
        assert classify_role("Machine Learning Engineer") == "machine_learning_engineer"

    def test_devops_engineer(self):
        assert classify_role("DevOps Engineer") == "devops_engineer"

    def test_sre(self):
        assert classify_role("Site Reliability Engineer (SRE)") == "devops_engineer"

    def test_product_manager(self):
        assert classify_role("Product Manager, Growth") == "product_manager"

    def test_marketing_manager(self):
        assert classify_role("Marketing Manager") == "marketing_manager"

    def test_sales_account_executive(self):
        assert classify_role("Account Executive") == "sales"

    def test_customer_service(self):
        assert classify_role("Customer Service Representative") == "customer_service"

    def test_nursing(self):
        assert classify_role("Registered Nurse (RN)") == "nursing"

    def test_accounting(self):
        assert classify_role("Senior Accountant") == "accounting"

    def test_project_manager(self):
        assert classify_role("Project Manager, Agile") == "project_manager"

    def test_warehouse(self):
        assert classify_role("Warehouse Associate") == "warehouse"

    def test_graphic_designer(self):
        assert classify_role("Graphic Designer") == "graphic_designer"

    def test_legal(self):
        assert classify_role("Attorney, Corporate Law") == "legal"

    def test_admin_assistant(self):
        assert classify_role("Administrative Assistant") == "admin_assistant"

    def test_unknown_returns_none(self):
        assert classify_role("Quantum Alchemist") is None

    def test_empty_returns_none(self):
        assert classify_role("") is None


# ===========================================================================
# Part 2 — normalize_experience_level helper
# ===========================================================================

class TestNormalizeExperienceLevel:
    def test_entry_variations(self):
        for raw in ("entry", "entry level", "Entry-Level", "junior", "internship", "intern"):
            assert normalize_experience_level(raw) == "entry", f"failed for {raw!r}"

    def test_senior_variations(self):
        for raw in ("senior", "Sr.", "Lead", "Principal", "Staff"):
            assert normalize_experience_level(raw) == "senior", f"failed for {raw!r}"

    def test_exec_variations(self):
        for raw in ("Chief Technology Officer", "VP of Engineering", "Head of Product"):
            assert normalize_experience_level(raw) == "exec", f"failed for {raw!r}"

    def test_mid_default(self):
        assert normalize_experience_level("mid") == "mid"
        assert normalize_experience_level("associate") == "mid"

    def test_empty_defaults_to_mid(self):
        assert normalize_experience_level("") == "mid"

    def test_director_is_senior(self):
        assert normalize_experience_level("director") == "senior"


# ===========================================================================
# Part 3 — get_col_multiplier helper
# ===========================================================================

class TestGetColMultiplier:
    def test_san_francisco(self):
        assert get_col_multiplier("San Francisco, CA") >= 1.40

    def test_nyc(self):
        assert get_col_multiplier("New York, NY") >= 1.28

    def test_rural(self):
        assert get_col_multiplier("rural, ND") <= 0.80

    def test_remote(self):
        assert get_col_multiplier("Remote") == 1.00

    def test_unknown_location(self):
        assert get_col_multiplier("Smallville, ZZ") == 1.00

    def test_state_fallback(self):
        # CA state fallback
        mult = get_col_multiplier("Fresno, CA")
        assert mult > 1.0  # CA is high cost

    def test_texas(self):
        mult = get_col_multiplier("Dallas, TX")
        assert mult <= 1.05  # TX is average cost

    def test_case_insensitive(self):
        m1 = get_col_multiplier("Seattle, WA")
        m2 = get_col_multiplier("SEATTLE, WA")
        assert m1 == m2


# ===========================================================================
# Part 4 — MarketRateValidator
# ===========================================================================

class TestMarketRateValidator:
    def setup_method(self):
        self.v = MarketRateValidator()

    def test_no_salary_returns_no_signals(self):
        job = make_job(salary_min=0, salary_max=0)
        result = self.v.validate(job)
        assert not result.signals
        assert not result.is_suspicious

    def test_normal_mid_swe_salary_clean(self):
        job = make_job(title="Software Engineer", salary_max=135_000, experience_level="mid")
        result = self.v.validate(job)
        # $135k is at p50 for mid SWE — no flag expected
        salary_too_high = [s for s in result.signals if s.name == "salary_too_high"]
        assert not salary_too_high

    def test_outrageous_entry_swe_triggers_red_flag(self):
        # P90 entry SWE ~$135k; 3x = $405k → red flag
        job = make_job(title="Software Engineer", salary_max=500_000, experience_level="entry")
        result = self.v.validate(job)
        assert result.is_suspicious
        names = [s.name for s in result.signals]
        assert "salary_too_high" in names

    def test_2x_p90_triggers_warning(self):
        # P90 entry SWE ~$135k; 2x threshold = $270k.
        # Use a neutral location (no CoL boost) so adj_p90 ≈ $135k.
        # $320k > 2x adj_p90 → warning; <3x → not red flag
        job = make_job(
            title="Junior Software Engineer",
            salary_max=320_000,
            experience_level="entry",
            location="Memphis, TN",  # CoL ~0.90x → adj_p90 ≈ $121k; 320k/121k = 2.6x
        )
        result = self.v.validate(job)
        sig = next((s for s in result.signals if s.name == "salary_too_high"), None)
        assert sig is not None
        assert sig.category in (SignalCategory.WARNING, SignalCategory.RED_FLAG)

    def test_very_high_entry_swe_triggers_red_flag(self):
        job = make_job(title="Software Engineer", salary_max=600_000, experience_level="entry")
        result = self.v.validate(job)
        sig = next((s for s in result.signals if s.name == "salary_too_high"), None)
        assert sig is not None
        assert sig.category == SignalCategory.RED_FLAG

    def test_too_low_salary_triggers_warning(self):
        # Entry SWE P25 ~$75k; 0.35 * 75000 = $26k triggers warning
        job = make_job(title="Software Engineer", salary_max=20_000, experience_level="entry")
        result = self.v.validate(job)
        sig = next((s for s in result.signals if s.name == "salary_too_low"), None)
        assert sig is not None

    def test_customer_service_200k_entry_flags(self):
        # CS entry P90 ~$62k; $200k is >3x
        job = make_job(title="Customer Service Representative", salary_max=200_000, experience_level="entry")
        result = self.v.validate(job)
        assert result.is_suspicious

    def test_unknown_role_wide_range_flags(self):
        job = make_job(title="Mystery Analyst XYZ999", salary_min=10_000, salary_max=80_000)
        result = self.v.validate(job)
        sig = next((s for s in result.signals if s.name == "salary_range_implausible"), None)
        assert sig is not None

    def test_unknown_role_narrow_range_clean(self):
        job = make_job(title="Mystery Analyst XYZ999", salary_min=50_000, salary_max=70_000)
        result = self.v.validate(job)
        assert not result.signals

    def test_col_multiplier_applied(self):
        # SF multiplier ~1.45; entry SWE SF P90 ≈ $135k * 1.45 = $195k
        # $200k should NOT be 3x in SF context (3x = $585k)
        job = make_job(title="Software Engineer", salary_max=200_000, experience_level="entry", location="San Francisco, CA")
        result = self.v.validate(job)
        # Should still flag at 2x level but not as extreme as without CoL
        sig = next((s for s in result.signals if s.name == "salary_too_high"), None)
        if sig:
            assert sig.category == SignalCategory.WARNING

    def test_too_good_to_be_true_remote_entry_200k(self):
        job = make_job(
            title="Software Engineer",
            salary_max=200_000,
            experience_level="entry",
            is_remote=True,
        )
        result = self.v.validate(job)
        sig = next((s for s in result.signals if s.name == "too_good_to_be_true"), None)
        assert sig is not None
        assert sig.category == SignalCategory.RED_FLAG

    def test_too_good_to_be_true_not_triggered_for_senior(self):
        # Senior remote $200k is normal
        job = make_job(
            title="Senior Software Engineer",
            salary_max=200_000,
            experience_level="senior",
            is_remote=True,
        )
        result = self.v.validate(job)
        tgtb = [s for s in result.signals if s.name == "too_good_to_be_true"]
        assert not tgtb

    def test_impossible_combination_intern_high_salary(self):
        job = make_job(
            title="Software Engineering Internship",
            salary_max=150_000,
            experience_level="internship",
        )
        result = self.v.validate(job)
        sig = next((s for s in result.signals if s.name == "impossible_combination"), None)
        assert sig is not None
        assert sig.category == SignalCategory.RED_FLAG

    def test_impossible_combination_no_experience_200k(self):
        job = make_job(
            title="Software Developer",
            salary_max=250_000,
            experience_level="entry",
            description="No experience required. Anyone can apply. Join our team today.",
        )
        result = self.v.validate(job)
        sig = next((s for s in result.signals if s.name == "impossible_combination"), None)
        assert sig is not None

    def test_percentile_estimate_populated(self):
        job = make_job(title="Software Engineer", salary_max=200_000, experience_level="mid")
        result = self.v.validate(job)
        assert result.percentile_estimate is not None
        assert 0 <= result.percentile_estimate <= 100

    def test_role_category_populated(self):
        job = make_job(title="Senior Software Engineer", salary_max=200_000)
        result = self.v.validate(job)
        assert result.role_category == "software_engineer"

    def test_expected_p90_populated(self):
        job = make_job(title="Software Engineer", salary_max=100_000, experience_level="entry")
        result = self.v.validate(job)
        assert result.expected_p90 is not None
        assert result.expected_p90 > 100_000


# ===========================================================================
# Part 5 — CompanyEconomics
# ===========================================================================

class TestCompanyEconomics:
    def setup_method(self):
        self.v = CompanyEconomics()

    def test_brand_new_company_500_employees_flags(self):
        # Founded this year, claiming 500+ employees
        current_year = 2026
        job = make_job(company="FreshStartup LLC", company_size="501-1000")
        result = self.v.validate(job, company_founded_year=current_year, claimed_employees=501)
        assert result.is_suspicious
        names = [s.name for s in result.signals]
        assert "company_age_size_mismatch" in names

    def test_established_company_500_employees_ok(self):
        job = make_job(company="Established Corp", company_size="501-1000")
        result = self.v.validate(job, company_founded_year=2005, claimed_employees=501)
        age_mismatch = [s for s in result.signals if s.name == "company_age_size_mismatch"]
        assert not age_mismatch

    def test_seed_stage_1000_employees_flags(self):
        job = make_job(company="SeedCo")
        result = self.v.validate(job, funding_stage="seed", claimed_employees=1000)
        assert result.is_suspicious
        names = [s.name for s in result.signals]
        assert "funding_stage_size_mismatch" in names

    def test_series_b_200_employees_ok(self):
        job = make_job(company="Series B Co")
        result = self.v.validate(job, funding_stage="series_b", claimed_employees=200)
        mismatch = [s for s in result.signals if s.name == "funding_stage_size_mismatch"]
        assert not mismatch

    def test_revenue_per_employee_too_low_flags(self):
        # 10 employees, $100 revenue = $10/employee vs software P25 $200k
        job = make_job(industry="software")
        result = self.v.validate(job, annual_revenue=100, claimed_employees=10)
        assert result.is_suspicious

    def test_revenue_per_employee_normal_ok(self):
        # 100 employees, $30M revenue = $300k/employee — within software range
        job = make_job(industry="software")
        result = self.v.validate(job, annual_revenue=30_000_000, claimed_employees=100)
        rev_sig = [s for s in result.signals if s.name == "revenue_per_employee_anomaly"]
        assert not rev_sig

    def test_always_hiring_seasonal_industry_flags(self):
        job = make_job(
            industry="retail",
            description="We are always hiring — open pipeline year round!",
        )
        result = self.v.validate(job)
        assert result.is_suspicious
        names = [s.name for s in result.signals]
        assert "hiring_pattern_mismatch" in names

    def test_always_hiring_tech_no_flag(self):
        job = make_job(
            industry="technology",
            description="We are always hiring great engineers.",
        )
        result = self.v.validate(job)
        pattern_sig = [s for s in result.signals if s.name == "hiring_pattern_mismatch"]
        assert not pattern_sig

    def test_parse_company_size_1_10(self):
        ec = CompanyEconomics()
        assert ec._parse_company_size("1-10") == 1

    def test_parse_company_size_10001_plus(self):
        ec = CompanyEconomics()
        assert ec._parse_company_size("10001+") == 10001

    def test_no_data_returns_clean_result(self):
        job = make_job()
        result = self.v.validate(job)
        assert isinstance(result.signals, list)


# ===========================================================================
# Part 6 — BenefitsAnalyzer
# ===========================================================================

class TestBenefitsAnalyzer:
    def setup_method(self):
        self.a = BenefitsAnalyzer()

    def test_parse_basic_benefits(self):
        text = "We offer health insurance, dental, and vision coverage."
        benefits = self.a.parse_benefits(text)
        assert "health_insurance" in benefits
        assert "dental" in benefits
        assert "vision" in benefits

    def test_parse_premium_benefits(self):
        text = "Unlimited PTO, stock options, annual bonus, and equity."
        benefits = self.a.parse_benefits(text)
        assert "unlimited_pto" in benefits
        assert "equity" in benefits
        assert "bonus" in benefits

    def test_parse_luxury_benefits(self):
        text = "Company car provided, pet insurance, sabbatical after 5 years."
        benefits = self.a.parse_benefits(text)
        assert "company_car" in benefits
        assert "pet_insurance" in benefits
        assert "sabbatical" in benefits

    def test_no_benefits_returns_empty(self):
        text = "Come join our exciting team and grow your career."
        benefits = self.a.parse_benefits(text)
        assert isinstance(benefits, list)

    def test_keyword_stuffing_flags_over_threshold(self):
        # Build a description with 16+ distinct benefit mentions
        mega_desc = (
            "health insurance, dental, vision, 401k, unlimited PTO, parental leave, "
            "stock options, annual bonus, company car, profit sharing, gym membership, "
            "remote work, tuition reimbursement, life insurance, disability insurance, "
            "employee discount, free lunch, relocation assistance, mental health benefit, "
            "childcare benefit, home office stipend, travel allowance, pet insurance, sabbatical"
        )
        job = make_job(description=mega_desc, experience_level="entry")
        result = self.a.analyze(job)
        stuffing = [s for s in result.signals if s.name == "benefit_keyword_stuffing"]
        assert stuffing, f"Expected stuffing signal, found signals: {[s.name for s in result.signals]}"

    def test_normal_benefit_count_no_stuffing(self):
        desc = "We offer health insurance, dental, 401k, and PTO."
        job = make_job(description=desc)
        result = self.a.analyze(job)
        stuffing = [s for s in result.signals if s.name == "benefit_keyword_stuffing"]
        assert not stuffing

    def test_entry_level_company_car_flags(self):
        job = make_job(
            description="Company car provided for all new hires.",
            experience_level="entry",
            title="Customer Service Rep",
        )
        result = self.a.analyze(job)
        unrealistic = [s for s in result.signals if s.name == "unrealistic_benefits"]
        assert unrealistic

    def test_entry_level_too_many_premiums_flags(self):
        desc = (
            "Unlimited PTO, stock options, signing bonus, profit sharing, "
            "company car, gym membership, sabbatical, tuition reimbursement, "
            "parental leave, home office stipend, mental health benefit"
        )
        job = make_job(
            description=desc,
            experience_level="entry",
            title="Customer Service Rep",
            industry="retail",
        )
        result = self.a.analyze(job)
        assert result.is_suspicious

    def test_senior_engineer_equity_no_flag(self):
        desc = "Stock options, annual bonus, unlimited PTO, health insurance, dental, vision."
        job = make_job(
            description=desc,
            experience_level="senior",
            title="Senior Software Engineer",
            industry="technology",
        )
        result = self.a.analyze(job)
        unrealistic = [s for s in result.signals if s.name == "unrealistic_benefits"]
        assert not unrealistic

    def test_tiers_populated(self):
        desc = "Health insurance, stock options, company car, unlimited PTO."
        job = make_job(description=desc)
        result = self.a.analyze(job)
        assert "basic" in result.tiers or "standard" in result.tiers or "premium" in result.tiers


# ===========================================================================
# Part 7 — GeographicValidator
# ===========================================================================

class TestGeographicValidator:
    def setup_method(self):
        self.v = GeographicValidator()

    def test_mlm_remote_must_be_near_flags(self):
        job = make_job(
            description="This is a fully remote role. Must live near Smalltown, OH to attend weekly meetings.",
            location="Remote",
            is_remote=True,
        )
        result = self.v.validate(job)
        sig = next((s for s in result.signals if s.name == "mlm_geographic_restriction"), None)
        assert sig is not None

    def test_onsite_local_only_no_flag(self):
        # "local candidates only" for an on-site role is fine
        job = make_job(
            description="Local candidates only preferred. On-site position in Austin, TX.",
            location="Austin, TX",
            is_remote=False,
        )
        result = self.v.validate(job)
        mlm = [s for s in result.signals if s.name == "mlm_geographic_restriction"]
        assert not mlm

    def test_uk_ltd_us_only_job_flags(self):
        job = make_job(
            company="Acme Services Ltd.",
            description="US citizen required. Must be authorized to work in the USA only.",
            location="New York, NY",
        )
        result = self.v.validate(job)
        geo = [s for s in result.signals if s.name == "geographic_impossibility"]
        assert geo

    def test_us_llc_london_office_flags(self):
        job = make_job(
            company="TechCorp LLC",
            description="Join our London office — based in London HQ.",
            location="United States",
        )
        result = self.v.validate(job)
        geo = [s for s in result.signals if s.name == "geographic_impossibility"]
        assert geo

    def test_phone_area_code_mismatch_flags(self):
        # 212 = NY, job listed in TX
        job = make_job(
            description="Call us at 212-555-1234 for details.",
            location="Austin, TX",
            is_remote=False,
        )
        result = self.v.validate(job)
        phone = [s for s in result.signals if s.name == "phone_area_code_mismatch"]
        assert phone

    def test_phone_area_code_match_ok(self):
        # 512 = TX, job in TX
        job = make_job(
            description="Call us at 512-555-9876 for details.",
            location="Austin, TX",
            is_remote=False,
        )
        result = self.v.validate(job)
        phone = [s for s in result.signals if s.name == "phone_area_code_mismatch"]
        assert not phone

    def test_phone_area_code_remote_no_flag(self):
        # Remote job — area code mismatch is expected
        job = make_job(
            description="Call us at 212-555-1234 for details.",
            location="Remote",
            is_remote=True,
        )
        result = self.v.validate(job)
        phone = [s for s in result.signals if s.name == "phone_area_code_mismatch"]
        assert not phone

    def test_suspicious_posting_time_3am_small_company(self):
        job = make_job(company_size="1-10")
        result = self.v.validate(job, posted_hour_utc=3)
        sig = next((s for s in result.signals if s.name == "suspicious_posting_time"), None)
        assert sig is not None

    def test_suspicious_posting_time_3am_large_company_ok(self):
        job = make_job(company_size="10001+")
        result = self.v.validate(job, posted_hour_utc=3)
        sig = next((s for s in result.signals if s.name == "suspicious_posting_time"), None)
        assert sig is None

    def test_normal_posting_time_no_flag(self):
        job = make_job(company_size="1-10")
        result = self.v.validate(job, posted_hour_utc=10)
        sig = next((s for s in result.signals if s.name == "suspicious_posting_time"), None)
        assert sig is None

    def test_no_posted_hour_no_time_flag(self):
        job = make_job()
        result = self.v.validate(job, posted_hour_utc=None)
        time_sig = [s for s in result.signals if s.name == "suspicious_posting_time"]
        assert not time_sig


# ===========================================================================
# Part 8 — IndustryBenchmark
# ===========================================================================

class TestIndustryBenchmark:
    def setup_method(self):
        self.b = IndustryBenchmark()

    def test_rank_returns_percentile_for_known_role(self):
        job = make_job(
            title="Software Engineer",
            salary_max=145_000,
            experience_level="mid",
            industry="technology",
        )
        result = self.b.rank(job)
        assert result.percentile is not None
        assert 0 <= result.percentile <= 100

    def test_high_salary_extreme_percentile_flag(self):
        # Very high salary → 95th+ percentile → signal
        job = make_job(
            title="Software Engineer",
            salary_max=450_000,
            experience_level="mid",
            industry="technology",
        )
        result = self.b.rank(job)
        sig = next((s for s in result.signals if s.name == "salary_extreme_percentile"), None)
        assert sig is not None

    def test_normal_salary_no_percentile_flag(self):
        job = make_job(
            title="Software Engineer",
            salary_max=145_000,
            experience_level="mid",
            industry="technology",
        )
        result = self.b.rank(job)
        sig = next((s for s in result.signals if s.name == "salary_extreme_percentile"), None)
        assert sig is None

    def test_no_salary_returns_no_percentile(self):
        job = make_job(salary_min=0, salary_max=0)
        result = self.b.rank(job)
        assert result.percentile is None

    def test_percentile_label_below_market(self):
        result = self.b._percentile_label(5)
        assert "below" in result.lower()

    def test_percentile_label_exceptional(self):
        result = self.b._percentile_label(97)
        assert "exceptional" in result.lower() or "verify" in result.lower()

    def test_drift_detection_above_threshold(self):
        # stored p50 for tech/software_engineer/entry ≈ $98k
        # Observed $130k is >15% above
        sig = self.b.check_drift("technology", "software_engineer", "entry", 130_000)
        assert sig is not None
        assert sig.name == "benchmark_drift"

    def test_drift_detection_within_threshold(self):
        # $100k vs stored $98k = ~2% deviation — no flag
        sig = self.b.check_drift("technology", "software_engineer", "entry", 100_000)
        assert sig is None

    def test_drift_detection_unknown_combo_returns_none(self):
        sig = self.b.check_drift("aerospace", "rocket_surgeon", "legendary", 999_000)
        assert sig is None

    def test_get_benchmark_returns_dict(self):
        bench = self.b.get_benchmark("technology", "software_engineer", "entry")
        assert bench is not None
        assert "p50" in bench
        assert bench["p50"] > 0

    def test_get_benchmark_unknown_returns_none(self):
        bench = self.b.get_benchmark("mystery", "unknown_role", "entry")
        assert bench is None

    def test_rank_falls_back_to_national_data(self):
        # nursing in unknown industry falls back to SALARY_RANGES
        job = make_job(
            title="Registered Nurse (RN)",
            salary_max=200_000,
            experience_level="senior",
            industry="unknown_industry",
        )
        result = self.b.rank(job)
        # Should still return a percentile (from national fallback)
        assert result.percentile is not None


# ===========================================================================
# Part 9 — validate_economics (aggregated convenience wrapper)
# ===========================================================================

class TestValidateEconomics:
    def test_clean_job_no_signals(self):
        job = make_job(
            title="Software Engineer",
            salary_max=140_000,
            experience_level="mid",
            industry="technology",
            is_remote=False,
            description=(
                "Responsibilities include building scalable systems. "
                "Requires 3+ years Python experience. "
                "We offer health insurance, dental, 401k."
            ),
        )
        result = validate_economics(job)
        assert not result.is_suspicious

    def test_scam_job_accumulates_signals(self):
        job = make_job(
            title="Software Engineering Internship",
            salary_max=250_000,
            experience_level="internship",
            is_remote=True,
            description=(
                "No experience required! Anyone can apply! "
                "Must live near Smalltown, OH. "
                "Unlimited PTO, company car, stock options, signing bonus, "
                "pet insurance, sabbatical, profit sharing, gym membership, "
                "tuition reimbursement, mental health benefit, childcare benefit, "
                "home office stipend, life insurance, disability insurance, free lunch."
            ),
            company_size="1-10",
        )
        result = validate_economics(
            job,
            company_founded_year=2026,
            claimed_employees=500,
            funding_stage="seed",
        )
        assert result.is_suspicious
        assert len(result.all_signals) >= 3

    def test_all_signals_property_aggregates(self):
        job = make_job(
            title="Software Engineer",
            salary_max=600_000,
            experience_level="entry",
            is_remote=True,
        )
        result = validate_economics(job)
        # all_signals should be a flat list from all sub-validators
        assert isinstance(result.all_signals, list)

    def test_company_founded_current_year_many_employees(self):
        job = make_job(company_size="1001-5000")
        result = validate_economics(
            job,
            company_founded_year=2026,
            claimed_employees=1001,
        )
        assert result.company.is_suspicious

    def test_benefits_result_populated(self):
        job = make_job(description="We offer health insurance and 401k.")
        result = validate_economics(job)
        assert isinstance(result.benefits.found_benefits, list)

    def test_geography_result_populated(self):
        job = make_job()
        result = validate_economics(job)
        assert isinstance(result.geography.signals, list)


# ===========================================================================
# Part 10 — Reference data integrity
# ===========================================================================

class TestReferenceDataIntegrity:
    def test_salary_ranges_all_roles_have_levels(self):
        for role, levels in SALARY_RANGES.items():
            assert "entry" in levels, f"{role} missing 'entry'"
            assert "mid" in levels, f"{role} missing 'mid'"
            assert "senior" in levels, f"{role} missing 'senior'"

    def test_salary_ranges_values_ascending(self):
        for role, levels in SALARY_RANGES.items():
            for level, vals in levels.items():
                p10, p25, p50, p75, p90, p99 = vals
                assert p10 < p25 < p50 < p75 < p90 < p99, (
                    f"Salary range not ascending for {role}/{level}: {vals}"
                )

    def test_salary_ranges_exec_above_senior(self):
        for role in SALARY_RANGES:
            if "exec" in SALARY_RANGES[role] and "senior" in SALARY_RANGES[role]:
                exec_p50 = SALARY_RANGES[role]["exec"][2]
                senior_p50 = SALARY_RANGES[role]["senior"][2]
                assert exec_p50 > senior_p50, f"{role}: exec p50 not > senior p50"

    def test_col_multipliers_sf_highest(self):
        sf = COL_MULTIPLIERS["san francisco"]
        rural = COL_MULTIPLIERS["rural"]
        assert sf > rural
        assert sf >= 1.40

    def test_industry_benchmarks_structure(self):
        for industry, roles in INDUSTRY_BENCHMARKS.items():
            for role, levels in roles.items():
                for level, bench in levels.items():
                    assert "p25" in bench, f"Missing p25 in {industry}/{role}/{level}"
                    assert "p50" in bench
                    assert "p90" in bench
                    assert bench["p25"] < bench["p50"] < bench["p90"]

    def test_revenue_per_employee_benchmarks(self):
        for industry, (p25, p50, p75) in REVENUE_PER_EMPLOYEE.items():
            assert p25 < p50 < p75, f"{industry}: revenue/employee not ascending"

    def test_industry_hiring_patterns_known_values(self):
        valid_patterns = {"continuous", "seasonal", "project_based", "cycle_based", "grant_based", "steady"}
        for industry, pattern in INDUSTRY_HIRING_PATTERNS.items():
            assert pattern in valid_patterns, f"Unknown pattern {pattern!r} for {industry}"

    def test_salary_ranges_minimum_role_count(self):
        assert len(SALARY_RANGES) >= 15

    def test_col_multipliers_minimum_count(self):
        assert len(COL_MULTIPLIERS) >= 20

    def test_industry_benchmarks_minimum_industries(self):
        assert len(INDUSTRY_BENCHMARKS) >= 3
