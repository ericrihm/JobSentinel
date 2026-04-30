"""Advanced tests for new Sentinel modules: signals (AI-informed), ecosystem, innovation."""

import json
import random

import pytest

from sentinel.models import JobPosting, ScamSignal, SignalCategory


# ===========================================================================
# TestNewSignals — AI-informed scam detection functions
# ===========================================================================


class TestAIGeneratedContent:
    """check_ai_generated_content: generic culture buzzwords vs specific descriptions."""

    def test_detects_heavy_generic_culture(self):
        from sentinel.signals import check_ai_generated_content

        # 4 generic culture phrases, no specifics → ratio well above 3.0
        desc = (
            "We are a dynamic team of passionate individuals in a fast-paced environment. "
            "We are looking for a go-getter who wants to make a difference. "
            "Join our collaborative culture and be part of something great. "
            "We work hard and play hard together as a family-like team."
        )
        job = JobPosting(description=desc)
        sig = check_ai_generated_content(job)
        assert sig is not None
        assert sig.name == "ai_generated_content"
        assert sig.category == SignalCategory.STRUCTURAL

    def test_no_signal_for_specific_description(self):
        from sentinel.signals import check_ai_generated_content

        desc = (
            "Join team Hermes to build our real-time analytics pipeline. "
            "Using Python 3.12, Apache Kafka, and BigQuery, you will own the ingestion layer. "
            "Must have 4+ years of experience with streaming data systems and a bachelor's degree. "
            "We offer health insurance, 401k, and stock options."
        )
        job = JobPosting(description=desc)
        assert check_ai_generated_content(job) is None

    def test_no_signal_for_short_description(self):
        from sentinel.signals import check_ai_generated_content

        job = JobPosting(description="Dynamic team. Be a rockstar.")
        # Under 30 words → returns None regardless
        assert check_ai_generated_content(job) is None

    def test_signal_weight_in_range(self):
        from sentinel.signals import check_ai_generated_content

        desc = (
            "We are a dynamic team of passionate individuals seeking a self-starter. "
            "Join our results-driven collaborative culture. Be a go-getter who makes a difference. "
            "Inclusive workplace for rockstar ninjas who work hard and play hard every day."
        )
        job = JobPosting(description=desc)
        sig = check_ai_generated_content(job)
        if sig is not None:
            assert 0.0 <= sig.weight <= 1.0
            assert 0.0 <= sig.confidence <= 1.0


class TestPhoneAnomaly:
    """check_phone_anomaly: premium rate numbers, normal descriptions."""

    def test_detects_premium_rate_900_number(self):
        from sentinel.signals import check_phone_anomaly

        job = JobPosting(description="Call us at 900-555-1234 to apply for this position.")
        sig = check_phone_anomaly(job)
        assert sig is not None
        assert sig.name == "phone_anomaly"
        assert sig.category == SignalCategory.WARNING
        assert "900" in sig.evidence or "900" in sig.detail

    def test_detects_premium_rate_976_number(self):
        from sentinel.signals import check_phone_anomaly

        job = JobPosting(description="Interested? Dial 976-123-4567 for more information.")
        sig = check_phone_anomaly(job)
        assert sig is not None
        assert sig.name == "phone_anomaly"

    def test_detects_regular_phone_in_description(self):
        from sentinel.signals import check_phone_anomaly

        job = JobPosting(description="Send your resume to 415-555-7890 or email us.")
        sig = check_phone_anomaly(job)
        assert sig is not None
        assert sig.name == "phone_anomaly"

    def test_no_signal_for_clean_description(self):
        from sentinel.signals import check_phone_anomaly

        job = JobPosting(
            description=(
                "Apply online at careers.acme.com. "
                "No phone calls please. We will reach out to qualified candidates."
            )
        )
        assert check_phone_anomaly(job) is None


class TestInterviewBypass:
    """check_interview_bypass: 'no interview', 'hired on the spot' vs normal."""

    def test_detects_no_interview_required(self):
        from sentinel.signals import check_interview_bypass

        job = JobPosting(description="No interview required — just submit your details to start.")
        sig = check_interview_bypass(job)
        assert sig is not None
        assert sig.name == "interview_bypass"
        assert sig.category == SignalCategory.RED_FLAG
        assert sig.weight == pytest.approx(0.75)

    def test_detects_hired_on_the_spot(self):
        from sentinel.signals import check_interview_bypass

        job = JobPosting(description="You will be hired on the spot at our open hiring event.")
        sig = check_interview_bypass(job)
        assert sig is not None
        assert sig.name == "interview_bypass"

    def test_detects_no_resume_needed(self):
        from sentinel.signals import check_interview_bypass

        job = JobPosting(description="No resume required. Get started today!")
        sig = check_interview_bypass(job)
        assert sig is not None

    def test_no_signal_for_standard_hiring(self):
        from sentinel.signals import check_interview_bypass

        job = JobPosting(
            description=(
                "Candidates will complete a phone screen, two technical rounds, "
                "and a final on-site interview with the engineering team."
            )
        )
        assert check_interview_bypass(job) is None


class TestMLMLanguage:
    """check_mlm_language: 'downline', 'recruit others', 'be your own boss'."""

    def test_detects_downline(self):
        from sentinel.signals import check_mlm_language

        job = JobPosting(description="Grow your downline and earn residual income.")
        sig = check_mlm_language(job)
        assert sig is not None
        assert sig.name == "mlm_language"
        assert sig.category == SignalCategory.RED_FLAG

    def test_detects_recruit_others(self):
        from sentinel.signals import check_mlm_language

        job = JobPosting(description="Recruit others to join your team and earn commissions.")
        sig = check_mlm_language(job)
        assert sig is not None
        assert sig.name == "mlm_language"

    def test_detects_be_your_own_boss(self):
        from sentinel.signals import check_mlm_language

        job = JobPosting(description="Be your own boss and enjoy unlimited earning potential!")
        sig = check_mlm_language(job)
        assert sig is not None

    def test_no_signal_for_legitimate_sales_role(self):
        from sentinel.signals import check_mlm_language

        job = JobPosting(
            description=(
                "We are looking for an Account Executive to manage enterprise sales pipelines. "
                "You will work with our SDR team to close Fortune 500 deals."
            )
        )
        assert check_mlm_language(job) is None


class TestReshippingScam:
    """check_reshipping_scam: 'receive packages', 'reship', 'forward packages'."""

    def test_detects_reship(self):
        from sentinel.signals import check_reshipping_scam

        job = JobPosting(
            description="You will receive packages at home and reship them to our warehouse."
        )
        sig = check_reshipping_scam(job)
        assert sig is not None
        assert sig.name == "reshipping_scam"
        assert sig.category == SignalCategory.RED_FLAG
        assert sig.weight == pytest.approx(0.9)

    def test_detects_forward_packages(self):
        from sentinel.signals import check_reshipping_scam

        job = JobPosting(description="Your job is to forward packages to our international warehouse daily.")
        sig = check_reshipping_scam(job)
        assert sig is not None
        assert sig.name == "reshipping_scam"

    def test_detects_receive_packages_at_home(self):
        from sentinel.signals import check_reshipping_scam

        job = JobPosting(description="Receive parcels at your home and prepare them for reshipment.")
        sig = check_reshipping_scam(job)
        assert sig is not None

    def test_no_signal_for_warehouse_logistics(self):
        from sentinel.signals import check_reshipping_scam

        job = JobPosting(
            description=(
                "Warehouse associate needed to manage inbound/outbound inventory at our facility. "
                "Forklift certification preferred. Monday–Friday, full-time."
            )
        )
        assert check_reshipping_scam(job) is None


class TestDataHarvesting:
    """check_data_harvesting: Google Forms links, 'fill out form at'."""

    def test_detects_google_forms_link(self):
        from sentinel.signals import check_data_harvesting

        job = JobPosting(
            description="Interested? Fill out the application at forms.gle/xyz123 to get started."
        )
        sig = check_data_harvesting(job)
        assert sig is not None
        assert sig.name == "data_harvesting"
        assert sig.category == SignalCategory.RED_FLAG

    def test_detects_docs_google_forms(self):
        from sentinel.signals import check_data_harvesting

        job = JobPosting(
            description="Apply now: docs.google.com/forms/d/abc456 — we need your personal info."
        )
        sig = check_data_harvesting(job)
        assert sig is not None
        assert sig.name == "data_harvesting"

    def test_detects_fill_out_form_at(self):
        from sentinel.signals import check_data_harvesting

        job = JobPosting(
            description="Please fill out our application form at typeform.com/to/apply123 now."
        )
        sig = check_data_harvesting(job)
        assert sig is not None

    def test_no_signal_for_normal_application(self):
        from sentinel.signals import check_data_harvesting

        job = JobPosting(
            description=(
                "Apply through our careers page at careers.acme.com/jobs/senior-engineer. "
                "Upload your resume and cover letter directly."
            )
        )
        assert check_data_harvesting(job) is None


class TestCompensationRedFlags:
    """check_compensation_red_flags: 'commission only', '1099', 'unpaid training'."""

    def test_detects_commission_only(self):
        from sentinel.signals import check_compensation_red_flags

        job = JobPosting(description="This is a commission-only role with no base salary provided.")
        sig = check_compensation_red_flags(job)
        assert sig is not None
        assert sig.name == "compensation_red_flags"
        assert sig.category == SignalCategory.WARNING

    def test_detects_1099_only(self):
        from sentinel.signals import check_compensation_red_flags

        job = JobPosting(description="All workers classified as 1099 only — no W-2 employees.")
        sig = check_compensation_red_flags(job)
        assert sig is not None
        assert sig.name == "compensation_red_flags"

    def test_detects_unpaid_training(self):
        from sentinel.signals import check_compensation_red_flags

        job = JobPosting(description="Training period is unpaid but you will learn valuable skills.")
        sig = check_compensation_red_flags(job)
        assert sig is not None

    def test_no_signal_for_standard_compensation(self):
        from sentinel.signals import check_compensation_red_flags

        job = JobPosting(
            description=(
                "We offer a competitive base salary of $95,000 with health insurance, "
                "401k matching, and an annual performance bonus."
            )
        )
        assert check_compensation_red_flags(job) is None


class TestCompanyNameSuspicious:
    """check_company_name_suspicious: all-caps name, 'Solutions International'."""

    def test_detects_all_caps_company_name(self):
        from sentinel.signals import check_company_name_suspicious

        job = JobPosting(company="GLOBALREACH")
        sig = check_company_name_suspicious(job)
        assert sig is not None
        assert sig.name == "company_name_suspicious"
        assert sig.category == SignalCategory.WARNING

    def test_detects_generic_suffix_solutions(self):
        from sentinel.signals import check_company_name_suspicious

        job = JobPosting(company="Apex Solutions")
        sig = check_company_name_suspicious(job)
        assert sig is not None
        assert sig.name == "company_name_suspicious"

    def test_detects_generic_suffix_international(self):
        from sentinel.signals import check_company_name_suspicious

        job = JobPosting(company="Horizon International")
        sig = check_company_name_suspicious(job)
        assert sig is not None

    def test_no_signal_for_known_company_pattern(self):
        from sentinel.signals import check_company_name_suspicious

        # Well-known tech company names don't use generic scam suffixes
        job = JobPosting(company="Stripe")
        assert check_company_name_suspicious(job) is None

    def test_no_signal_for_empty_company(self):
        from sentinel.signals import check_company_name_suspicious

        job = JobPosting(company="")
        # Empty company → None (handled at top of function)
        assert check_company_name_suspicious(job) is None


# ===========================================================================
# TestEcosystem
# ===========================================================================


class TestEcosystem:
    def test_publish_observation_writes_to_jsonl(self, tmp_path, monkeypatch):
        """publish_observation appends a valid JSON line to the observations file."""
        obs_path = tmp_path / "observations.jsonl"
        import sentinel.ecosystem as eco
        monkeypatch.setattr(eco, "OBSERVATIONS_PATH", obs_path)

        eco.publish_observation("success", "test evidence", "test context")

        assert obs_path.exists()
        lines = obs_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["subject"] == "sentinel"
        assert entry["category"] == "success"
        assert entry["evidence"] == "test evidence"
        assert entry["context"] == "test context"

    def test_publish_event_writes_correct_format(self, tmp_path, monkeypatch):
        """publish_event writes a properly structured event entry."""
        events_path = tmp_path / "events.jsonl"
        import sentinel.ecosystem as eco
        monkeypatch.setattr(eco, "EVENTS_PATH", events_path)

        eco.publish_event("sentinel_test", {"key": "value", "score": 0.5})

        assert events_path.exists()
        lines = events_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "sentinel_test"
        assert entry["source"] == "sentinel"
        assert entry["data"] == {"key": "value", "score": 0.5}
        assert "ts" in entry

    def test_publish_detection_result_creates_valid_observation(self, tmp_path, monkeypatch):
        """publish_detection_result calls publish_observation with scam pattern data."""
        obs_path = tmp_path / "observations.jsonl"
        import sentinel.ecosystem as eco
        monkeypatch.setattr(eco, "OBSERVATIONS_PATH", obs_path)

        eco.publish_detection_result(scam_score=0.87, signal_count=5, risk_level="scam")

        assert obs_path.exists()
        entry = json.loads(obs_path.read_text().strip().splitlines()[0])
        assert "0.87" in entry["evidence"]
        assert "5" in entry["evidence"]
        assert "scam" in entry["evidence"]

    def test_publish_observation_multiple_appends(self, tmp_path, monkeypatch):
        """Multiple publish_observation calls all append to the same file."""
        obs_path = tmp_path / "observations.jsonl"
        import sentinel.ecosystem as eco
        monkeypatch.setattr(eco, "OBSERVATIONS_PATH", obs_path)

        eco.publish_observation("success", "first")
        eco.publish_observation("failure", "second")
        eco.publish_observation("partial", "third")

        lines = obs_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_read_ecosystem_context_returns_dict_when_no_ecosystem(self):
        """read_ecosystem_context returns a dict even when no ecosystem tooling is present."""
        from sentinel.ecosystem import read_ecosystem_context

        result = read_ecosystem_context()
        assert isinstance(result, dict)


# ===========================================================================
# TestInnovationEngine
# ===========================================================================


class TestImprovementArm:
    def test_sample_returns_value_in_0_1(self):
        """ImprovementArm.sample returns a float in [0, 1]."""
        from sentinel.innovation import ImprovementArm

        arm = ImprovementArm(name="test_arm", description="test")
        for _ in range(50):
            val = arm.sample()
            assert 0.0 <= val <= 1.0

    def test_mean_uniform_prior(self):
        """Default Beta(1,1) has mean = 0.5."""
        from sentinel.innovation import ImprovementArm

        arm = ImprovementArm(name="test_arm", description="test")
        assert arm.mean == pytest.approx(0.5)

    def test_mean_after_successes(self):
        """Alpha=9, beta=1 → mean = 0.9."""
        from sentinel.innovation import ImprovementArm

        arm = ImprovementArm(name="strong_arm", description="test", alpha=9.0, beta=1.0)
        assert arm.mean == pytest.approx(0.9)

    def test_mean_after_failures(self):
        """Alpha=1, beta=9 → mean ≈ 0.1."""
        from sentinel.innovation import ImprovementArm

        arm = ImprovementArm(name="weak_arm", description="test", alpha=1.0, beta=9.0)
        assert arm.mean == pytest.approx(0.1)

    def test_sample_biased_high_when_alpha_large(self):
        """With alpha=50, beta=1 the sample distribution should be concentrated near 1."""
        from sentinel.innovation import ImprovementArm

        arm = ImprovementArm(name="biased_arm", description="test", alpha=50.0, beta=1.0)
        samples = [arm.sample() for _ in range(200)]
        assert sum(samples) / len(samples) > 0.8


class TestInnovationEngineCore:
    def _make_engine(self, tmp_path):
        """Create an InnovationEngine backed by a temp DB and temp state file."""
        from sentinel.db import SentinelDB
        from sentinel.innovation import InnovationEngine

        db_path = str(tmp_path / "test_innovation.db")
        db = SentinelDB(db_path)
        db.__enter__()
        engine = InnovationEngine(db=db)
        # Redirect state file to tmp_path to avoid touching home dir
        engine.STATE_PATH = tmp_path / "innovation_state.json"
        return engine, db

    def test_select_strategy_returns_an_arm(self, tmp_path):
        """select_strategy returns a valid ImprovementArm."""
        from sentinel.innovation import ImprovementArm

        engine, db = self._make_engine(tmp_path)
        arm = engine.select_strategy()
        assert isinstance(arm, ImprovementArm)
        assert arm.name in {a.name for a in engine.STRATEGIES}
        db.__exit__(None, None, None)

    def test_run_cycle_returns_results_list(self, tmp_path, monkeypatch):
        """run_cycle returns a non-empty list of ImprovementResult objects."""
        from sentinel.innovation import ImprovementResult
        import sentinel.ecosystem as eco

        # Monkeypatch ecosystem writes to avoid touching home dir
        obs_path = tmp_path / "obs.jsonl"
        events_path = tmp_path / "events.jsonl"
        monkeypatch.setattr(eco, "OBSERVATIONS_PATH", obs_path)
        monkeypatch.setattr(eco, "EVENTS_PATH", events_path)

        engine, db = self._make_engine(tmp_path)
        results = engine.run_cycle(max_strategies=2)
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, ImprovementResult)
        db.__exit__(None, None, None)

    def test_save_and_load_state_round_trip(self, tmp_path):
        """_save_state persists arm posteriors; _load_state restores them."""
        from sentinel.innovation import InnovationEngine
        from sentinel.db import SentinelDB

        db_path = str(tmp_path / "rt.db")
        db = SentinelDB(db_path)
        db.__enter__()

        engine = InnovationEngine(db=db)
        engine.STATE_PATH = tmp_path / "state.json"

        # Mutate one arm
        target_arm = engine.STRATEGIES[0]
        target_arm.alpha = 7.0
        target_arm.beta = 3.0
        target_arm.attempts = 4

        engine._save_state()

        # Create a fresh engine pointing at the same state file
        engine2 = InnovationEngine(db=db)
        engine2.STATE_PATH = tmp_path / "state.json"
        engine2._load_state()

        restored = next(a for a in engine2.STRATEGIES if a.name == target_arm.name)
        assert restored.alpha == pytest.approx(7.0)
        assert restored.beta == pytest.approx(3.0)
        assert restored.attempts == 4
        db.__exit__(None, None, None)

    def test_state_file_created_after_save(self, tmp_path):
        """_save_state creates the JSON file on disk."""
        from sentinel.db import SentinelDB
        from sentinel.innovation import InnovationEngine

        db_path = str(tmp_path / "stest.db")
        db = SentinelDB(db_path)
        db.__enter__()
        engine = InnovationEngine(db=db)
        engine.STATE_PATH = tmp_path / "innovation_state.json"
        engine._save_state()
        assert engine.STATE_PATH.exists()
        data = json.loads(engine.STATE_PATH.read_text())
        assert len(data) == len(engine.STRATEGIES)
        db.__exit__(None, None, None)

    def test_strategy_rankings_sorted_by_mean(self, tmp_path):
        """get_strategy_rankings returns strategies sorted descending by mean."""
        engine, db = self._make_engine(tmp_path)
        # Give one arm a much higher mean
        engine.STRATEGIES[2].alpha = 20.0
        engine.STRATEGIES[2].beta = 1.0

        rankings = engine.get_strategy_rankings()
        means = [r["mean"] for r in rankings]
        assert means == sorted(means, reverse=True)
        db.__exit__(None, None, None)

    def test_get_report_returns_correct_structure(self, tmp_path):
        """get_report returns a dict with the expected keys."""
        engine, db = self._make_engine(tmp_path)
        report = engine.get_report()
        assert isinstance(report, dict)
        for key in ("flywheel_grade", "precision", "recall", "strategies", "total_cycles"):
            assert key in report
        assert isinstance(report["strategies"], list)
        assert isinstance(report["total_cycles"], int)
        db.__exit__(None, None, None)

    def test_total_cycles_increments_after_run(self, tmp_path, monkeypatch):
        """total_cycles in get_report reflects the sum of all arm attempts."""
        import sentinel.ecosystem as eco

        obs_path = tmp_path / "obs.jsonl"
        events_path = tmp_path / "events.jsonl"
        monkeypatch.setattr(eco, "OBSERVATIONS_PATH", obs_path)
        monkeypatch.setattr(eco, "EVENTS_PATH", events_path)

        engine, db = self._make_engine(tmp_path)
        before = engine.get_report()["total_cycles"]
        engine.run_cycle(max_strategies=1)
        after = engine.get_report()["total_cycles"]
        assert after > before
        db.__exit__(None, None, None)


class TestInnovationStrategies:
    """Individual strategy methods return ImprovementResult objects."""

    def _make_engine(self, tmp_path):
        from sentinel.db import SentinelDB
        from sentinel.innovation import InnovationEngine

        db_path = str(tmp_path / "strat.db")
        db = SentinelDB(db_path)
        db.__enter__()
        engine = InnovationEngine(db=db)
        engine.STATE_PATH = tmp_path / "state.json"
        return engine, db

    def test_review_false_positives_returns_result(self, tmp_path):
        from sentinel.innovation import ImprovementResult

        engine, db = self._make_engine(tmp_path)
        result = engine._review_false_positives(0.8)
        assert isinstance(result, ImprovementResult)
        assert result.strategy == "false_positive_review"
        db.__exit__(None, None, None)

    def test_review_false_negatives_returns_result(self, tmp_path):
        from sentinel.innovation import ImprovementResult

        engine, db = self._make_engine(tmp_path)
        result = engine._review_false_negatives(0.8)
        assert isinstance(result, ImprovementResult)
        assert result.strategy == "false_negative_review"
        db.__exit__(None, None, None)

    def test_optimize_weights_returns_result(self, tmp_path):
        from sentinel.innovation import ImprovementResult

        engine, db = self._make_engine(tmp_path)
        result = engine._optimize_weights(0.8)
        assert isinstance(result, ImprovementResult)
        assert result.strategy == "weight_optimization"
        db.__exit__(None, None, None)

    def test_mine_patterns_returns_result_insufficient_data(self, tmp_path):
        """With fewer than 5 scam reports, pattern mining should return failure."""
        from sentinel.innovation import ImprovementResult

        engine, db = self._make_engine(tmp_path)
        result = engine._mine_patterns(0.8)
        assert isinstance(result, ImprovementResult)
        assert result.strategy == "pattern_mining"
        assert result.success is False
        db.__exit__(None, None, None)

    def test_check_regression_returns_result(self, tmp_path):
        from sentinel.innovation import ImprovementResult

        engine, db = self._make_engine(tmp_path)
        result = engine._check_regression(0.8)
        assert isinstance(result, ImprovementResult)
        assert result.strategy == "regression_check"
        db.__exit__(None, None, None)

    def test_correlate_signals_returns_result(self, tmp_path):
        from sentinel.innovation import ImprovementResult

        engine, db = self._make_engine(tmp_path)
        result = engine._correlate_signals(0.8)
        assert isinstance(result, ImprovementResult)
        assert result.strategy == "cross_signal_correlation"
        assert result.success is True
        db.__exit__(None, None, None)

    def test_expand_keywords_returns_result(self, tmp_path):
        from sentinel.innovation import ImprovementResult

        engine, db = self._make_engine(tmp_path)
        result = engine._expand_keywords(0.8)
        assert isinstance(result, ImprovementResult)
        assert result.strategy == "keyword_expansion"
        db.__exit__(None, None, None)

    def test_tune_thresholds_returns_result(self, tmp_path):
        from sentinel.innovation import ImprovementResult

        engine, db = self._make_engine(tmp_path)
        result = engine._tune_thresholds(0.8)
        assert isinstance(result, ImprovementResult)
        assert result.strategy == "threshold_tuning"
        db.__exit__(None, None, None)
