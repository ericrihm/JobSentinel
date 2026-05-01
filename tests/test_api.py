"""Comprehensive tests for sentinel.api — FastAPI REST endpoints."""

import os

import pytest

import sentinel.config
from sentinel.db import SentinelDB
from sentinel.knowledge import KnowledgeBase


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient backed by a temporary SQLite database."""
    db_path = str(tmp_path / "api_test.db")
    mock_config = sentinel.config.SentinelConfig(db_path=db_path)
    monkeypatch.setattr(sentinel.config, "get_config", lambda: mock_config)
    # Invalidate the cached config so SentinelDB() picks up our mock
    monkeypatch.setattr(sentinel.config, "_config", None)

    from sentinel.api import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    yield TestClient(app)


@pytest.fixture
def seeded_client(tmp_path, monkeypatch):
    """FastAPI TestClient with the default scam patterns seeded into the DB."""
    db_path = str(tmp_path / "api_seeded.db")
    mock_config = sentinel.config.SentinelConfig(db_path=db_path)
    monkeypatch.setattr(sentinel.config, "get_config", lambda: mock_config)
    monkeypatch.setattr(sentinel.config, "_config", None)

    # Seed patterns before starting the app
    db = SentinelDB(path=db_path)
    kb = KnowledgeBase(db=db)
    kb.seed_default_patterns()
    db.close()

    from sentinel.api import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    yield TestClient(app)


# ===========================================================================
# POST /api/analyze
# ===========================================================================


class TestAnalyzeEndpoint:
    def test_analyze_scam_text_returns_200(self, client):
        """A clear scam description should return HTTP 200."""
        response = client.post(
            "/api/analyze",
            json={"text": "Send $50 fee to start working from home guaranteed income"},
        )
        assert response.status_code == 200

    def test_analyze_scam_text_high_score(self, client):
        """A clear scam description should produce a scam_score > 0."""
        response = client.post(
            "/api/analyze",
            json={"text": "Send $50 fee to start. Earn GUARANTEED $5,000/week! No experience needed."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "scam_score" in data
        assert data["scam_score"] > 0

    def test_analyze_scam_text_risk_level_present(self, client):
        """Response should include risk_level field."""
        response = client.post(
            "/api/analyze",
            json={"text": "Send $99 registration fee. Earn $5000 per week guaranteed!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "risk_level" in data
        assert data["risk_level"] in ("safe", "low", "suspicious", "high", "scam")

    def test_analyze_obvious_scam_text_labeled_scam(self, client):
        """A heavily scam-laden text should be labeled scam or high risk."""
        response = client.post(
            "/api/analyze",
            json={
                "text": (
                    "Earn GUARANTEED $5,000/week working from home! "
                    "No experience required. Send $99 registration fee to get started. "
                    "Provide your SSN and bank account. Hire immediately, no interview needed."
                )
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] in ("high", "scam")

    def test_analyze_response_has_required_fields(self, client):
        """Response dict should contain all required fields."""
        response = client.post(
            "/api/analyze",
            json={"text": "Software Engineer at Google, competitive salary, excellent benefits."},
        )
        assert response.status_code == 200
        data = response.json()
        required = {"scam_score", "confidence", "risk_level", "risk_label",
                    "red_flags", "warnings", "signal_count"}
        assert required.issubset(set(data.keys()))

    def test_analyze_with_title_and_company(self, client):
        """Text with title and company context should still return valid analysis."""
        response = client.post(
            "/api/analyze",
            json={
                "text": "Join our dynamic team for exciting opportunities!",
                "title": "Work From Home Agent",
                "company": "Global Opportunity LLC",
            },
        )
        assert response.status_code == 200
        assert "scam_score" in response.json()

    def test_analyze_missing_all_inputs_returns_422(self, client):
        """Sending an empty payload (no text/url/job_data) should return 422."""
        response = client.post("/api/analyze", json={})
        assert response.status_code == 422

    def test_analyze_invalid_url_scheme_returns_422(self, client):
        """A URL without http/https scheme should be rejected with 422."""
        response = client.post(
            "/api/analyze",
            json={"url": "ftp://not-a-valid-url.com/job/123"},
        )
        assert response.status_code == 422

    def test_analyze_legit_text_lower_risk(self, client):
        """A clearly legitimate posting should have a lower scam score."""
        response = client.post(
            "/api/analyze",
            json={
                "text": (
                    "Senior Software Engineer at Google. Competitive salary $180k-$250k. "
                    "5+ years Python experience required. Full healthcare, 401k. "
                    "Interview process: phone screen, technical, system design."
                )
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Legit posting should not be flagged as definite scam with very high confidence
        assert data["risk_level"] not in ("scam",) or data["scam_score"] < 0.99

    def test_analyze_response_signal_count_nonnegative(self, client):
        """signal_count in response should be >= 0."""
        response = client.post(
            "/api/analyze",
            json={"text": "Looking for enthusiastic team members to join our company."},
        )
        assert response.status_code == 200
        assert response.json()["signal_count"] >= 0

    def test_analyze_red_flags_is_list(self, client):
        """red_flags in response should be a list."""
        response = client.post(
            "/api/analyze",
            json={"text": "Work from home earn $5000/week guaranteed, no experience needed."},
        )
        assert response.status_code == 200
        assert isinstance(response.json()["red_flags"], list)


# ===========================================================================
# POST /api/report
# ===========================================================================


class TestReportEndpoint:
    def test_report_scam_returns_200(self, client):
        """Submitting a valid scam report should return HTTP 200."""
        response = client.post(
            "/api/report",
            json={
                "url": "https://www.linkedin.com/jobs/view/9999999",
                "is_scam": True,
                "reason": "Asked for upfront payment",
            },
        )
        assert response.status_code == 200

    def test_report_scam_recorded_true(self, client):
        """Response should have recorded=True."""
        response = client.post(
            "/api/report",
            json={
                "url": "https://www.linkedin.com/jobs/view/8888888",
                "is_scam": True,
                "reason": "Guaranteed income claim",
            },
        )
        assert response.status_code == 200
        assert response.json()["recorded"] is True

    def test_report_legitimate_job(self, client):
        """Reporting a job as legitimate should also return 200 with recorded=True."""
        response = client.post(
            "/api/report",
            json={
                "url": "https://www.linkedin.com/jobs/view/1111111",
                "is_scam": False,
                "reason": "This is a real Google job",
            },
        )
        assert response.status_code == 200
        assert response.json()["recorded"] is True

    def test_report_verdict_matches_is_scam(self, client):
        """Verdict field should match the is_scam flag."""
        response = client.post(
            "/api/report",
            json={
                "url": "https://www.linkedin.com/jobs/view/7777777",
                "is_scam": True,
                "reason": "",
            },
        )
        assert response.json()["verdict"] == "scam"

        response2 = client.post(
            "/api/report",
            json={
                "url": "https://www.linkedin.com/jobs/view/6666666",
                "is_scam": False,
                "reason": "",
            },
        )
        assert response2.json()["verdict"] == "legitimate"

    def test_report_without_reason(self, client):
        """Report without a reason should still succeed."""
        response = client.post(
            "/api/report",
            json={
                "url": "https://www.linkedin.com/jobs/view/5555555",
                "is_scam": True,
            },
        )
        assert response.status_code == 200

    def test_report_missing_url_returns_422(self, client):
        """Missing required url field should return 422."""
        response = client.post(
            "/api/report",
            json={"is_scam": True, "reason": "test"},
        )
        assert response.status_code == 422

    def test_report_missing_is_scam_returns_422(self, client):
        """Missing required is_scam field should return 422."""
        response = client.post(
            "/api/report",
            json={"url": "https://example.com/job/1", "reason": "test"},
        )
        assert response.status_code == 422

    def test_report_response_has_message(self, client):
        """Response should contain a message field."""
        response = client.post(
            "/api/report",
            json={
                "url": "https://www.linkedin.com/jobs/view/4444444",
                "is_scam": True,
            },
        )
        assert response.status_code == 200
        assert "message" in response.json()
        assert len(response.json()["message"]) > 0


# ===========================================================================
# GET /api/patterns
# ===========================================================================


class TestPatternsEndpoint:
    def test_patterns_returns_200(self, client):
        """GET /api/patterns should return HTTP 200."""
        response = client.get("/api/patterns")
        assert response.status_code == 200

    def test_patterns_has_patterns_key(self, client):
        """Response should contain a 'patterns' key."""
        response = client.get("/api/patterns")
        assert "patterns" in response.json()

    def test_patterns_has_count_key(self, client):
        """Response should contain a 'count' key."""
        response = client.get("/api/patterns")
        assert "count" in response.json()

    def test_patterns_count_matches_list_length(self, client):
        """count field should equal len(patterns)."""
        response = client.get("/api/patterns")
        data = response.json()
        assert data["count"] == len(data["patterns"])

    def test_patterns_with_seeded_db_returns_patterns(self, seeded_client):
        """With a seeded DB the patterns list should be non-empty."""
        response = seeded_client.get("/api/patterns")
        data = response.json()
        assert data["count"] > 0
        assert len(data["patterns"]) > 0

    def test_patterns_each_item_has_required_fields(self, seeded_client):
        """Each pattern object should have all required fields."""
        response = seeded_client.get("/api/patterns")
        patterns = response.json()["patterns"]
        assert len(patterns) > 0
        required_fields = {"pattern_id", "name", "description", "category",
                           "status", "observations", "bayesian_score", "keywords"}
        for pattern in patterns:
            assert required_fields.issubset(set(pattern.keys()))

    def test_patterns_status_filter_active(self, seeded_client):
        """Filtering by status=active should return only active patterns."""
        response = seeded_client.get("/api/patterns?status=active")
        assert response.status_code == 200
        data = response.json()
        for p in data["patterns"]:
            assert p["status"] == "active"

    def test_patterns_status_filter_all(self, seeded_client):
        """status=all should return patterns of all statuses."""
        response = seeded_client.get("/api/patterns?status=all")
        assert response.status_code == 200
        assert "patterns" in response.json()

    def test_patterns_category_filter(self, seeded_client):
        """Filtering by category should restrict results to that category."""
        response = seeded_client.get("/api/patterns?category=red_flag")
        assert response.status_code == 200
        data = response.json()
        for p in data["patterns"]:
            assert p["category"] == "red_flag"

    def test_patterns_is_list(self, client):
        """patterns field should always be a list (even empty)."""
        response = client.get("/api/patterns")
        assert isinstance(response.json()["patterns"], list)


# ===========================================================================
# GET /api/stats
# ===========================================================================


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        """GET /api/stats should return HTTP 200."""
        response = client.get("/api/stats")
        assert response.status_code == 200

    def test_stats_has_required_keys(self, client):
        """Stats response should contain all documented fields."""
        response = client.get("/api/stats")
        data = response.json()
        required_keys = {
            "total_jobs_analyzed",
            "scam_jobs_detected",
            "avg_scam_score",
            "total_user_reports",
            "scam_reports",
            "prediction_accuracy",
            "active_patterns",
            "total_companies",
            "verified_companies",
            "last_flywheel_cycle",
        }
        assert required_keys.issubset(set(data.keys()))

    def test_stats_empty_db_all_zeros(self, client):
        """Fresh DB should report zero counts."""
        response = client.get("/api/stats")
        data = response.json()
        assert data["total_jobs_analyzed"] == 0
        assert data["total_user_reports"] == 0
        assert data["scam_jobs_detected"] == 0

    def test_stats_prediction_accuracy_in_range(self, client):
        """prediction_accuracy should be between 0.0 and 1.0."""
        response = client.get("/api/stats")
        acc = response.json()["prediction_accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_stats_avg_scam_score_in_range(self, client):
        """avg_scam_score should be between 0.0 and 1.0."""
        response = client.get("/api/stats")
        score = response.json()["avg_scam_score"]
        assert 0.0 <= score <= 1.0

    def test_stats_last_flywheel_cycle_is_dict(self, client):
        """last_flywheel_cycle should be a dict (possibly empty)."""
        response = client.get("/api/stats")
        assert isinstance(response.json()["last_flywheel_cycle"], dict)

    def test_stats_reflects_analyzed_jobs(self, client):
        """After analyzing a job, total_jobs_analyzed should increment."""
        # Analyze a job first
        client.post(
            "/api/analyze",
            json={"text": "Send $50 registration fee to join our team."},
        )
        response = client.get("/api/stats")
        assert response.json()["total_jobs_analyzed"] >= 1


# ===========================================================================
# GET /api/health
# ===========================================================================


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """GET /api/health should return HTTP 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_has_healthy_key(self, client):
        """Health response should contain 'healthy' key."""
        response = client.get("/api/health")
        assert "healthy" in response.json()

    def test_health_has_required_keys(self, client):
        """Health response should contain all documented fields."""
        response = client.get("/api/health")
        data = response.json()
        required = {
            "status",
            "healthy",
            "grade",
            "precision",
            "recall",
            "f1",
            "active_patterns",
            "total_jobs_analyzed",
            "total_user_reports",
            "regression_alarm",
            "cusum_statistic",
            "cycle_count",
            "checked_at",
        }
        assert required.issubset(set(data.keys()))

    def test_health_fresh_db_is_cold_start(self, client):
        """A fresh DB with no data should report cold_start=True, healthy=False."""
        response = client.get("/api/health")
        data = response.json()
        assert data["cold_start"] is True
        assert data["healthy"] is False
        assert data["grade"] == "N/A"

    def test_health_status_field(self, client):
        """status should be 'ok' when healthy is True."""
        response = client.get("/api/health")
        data = response.json()
        if data["healthy"]:
            assert data["status"] == "ok"
        else:
            assert data["status"] == "degraded"

    def test_health_grade_is_letter(self, client):
        """grade should be one of A/B/C/D/F."""
        response = client.get("/api/health")
        assert response.json()["grade"] in ("A", "B", "C", "D", "F", "N/A")

    def test_health_regression_alarm_is_bool(self, client):
        """regression_alarm should be a boolean."""
        response = client.get("/api/health")
        assert isinstance(response.json()["regression_alarm"], bool)

    def test_health_cusum_statistic_is_float(self, client):
        """cusum_statistic should be a float."""
        response = client.get("/api/health")
        assert isinstance(response.json()["cusum_statistic"], float)

    def test_health_checked_at_is_string(self, client):
        """checked_at should be a non-empty ISO timestamp string."""
        response = client.get("/api/health")
        checked_at = response.json()["checked_at"]
        assert isinstance(checked_at, str)
        assert len(checked_at) > 0

    def test_health_precision_in_range(self, client):
        """precision should be between 0.0 and 1.0."""
        response = client.get("/api/health")
        precision = response.json()["precision"]
        assert 0.0 <= precision <= 1.0


# ===========================================================================
# POST /api/v1/analyze-url
# ===========================================================================


class TestAnalyzeUrlEndpoint:
    def test_analyze_url_returns_200(self, client):
        """POST /api/v1/analyze-url with a valid https URL should return 200."""
        response = client.post(
            "/api/v1/analyze-url",
            json={"url": "https://www.google.com"},
        )
        assert response.status_code == 200

    def test_analyze_url_response_has_required_fields(self, client):
        """Response should contain url, domain_analysis, and reputation fields."""
        response = client.post(
            "/api/v1/analyze-url",
            json={"url": "https://bit.ly/fake-job"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "url" in data
        assert "domain_analysis" in data
        assert "reputation" in data

    def test_analyze_url_domain_analysis_has_risk_score(self, client):
        """domain_analysis sub-dict should contain a risk_score field."""
        response = client.post(
            "/api/v1/analyze-url",
            json={"url": "https://bit.ly/scamjob1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data["domain_analysis"]

    def test_analyze_url_domain_analysis_has_flags(self, client):
        """domain_analysis should contain a flags list."""
        response = client.post(
            "/api/v1/analyze-url",
            json={"url": "https://jobs-apply.xyz/hiring-now"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["domain_analysis"]["flags"], list)

    def test_analyze_url_shortener_flagged(self, client):
        """A URL shortener domain should be flagged as is_shortener=True."""
        response = client.post(
            "/api/v1/analyze-url",
            json={"url": "https://bit.ly/apply-now"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["domain_analysis"]["is_shortener"] is True

    def test_analyze_url_missing_url_returns_422(self, client):
        """Missing url field should return 422."""
        response = client.post("/api/v1/analyze-url", json={})
        assert response.status_code == 422

    def test_analyze_url_no_http_scheme_returns_422(self, client):
        """URL without http/https scheme should return 422."""
        response = client.post(
            "/api/v1/analyze-url",
            json={"url": "ftp://badurl.com"},
        )
        assert response.status_code == 422

    def test_analyze_url_reputation_has_is_malicious(self, client):
        """reputation sub-dict should contain is_malicious boolean."""
        response = client.post(
            "/api/v1/analyze-url",
            json={"url": "https://example.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_malicious" in data["reputation"]
        assert isinstance(data["reputation"]["is_malicious"], bool)


# ===========================================================================
# POST /api/v1/verify-company
# ===========================================================================


class TestVerifyCompanyEndpoint:
    def test_verify_company_returns_200(self, client):
        """POST /api/v1/verify-company with a valid company name should return 200."""
        response = client.post(
            "/api/v1/verify-company",
            json={"company_name": "Google"},
        )
        assert response.status_code == 200

    def test_verify_company_response_has_required_fields(self, client):
        """Response should contain domain, existence, and company_name fields."""
        response = client.post(
            "/api/v1/verify-company",
            json={"company_name": "Microsoft"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "domain" in data
        assert "existence" in data
        assert "company_name" in data

    def test_verify_known_company_is_known_true(self, client):
        """A well-known Fortune 500 company should have existence.is_known=True."""
        response = client.post(
            "/api/v1/verify-company",
            json={"company_name": "google"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["existence"]["is_known"] is True

    def test_verify_fake_company_not_known(self, client):
        """A made-up company name should not be flagged as known."""
        response = client.post(
            "/api/v1/verify-company",
            json={"company_name": "xyzFakeCompanyNotReal99"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["existence"]["is_known"] is False

    def test_verify_company_existence_has_confidence(self, client):
        """existence sub-dict should contain a confidence score."""
        response = client.post(
            "/api/v1/verify-company",
            json={"company_name": "Amazon"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data["existence"]
        assert 0.0 <= data["existence"]["confidence"] <= 1.0

    def test_verify_company_missing_name_returns_422(self, client):
        """Missing company_name field should return 422."""
        response = client.post("/api/v1/verify-company", json={})
        assert response.status_code == 422

    def test_verify_company_with_url(self, client):
        """Passing company_url should add domain verification to response."""
        response = client.post(
            "/api/v1/verify-company",
            json={
                "company_name": "Stripe",
                "company_url": "https://stripe.com",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["company_url"] == "https://stripe.com"
        assert "domain" in data

    def test_verify_company_linkedin_none_when_not_provided(self, client):
        """linkedin field should be None when linkedin_url not provided."""
        response = client.post(
            "/api/v1/verify-company",
            json={"company_name": "Apple"},
        )
        assert response.status_code == 200
        assert response.json()["linkedin"] is None

    def test_verify_company_existence_has_flags_list(self, client):
        """existence.flags should always be a list."""
        response = client.post(
            "/api/v1/verify-company",
            json={"company_name": "Global Solutions LLC"},
        )
        assert response.status_code == 200
        assert isinstance(response.json()["existence"]["flags"], list)


# ===========================================================================
# POST /api/v1/analyze-links
# ===========================================================================


class TestAnalyzeLinksEndpoint:
    def test_analyze_links_returns_200(self, client):
        """POST /api/v1/analyze-links with valid text should return 200."""
        response = client.post(
            "/api/v1/analyze-links",
            json={"text": "Apply at https://example.com or email hr@company.com"},
        )
        assert response.status_code == 200

    def test_analyze_links_response_has_required_fields(self, client):
        """Response should contain urls_found (int) and results (list) fields."""
        response = client.post(
            "/api/v1/analyze-links",
            json={"text": "Visit https://bit.ly/job123 to apply"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "urls_found" in data
        assert "results" in data
        assert isinstance(data["urls_found"], int)
        assert isinstance(data["results"], list)

    def test_analyze_links_detects_shortener_url(self, client):
        """A bit.ly link in text should be detected and flagged as shortener."""
        response = client.post(
            "/api/v1/analyze-links",
            json={"text": "Click here to apply: https://bit.ly/apply-now123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["urls_found"] >= 1
        shorteners = [
            r for r in data["results"]
            if r["domain_analysis"].get("is_shortener")
        ]
        assert len(shorteners) >= 1

    def test_analyze_links_no_urls_returns_zero(self, client):
        """Text with no URLs should return urls_found=0 and empty results."""
        response = client.post(
            "/api/v1/analyze-links",
            json={"text": "Please call us at our office to apply for this position."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["urls_found"] == 0
        assert data["results"] == []

    def test_analyze_links_multiple_urls(self, client):
        """Text with multiple URLs should return one result per URL."""
        response = client.post(
            "/api/v1/analyze-links",
            json={
                "text": (
                    "Apply at https://example.com/jobs or visit "
                    "https://bit.ly/scamjob for more info."
                )
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["urls_found"] >= 1

    def test_analyze_links_each_result_has_domain_analysis(self, client):
        """Each result entry should contain domain_analysis and reputation."""
        response = client.post(
            "/api/v1/analyze-links",
            json={"text": "Apply here: https://jobs-apply.xyz/now"},
        )
        assert response.status_code == 200
        data = response.json()
        for result in data["results"]:
            assert "domain_analysis" in result
            assert "reputation" in result
            assert "url" in result

    def test_analyze_links_missing_text_returns_422(self, client):
        """Missing text field should return 422."""
        response = client.post("/api/v1/analyze-links", json={})
        assert response.status_code == 422

    def test_analyze_links_script_tag_rejected(self, client):
        """Text containing <script> should be rejected with 422."""
        response = client.post(
            "/api/v1/analyze-links",
            json={"text": "Apply here <script>alert(1)</script> https://example.com"},
        )
        assert response.status_code == 422

    def test_analyze_links_risk_score_in_range(self, client):
        """domain_analysis.risk_score for each URL should be between 0.0 and 1.0."""
        response = client.post(
            "/api/v1/analyze-links",
            json={"text": "Join us: https://bit.ly/job-scam and https://amazon-jobs.xyz"},
        )
        assert response.status_code == 200
        for result in response.json()["results"]:
            score = result["domain_analysis"]["risk_score"]
            assert 0.0 <= score <= 1.0
