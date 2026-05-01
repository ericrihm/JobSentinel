"""Unit tests for sentinel/link_analyzer.py — URL/link analysis module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from sentinel.link_analyzer import (
    LinkAnalyzer,
    _levenshtein,
    _parse_domain,
    _get_tld,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _analyzer() -> LinkAnalyzer:
    return LinkAnalyzer()


# ===========================================================================
# Module-level helpers
# ===========================================================================

class TestLevenshtein:
    def test_identical_strings(self):
        assert _levenshtein("google", "google") == 0

    def test_empty_a(self):
        assert _levenshtein("", "abc") == 3

    def test_empty_b(self):
        assert _levenshtein("abc", "") == 3

    def test_single_substitution(self):
        assert _levenshtein("gogle", "google") == 1

    def test_insertion(self):
        assert _levenshtein("linkin", "linkedin") == 2


class TestParseDomain:
    def test_full_https_url(self):
        assert _parse_domain("https://www.example.com/path") == "example.com"

    def test_strips_www(self):
        assert _parse_domain("http://www.google.com") == "google.com"

    def test_no_scheme(self):
        assert _parse_domain("example.com") == "example.com"

    def test_empty_string(self):
        assert _parse_domain("") == ""

    def test_subdomain_preserved(self):
        domain = _parse_domain("https://careers.amazon.com/jobs/1")
        assert domain == "careers.amazon.com"


class TestGetTld:
    def test_com_tld(self):
        assert _get_tld("example.com") == ".com"

    def test_xyz_tld(self):
        assert _get_tld("malicious.xyz") == ".xyz"

    def test_no_dot(self):
        assert _get_tld("localhost") == ""

    def test_subdomain(self):
        assert _get_tld("sub.example.co") == ".co"


# ===========================================================================
# extract_urls
# ===========================================================================

class TestExtractUrls:
    def setup_method(self):
        self.analyzer = _analyzer()

    def test_https_url(self):
        urls = self.analyzer.extract_urls("Visit https://example.com for details.")
        assert "https://example.com" in urls

    def test_http_url(self):
        urls = self.analyzer.extract_urls("See http://jobs.example.org/apply")
        assert any("jobs.example.org" in u for u in urls)

    def test_shortener_url(self):
        urls = self.analyzer.extract_urls("Apply here: bit.ly/abcXYZ123")
        assert any("bit.ly" in u for u in urls)

    def test_bare_domain(self):
        urls = self.analyzer.extract_urls("Visit scamjobs.xyz for more info.")
        assert any("scamjobs.xyz" in u for u in urls)

    def test_no_urls(self):
        urls = self.analyzer.extract_urls("No links here, plain text only.")
        assert urls == []

    def test_empty_string(self):
        assert self.analyzer.extract_urls("") == []

    def test_deduplication(self):
        text = "See https://example.com and https://example.com again."
        urls = self.analyzer.extract_urls(text)
        assert urls.count("https://example.com") == 1

    def test_preserves_order(self):
        text = "First https://alpha.com then https://beta.com"
        urls = self.analyzer.extract_urls(text)
        alpha_idx = next(i for i, u in enumerate(urls) if "alpha.com" in u)
        beta_idx = next(i for i, u in enumerate(urls) if "beta.com" in u)
        assert alpha_idx < beta_idx

    def test_strips_trailing_punctuation(self):
        urls = self.analyzer.extract_urls("Visit https://example.com.")
        # Trailing period should be stripped
        for u in urls:
            assert not u.endswith(".")

    def test_email_not_extracted_as_url(self):
        urls = self.analyzer.extract_urls("Contact us: recruiter@gmail.com for details.")
        # Email addresses should not be picked up as bare domain URLs
        assert not any(u.startswith("@") for u in urls)


# ===========================================================================
# analyze_domain
# ===========================================================================

class TestAnalyzeDomain:
    def setup_method(self):
        self.analyzer = _analyzer()

    def test_clean_domain_low_risk(self):
        result = self.analyzer.analyze_domain("https://stripe.com/jobs")
        assert result["risk_score"] < 0.4
        assert not result["is_high_risk_tld"]
        assert not result["is_shortener"]

    def test_high_risk_tld_flagged(self):
        result = self.analyzer.analyze_domain("https://jobs.xyz/apply")
        assert result["is_high_risk_tld"] is True
        assert "high_risk_tld" in result["flags"]
        assert result["risk_score"] >= 0.35

    def test_shortener_flagged(self):
        result = self.analyzer.analyze_domain("bit.ly/abc123")
        assert result["is_shortener"] is True
        assert "shortened_url" in result["flags"]

    def test_free_email_domain_flagged(self):
        result = self.analyzer.analyze_domain("gmail.com")
        assert result["is_free_email_domain"] is True
        assert "free_email_domain" in result["flags"]

    def test_brand_impersonation_detected(self):
        # "gooogle" is distance 1 from "google"
        result = self.analyzer.analyze_domain("https://gooogle.com/jobs")
        assert result["brand_impersonation"] == "google"
        assert any("brand_impersonation" in f for f in result["flags"])
        assert result["risk_score"] >= 0.50

    def test_exact_brand_not_flagged(self):
        # linkedin.com itself should not be flagged as brand impersonation
        result = self.analyzer.analyze_domain("https://linkedin.com/company/acme")
        assert result["brand_impersonation"] is None

    def test_risk_score_capped_at_one(self):
        # A URL that hits multiple risk factors should not exceed 1.0
        result = self.analyzer.analyze_domain("https://amazzon.xyz/apply")
        assert result["risk_score"] <= 1.0

    def test_empty_url(self):
        result = self.analyzer.analyze_domain("")
        assert "unparseable_domain" in result["flags"]
        assert result["risk_score"] > 0.0

    def test_result_has_all_keys(self):
        result = self.analyzer.analyze_domain("https://example.com")
        for key in ("url", "domain", "tld", "risk_score", "flags",
                    "is_free_email_domain", "is_shortener", "is_high_risk_tld",
                    "looks_autogenerated", "brand_impersonation"):
            assert key in result

    def test_autogenerated_domain_flag(self):
        # A domain with long random hex-like label + high digit ratio
        result = self.analyzer.analyze_domain("https://a3b9c2d1.top/go")
        # Should trigger at least autogenerated or high_risk_tld
        assert result["risk_score"] > 0.0


# ===========================================================================
# check_url_reputation — local blocklist (no API keys needed)
# ===========================================================================

class TestCheckUrlReputation:
    def setup_method(self):
        self.analyzer = _analyzer()

    def test_clean_url_not_malicious(self):
        result = self.analyzer.check_url_reputation("https://stripe.com/careers")
        assert result["is_malicious"] is False
        assert result["url"] == "https://stripe.com/careers"

    def test_local_blocklist_work_from_home(self):
        result = self.analyzer.check_url_reputation("http://workfromhome123.xyz/apply")
        assert result["is_malicious"] is True
        assert result["source"] == "local_blocklist"
        assert result["threat_type"] == "local_blocklist"

    def test_local_blocklist_job_apply_scam(self):
        result = self.analyzer.check_url_reputation("https://jobapply.xyz/now")
        assert result["is_malicious"] is True

    def test_result_keys_present(self):
        result = self.analyzer.check_url_reputation("https://example.com")
        for key in ("url", "is_malicious", "threat_type", "source", "checked"):
            assert key in result

    def test_no_api_keys_unchecked(self):
        # Without env vars, checked stays False unless local blocklist fires
        with patch.dict("os.environ", {}, clear=True):
            result = self.analyzer.check_url_reputation("https://legitimate-company.com")
        assert result["checked"] is False or result["is_malicious"] is False

    def test_gsb_key_triggers_check(self):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"matches": []}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.dict("os.environ", {"GOOGLE_SAFE_BROWSING_API_KEY": "test_key"}):
            with patch("urllib.request.urlopen", return_value=mock_response):
                result = self.analyzer.check_url_reputation("https://example.com")
        assert result["source"] == "google_safe_browsing"
        assert result["is_malicious"] is False

    def test_gsb_malicious_match(self):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"matches": [{"threatType": "SOCIAL_ENGINEERING"}]}'
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch.dict("os.environ", {"GOOGLE_SAFE_BROWSING_API_KEY": "test_key"}):
            with patch("urllib.request.urlopen", return_value=mock_response):
                result = self.analyzer.check_url_reputation("https://evil.example.com")
        assert result["is_malicious"] is True
        assert result["threat_type"] == "SOCIAL_ENGINEERING"


# ===========================================================================
# analyze_redirect_chain
# ===========================================================================

class TestAnalyzeRedirectChain:
    def setup_method(self):
        self.analyzer = _analyzer()

    def test_no_redirect_returns_same_url(self):
        url = "https://example.com/jobs"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}

        with patch("httpx.head", return_value=mock_resp):
            result = self.analyzer.analyze_redirect_chain(url)

        assert result["url"] == url
        assert result["final_url"] == url
        assert result["hop_count"] == 0

    def test_cross_domain_redirect_flagged(self):
        url = "https://bit.ly/abc123"

        responses = [
            MagicMock(status_code=302, headers={"location": "https://evil.xyz/landing"}),
            MagicMock(status_code=200, headers={}),
        ]
        with patch("httpx.head", side_effect=responses):
            result = self.analyzer.analyze_redirect_chain(url)

        assert result["hop_count"] >= 1
        assert "cross_domain_redirect" in result["flags"]
        assert result["is_suspicious"] is True

    def test_high_risk_tld_final_destination(self):
        url = "https://shorturl.example.com/go"

        responses = [
            MagicMock(status_code=301, headers={"location": "https://scam.top/apply"}),
            MagicMock(status_code=200, headers={}),
        ]
        with patch("httpx.head", side_effect=responses):
            result = self.analyzer.analyze_redirect_chain(url)

        assert "redirect_to_high_risk_tld" in result["flags"]

    def test_network_error_gracefully_handled(self):
        with patch("httpx.head", side_effect=Exception("connection refused")):
            result = self.analyzer.analyze_redirect_chain("https://example.com")
        # Should not raise; error field may or may not be set depending on chain length
        assert "url" in result

    def test_result_has_all_keys(self):
        mock_resp = MagicMock(status_code=200, headers={})
        with patch("httpx.head", return_value=mock_resp):
            result = self.analyzer.analyze_redirect_chain("https://example.com")
        for key in ("url", "final_url", "hop_count", "domains_visited",
                    "is_suspicious", "flags", "error"):
            assert key in result


# ===========================================================================
# check_domain_age
# ===========================================================================

class TestCheckDomainAge:
    def setup_method(self):
        self.analyzer = _analyzer()

    def test_library_success_new_domain(self):
        from datetime import datetime, UTC, timedelta
        recent = datetime.now(UTC) - timedelta(days=30)

        with patch.object(self.analyzer, "_whois_via_library", return_value=recent):
            result = self.analyzer.check_domain_age("newdomain.com")

        assert result["is_new_domain"] is True
        assert result["age_days"] < 90

    def test_library_success_old_domain(self):
        from datetime import datetime, UTC, timedelta
        old = datetime.now(UTC) - timedelta(days=3650)

        with patch.object(self.analyzer, "_whois_via_library", return_value=old):
            result = self.analyzer.check_domain_age("established.com")

        assert result["is_new_domain"] is False
        assert result["age_days"] >= 90

    def test_whois_unavailable_returns_error(self):
        with patch.object(self.analyzer, "_whois_via_library", return_value=None):
            with patch.object(self.analyzer, "_whois_via_subprocess", return_value=None):
                result = self.analyzer.check_domain_age("unknown.example.com")
        assert result["error"] is not None
        assert result["age_days"] is None

    def test_result_has_all_keys(self):
        with patch.object(self.analyzer, "_whois_via_library", return_value=None):
            with patch.object(self.analyzer, "_whois_via_subprocess", return_value=None):
                result = self.analyzer.check_domain_age("example.com")
        for key in ("domain", "age_days", "registered_date", "is_new_domain", "error"):
            assert key in result


# ===========================================================================
# extract_link_signals (pipeline integration)
# ===========================================================================

class TestExtractLinkSignals:
    def test_returns_list(self):
        from sentinel.link_analyzer import extract_link_signals
        from sentinel.models import JobPosting

        job = JobPosting(
            url="https://example.com/job",
            title="Remote Work",
            description="No experience needed. Apply at bit.ly/scamjob now!",
        )
        with patch.object(LinkAnalyzer, "check_domain_age", return_value={
            "age_days": None, "is_new_domain": False, "registered_date": None, "error": "failed"
        }):
            with patch.object(LinkAnalyzer, "analyze_redirect_chain", return_value={
                "hop_count": 0, "is_suspicious": False, "flags": [], "final_url": "bit.ly/scamjob",
                "domains_visited": [], "error": None
            }):
                signals = extract_link_signals(job)

        assert isinstance(signals, list)

    def test_shortened_url_signal(self):
        from sentinel.link_analyzer import extract_link_signals
        from sentinel.models import JobPosting

        job = JobPosting(
            url="https://example.com/job",
            title="Job",
            description="Apply at bit.ly/abcdef for more info.",
        )
        with patch.object(LinkAnalyzer, "check_domain_age", return_value={
            "age_days": None, "is_new_domain": False, "registered_date": None, "error": "failed"
        }):
            with patch.object(LinkAnalyzer, "analyze_redirect_chain", return_value={
                "hop_count": 0, "is_suspicious": False, "flags": [], "final_url": "https://example.com",
                "domains_visited": ["example.com"], "error": None
            }):
                signals = extract_link_signals(job)

        names = [s.name for s in signals]
        assert "shortened_url" in names

    def test_empty_job_returns_empty(self):
        from sentinel.link_analyzer import extract_link_signals
        from sentinel.models import JobPosting

        job = JobPosting(url="https://example.com/job", title="", description="")
        result = extract_link_signals(job)
        assert result == []

    def test_no_urls_returns_empty(self):
        from sentinel.link_analyzer import extract_link_signals
        from sentinel.models import JobPosting

        job = JobPosting(
            url="https://example.com/job",
            title="Software Engineer",
            description="Write code and build products. No links here.",
        )
        result = extract_link_signals(job)
        assert result == []
