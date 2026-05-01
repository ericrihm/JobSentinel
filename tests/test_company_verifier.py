"""Unit tests for sentinel/company_verifier.py — company legitimacy verification."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from sentinel.company_verifier import (
    CompanyVerifier,
    _normalize_for_compare,
    _domain_matches_company,
    _levenshtein,
    _is_misspelled_brand,
)
from sentinel.models import JobPosting, CompanyProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job(company: str = "TestCorp", location: str = "New York, NY", **kwargs) -> JobPosting:
    defaults = dict(
        url="https://example.com/job/1",
        title="Software Engineer",
        company=company,
        location=location,
        description="Write code.",
        company_linkedin_url="",
    )
    defaults.update(kwargs)
    return JobPosting(**defaults)


def _verifier() -> CompanyVerifier:
    return CompanyVerifier()


def _offline_domain_stub() -> dict:
    """verify_domain stub for offline testing — pretends DNS lookup failed."""
    return {
        "domain": "example.com",
        "resolves": False,
        "has_https": False,
        "name_matches_domain": False,
        "is_recent_domain": False,
        "whois_age_days": 0,
        "flags": ["domain_does_not_resolve"],
    }


# ===========================================================================
# Module-level helpers
# ===========================================================================

class TestNormalizeForCompare:
    def test_lowercases(self):
        assert _normalize_for_compare("Google") == "google"

    def test_strips_spaces(self):
        assert _normalize_for_compare("J P Morgan") == "jpmorgan"

    def test_strips_punctuation(self):
        assert _normalize_for_compare("AT&T") == "att"

    def test_strips_unicode_accents(self):
        assert _normalize_for_compare("Résumé") == "resume"

    def test_empty_string(self):
        assert _normalize_for_compare("") == ""


class TestDomainMatchesCompany:
    def test_exact_match(self):
        assert _domain_matches_company("stripe", "stripe.com") is True

    def test_partial_match(self):
        assert _domain_matches_company("Google", "google.com") is True

    def test_company_in_domain_base(self):
        # amazon.com -> domain_base = "amazon", which matches "amazon"
        assert _domain_matches_company("Amazon", "amazon.com") is True

    def test_mismatch(self):
        assert _domain_matches_company("Apple", "scamsite.xyz") is False

    def test_empty_company(self):
        assert _domain_matches_company("", "example.com") is False

    def test_empty_domain(self):
        assert _domain_matches_company("Stripe", "") is False


class TestIsMisspelledBrand:
    def test_detects_typo(self):
        # "Googgle" is close to "google"
        result = _is_misspelled_brand("Googgle")
        assert result == "google"

    def test_exact_match_not_flagged(self):
        # Exact match should not be considered a misspelling
        assert _is_misspelled_brand("google") is None

    def test_unrelated_name_not_flagged(self):
        assert _is_misspelled_brand("QuantumLeapAnalytics") is None

    def test_amazon_typo(self):
        result = _is_misspelled_brand("Amazzon")
        assert result == "amazon"


# ===========================================================================
# verify_domain
# ===========================================================================

class TestVerifyDomain:
    def setup_method(self):
        self.verifier = _verifier()

    def test_no_url_returns_no_domain_flag(self):
        result = self.verifier.verify_domain("Acme Corp", "")
        assert "no_domain_provided" in result["flags"]
        assert result["domain"] == ""

    def test_dns_resolution_failure(self):
        with patch("socket.getaddrinfo", side_effect=OSError("Name not resolved")):
            result = self.verifier.verify_domain("Scam Co", "https://definitely-not-real-scam99.xyz")
        assert result["resolves"] is False
        assert "domain_does_not_resolve" in result["flags"]

    def test_domain_extracted_correctly(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, None)]):
            with patch("ssl.create_default_context"):
                with patch("socket.create_connection", side_effect=Exception("no https")):
                    with patch.object(self.verifier, "_get_whois_age", return_value=0):
                        result = self.verifier.verify_domain("Stripe", "https://stripe.com/careers")
        assert result["domain"] == "stripe.com"

    def test_name_matches_domain(self):
        with patch("socket.getaddrinfo", return_value=[()]):
            with patch("socket.create_connection", side_effect=Exception("ssl fail")):
                with patch.object(self.verifier, "_get_whois_age", return_value=1000):
                    result = self.verifier.verify_domain("Stripe", "https://stripe.com")
        assert result["name_matches_domain"] is True

    def test_name_mismatch_flagged(self):
        with patch("socket.getaddrinfo", return_value=[()]):
            with patch("socket.create_connection", side_effect=Exception("ssl fail")):
                with patch.object(self.verifier, "_get_whois_age", return_value=1000):
                    result = self.verifier.verify_domain("Apple Inc", "https://totallydifferent.com")
        if result["resolves"]:
            assert "domain_name_mismatch" in result["flags"]

    def test_recently_registered_domain_flagged(self):
        with patch("socket.getaddrinfo", return_value=[()]):
            with patch("socket.create_connection", side_effect=Exception("ssl fail")):
                with patch.object(self.verifier, "_get_whois_age", return_value=30):
                    result = self.verifier.verify_domain("NewBiz", "https://newbiz.com")
        assert result["is_recent_domain"] is True
        assert "recently_registered_domain" in result["flags"]

    def test_result_has_all_keys(self):
        result = self.verifier.verify_domain("Corp", "")
        for key in ("domain", "resolves", "has_https", "name_matches_domain",
                    "is_recent_domain", "whois_age_days", "flags"):
            assert key in result


# ===========================================================================
# check_company_exists
# ===========================================================================

class TestCheckCompanyExists:
    def setup_method(self):
        self.verifier = _verifier()

    def test_known_company_recognized(self):
        result = self.verifier.check_company_exists("google")
        assert result["is_known"] is True
        assert result["confidence"] >= 0.90

    def test_known_company_case_insensitive(self):
        result = self.verifier.check_company_exists("Google")
        assert result["is_known"] is True

    def test_known_company_with_suffix(self):
        result = self.verifier.check_company_exists("Google LLC")
        assert result["is_known"] is True

    def test_empty_company_name(self):
        result = self.verifier.check_company_exists("")
        assert "empty_company_name" in result["flags"]
        assert result["confidence"] == 0.0

    def test_whitespace_only_name(self):
        result = self.verifier.check_company_exists("   ")
        assert "empty_company_name" in result["flags"]

    def test_misspelled_brand_detected(self):
        # "Googgle" normalizes to "googgle" - distance 1 from "google" (threshold=1)
        with patch.object(self.verifier, "_load_scam_company_names", return_value=[]):
            result = self.verifier.check_company_exists("Googgle")
        assert result["misspelled_brand"] == "google"
        assert "misspelled_brand_name" in result["flags"]

    def test_generic_buzzword_name_flagged(self):
        with patch.object(self.verifier, "_load_scam_company_names", return_value=[]):
            result = self.verifier.check_company_exists("Alpha Solutions")
        assert "generic_buzzword_name" in result["flags"]
        assert result["confidence"] < 0.50

    def test_all_caps_name_flagged(self):
        with patch.object(self.verifier, "_load_scam_company_names", return_value=[]):
            result = self.verifier.check_company_exists("XYZCORPINC")
        assert "all_caps_name" in result["flags"]

    def test_result_keys_present(self):
        result = self.verifier.check_company_exists("Stripe")
        for key in ("is_known", "confidence", "flags", "misspelled_brand", "matched_known_name"):
            assert key in result


# ===========================================================================
# verify_linkedin_presence
# ===========================================================================

class TestVerifyLinkedinPresence:
    def setup_method(self):
        self.verifier = _verifier()

    def test_valid_linkedin_url(self):
        result = self.verifier.verify_linkedin_presence("https://linkedin.com/company/stripe")
        assert result["is_valid_format"] is True
        assert result["slug"] == "stripe"
        assert result["slug_looks_legitimate"] is True

    def test_invalid_url_format(self):
        result = self.verifier.verify_linkedin_presence("https://notlinkedin.com/company/test")
        assert result["is_valid_format"] is False
        assert "invalid_linkedin_url_format" in result["flags"]

    def test_empty_url(self):
        result = self.verifier.verify_linkedin_presence("")
        assert "no_linkedin_url" in result["flags"]

    def test_autogenerated_slug_flagged(self):
        result = self.verifier.verify_linkedin_presence(
            "https://linkedin.com/company/company-1234567"
        )
        assert "auto_generated_linkedin_slug" in result["flags"]
        assert result["slug_looks_legitimate"] is False

    def test_slug_too_short(self):
        result = self.verifier.verify_linkedin_presence(
            "https://linkedin.com/company/ab"
        )
        assert "linkedin_slug_too_short" in result["flags"]

    def test_result_keys_present(self):
        result = self.verifier.verify_linkedin_presence("https://linkedin.com/company/test-co")
        for key in ("is_valid_format", "slug", "slug_looks_legitimate", "flags"):
            assert key in result


# ===========================================================================
# check_address_legitimacy
# ===========================================================================

class TestCheckAddressLegitimacy:
    def setup_method(self):
        self.verifier = _verifier()

    def test_clean_address_no_flags(self):
        result = self.verifier.check_address_legitimacy("123 Main St, San Francisco, CA 94101")
        assert not result["is_virtual_office"]
        assert not result["is_po_box"]
        assert not result["is_generic_remote"]

    def test_virtual_office_detected(self):
        result = self.verifier.check_address_legitimacy("Regus, 500 7th Ave Suite 2000, New York")
        assert result["is_virtual_office"] is True
        assert "virtual_office_address" in result["flags"]

    def test_wework_detected(self):
        result = self.verifier.check_address_legitimacy("WeWork 225 Bush St, San Francisco CA")
        assert result["is_virtual_office"] is True

    def test_po_box_detected(self):
        result = self.verifier.check_address_legitimacy("P.O. Box 1234, Austin TX 78701")
        assert result["is_po_box"] is True
        assert "po_box_only" in result["flags"]

    def test_po_box_variant(self):
        result = self.verifier.check_address_legitimacy("PO Box 99")
        assert result["is_po_box"] is True

    def test_generic_remote_detected(self):
        result = self.verifier.check_address_legitimacy("Remote")
        assert result["is_generic_remote"] is True
        assert "generic_remote_no_company_location" in result["flags"]

    def test_work_from_home_detected(self):
        result = self.verifier.check_address_legitimacy("Work From Home")
        assert result["is_generic_remote"] is True

    def test_empty_location(self):
        result = self.verifier.check_address_legitimacy("")
        assert "no_location_provided" in result["flags"]

    def test_result_keys_present(self):
        result = self.verifier.check_address_legitimacy("123 Main St, NY")
        for key in ("is_virtual_office", "is_po_box", "is_generic_remote", "flags"):
            assert key in result


# ===========================================================================
# full_verification
# ===========================================================================

class TestFullVerification:
    def setup_method(self):
        self.verifier = _verifier()

    def test_returns_company_profile(self):
        job = _job(company="Google")
        with patch.object(self.verifier, "verify_domain", return_value={
            "domain": "google.com", "resolves": True, "has_https": True,
            "name_matches_domain": True, "is_recent_domain": False,
            "whois_age_days": 10000, "flags": []
        }):
            profile = self.verifier.full_verification(job)
        assert isinstance(profile, CompanyProfile)

    def test_known_company_is_verified(self):
        job = _job(company="Stripe")
        with patch.object(self.verifier, "verify_domain", return_value={
            "domain": "stripe.com", "resolves": True, "has_https": True,
            "name_matches_domain": True, "is_recent_domain": False,
            "whois_age_days": 5000, "flags": []
        }):
            profile = self.verifier.full_verification(job)
        assert profile.is_verified is True
        assert profile.verification_source == "known_companies_list"

    def test_empty_company_returns_unverified(self):
        job = _job(company="")
        profile = self.verifier.full_verification(job)
        assert isinstance(profile, CompanyProfile)
        assert profile.is_verified is False


# ===========================================================================
# extract_verification_signals
# ===========================================================================

class TestExtractVerificationSignals:
    def setup_method(self):
        self.verifier = _verifier()

    def test_returns_list(self):
        job = _job(company="Google")
        with patch.object(self.verifier, "verify_domain", return_value=_offline_domain_stub()):
            signals = self.verifier.extract_verification_signals(job)
        assert isinstance(signals, list)

    def test_empty_company_returns_empty(self):
        job = _job(company="")
        signals = self.verifier.extract_verification_signals(job)
        assert signals == []

    def test_misspelled_brand_triggers_signal(self):
        # "Googgle" (bare) normalizes to "googgle" - distance 1 from "google"
        job = _job(company="Googgle")
        with patch.object(self.verifier, "verify_domain", return_value=_offline_domain_stub()):
            signals = self.verifier.extract_verification_signals(job)
        names = [s.name for s in signals]
        assert "suspicious_company_name" in names

    def test_virtual_office_triggers_signal(self):
        job = _job(company="Acme Corp", location="Regus, 123 Main St Suite 400")
        with patch.object(self.verifier, "verify_domain", return_value=_offline_domain_stub()):
            signals = self.verifier.extract_verification_signals(job)
        names = [s.name for s in signals]
        assert "virtual_office_address" in names

    def test_verified_company_adds_positive_signal(self):
        job = _job(company="Amazon")
        with patch.object(self.verifier, "verify_domain", return_value=_offline_domain_stub()):
            signals = self.verifier.extract_verification_signals(job)
        names = [s.name for s in signals]
        assert "company_verified" in names
