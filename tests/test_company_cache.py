"""Tests for company validation caching (validator.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from sentinel.models import CompanyProfile
from sentinel.validator import (
    CACHE_TTL_DAYS,
    _cached_to_profile,
    _is_cache_fresh,
    _now_iso,
    validate_company,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _iso(days_ago: int = 0) -> str:
    """Return an ISO timestamp offset by *days_ago* from now."""
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return ts.isoformat()


def _make_cached_row(name: str = "TestCo", days_ago: int = 1) -> dict:
    return {
        "name": name,
        "domain": "testco.com",
        "employee_count": 500,
        "is_verified": 1,
        "linkedin_url": "https://linkedin.com/company/testco",
        "glassdoor_rating": 4.2,
        "whois_age_days": 3650,
        "last_checked": _iso(days_ago),
    }


# ---------------------------------------------------------------------------
# _is_cache_fresh
# ---------------------------------------------------------------------------

def test_is_cache_fresh_true():
    """A timestamp from yesterday is within TTL."""
    assert _is_cache_fresh(_iso(days_ago=1)) is True


def test_is_cache_fresh_stale():
    """A timestamp older than CACHE_TTL_DAYS is stale."""
    assert _is_cache_fresh(_iso(days_ago=CACHE_TTL_DAYS + 1)) is False


def test_is_cache_fresh_empty():
    """An empty string is never fresh."""
    assert _is_cache_fresh("") is False
    assert _is_cache_fresh(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _cached_to_profile
# ---------------------------------------------------------------------------

def test_cached_to_profile_maps_fields():
    row = _make_cached_row()
    profile = _cached_to_profile(row)
    assert isinstance(profile, CompanyProfile)
    assert profile.name == "TestCo"
    assert profile.domain == "testco.com"
    assert profile.employee_count == 500
    assert profile.is_verified is True
    assert profile.whois_age_days == 3650
    assert profile.verification_source == "cache"


# ---------------------------------------------------------------------------
# validate_company — cache behaviour
# ---------------------------------------------------------------------------

def test_cache_hit_returns_cached(tmp_path):
    """When a fresh cache entry exists, validate_company returns it without
    calling any external checks."""
    row = _make_cached_row(days_ago=1)

    mock_db = MagicMock()
    mock_db.get_company.return_value = row

    with patch("sentinel.db.SentinelDB", return_value=mock_db):
        profile = validate_company("TestCo")

    assert profile.name == "TestCo"
    assert profile.verification_source == "cache"
    # No save should have been called (we used the cache)
    mock_db.save_company.assert_not_called()


def test_cache_miss_validates_fresh(tmp_path):
    """When there is no cache entry, validate_company performs fresh validation
    and saves the result."""
    mock_db = MagicMock()
    mock_db.get_company.return_value = None  # cache miss

    with patch("sentinel.db.SentinelDB", return_value=mock_db):
        profile = validate_company("google")

    assert isinstance(profile, CompanyProfile)
    assert profile.name == "google"
    # Should have persisted the fresh result
    mock_db.save_company.assert_called_once()


def test_cache_ttl_expiry(tmp_path):
    """An expired cache entry triggers fresh validation."""
    stale_row = _make_cached_row(days_ago=CACHE_TTL_DAYS + 2)

    mock_db = MagicMock()
    mock_db.get_company.return_value = stale_row

    with patch("sentinel.db.SentinelDB", return_value=mock_db):
        profile = validate_company("TestCo")

    # Fresh validation ran → save_company should have been called
    mock_db.save_company.assert_called_once()
    # verification_source is NOT "cache" because a fresh run happened
    assert profile.verification_source != "cache"


def test_refresh_bypasses_cache(tmp_path):
    """refresh=True skips cache lookup entirely."""
    row = _make_cached_row(days_ago=0)  # perfectly fresh

    mock_db = MagicMock()
    mock_db.get_company.return_value = row

    with patch("sentinel.db.SentinelDB", return_value=mock_db):
        profile = validate_company("TestCo", refresh=True)

    # Cache was never consulted for a hit
    mock_db.get_company.assert_not_called()
    # Fresh validation ran and result was saved
    mock_db.save_company.assert_called_once()
    assert profile.verification_source != "cache"


def test_db_failure_does_not_break_validation():
    """If the DB raises, validate_company still returns a valid profile."""
    with patch("sentinel.db.SentinelDB", side_effect=Exception("db down")):
        profile = validate_company("google")

    assert isinstance(profile, CompanyProfile)
    assert profile.name == "google"
