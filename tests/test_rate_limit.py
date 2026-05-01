"""Tests for in-memory rate limiter and API key authentication."""

import os
import time

import pytest

import sentinel.config
from sentinel.api import RateLimiter


# ===========================================================================
# Unit tests — RateLimiter class
# ===========================================================================


class TestRateLimiterUnit:
    def test_rate_limiter_allows_under_limit(self):
        """Requests below the rpm threshold are all allowed."""
        limiter = RateLimiter(rpm=5)
        for _ in range(5):
            assert limiter.is_allowed("127.0.0.1") is True

    def test_rate_limiter_blocks_over_limit(self):
        """The (rpm+1)-th request within the window is denied."""
        limiter = RateLimiter(rpm=3)
        for _ in range(3):
            limiter.is_allowed("10.0.0.1")
        assert limiter.is_allowed("10.0.0.1") is False

    def test_rate_limiter_resets_after_window(self, monkeypatch):
        """After the 60-second window elapses, the counter resets."""
        limiter = RateLimiter(rpm=2)

        # Exhaust the limit using a fixed fake time
        fake_now = 1000.0

        def fake_monotonic():
            return fake_now

        monkeypatch.setattr(time, "monotonic", fake_monotonic)

        limiter.is_allowed("192.168.1.1")
        limiter.is_allowed("192.168.1.1")
        assert limiter.is_allowed("192.168.1.1") is False

        # Advance time beyond the 60-second window
        fake_now += 61.0
        assert limiter.is_allowed("192.168.1.1") is True

    def test_rate_limiter_independent_per_ip(self):
        """Different IPs have independent counters."""
        limiter = RateLimiter(rpm=1)
        assert limiter.is_allowed("1.1.1.1") is True
        assert limiter.is_allowed("2.2.2.2") is True
        # First IP is now exhausted, second is still fresh
        assert limiter.is_allowed("1.1.1.1") is False
        assert limiter.is_allowed("2.2.2.2") is False


# ===========================================================================
# FastAPI integration tests
# ===========================================================================


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """TestClient with a temp DB and a generous rate limit so tests don't trip it."""
    db_path = str(tmp_path / "rl_test.db")
    mock_config = sentinel.config.SentinelConfig(db_path=db_path, rate_limit_rpm=1000)
    monkeypatch.setattr(sentinel.config, "get_config", lambda: mock_config)
    monkeypatch.setattr(sentinel.config, "_config", None)
    monkeypatch.delenv("SENTINEL_API_KEY", raising=False)

    from sentinel.api import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    yield TestClient(app)


@pytest.fixture
def rate_limited_client(tmp_path, monkeypatch):
    """TestClient with rpm=1 so the second request triggers a 429."""
    db_path = str(tmp_path / "rl_strict.db")
    mock_config = sentinel.config.SentinelConfig(db_path=db_path, rate_limit_rpm=1)
    monkeypatch.setattr(sentinel.config, "get_config", lambda: mock_config)
    monkeypatch.setattr(sentinel.config, "_config", None)
    monkeypatch.delenv("SENTINEL_API_KEY", raising=False)

    from sentinel.api import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    yield TestClient(app)


@pytest.fixture
def keyed_client(tmp_path, monkeypatch):
    """TestClient with SENTINEL_API_KEY set to 'secret-key'."""
    db_path = str(tmp_path / "rl_keyed.db")
    mock_config = sentinel.config.SentinelConfig(db_path=db_path, rate_limit_rpm=1000)
    monkeypatch.setattr(sentinel.config, "get_config", lambda: mock_config)
    monkeypatch.setattr(sentinel.config, "_config", None)
    monkeypatch.setenv("SENTINEL_API_KEY", "secret-key")

    from sentinel.api import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    yield TestClient(app)


class TestApiKeyAuth:
    def test_api_key_not_required_when_unset(self, api_client):
        """When SENTINEL_API_KEY is not set, requests without a key are accepted."""
        resp = api_client.get("/api/health")
        assert resp.status_code == 200

    def test_api_key_required_when_set(self, keyed_client):
        """When SENTINEL_API_KEY is set, requests without X-API-Key get 401."""
        resp = keyed_client.get("/api/health")
        assert resp.status_code == 401

    def test_api_key_wrong_value_returns_401(self, keyed_client):
        """Wrong key value also yields 401."""
        resp = keyed_client.get("/api/health", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_api_key_correct_value_passes(self, keyed_client):
        """Correct key value passes through to the endpoint."""
        resp = keyed_client.get("/api/health", headers={"X-API-Key": "secret-key"})
        assert resp.status_code == 200


class TestApiRateLimit:
    def test_api_returns_429_on_rate_limit(self, rate_limited_client):
        """Second request from the same IP with rpm=1 gets 429."""
        # First request should pass
        r1 = rate_limited_client.get("/api/health")
        assert r1.status_code == 200

        # Second request (same IP in the test client) should be rate limited
        r2 = rate_limited_client.get("/api/health")
        assert r2.status_code == 429

    def test_429_includes_retry_after_header(self, rate_limited_client):
        """The 429 response includes a Retry-After header."""
        rate_limited_client.get("/api/health")
        resp = rate_limited_client.get("/api/health")
        assert resp.status_code == 429
        assert resp.headers.get("retry-after") == "60"
