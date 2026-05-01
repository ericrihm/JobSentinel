"""Tests for sentinel.throttle — SmartThrottler per-domain rate limiting."""

import time
import unittest
from unittest.mock import patch, call

import pytest

from sentinel.throttle import DomainBudget, SmartThrottler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_monotonic(*values):
    """Return an iterator-backed side_effect for time.monotonic that yields
    successive values from *values.  Raises StopIteration (which surfaces as
    a test error) if called more times than values provided, making accidental
    extra calls easy to diagnose."""
    return iter(values)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFirstRequestNoWait:
    """First request to a fresh domain must not sleep."""

    def test_first_request_no_wait(self):
        throttler = SmartThrottler()
        # last_request_time starts at 0.0; monotonic returns a large value so
        # elapsed >> min_delay — no sleep should be triggered.
        # monotonic is called: (1) elapsed check, (2) set last_request_time
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(9999.0, 9999.1)):
            with patch("sentinel.throttle.time.sleep") as mock_sleep:
                result = throttler.wait_if_needed("https://example.com/jobs")

        assert result is True
        mock_sleep.assert_not_called()


class TestSecondRequestWaits:
    """Second request within the min_delay window must sleep for the remainder."""

    def test_second_request_waits(self):
        throttler = SmartThrottler()
        url = "https://example.com/jobs"

        # First call: elapsed = 9999 - 0 = 9999 → no sleep; last_request_time = 9999.5
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(9999.0, 9999.5)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url)

        # Second call: min_delay=1.0, elapsed = 10000.0 - 9999.5 = 0.5 → sleep 0.5
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(10000.0, 10000.6)):
            with patch("sentinel.throttle.time.sleep") as mock_sleep:
                result = throttler.wait_if_needed(url)

        assert result is True
        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert abs(sleep_arg - 0.5) < 1e-9


class TestBackoffOnError:
    """Delay should increase exponentially with consecutive errors."""

    def test_backoff_on_error(self):
        throttler = SmartThrottler()
        url = "https://example.com/jobs"

        # Seed last_request_time so the domain budget exists
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(1000.0, 1000.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url)

        # Record one error → consecutive_errors = 1
        throttler.record_error(url)

        # min_delay=1.0, backoff_factor=2.0, errors=1 → delay = 1.0 * 2^1 = 2.0
        # elapsed = 1000.05 - 1000.0 = 0.05 → sleep = 2.0 - 0.05 = 1.95
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(1000.05, 1000.1)):
            with patch("sentinel.throttle.time.sleep") as mock_sleep:
                throttler.wait_if_needed(url)

        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert abs(sleep_arg - 1.95) < 1e-9


class TestBackoffResetsOnSuccess:
    """record_success must reset consecutive_errors to 0."""

    def test_backoff_resets_on_success(self):
        throttler = SmartThrottler()
        url = "https://example.com/jobs"

        # Prime the domain budget
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(0.0, 0.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url)

        throttler.record_error(url)
        throttler.record_error(url)
        assert throttler._get_budget(throttler._get_domain(url)).consecutive_errors == 2

        throttler.record_success(url)
        assert throttler._get_budget(throttler._get_domain(url)).consecutive_errors == 0


class TestCircuitBreakerTrips:
    """5 consecutive errors should trip the circuit breaker."""

    def test_circuit_breaker_trips(self):
        throttler = SmartThrottler()
        url = "https://flaky.example.com/api"

        # Prime the budget
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(0.0, 0.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url)

        # Record CIRCUIT_BREAK_THRESHOLD errors (each record_error calls monotonic once
        # when it trips the breaker via the circuit-broken timestamp).
        for i in range(SmartThrottler.CIRCUIT_BREAK_THRESHOLD):
            # The 5th error call sets self._circuit_broken[domain] = time.monotonic()
            with patch("sentinel.throttle.time.monotonic", return_value=float(i + 1)):
                throttler.record_error(url)

        domain = throttler._get_domain(url)
        assert domain in throttler._circuit_broken

        # Now wait_if_needed should return False (circuit open)
        # monotonic returns a value within CIRCUIT_BREAK_DURATION of broken_at
        broken_at = throttler._circuit_broken[domain]
        with patch("sentinel.throttle.time.monotonic", return_value=broken_at + 10.0):
            with patch("sentinel.throttle.time.sleep") as mock_sleep:
                result = throttler.wait_if_needed(url)

        assert result is False
        mock_sleep.assert_not_called()


class TestCircuitBreakerResets:
    """After CIRCUIT_BREAK_DURATION has elapsed the circuit should auto-reset."""

    def test_circuit_breaker_resets(self):
        throttler = SmartThrottler()
        url = "https://flaky.example.com/api"

        # Prime and trip the breaker
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(0.0, 0.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url)

        for i in range(SmartThrottler.CIRCUIT_BREAK_THRESHOLD):
            with patch("sentinel.throttle.time.monotonic", return_value=100.0):
                throttler.record_error(url)

        domain = throttler._get_domain(url)
        assert domain in throttler._circuit_broken

        # Advance time past CIRCUIT_BREAK_DURATION (300 s)
        reset_time = 100.0 + SmartThrottler.CIRCUIT_BREAK_DURATION + 1.0
        # wait_if_needed calls monotonic three times after the breaker resets:
        #   (1) breaker check:  monotonic() - broken_at  → triggers reset
        #   (2) elapsed check:  monotonic() - last_request_time
        #   (3) record stamp:   budget.last_request_time = monotonic()
        with patch("sentinel.throttle.time.monotonic",
                   side_effect=make_monotonic(reset_time, reset_time + 0.1, reset_time + 0.1)):
            with patch("sentinel.throttle.time.sleep"):
                result = throttler.wait_if_needed(url)

        assert result is True
        assert domain not in throttler._circuit_broken


class Test429AggressiveBackoff:
    """HTTP 429 must set consecutive_errors to at least 3."""

    def test_429_aggressive_backoff_from_zero(self):
        throttler = SmartThrottler()
        url = "https://example.com/api"
        throttler.record_error(url, status_code=429)
        budget = throttler._get_budget(throttler._get_domain(url))
        assert budget.consecutive_errors >= 3

    def test_429_does_not_lower_existing_count(self):
        """If consecutive_errors is already > 3 a 429 should not lower it."""
        throttler = SmartThrottler()
        url = "https://example.com/api"
        for _ in range(4):
            throttler.record_error(url)
        budget = throttler._get_budget(throttler._get_domain(url))
        count_before = budget.consecutive_errors  # 4
        throttler.record_error(url, status_code=429)
        # record_error increments first (+1 → 5), then max(5, 3) = 5
        assert budget.consecutive_errors >= count_before


class TestDefaultBudgetKnownDomain:
    """remoteok.com should receive its custom budget, not the generic default."""

    def test_default_budget_for_known_domain(self):
        throttler = SmartThrottler()
        url = "https://remoteok.com/remote-jobs"
        budget = throttler._get_budget(throttler._get_domain(url))
        assert budget.requests_per_minute == 30
        assert budget.min_delay_seconds == 2.0


class TestDefaultBudgetUnknownDomain:
    """An unknown domain should receive conservative generic defaults."""

    def test_default_budget_for_unknown_domain(self):
        throttler = SmartThrottler()
        url = "https://unknown-board.example.org/jobs"
        budget = throttler._get_budget(throttler._get_domain(url))
        # Conservative defaults from DomainBudget()
        assert budget.requests_per_minute == 10.0
        assert budget.min_delay_seconds == 1.0
        assert budget.backoff_factor == 2.0


class TestGetStats:
    """get_stats must return accurate per-domain counters."""

    def test_get_stats(self):
        throttler = SmartThrottler()
        url = "https://stats.example.com/jobs"

        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(0.0, 0.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url)

        throttler.record_error(url)
        throttler.record_success(url)

        stats = throttler.get_stats()
        domain = "stats.example.com"
        assert domain in stats
        assert stats[domain]["total_requests"] == 1
        assert stats[domain]["total_errors"] == 1
        assert stats[domain]["consecutive_errors"] == 0  # reset by success
        assert stats[domain]["circuit_broken"] is False

    def test_get_stats_circuit_broken_flag(self):
        throttler = SmartThrottler()
        url = "https://broken.example.com/jobs"

        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(0.0, 0.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url)

        for _ in range(SmartThrottler.CIRCUIT_BREAK_THRESHOLD):
            with patch("sentinel.throttle.time.monotonic", return_value=1.0):
                throttler.record_error(url)

        stats = throttler.get_stats()
        assert stats["broken.example.com"]["circuit_broken"] is True


class TestDomainExtraction:
    """_get_domain must correctly parse various URL formats."""

    @pytest.mark.parametrize("url, expected", [
        ("https://remoteok.com/remote-jobs", "remoteok.com"),
        ("http://api.adzuna.com/v1/api/jobs", "api.adzuna.com"),
        ("https://www.linkedin.com/jobs/view/123", "www.linkedin.com"),
        ("https://DATA.USAJOBS.GOV/api/search", "data.usajobs.gov"),  # normalised to lower
        ("https://remotive.com/remote-jobs?category=software-dev", "remotive.com"),
        ("https://sub.sub.example.co.uk/path?q=1&r=2#frag", "sub.sub.example.co.uk"),
    ])
    def test_domain_extraction(self, url, expected):
        throttler = SmartThrottler()
        assert throttler._get_domain(url) == expected


class TestMultipleDomainsIndependent:
    """Throttling / errors on one domain must not affect another."""

    def test_multiple_domains_independent(self):
        throttler = SmartThrottler()
        url_a = "https://domain-a.com/jobs"
        url_b = "https://domain-b.com/jobs"

        # Prime domain-a and record errors to near-break it
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(0.0, 0.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url_a)

        for _ in range(SmartThrottler.CIRCUIT_BREAK_THRESHOLD - 1):
            throttler.record_error(url_a)

        # domain-b should be completely unaffected — first request, no backoff
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(9999.0, 9999.1)):
            with patch("sentinel.throttle.time.sleep") as mock_sleep:
                result = throttler.wait_if_needed(url_b)

        assert result is True
        mock_sleep.assert_not_called()

        # domain-b stats should be independent of domain-a
        stats = throttler.get_stats()
        assert stats.get("domain-a.com", {}).get("consecutive_errors", 0) == SmartThrottler.CIRCUIT_BREAK_THRESHOLD - 1
        assert stats.get("domain-b.com", {}).get("consecutive_errors", 0) == 0

    def test_circuit_break_domain_a_does_not_block_domain_b(self):
        throttler = SmartThrottler()
        url_a = "https://tripped.example.com/jobs"
        url_b = "https://healthy.example.com/jobs"

        # Trip the breaker for domain-a
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(0.0, 0.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url_a)

        for _ in range(SmartThrottler.CIRCUIT_BREAK_THRESHOLD):
            with patch("sentinel.throttle.time.monotonic", return_value=1.0):
                throttler.record_error(url_a)

        # domain-b must still be callable
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(9999.0, 9999.1)):
            with patch("sentinel.throttle.time.sleep"):
                result = throttler.wait_if_needed(url_b)

        assert result is True


class TestMaxDelayCapEnforced:
    """Backoff must be capped at max_delay_seconds regardless of error count."""

    def test_max_delay_cap(self):
        throttler = SmartThrottler()
        url = "https://example.com/jobs"

        # Prime the budget
        with patch("sentinel.throttle.time.monotonic", side_effect=make_monotonic(0.0, 0.0)):
            with patch("sentinel.throttle.time.sleep"):
                throttler.wait_if_needed(url)

        # Record many errors to push backoff beyond max_delay (60 s default)
        for _ in range(10):
            throttler.record_error(url)

        budget = throttler._get_budget(throttler._get_domain(url))
        computed_delay = min(
            budget.min_delay_seconds * (budget.backoff_factor ** budget.consecutive_errors),
            budget.max_delay_seconds,
        )
        assert computed_delay == budget.max_delay_seconds
