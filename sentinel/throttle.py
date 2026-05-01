import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclass
class DomainBudget:
    """Per-domain request budget and timing."""
    requests_per_minute: float = 10.0
    min_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_factor: float = 2.0
    last_request_time: float = 0.0
    consecutive_errors: int = 0
    total_requests: int = 0
    total_errors: int = 0

class SmartThrottler:
    """Per-domain rate limiting with exponential backoff and circuit breaking."""

    # Default budgets per known domain
    DEFAULT_BUDGETS = {
        "remoteok.com": DomainBudget(requests_per_minute=30, min_delay_seconds=2.0),
        "api.adzuna.com": DomainBudget(requests_per_minute=20, min_delay_seconds=3.0),
        "www.themuse.com": DomainBudget(requests_per_minute=8, min_delay_seconds=7.5),  # 500/hr
        "data.usajobs.gov": DomainBudget(requests_per_minute=10, min_delay_seconds=6.0),
        "remotive.com": DomainBudget(requests_per_minute=20, min_delay_seconds=3.0),
        "www.linkedin.com": DomainBudget(requests_per_minute=5, min_delay_seconds=12.0),
    }

    CIRCUIT_BREAK_THRESHOLD = 5  # consecutive errors before circuit break
    CIRCUIT_BREAK_DURATION = 300  # seconds to wait after circuit break

    def __init__(self):
        self._budgets: dict[str, DomainBudget] = {}
        self._circuit_broken: dict[str, float] = {}  # domain -> broken_at timestamp

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc.lower()

    def _get_budget(self, domain: str) -> DomainBudget:
        """Get or create budget for domain."""
        if domain not in self._budgets:
            # Check defaults, or create conservative default
            if domain in self.DEFAULT_BUDGETS:
                self._budgets[domain] = DomainBudget(**{
                    k: v for k, v in self.DEFAULT_BUDGETS[domain].__dict__.items()
                })
            else:
                self._budgets[domain] = DomainBudget()
        return self._budgets[domain]

    def wait_if_needed(self, url: str) -> bool:
        """Wait the appropriate time before making a request. Returns False if circuit broken."""
        domain = self._get_domain(url)

        # Check circuit breaker
        if domain in self._circuit_broken:
            broken_at = self._circuit_broken[domain]
            if time.monotonic() - broken_at < self.CIRCUIT_BREAK_DURATION:
                logger.warning("Circuit broken for %s, skipping", domain)
                return False
            else:
                del self._circuit_broken[domain]
                logger.info("Circuit breaker reset for %s", domain)

        budget = self._get_budget(domain)

        # Calculate delay with exponential backoff for errors
        delay = budget.min_delay_seconds
        if budget.consecutive_errors > 0:
            delay = min(
                budget.min_delay_seconds * (budget.backoff_factor ** budget.consecutive_errors),
                budget.max_delay_seconds
            )

        # Wait if needed
        elapsed = time.monotonic() - budget.last_request_time
        if elapsed < delay:
            sleep_time = delay - elapsed
            logger.debug("Throttling %s: sleeping %.1fs", domain, sleep_time)
            time.sleep(sleep_time)

        budget.last_request_time = time.monotonic()
        budget.total_requests += 1
        return True

    def record_success(self, url: str):
        """Record a successful request."""
        domain = self._get_domain(url)
        budget = self._get_budget(domain)
        budget.consecutive_errors = 0

    def record_error(self, url: str, status_code: int | None = None):
        """Record a failed request. Triggers backoff and potentially circuit break."""
        domain = self._get_domain(url)
        budget = self._get_budget(domain)
        budget.consecutive_errors += 1
        budget.total_errors += 1

        # Rate limit response — extra cautious
        if status_code == 429:
            budget.consecutive_errors = max(budget.consecutive_errors, 3)
            logger.warning("Rate limited by %s (429), backing off", domain)

        # Circuit break after too many consecutive errors
        if budget.consecutive_errors >= self.CIRCUIT_BREAK_THRESHOLD:
            self._circuit_broken[domain] = time.monotonic()
            logger.warning("Circuit breaker tripped for %s after %d errors",
                         domain, budget.consecutive_errors)

    def get_stats(self) -> dict[str, dict]:
        """Return per-domain statistics."""
        stats = {}
        for domain, budget in self._budgets.items():
            stats[domain] = {
                "total_requests": budget.total_requests,
                "total_errors": budget.total_errors,
                "consecutive_errors": budget.consecutive_errors,
                "circuit_broken": domain in self._circuit_broken,
            }
        return stats
