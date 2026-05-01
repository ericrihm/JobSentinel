"""Company registry for ATS-based job scraping.

Maps companies to their ATS platform and board token/slug, enabling
the hiring.cafe-style approach of scraping directly from company
career pages rather than from job board aggregators.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent / "company_registry.json"

_registry_cache: dict | None = None


def load_registry() -> dict[str, list[str]]:
    """Load the company registry from disk, caching the result."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache

    try:
        data = json.loads(_REGISTRY_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Company registry not found or invalid at %s", _REGISTRY_PATH)
        return {}

    result: dict[str, list[str]] = {}
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            result[key] = value

    _registry_cache = result
    return result


def get_companies(ats: str) -> list[str]:
    """Return the list of company slugs for a given ATS platform."""
    return load_registry().get(ats, [])


def all_ats_platforms() -> list[str]:
    """Return all ATS platform names in the registry."""
    return list(load_registry().keys())
