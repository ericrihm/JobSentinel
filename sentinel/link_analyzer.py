"""URL and link analysis module for detecting scam indicators in job postings."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import socket
import subprocess
import urllib.parse
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.models import JobPosting, ScamSignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TLDs associated with high abuse rates (source: Spamhaus / SURBL research)
_HIGH_RISK_TLDS: frozenset[str] = frozenset({
    ".xyz", ".top", ".club", ".work", ".click", ".link",
    ".info", ".biz", ".online", ".site", ".live",
})

# Free/personal email and messaging domains — suspicious for business use
_FREE_EMAIL_DOMAINS: frozenset[str] = frozenset({
    "gmail.com", "outlook.com", "hotmail.com", "yahoo.com",
    "protonmail.com", "proton.me", "mail.com", "ymail.com",
    "icloud.com", "me.com", "aol.com", "live.com", "msn.com",
    "googlemail.com",
})

# URL shorteners and redirect services
_SHORTENER_DOMAINS: frozenset[str] = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "goo.gl",
    "buff.ly", "short.io", "rebrand.ly", "cutt.ly", "tiny.cc",
    "is.gd", "v.gd", "shrtco.de", "shorturl.at", "clck.ru",
    "lnkd.in",
})

# Well-known brands to check for typosquatting
_KNOWN_BRANDS: list[str] = [
    "google", "linkedin", "amazon", "microsoft", "apple",
    "facebook", "meta", "netflix", "twitter", "instagram",
    "paypal", "ebay", "walmart", "indeed", "glassdoor",
    "monster", "ziprecruiter", "workday", "salesforce",
]

# Local blocklist — patterns seen in scam campaigns
_LOCAL_BLOCKLIST: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"jobs?[-.]?(apply|offer|hire|work)\.(?:xyz|top|click|live|site|online)",
        r"work[-.]?from[-.]?home\d*\.",
        r"earn[-.]?\$\d+",
        r"easy[-.]?money\d*\.",
        r"get[-.]?hired[-.]?(now|today|fast)",
        r"career[-.]?opportunity\d+\.",
        r"(apply|recruit)[-_]?[a-z]{6,12}\.(?:xyz|top|click|biz|info)",
    ]
]

# Regex for bare domain detection (used in auto-generated domain heuristic)
_RANDOM_SUBDOMAIN = re.compile(r"^[a-z0-9]{8,}$")
_EXCESSIVE_HYPHENS = re.compile(r"-{2,}|(?:[a-z0-9]-){4,}")

# ---------------------------------------------------------------------------
# URL extraction regex
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"""
    (?:
        # Full URLs with scheme
        https?://
        (?:[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+)
    |
        # Shortened URLs: bit.ly/xxx, t.co/xxx etc.
        (?:bit\.ly|tinyurl\.com|t\.co|ow\.ly|goo\.gl|buff\.ly|
           short\.io|rebrand\.ly|cutt\.ly|tiny\.cc|is\.gd|v\.gd|
           shrtco\.de|lnkd\.in)/[a-zA-Z0-9_\-/?=&%+.#]+
    |
        # Bare domains with common TLDs (not preceded by @)
        (?<![/@])
        (?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+
        (?:com|net|org|io|co|ai|dev|app|jobs|work|xyz|top|club|
           click|link|info|biz|online|site|live|me|us|uk|ca|au|de)
        (?:/[a-zA-Z0-9_\-/?=&%+.#]*)?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Levenshtein distance (same impl as signals.py for consistency)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def _parse_domain(url: str) -> str:
    """Return the netloc/host portion of a URL, lowercased, without port."""
    # Ensure the URL has a scheme so urlparse works correctly
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        return host.lower().lstrip("www.")
    except Exception:
        return ""


def _get_tld(domain: str) -> str:
    """Return the last label of a domain including the dot (e.g., '.com')."""
    parts = domain.rsplit(".", 1)
    return "." + parts[-1] if len(parts) == 2 else ""


# ---------------------------------------------------------------------------
# LinkAnalyzer class
# ---------------------------------------------------------------------------

class LinkAnalyzer:
    """Analyze URLs found in job postings for scam indicators."""

    # ------------------------------------------------------------------
    # 1. URL extraction
    # ------------------------------------------------------------------

    def extract_urls(self, text: str) -> list[str]:
        """Extract all URLs from job description text.

        Handles http/https full URLs, known URL shorteners, and bare domains.
        Returns deduplicated list preserving first-seen order.
        """
        matches = _URL_RE.findall(text)
        seen: set[str] = set()
        results: list[str] = []
        for url in matches:
            url = url.strip().rstrip(".,;:!?\"')")
            if url and url not in seen:
                seen.add(url)
                results.append(url)
        return results

    # ------------------------------------------------------------------
    # 2. Domain analysis
    # ------------------------------------------------------------------

    def analyze_domain(self, url: str) -> dict:
        """Analyze a URL's domain for risk indicators.

        Returns a dict with keys:
            url, domain, tld, risk_score (0.0–1.0), flags (list[str]),
            is_free_email_domain, is_shortener, is_high_risk_tld,
            looks_autogenerated, brand_impersonation (str|None)
        """
        domain = _parse_domain(url)
        result: dict = {
            "url": url,
            "domain": domain,
            "tld": "",
            "risk_score": 0.0,
            "flags": [],
            "is_free_email_domain": False,
            "is_shortener": False,
            "is_high_risk_tld": False,
            "looks_autogenerated": False,
            "brand_impersonation": None,
        }

        if not domain:
            result["flags"].append("unparseable_domain")
            result["risk_score"] = 0.3
            return result

        tld = _get_tld(domain)
        result["tld"] = tld

        # --- Free email / messaging domains (used as business contact) ---
        if domain in _FREE_EMAIL_DOMAINS:
            result["is_free_email_domain"] = True
            result["flags"].append("free_email_domain")
            result["risk_score"] += 0.35

        # --- URL shortener ---
        if domain in _SHORTENER_DOMAINS:
            result["is_shortener"] = True
            result["flags"].append("shortened_url")
            result["risk_score"] += 0.25

        # --- High-risk TLD ---
        if tld in _HIGH_RISK_TLDS:
            result["is_high_risk_tld"] = True
            result["flags"].append("high_risk_tld")
            result["risk_score"] += 0.35

        # --- Auto-generated domain heuristics ---
        sld = domain.rsplit(".", 1)[0] if "." in domain else domain
        # Strip one level of subdomain to get the registered domain label
        registered_label = sld.split(".")[-1] if "." in sld else sld

        autogen_score = 0
        if _RANDOM_SUBDOMAIN.match(registered_label) and len(registered_label) >= 8:
            autogen_score += 1
        if _EXCESSIVE_HYPHENS.search(registered_label):
            autogen_score += 1
        # Very long subdomain chain (e.g., a.b.c.d.example.com)
        if domain.count(".") >= 4:
            autogen_score += 1
        # Lots of digits interspersed
        digit_ratio = sum(c.isdigit() for c in registered_label) / max(len(registered_label), 1)
        if digit_ratio > 0.4:
            autogen_score += 1

        if autogen_score >= 2:
            result["looks_autogenerated"] = True
            result["flags"].append("autogenerated_domain")
            result["risk_score"] += 0.30

        # --- Brand impersonation via Levenshtein ---
        # Check the registered domain label against known brands
        best_brand: str | None = None
        best_dist = 999
        for brand in _KNOWN_BRANDS:
            # Skip exact matches — the real brand isn't impersonation
            if registered_label == brand:
                best_brand = None
                best_dist = 999
                break
            dist = _levenshtein(registered_label, brand)
            # Distance of 1–2 with a non-matching TLD or different domain is suspicious
            if 0 < dist <= 2 and dist < best_dist:
                best_dist = dist
                best_brand = brand

        if best_brand is not None:
            result["brand_impersonation"] = best_brand
            result["flags"].append(f"brand_impersonation:{best_brand}")
            result["risk_score"] += 0.50

        result["risk_score"] = min(result["risk_score"], 1.0)
        return result

    # ------------------------------------------------------------------
    # 3. URL reputation check
    # ------------------------------------------------------------------

    def check_url_reputation(self, url: str) -> dict:
        """Check a URL against threat intelligence sources.

        Checks (in order of priority):
        1. Google Safe Browsing API (if GOOGLE_SAFE_BROWSING_API_KEY is set)
        2. PhishTank API (if PHISHTANK_API_KEY is set)
        3. Local blocklist patterns (always active)

        Returns a dict with: url, is_malicious (bool), threat_type (str|None),
        source (str), checked (bool)
        """
        result: dict = {
            "url": url,
            "is_malicious": False,
            "threat_type": None,
            "source": "none",
            "checked": False,
        }

        # --- Google Safe Browsing ---
        gsb_key = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "")
        if gsb_key:
            gsb_result = self._check_google_safe_browsing(url, gsb_key)
            if gsb_result is not None:
                result.update(gsb_result)
                result["checked"] = True
                if result["is_malicious"]:
                    return result

        # --- PhishTank ---
        pt_key = os.environ.get("PHISHTANK_API_KEY", "")
        if pt_key:
            pt_result = self._check_phishtank(url, pt_key)
            if pt_result is not None:
                result.update(pt_result)
                result["checked"] = True
                if result["is_malicious"]:
                    return result

        # --- Local blocklist (always runs) ---
        local_result = self._check_local_blocklist(url)
        if local_result["is_malicious"]:
            result.update(local_result)
            result["checked"] = True

        return result

    def _check_google_safe_browsing(self, url: str, api_key: str) -> dict | None:
        """Query Google Safe Browsing API v4. Returns None on failure."""
        try:
            import urllib.request

            endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
            payload = json.dumps({
                "client": {"clientId": "sentinel-job-analyzer", "clientVersion": "1.0"},
                "threatInfo": {
                    "threatTypes": [
                        "MALWARE", "SOCIAL_ENGINEERING",
                        "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION",
                    ],
                    "platformTypes": ["ANY_PLATFORM"],
                    "threatEntryTypes": ["URL"],
                    "threatEntries": [{"url": url}],
                },
            }).encode()

            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            matches = data.get("matches", [])
            if matches:
                threat_type = matches[0].get("threatType", "UNKNOWN")
                return {
                    "is_malicious": True,
                    "threat_type": threat_type,
                    "source": "google_safe_browsing",
                }
            return {"is_malicious": False, "threat_type": None, "source": "google_safe_browsing"}

        except Exception as exc:
            logger.debug("Google Safe Browsing check failed for %s: %s", url, exc)
            return None

    def _check_phishtank(self, url: str, api_key: str) -> dict | None:
        """Query PhishTank API. Returns None on failure."""
        try:
            import urllib.request

            data = urllib.parse.urlencode({
                "url": url,
                "format": "json",
                "app_key": api_key,
            }).encode()

            req = urllib.request.Request(
                "https://checkurl.phishtank.com/checkurl/",
                data=data,
                headers={"User-Agent": "sentinel-job-analyzer/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())

            results = body.get("results", {})
            in_database = results.get("in_database", False)
            valid = results.get("valid", False)

            if in_database and valid:
                return {
                    "is_malicious": True,
                    "threat_type": "phishing",
                    "source": "phishtank",
                }
            return {"is_malicious": False, "threat_type": None, "source": "phishtank"}

        except Exception as exc:
            logger.debug("PhishTank check failed for %s: %s", url, exc)
            return None

    def _check_local_blocklist(self, url: str) -> dict:
        """Check URL against local regex blocklist patterns."""
        for pattern in _LOCAL_BLOCKLIST:
            if pattern.search(url):
                return {
                    "is_malicious": True,
                    "threat_type": "local_blocklist",
                    "source": "local_blocklist",
                }
        return {"is_malicious": False, "threat_type": None, "source": "local_blocklist"}

    # ------------------------------------------------------------------
    # 4. Redirect chain analysis
    # ------------------------------------------------------------------

    def analyze_redirect_chain(self, url: str) -> dict:
        """Follow redirect chain using HEAD requests.

        Uses httpx if available, falls back to urllib. Max 10 hops, 5s timeout.

        Returns a dict with: url, final_url, hop_count, domains_visited (list),
        is_suspicious (bool), flags (list[str]), error (str|None)
        """
        result: dict = {
            "url": url,
            "final_url": url,
            "hop_count": 0,
            "domains_visited": [],
            "is_suspicious": False,
            "flags": [],
            "error": None,
        }

        try:
            import httpx
            chain = self._follow_redirects_httpx(url, max_hops=10, timeout=5.0)
        except ImportError:
            logger.debug("httpx not available; falling back to urllib for redirect analysis")
            try:
                chain = self._follow_redirects_urllib(url, max_hops=10, timeout=5)
            except Exception as exc:
                result["error"] = str(exc)
                logger.debug("Redirect chain analysis failed for %s: %s", url, exc)
                return result
        except Exception as exc:
            result["error"] = str(exc)
            logger.debug("Redirect chain analysis failed for %s: %s", url, exc)
            return result

        if not chain:
            return result

        result["final_url"] = chain[-1]
        result["hop_count"] = len(chain) - 1
        domains_visited = [_parse_domain(u) for u in chain]
        result["domains_visited"] = [d for d in domains_visited if d]

        start_domain = domains_visited[0] if domains_visited else ""
        end_domain = domains_visited[-1] if domains_visited else ""

        # Flag: excessive redirects
        if result["hop_count"] > 3:
            result["is_suspicious"] = True
            result["flags"].append("excessive_redirects")

        # Flag: cross-domain redirect
        if start_domain and end_domain and start_domain != end_domain:
            result["flags"].append("cross_domain_redirect")
            result["is_suspicious"] = True

        # Flag: landed on a high-risk TLD
        final_tld = _get_tld(end_domain)
        if final_tld in _HIGH_RISK_TLDS:
            result["flags"].append("redirect_to_high_risk_tld")
            result["is_suspicious"] = True

        return result

    def _follow_redirects_httpx(
        self, url: str, max_hops: int = 10, timeout: float = 5.0
    ) -> list[str]:
        """Follow redirects with httpx (HEAD only). Returns ordered URL chain."""
        import httpx

        chain = [url]
        current = url
        for _ in range(max_hops):
            try:
                resp = httpx.head(
                    current,
                    follow_redirects=False,
                    timeout=timeout,
                    headers={"User-Agent": "sentinel-link-analyzer/1.0"},
                )
            except Exception:
                break

            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("location", "")
                if not location or location == current:
                    break
                # Handle relative redirects
                if not location.startswith("http"):
                    location = urllib.parse.urljoin(current, location)
                chain.append(location)
                current = location
            else:
                break

        return chain

    def _follow_redirects_urllib(
        self, url: str, max_hops: int = 10, timeout: int = 5
    ) -> list[str]:
        """Follow redirects with urllib (HEAD only). Returns ordered URL chain."""
        import http.client
        import ssl

        chain = [url]
        current = url
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        for _ in range(max_hops):
            parsed = urllib.parse.urlparse(current)
            host = parsed.netloc
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query

            try:
                if parsed.scheme == "https":
                    conn = http.client.HTTPSConnection(host, timeout=timeout, context=ctx)
                else:
                    conn = http.client.HTTPConnection(host, timeout=timeout)

                conn.request("HEAD", path, headers={"User-Agent": "sentinel-link-analyzer/1.0"})
                resp = conn.getresponse()
                conn.close()
            except Exception:
                break

            if resp.status in (301, 302, 303, 307, 308):
                location = resp.getheader("location", "")
                if not location or location == current:
                    break
                if not location.startswith("http"):
                    location = urllib.parse.urljoin(current, location)
                chain.append(location)
                current = location
            else:
                break

        return chain

    # ------------------------------------------------------------------
    # 5. Domain age check
    # ------------------------------------------------------------------

    def check_domain_age(self, domain: str) -> dict:
        """Estimate domain registration age using WHOIS data.

        Tries python-whois library first, then subprocess `whois`, then
        socket-based fallback. Domains < 90 days old are flagged suspicious.

        Returns: domain, age_days (int|None), registered_date (str|None),
        is_new_domain (bool), error (str|None)
        """
        result: dict = {
            "domain": domain,
            "age_days": None,
            "registered_date": None,
            "is_new_domain": False,
            "error": None,
        }

        # Try python-whois library
        registered_date = self._whois_via_library(domain)

        # Fall back to subprocess whois
        if registered_date is None:
            registered_date = self._whois_via_subprocess(domain)

        if registered_date is None:
            result["error"] = "could not determine domain registration date"
            logger.debug("WHOIS lookup failed for %s", domain)
            return result

        result["registered_date"] = registered_date.isoformat()
        age_days = (datetime.now(UTC) - registered_date).days
        result["age_days"] = age_days
        result["is_new_domain"] = age_days < 90

        return result

    def _whois_via_library(self, domain: str) -> datetime | None:
        """Try the `whois` pip package (python-whois)."""
        try:
            import whois as python_whois  # type: ignore
            w = python_whois.whois(domain)
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if creation_date is None:
                return None
            if isinstance(creation_date, datetime):
                if creation_date.tzinfo is None:
                    creation_date = creation_date.replace(tzinfo=UTC)
                return creation_date
        except Exception as exc:
            logger.debug("python-whois failed for %s: %s", domain, exc)
        return None

    def _whois_via_subprocess(self, domain: str) -> datetime | None:
        """Call system `whois` command and parse creation date from output."""
        try:
            result = subprocess.run(
                ["whois", domain],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
            logger.debug("whois subprocess failed for %s: %s", domain, exc)
            return None

        # Common WHOIS date field patterns
        date_patterns = [
            re.compile(
                r"(?:creation date|created|registered on|domain registered|"
                r"registration date|created on)[:\s]+([^\r\n]+)",
                re.IGNORECASE,
            ),
            re.compile(r"crdate[:\s]+([^\r\n]+)", re.IGNORECASE),
        ]
        date_formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%d-%b-%Y",
            "%d/%m/%Y",
            "%Y.%m.%d",
            "%d.%m.%Y",
        ]

        for dp in date_patterns:
            m = dp.search(output)
            if not m:
                continue
            raw = m.group(1).strip()
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(raw[:len(fmt)], fmt).replace(tzinfo=UTC)
                    return dt
                except ValueError:
                    continue

        return None

    # ------------------------------------------------------------------
    # 6. Analyze all URLs in text
    # ------------------------------------------------------------------

    def analyze_all(self, text: str) -> list[dict]:
        """Run all checks on every URL found in text.

        Returns a list of consolidated result dicts, one per unique URL.
        Each dict contains: url, domain_analysis, reputation, redirect_chain
        (redirect_chain only checked for non-shortener, non-free-email URLs
        to avoid unnecessary network calls).
        """
        urls = self.extract_urls(text)
        results = []

        for url in urls:
            domain_analysis = self.analyze_domain(url)
            reputation = self.check_url_reputation(url)

            # Only follow redirects for shorteners or suspicious domains
            # to limit network exposure in a pipeline context
            should_check_redirects = (
                domain_analysis["is_shortener"]
                or domain_analysis["is_high_risk_tld"]
                or domain_analysis["looks_autogenerated"]
            )
            redirect_chain: dict | None = None
            if should_check_redirects:
                redirect_chain = self.analyze_redirect_chain(url)

            # Domain age — only for non-shortener, non-free-email URLs
            domain_age: dict | None = None
            domain = domain_analysis["domain"]
            if domain and not domain_analysis["is_shortener"] and not domain_analysis["is_free_email_domain"]:
                domain_age = self.check_domain_age(domain)

            results.append({
                "url": url,
                "domain_analysis": domain_analysis,
                "reputation": reputation,
                "redirect_chain": redirect_chain,
                "domain_age": domain_age,
            })

        return results


# ---------------------------------------------------------------------------
# Pipeline integration function
# ---------------------------------------------------------------------------

def extract_link_signals(job: "JobPosting") -> list["ScamSignal"]:
    """Run URL/link analysis on a job posting and return ScamSignals.

    Designed to be called from the signal extraction pipeline alongside
    the functions in sentinel.signals.

    Signal names and weights:
        suspicious_redirect_chain  0.72  — >3 hops or cross-domain redirect
        new_domain                 0.65  — domain registered < 90 days ago
        high_risk_tld              0.55  — .xyz, .top, .club, .work, etc.
        brand_impersonation_url    0.85  — typosquatting a known brand
        url_reputation_bad         0.90  — confirmed by Safe Browsing/PhishTank/blocklist
        shortened_url              0.45  — bit.ly, tinyurl, etc.
    """
    from sentinel.models import ScamSignal, SignalCategory

    analyzer = LinkAnalyzer()
    text = f"{job.title} {job.description}".strip()

    if not text:
        return []

    urls = analyzer.extract_urls(text)
    if not urls:
        return []

    signals: list[ScamSignal] = []
    seen_signal_names: set[str] = set()

    def _add_signal(name: str, **kwargs) -> None:
        """Add a signal, deduplicating by name (keep highest weight)."""
        if name not in seen_signal_names:
            seen_signal_names.add(name)
            signals.append(ScamSignal(name=name, **kwargs))

    for url in urls:
        # --- Domain analysis ---
        da = analyzer.analyze_domain(url)

        if da["is_shortener"]:
            _add_signal(
                "shortened_url",
                category=SignalCategory.STRUCTURAL,
                weight=0.45,
                confidence=0.80,
                detail="Job posting contains a URL shortener link — destination is hidden",
                evidence=url,
            )

        if da["is_high_risk_tld"]:
            _add_signal(
                "high_risk_tld",
                category=SignalCategory.WARNING,
                weight=0.55,
                confidence=0.70,
                detail=f"URL uses a high-risk TLD ({da['tld']})",
                evidence=url,
            )

        if da["brand_impersonation"]:
            _add_signal(
                "brand_impersonation_url",
                category=SignalCategory.RED_FLAG,
                weight=0.85,
                confidence=0.78,
                detail=(
                    f"URL domain appears to impersonate '{da['brand_impersonation']}' "
                    f"(domain: {da['domain']})"
                ),
                evidence=url,
            )

        # --- URL reputation ---
        rep = analyzer.check_url_reputation(url)
        if rep["is_malicious"]:
            _add_signal(
                "url_reputation_bad",
                category=SignalCategory.RED_FLAG,
                weight=0.90,
                confidence=0.88,
                detail=(
                    f"URL flagged as malicious by {rep['source']} "
                    f"(threat: {rep['threat_type'] or 'unknown'})"
                ),
                evidence=url,
            )

        # --- Redirect chain (only for shorteners / suspicious domains) ---
        if da["is_shortener"] or da["is_high_risk_tld"]:
            try:
                rc = analyzer.analyze_redirect_chain(url)
                if rc["is_suspicious"] and rc["hop_count"] > 0:
                    _add_signal(
                        "suspicious_redirect_chain",
                        category=SignalCategory.RED_FLAG,
                        weight=0.72,
                        confidence=0.65,
                        detail=(
                            f"URL redirect chain is suspicious: {rc['hop_count']} hop(s), "
                            f"flags: {', '.join(rc['flags'])}"
                        ),
                        evidence=f"{url} -> {rc['final_url']}",
                    )
            except Exception as exc:
                logger.debug("Redirect chain check failed for %s: %s", url, exc)

        # --- Domain age ---
        domain = da["domain"]
        if domain and not da["is_shortener"] and not da["is_free_email_domain"]:
            try:
                age = analyzer.check_domain_age(domain)
                if age["is_new_domain"] and age["age_days"] is not None:
                    _add_signal(
                        "new_domain",
                        category=SignalCategory.WARNING,
                        weight=0.65,
                        confidence=0.60,
                        detail=(
                            f"Domain '{domain}' was registered only {age['age_days']} days ago "
                            f"(< 90 days threshold)"
                        ),
                        evidence=f"{domain} registered {age['registered_date']}",
                    )
            except Exception as exc:
                logger.debug("Domain age check failed for %s: %s", domain, exc)

    return signals
