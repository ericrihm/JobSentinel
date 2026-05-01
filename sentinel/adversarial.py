"""Adversarial evasion detection for JobSentinel.

Scammers evade regex-based signal extractors via:
  - Unicode confusables / homoglyphs (Cyrillic а → Latin a)
  - Leetspeak (b1tc0in, guaran+eed)
  - Zero-width characters inserted into keywords
  - Mixed scripts in the same word
  - Deliberate whitespace splits ("pa yment")

This module provides two components:
  1. TextNormalizer — lightweight preprocessing to strip obfuscation before
     signal extraction runs (< 1ms per job, stdlib-only).
  2. EvasionDetector — detects *when* obfuscation was present, surfaces those
     observations as ScamSignals, and monitors per-signal firing-rate decay.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, datetime

from sentinel.models import ScamSignal, SignalCategory

# ---------------------------------------------------------------------------
# Unicode confusable mapping
# ---------------------------------------------------------------------------
# Covers the most common Cyrillic / Greek / Latin lookalikes exploited in scam text.
# Extend as new evasion techniques are observed in the wild.

_CONFUSABLE_MAP: dict[int, str] = {
    # Cyrillic → Latin
    ord("а"): "a",  # U+0430
    ord("е"): "e",  # U+0435
    ord("о"): "o",  # U+043E
    ord("р"): "p",  # U+0440
    ord("с"): "c",  # U+0441
    ord("х"): "x",  # U+0445
    ord("у"): "y",  # U+0443
    ord("і"): "i",  # U+0456 Ukrainian і
    ord("ї"): "i",  # U+0457 Ukrainian ї
    ord("В"): "B",  # U+0412
    ord("Н"): "H",  # U+041D
    ord("К"): "K",  # U+041A
    ord("М"): "M",  # U+041C
    ord("О"): "O",  # U+041E
    ord("Р"): "P",  # U+0420
    ord("С"): "C",  # U+0421
    ord("Т"): "T",  # U+0422
    ord("Х"): "X",  # U+0425
    ord("А"): "A",  # U+0410
    ord("Е"): "E",  # U+0415
    # Greek → Latin
    ord("ο"): "o",  # U+03BF omicron
    ord("Ο"): "O",  # U+039F capital omicron
    ord("α"): "a",  # U+03B1 alpha
    ord("Α"): "A",  # U+0391 capital alpha
    ord("ν"): "v",  # U+03BD nu (visually close to v)
    ord("κ"): "k",  # U+03BA kappa
    # Common additional homoglyphs
    ord("Ι"): "I",  # U+0399 Greek capital iota
    ord("ι"): "i",  # U+03B9 Greek iota
    ord("ρ"): "p",  # U+03C1 Greek rho
    ord("ϲ"): "c",  # U+03F2 lunate sigma
    # Fullwidth ASCII → ASCII (U+FF01–U+FF5E)
    **{0xFF01 + i: chr(0x21 + i) for i in range(94)},
}

_CONFUSABLE_TABLE = str.maketrans(_CONFUSABLE_MAP)

# ---------------------------------------------------------------------------
# Zero-width and invisible characters
# ---------------------------------------------------------------------------

_INVISIBLE_CHARS = re.compile(
    r"[​‌‍‎‏"  # zero-width space / joiners / marks
    r"﻿"                            # BOM / zero-width no-break space
    r"­"                            # soft hyphen
    r"⁠⁡⁢⁣⁤"   # word joiners / invisible operators
    r"᠎"                            # Mongolian vowel separator (invisible)
    r"͏"                            # combining grapheme joiner
    r"]",
    re.UNICODE,
)

# Non-breaking and unusual whitespace → regular space
_UNUSUAL_WHITESPACE = re.compile(
    r"[   -     　]",
    re.UNICODE,
)

# ---------------------------------------------------------------------------
# Leetspeak expansion table
# ---------------------------------------------------------------------------

_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",   # 1 is most commonly used as i in leetspeak (b1tc0in → bitcoin)
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "+": "t",
    "!": "i",
    "|": "l",
    "€": "e",
    "£": "l",
}

# Build a single character-level translation table for pure digit/symbol leet
_LEET_TABLE = str.maketrans(_LEET_MAP)


# ---------------------------------------------------------------------------
# Scam keywords — used for near-miss and evasion detection
# ---------------------------------------------------------------------------

_SCAM_KEYWORDS = [
    "bitcoin", "payment", "guaranteed", "upfront", "social security",
    "bank account", "wire transfer", "no experience", "earn money",
    "work from home", "cash app", "western union", "moneygram",
    "gift card", "send money", "fee required", "registration fee",
    "training fee", "starter kit", "crypto", "ethereum",
]


def _keyword_variants(keyword: str) -> list[str]:
    """Return common obfuscated variants of a keyword for near-miss matching."""
    variants = [keyword]
    # space-split variant ("pay ment")
    for i in range(1, len(keyword)):
        variants.append(keyword[:i] + " " + keyword[i:])
    return variants


# ---------------------------------------------------------------------------
# TextNormalizer
# ---------------------------------------------------------------------------


class TextNormalizer:
    """Lightweight, fast text normalizer for adversarial evasion prevention.

    Designed to run before every call to ``extract_signals``.  All operations
    are O(n) on the character count of the input and involve no network I/O or
    heavy computation.

    Usage::

        normalizer = TextNormalizer()
        clean = normalizer.normalize(raw_text)
        signals = extract_signals(job_with_clean_text)
    """

    def normalize(self, text: str) -> str:
        """Apply the full normalization pipeline and return the cleaned text.

        Pipeline order (each step feeds the next):
          1. Strip zero-width / invisible characters
          2. NFKD normalization — decomposes ligatures, compatibility chars, fullwidth
          3. Strip combining marks exposed by NFKD decomposition
          4. Unusual whitespace → regular space
          5. Confusable / homoglyph → ASCII equivalent
          6. Leetspeak digit/symbol expansion
          7. Collapse repeated whitespace
        """
        if not text:
            return text

        # 1. Remove zero-width and invisible characters before normalization so
        # they cannot hide inside decomposed sequences.
        text = _INVISIBLE_CHARS.sub("", text)

        # 2. NFKD decomposes ligatures (ﬁ→fi), fullwidth (Ａ→A), and compatibility
        # characters that NFC would leave opaque to the homoglyph table below.
        text = unicodedata.normalize("NFKD", text)

        # 3. Drop combining marks (Mn category) that NFKD exposed — e.g. café→cafe.
        # This strips accent-based obfuscation while preserving base letters.
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        # 4. Unusual whitespace → regular space
        text = _UNUSUAL_WHITESPACE.sub(" ", text)

        # 5. Confusable / homoglyph normalisation
        text = text.translate(_CONFUSABLE_TABLE)

        # 6. Leetspeak digit/symbol → letter (only outside obvious numeric contexts)
        text = self._expand_leet(text)

        # 7. Collapse repeated whitespace
        text = re.sub(r"  +", " ", text)

        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_leet(text: str) -> str:
        """Replace leet characters with their letter equivalents.

        We apply per-token expansion so that legitimate numbers like "3 years"
        are not corrupted while "b1tc0in" → "bitcoin".  A token that is purely
        numeric (e.g. "2024") is left unchanged.
        """
        tokens = text.split(" ")
        result = []
        for token in tokens:
            # Keep purely numeric tokens (dates, years, salaries) intact
            if token.isdigit():
                result.append(token)
            else:
                result.append(token.translate(_LEET_TABLE))
        return " ".join(result)


# ---------------------------------------------------------------------------
# EvasionDetector
# ---------------------------------------------------------------------------

# Patterns for deliberate misspelling of known scam keywords
_MISSPELL_PATTERNS = [
    # bitcoin variants
    re.compile(r"\bb[i1!|][t+][c©][o0][i1!|]n\b", re.IGNORECASE),
    # payment variants with inserted space/char
    re.compile(r"\bpa\s*y\s*ment\b", re.IGNORECASE),
    re.compile(r"\bp[@a]yment\b", re.IGNORECASE),
    # guaranteed variants
    re.compile(r"\bguaran[t+]ee[d]?\b", re.IGNORECASE),
    re.compile(r"\bgu4ran[t+]eed\b", re.IGNORECASE),
    # crypto obfuscation
    re.compile(r"\bcrypt[o0]\b", re.IGNORECASE),
    # upfront variants
    re.compile(r"\bupfr[o0]nt\b", re.IGNORECASE),
    re.compile(r"\bup\s+fr[o0]nt\b", re.IGNORECASE),
    # fee variants
    re.compile(r"\bf[e3][e3]\s+r[e3]quir[e3]d\b", re.IGNORECASE),
    # cash app / zelle variants
    re.compile(r"\bc[a@]sh\s*[a@]pp\b", re.IGNORECASE),
]

# Script-range detectors for mixed-script detection
_CYRILLIC_RANGE = re.compile(r"[Ѐ-ӿ]")
_GREEK_RANGE = re.compile(r"[Ͱ-Ͽ]")
_LATIN_RANGE = re.compile(r"[A-Za-z]")
_CJK_RANGE = re.compile(r"[一-鿿぀-ヿ]")
_ARABIC_RANGE = re.compile(r"[؀-ۿ]")


def _script_flags(word: str) -> set[str]:
    """Return the set of scripts present in *word*."""
    scripts: set[str] = set()
    if _LATIN_RANGE.search(word):
        scripts.add("latin")
    if _CYRILLIC_RANGE.search(word):
        scripts.add("cyrillic")
    if _GREEK_RANGE.search(word):
        scripts.add("greek")
    if _CJK_RANGE.search(word):
        scripts.add("cjk")
    if _ARABIC_RANGE.search(word):
        scripts.add("arabic")
    return scripts


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class EvasionDetector:
    """Detects adversarial evasion attempts in job posting text.

    Instances are stateless — all state lives in the DB or is computed freshly
    from inputs.  Safe to instantiate once and reuse across many jobs.
    """

    def __init__(self) -> None:
        self._normalizer = TextNormalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_evasion_attempts(
        self, text: str, normalized_text: str
    ) -> list[ScamSignal]:
        """Detect adversarial evasion in *text* (original) vs *normalized_text*.

        Returns a (possibly empty) list of ScamSignals representing:
          - ``evasion_attempt``  — significant normalization delta
          - ``unicode_anomaly``  — mixed scripts in the same word
          - ``misspelled_scam_keyword`` — deliberate misspelling / leet of scam words
        """
        signals: list[ScamSignal] = []

        # 1. Significant normalization delta
        if self._significant_delta(text, normalized_text):
            hidden = self._find_hidden_content(text, normalized_text)
            signals.append(
                ScamSignal(
                    name="evasion_attempt",
                    category=SignalCategory.RED_FLAG,
                    weight=0.70,
                    confidence=0.80,
                    detail="Text contains obfuscation that normalization uncovered",
                    evidence=hidden[:200] if hidden else "normalization delta detected",
                )
            )

        # 2. Unicode anomalies (mixed script in same word)
        mixed = self._mixed_script_words(text)
        if mixed:
            signals.append(
                ScamSignal(
                    name="unicode_anomaly",
                    category=SignalCategory.WARNING,
                    weight=0.55,
                    confidence=0.75,
                    detail=f"Mixed-script characters in {len(mixed)} word(s): {', '.join(mixed[:3])}",
                    evidence=", ".join(mixed[:5]),
                )
            )

        # 3. Deliberate keyword misspellings / leet
        misspelled = self._find_misspellings(text)
        if misspelled:
            signals.append(
                ScamSignal(
                    name="misspelled_scam_keyword",
                    category=SignalCategory.RED_FLAG,
                    weight=0.70,
                    confidence=0.78,
                    detail=f"Deliberate misspelling of scam keyword(s): {', '.join(misspelled[:3])}",
                    evidence=", ".join(misspelled[:5]),
                )
            )

        return signals

    def track_signal_decay(self, db: object) -> list[dict]:
        """Monitor per-signal firing rates and flag suspicious decay.

        A signal that historically fired at rate R and now fires at R * 0.25
        may be under active evasion.

        Args:
            db: SentinelDB instance.

        Returns:
            List of dicts with keys:
              signal_name, baseline_rate, recent_rate, decay_rate, suspicious
        """
        from sentinel.db import SentinelDB
        assert isinstance(db, SentinelDB)

        history = db.get_signal_rate_history()
        if not history:
            return []

        # Group by signal name, separate into "baseline" (older half) vs "recent" (newer half)
        from collections import defaultdict
        by_signal: dict[str, list[dict]] = defaultdict(list)
        for row in history:
            by_signal[row["signal_name"]].append(row)

        results = []
        for signal_name, rows in by_signal.items():
            # Need at least 4 windows to make a meaningful comparison
            if len(rows) < 4:
                continue

            # Sort chronologically
            rows_sorted = sorted(rows, key=lambda r: r["window_start"])
            half = len(rows_sorted) // 2
            baseline_rows = rows_sorted[:half]
            recent_rows = rows_sorted[half:]

            def _rate(window: list[dict]) -> float:
                total = sum(r["total_jobs"] for r in window)
                fires = sum(r["fire_count"] for r in window)
                return fires / total if total > 0 else 0.0

            baseline_rate = _rate(baseline_rows)
            recent_rate = _rate(recent_rows)

            # Avoid division-by-zero; also skip signals that barely fired
            if baseline_rate < 0.01:
                continue

            decay_rate = (baseline_rate - recent_rate) / baseline_rate
            suspicious = decay_rate > 0.50  # > 50% drop is suspicious

            results.append(
                {
                    "signal_name": signal_name,
                    "baseline_rate": round(baseline_rate, 4),
                    "recent_rate": round(recent_rate, 4),
                    "decay_rate": round(decay_rate, 4),
                    "suspicious": suspicious,
                }
            )

        return results

    def detect_near_misses(
        self, signals: list[ScamSignal], score: float
    ) -> list[dict]:
        """Identify partial keyword matches that almost triggered signals.

        At high volume a cluster of near-misses for the same keyword suggests
        scammers are deliberately staying just below the detection threshold.

        Args:
            signals: The ScamSignals that *did* fire for this job.
            score: The current scam score for the job (0–1).

        Returns:
            List of dicts with keys:
              signal_name, partial_match, severity
            ``severity`` is "high" when score >= 0.4 and "low" otherwise.
        """
        near_misses: list[dict] = []
        {s.name for s in signals}

        # Near-miss only interesting when score is non-trivial
        if score < 0.1:
            return near_misses

        # For each signal that did NOT fire, check if evidence in fired signals
        # partially overlaps with scam keywords — these are the "almost" hits.
        all_evidence = " ".join(s.evidence.lower() for s in signals)

        for keyword in _SCAM_KEYWORDS:
            # Skip keywords already fully matched by a fired signal
            if keyword in all_evidence:
                continue

            # Partial match: find the *longest* prefix of keyword (at least 60%
            # of its length) that appears in the evidence.  Longest-first gives
            # more meaningful / suspicious near-miss strings (e.g. "bitco" rather
            # than "bitc").
            threshold = int(len(keyword) * 0.6)
            for length in range(len(keyword) - 1, threshold - 1, -1):
                fragment = keyword[:length]
                if fragment in all_evidence and len(fragment) >= 4:
                    near_misses.append(
                        {
                            "signal_name": "near_miss",
                            "partial_match": fragment,
                            "severity": "high" if score >= 0.4 else "low",
                        }
                    )
                    break  # one near-miss per keyword is enough

        return near_misses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _significant_delta(original: str, normalized: str) -> bool:
        """Return True if normalization changed more than 1% of characters."""
        if not original:
            return False
        changed = sum(1 for a, b in zip(original, normalized, strict=False) if a != b)
        # Also count length difference (invisible chars were removed)
        length_diff = abs(len(original) - len(normalized))
        total_changes = changed + length_diff
        return total_changes / max(len(original), 1) > 0.01

    @staticmethod
    def _find_hidden_content(original: str, normalized: str) -> str:
        """Return a short description of what changed during normalization."""
        parts = []
        if len(original) != len(normalized):
            parts.append(f"length {len(original)}→{len(normalized)}")

        # Find changed character positions (compare up to shorter length)
        diff_chars: list[str] = []
        for a, b in zip(original, normalized, strict=False):
            if a != b and a not in diff_chars:
                diff_chars.append(a)
        if diff_chars:
            samples = [f"U+{ord(c):04X}" for c in diff_chars[:5]]
            parts.append("changed: " + ", ".join(samples))

        return "; ".join(parts)

    @staticmethod
    def _mixed_script_words(text: str) -> list[str]:
        """Return words that contain characters from more than one script."""
        mixed: list[str] = []
        for word in re.findall(r"\S+", text):
            if len(word) < 3:
                continue
            scripts = _script_flags(word)
            if len(scripts) >= 2:
                mixed.append(word)
        return mixed

    @staticmethod
    def _find_misspellings(text: str) -> list[str]:
        """Return matched leet/misspelling evidence strings."""
        found: list[str] = []
        for pattern in _MISSPELL_PATTERNS:
            m = pattern.search(text)
            if m:
                found.append(m.group(0))
        return found

# ---------------------------------------------------------------------------
# Adversarial Training — TextAttack-style signal hardening
# ---------------------------------------------------------------------------

import random as _random

_ATTACK_STRATEGIES = [
    'homoglyph',
    'zwc_insert',
    'leetspeak',
    'whitespace_split',
    'case_swap',
    'char_duplicate',
    'char_drop',
]


class AdversarialTrainer:
    """Generate adversarial variants of scam text and test signal resilience.

    For each known scam keyword or pattern, applies multiple perturbation
    strategies and checks whether signal extraction still detects it after
    normalization.  Reports survival rates per signal and per attack type.
    """

    def __init__(self, normalizer: TextNormalizer | None = None) -> None:
        self._normalizer = normalizer or TextNormalizer()
        self._rng = _random.Random(42)

    def _apply_homoglyph(self, text: str) -> str:
        result = list(text)
        reverse_map: dict[str, list[str]] = {}
        for cp, latin in _CONFUSABLE_MAP.items():
            reverse_map.setdefault(latin.lower(), []).append(chr(cp))
        for i, ch in enumerate(result):
            if ch.lower() in reverse_map and self._rng.random() < 0.4:
                result[i] = self._rng.choice(reverse_map[ch.lower()])
        return ''.join(result)

    def _apply_zwc_insert(self, text: str) -> str:
        zwc = '​'
        result = []
        for ch in text:
            result.append(ch)
            if ch.isalpha() and self._rng.random() < 0.3:
                result.append(zwc)
        return ''.join(result)

    def _apply_leetspeak(self, text: str) -> str:
        leet = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'}
        result = []
        for ch in text:
            if ch.lower() in leet and self._rng.random() < 0.4:
                result.append(leet[ch.lower()])
            else:
                result.append(ch)
        return ''.join(result)

    def _apply_whitespace_split(self, text: str) -> str:
        words = text.split()
        result = []
        for w in words:
            if len(w) > 4 and self._rng.random() < 0.3:
                pos = self._rng.randint(1, len(w) - 2)
                result.append(w[:pos] + ' ' + w[pos:])
            else:
                result.append(w)
        return ' '.join(result)

    def _apply_case_swap(self, text: str) -> str:
        result = []
        for ch in text:
            if ch.isalpha() and self._rng.random() < 0.3:
                result.append(ch.swapcase())
            else:
                result.append(ch)
        return ''.join(result)

    def _apply_char_duplicate(self, text: str) -> str:
        result = []
        for ch in text:
            result.append(ch)
            if ch.isalpha() and self._rng.random() < 0.15:
                result.append(ch)
        return ''.join(result)

    def _apply_char_drop(self, text: str) -> str:
        result = []
        for ch in text:
            if ch.isalpha() and self._rng.random() < 0.1:
                continue
            result.append(ch)
        return ''.join(result)

    def perturb(self, text: str, strategy: str) -> str:
        dispatch = {
            'homoglyph': self._apply_homoglyph,
            'zwc_insert': self._apply_zwc_insert,
            'leetspeak': self._apply_leetspeak,
            'whitespace_split': self._apply_whitespace_split,
            'case_swap': self._apply_case_swap,
            'char_duplicate': self._apply_char_duplicate,
            'char_drop': self._apply_char_drop,
        }
        fn = dispatch.get(strategy)
        if fn is None:
            return text
        return fn(text)

    def generate_variants(
        self,
        text: str,
        strategies: list[str] | None = None,
        n_per_strategy: int = 3,
    ) -> list[dict]:
        strategies = strategies or _ATTACK_STRATEGIES
        variants: list[dict] = []
        for strat in strategies:
            for _ in range(n_per_strategy):
                self._rng = _random.Random(self._rng.randint(0, 2**32))
                perturbed = self.perturb(text, strat)
                normalized = self._normalizer.normalize(perturbed)
                variants.append({
                    'strategy': strat,
                    'original': text,
                    'perturbed': perturbed,
                    'normalized': normalized,
                    'recovered': normalized.lower().strip() == text.lower().strip(),
                })
        return variants

    def test_signal_resilience(
        self,
        text: str,
        signal_extractor,
        strategies: list[str] | None = None,
    ) -> dict:
        """Test whether signal extraction survives adversarial perturbations.

        Args:
            text: Original scam job posting text.
            signal_extractor: Callable that takes text and returns a list of
                              signal names (or ScamSignal objects with .name).
            strategies: List of attack strategies to test.

        Returns:
            Dict with per-strategy and per-signal survival rates.
        """
        from sentinel.models import JobPosting

        baseline_job = JobPosting(description=text)
        baseline_signals = signal_extractor(baseline_job)
        baseline_names = set()
        for s in baseline_signals:
            name = s.name if hasattr(s, 'name') else str(s)
            baseline_names.add(name)

        if not baseline_names:
            return {'baseline_signals': 0, 'strategies': {}, 'overall_survival': 1.0}

        strategies = strategies or _ATTACK_STRATEGIES
        strategy_results: dict[str, dict] = {}
        signal_survival: dict[str, list[bool]] = {n: [] for n in baseline_names}

        for strat in strategies:
            variants = self.generate_variants(text, [strat], n_per_strategy=5)
            survived_total = 0
            tested_total = 0

            for v in variants:
                attacked_job = JobPosting(description=v['normalized'])
                attacked_signals = signal_extractor(attacked_job)
                attacked_names = set()
                for s in attacked_signals:
                    name = s.name if hasattr(s, 'name') else str(s)
                    attacked_names.add(name)

                for bn in baseline_names:
                    hit = bn in attacked_names
                    signal_survival[bn].append(hit)
                    tested_total += 1
                    if hit:
                        survived_total += 1

            strategy_results[strat] = {
                'survival_rate': survived_total / tested_total if tested_total else 1.0,
                'variants_tested': len(variants),
            }

        per_signal = {}
        for name, hits in signal_survival.items():
            per_signal[name] = sum(hits) / len(hits) if hits else 1.0

        all_hits = [h for hits in signal_survival.values() for h in hits]
        overall = sum(all_hits) / len(all_hits) if all_hits else 1.0

        return {
            'baseline_signals': len(baseline_names),
            'strategies': strategy_results,
            'per_signal_survival': per_signal,
            'overall_survival': round(overall, 4),
            'weak_signals': [n for n, r in per_signal.items() if r < 0.6],
        }
