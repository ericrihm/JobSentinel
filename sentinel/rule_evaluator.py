"""Offline rule quality evaluator for JobSentinel scam detection patterns.

Implements LLM-as-judge concept adapted for offline use: assesses pattern
quality via multiple heuristics without requiring an external API call.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score weights for composite scoring
# ---------------------------------------------------------------------------

_SPECIFICITY_WEIGHT = 0.25
_COVERAGE_WEIGHT    = 0.25
_PRECISION_WEIGHT   = 0.35
_NOVELTY_WEIGHT     = 0.15

# Minimum observations before coverage/precision scores are trustworthy
_MIN_OBS_FOR_TRUST = 5


def _parse_keywords(raw: Any) -> list[str]:
    """Normalise the keywords field — may be a JSON string, a list, or None."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(k).strip().lower() for k in raw if str(k).strip()]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(k).strip().lower() for k in parsed if str(k).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fall back: comma-separated plain string
        return [k.strip().lower() for k in raw.split(",") if k.strip()]
    return []


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def _specificity_score(pattern: dict) -> float:
    """Score how specific/targeted the pattern is.

    Longer regex and more keywords both indicate a more precise pattern
    (less likely to produce false positives via over-broad matching).

    Returns a value in [0, 1].
    """
    regex: str = pattern.get("regex") or ""
    keywords: list[str] = _parse_keywords(pattern.get("keywords"))

    # Regex component: score based on length and feature richness
    regex_score = 0.0
    if regex:
        # Penalise trivially short / catch-all patterns
        length_score = min(len(regex) / 80.0, 1.0)  # saturates at 80 chars
        # Bonus for anchors, word boundaries, groups — signs of careful crafting
        feature_bonus = 0.0
        for feature in (r"\b", r"(?:", r"(?i", r"^", r"$", r"\d", r"\w"):
            if feature in regex:
                feature_bonus += 0.05
        feature_bonus = min(feature_bonus, 0.25)
        regex_score = min(length_score + feature_bonus, 1.0)

    # Keywords component: more unique keywords = more specific
    kw_count = len(set(keywords))
    kw_score = min(kw_count / 10.0, 1.0)  # saturates at 10 keywords

    if regex and keywords:
        return 0.5 * regex_score + 0.5 * kw_score
    elif regex:
        return regex_score
    elif keywords:
        return kw_score
    # Pattern has neither regex nor keywords — worst case
    return 0.0


def _coverage_score(pattern: dict) -> float:
    """Fraction of known scam jobs caught — true_positives / max(observations, 1).

    Shrunk toward 0.5 when observations are low to avoid rewarding lucky streaks
    on tiny samples.

    Returns a value in [0, 1].
    """
    obs = int(pattern.get("observations") or 0)
    tp  = int(pattern.get("true_positives") or 0)

    if obs == 0:
        return 0.0

    raw = tp / obs  # recall-like: what fraction of obs were true positives

    # Bayesian shrinkage toward 0.5 when data is sparse
    if obs < _MIN_OBS_FOR_TRUST:
        # Shrink toward prior mean (0.5) inversely proportional to obs count
        shrink = (1.0 - obs / _MIN_OBS_FOR_TRUST)
        raw = raw * (1.0 - shrink) + 0.5 * shrink

    return max(0.0, min(1.0, raw))


def _precision_score(pattern: dict) -> float:
    """TP / (TP + FP) with Laplace smoothing when counts are low.

    Returns a value in [0, 1].
    """
    tp = int(pattern.get("true_positives") or 0)
    fp = int(pattern.get("false_positives") or 0)
    obs = int(pattern.get("observations") or 0)

    # Prefer Bayesian score if present and observations are sparse
    if obs < _MIN_OBS_FOR_TRUST:
        alpha = float(pattern.get("alpha") or 1.0)
        beta  = float(pattern.get("beta")  or 1.0)
        return alpha / (alpha + beta)

    total = tp + fp
    if total == 0:
        # No recorded outcomes — return Bayesian fallback
        alpha = float(pattern.get("alpha") or 1.0)
        beta  = float(pattern.get("beta")  or 1.0)
        return alpha / (alpha + beta)

    # Laplace smoothing: add 1 to numerator and 2 to denominator
    return (tp + 1) / (total + 2)


def _novelty_score(pattern: dict, active_patterns: list[dict]) -> float:
    """How different is this pattern from existing active patterns?

    Computed as 1 minus the maximum Jaccard similarity between this pattern's
    keyword set and any active pattern's keyword set.

    A score of 1.0 means completely novel; 0.0 means exact duplicate.

    Returns a value in [0, 1].
    """
    my_keywords = set(_parse_keywords(pattern.get("keywords")))
    my_regex    = (pattern.get("regex") or "").strip()

    if not active_patterns:
        return 1.0  # nothing to compare against — fully novel

    max_similarity = 0.0

    for ap in active_patterns:
        # Skip if comparing the pattern against itself
        if ap.get("pattern_id") == pattern.get("pattern_id"):
            continue

        their_keywords = set(_parse_keywords(ap.get("keywords")))

        # Keyword Jaccard similarity
        if my_keywords or their_keywords:
            union = my_keywords | their_keywords
            intersection = my_keywords & their_keywords
            jaccard = len(intersection) / len(union) if union else 0.0
        else:
            jaccard = 0.0

        # Regex character-level overlap (simple, cheap)
        their_regex = (ap.get("regex") or "").strip()
        if my_regex and their_regex:
            # Normalised common prefix + bigram overlap as a cheap proxy
            min_len = min(len(my_regex), len(their_regex))
            common_chars = sum(a == b for a, b in zip(my_regex, their_regex))
            regex_sim = common_chars / max(len(my_regex), len(their_regex)) if my_regex else 0.0
            combined = 0.6 * jaccard + 0.4 * regex_sim
        else:
            combined = jaccard

        max_similarity = max(max_similarity, combined)

    return max(0.0, min(1.0, 1.0 - max_similarity))


# ---------------------------------------------------------------------------
# RuleEvaluator
# ---------------------------------------------------------------------------

class RuleEvaluator:
    """Offline quality evaluator for scam detection rules/patterns.

    Scores candidate patterns across four orthogonal dimensions and produces a
    composite score plus a promote/reject recommendation — all without making
    any external API calls.

    Usage::

        evaluator = RuleEvaluator()
        result = evaluator.evaluate_candidate(candidate_pattern, active_patterns)
        ranked  = evaluator.rank_candidates(candidate_list, active_patterns)
    """

    # Composite score threshold required to recommend promotion
    PROMOTE_THRESHOLD: float = 0.55

    def evaluate_candidate(
        self,
        pattern: dict,
        active_patterns: list[dict],
    ) -> dict:
        """Evaluate a single candidate pattern for quality.

        Args:
            pattern:         Pattern dict (keys match db.get_patterns() output).
            active_patterns: List of currently active pattern dicts for novelty
                             comparison.

        Returns:
            Dict with keys:
            - specificity_score  (float [0,1])
            - coverage_score     (float [0,1])
            - precision_score    (float [0,1])
            - novelty_score      (float [0,1])
            - composite_score    (float [0,1])
            - recommendation     ("promote" | "reject")
            - reason             (human-readable string)
        """
        spec    = _specificity_score(pattern)
        cov     = _coverage_score(pattern)
        prec    = _precision_score(pattern)
        novelty = _novelty_score(pattern, active_patterns)

        composite = (
            _SPECIFICITY_WEIGHT * spec
            + _COVERAGE_WEIGHT   * cov
            + _PRECISION_WEIGHT  * prec
            + _NOVELTY_WEIGHT    * novelty
        )
        composite = round(composite, 4)

        recommend = "promote" if composite >= self.PROMOTE_THRESHOLD else "reject"

        # Build a human-readable reason
        issues: list[str] = []
        strengths: list[str] = []
        if spec < 0.3:
            issues.append("low specificity (vague regex/few keywords)")
        elif spec >= 0.7:
            strengths.append("highly specific pattern")
        if cov < 0.3:
            issues.append("low coverage (catches few true positives)")
        elif cov >= 0.7:
            strengths.append("good coverage")
        if prec < 0.5:
            issues.append("precision below 50%")
        elif prec >= 0.8:
            strengths.append("high precision")
        if novelty < 0.3:
            issues.append("highly similar to existing active patterns")
        elif novelty >= 0.8:
            strengths.append("novel pattern not covered by existing rules")

        if recommend == "promote":
            reason = "Meets quality gate."
            if strengths:
                reason += " Strengths: " + ", ".join(strengths) + "."
        else:
            reason = "Below quality threshold."
            if issues:
                reason += " Issues: " + ", ".join(issues) + "."

        return {
            "pattern_id":       pattern.get("pattern_id", ""),
            "specificity_score": round(spec, 4),
            "coverage_score":    round(cov, 4),
            "precision_score":   round(prec, 4),
            "novelty_score":     round(novelty, 4),
            "composite_score":   composite,
            "recommendation":    recommend,
            "reason":            reason,
        }

    def rank_candidates(
        self,
        candidates: list[dict],
        active_patterns: list[dict],
    ) -> list[dict]:
        """Evaluate and rank a list of candidate patterns by composite score.

        Args:
            candidates:      List of candidate pattern dicts.
            active_patterns: List of currently active pattern dicts.

        Returns:
            List of evaluation result dicts (same shape as evaluate_candidate()
            output), sorted by composite_score descending.
        """
        results = [
            self.evaluate_candidate(p, active_patterns)
            for p in candidates
        ]
        results.sort(key=lambda r: r["composite_score"], reverse=True)
        return results
