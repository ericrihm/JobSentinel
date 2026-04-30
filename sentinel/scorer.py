"""Bayesian scam scoring with Thompson Sampling for signal weight learning."""

import json
import math
import os
import random

from sentinel.models import JobPosting, RiskLevel, ScamSignal, SignalCategory, ValidationResult


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_signals(signals: list[ScamSignal]) -> tuple[float, float]:
    """Compute scam score and confidence from extracted signals.

    Uses weighted Bayesian combination:
    - Positive signals reduce score
    - Red flags and warnings increase score
    - Confidence is based on number and agreement of signals

    Returns (scam_score 0-1, confidence 0-1)
    """
    if not signals:
        return 0.0, 0.0

    _POSITIVE = SignalCategory.POSITIVE

    # Log-odds accumulation: start at log-odds 0 (= 50% prior)
    log_odds = 0.0
    for s in signals:
        w = max(1e-6, min(1.0 - 1e-6, s.weight))
        # Positive signals move log-odds toward "legitimate"
        if s.category == _POSITIVE:
            # Positive signal: weight is chance-of-scam, so (1-w)/w is the
            # legitimacy odds ratio — subtract it to push score toward 0.
            log_odds -= math.log((1.0 - w) / w)
        else:
            log_odds += math.log(w / (1.0 - w))

    scam_score = 1.0 / (1.0 + math.exp(-log_odds))

    # Confidence: grows with signal count, penalised when positive and negative
    # signals strongly disagree (mixed evidence lowers confidence).
    n_scam = sum(1 for s in signals if s.category != _POSITIVE)
    n_pos = sum(1 for s in signals if s.category == _POSITIVE)
    total = len(signals)

    # Base confidence from signal count (asymptote ~0.95)
    base_conf = 1.0 - math.exp(-0.3 * total)

    # Agreement factor: 1.0 when all same-direction, lower when mixed
    if total > 0:
        majority = max(n_scam, n_pos)
        agreement = majority / total
    else:
        agreement = 1.0

    confidence = round(base_conf * agreement, 4)
    scam_score = round(scam_score, 4)

    return scam_score, confidence


def classify_risk(scam_score: float) -> RiskLevel:
    """Map score to risk level."""
    if scam_score < 0.2:
        return RiskLevel.SAFE
    if scam_score < 0.4:
        return RiskLevel.LOW
    if scam_score < 0.6:
        return RiskLevel.SUSPICIOUS
    if scam_score < 0.8:
        return RiskLevel.HIGH
    return RiskLevel.SCAM


def build_result(
    job: JobPosting,
    signals: list[ScamSignal],
    analysis_time_ms: float = 0.0,
) -> ValidationResult:
    """Full pipeline: score signals, classify risk, build result."""
    scam_score, confidence = score_signals(signals)
    risk_level = classify_risk(scam_score)
    return ValidationResult(
        job=job,
        signals=signals,
        scam_score=scam_score,
        confidence=confidence,
        risk_level=risk_level,
        analysis_time_ms=analysis_time_ms,
    )


# ---------------------------------------------------------------------------
# Bayesian weight learner
# ---------------------------------------------------------------------------

class SignalWeightTracker:
    """Bayesian weight learner for signal effectiveness.

    Uses Beta(alpha, beta) posteriors per signal name.
    Updated when user reports confirm/deny our predictions.
    Thompson Sampling for exploration.
    """

    def __init__(self) -> None:
        # Maps signal name -> (alpha, beta); Beta(1,1) = uniform prior
        self._weights: dict[str, tuple[float, float]] = {}

    def _get_posterior(self, signal_name: str) -> tuple[float, float]:
        return self._weights.get(signal_name, (1.0, 1.0))

    def update(self, signal_name: str, was_correct: bool) -> None:
        """Update posterior: alpha += was_correct, beta += (not was_correct)."""
        alpha, beta = self._get_posterior(signal_name)
        if was_correct:
            alpha += 1.0
        else:
            beta += 1.0
        self._weights[signal_name] = (alpha, beta)

    def get_weight(self, signal_name: str) -> float:
        """Thompson sample from Beta posterior.

        Samples a weight from the current posterior — balances exploitation
        (high-precision signals get high weight) with exploration (uncertain
        signals still get occasional high samples, keeping them in play).
        """
        alpha, beta = self._get_posterior(signal_name)
        # random.betavariate is stdlib; no numpy needed
        return random.betavariate(alpha, beta)

    def get_ranking(self) -> list[tuple[str, float, float]]:
        """Return signals ranked by mean effectiveness: (name, mean, confidence).

        Mean = alpha / (alpha + beta).
        Confidence here is a normalised measure of how many observations have
        been made; it saturates toward 1.0 as alpha + beta grows.
        """
        rows: list[tuple[str, float, float]] = []
        for name, (alpha, beta) in self._weights.items():
            n = alpha + beta - 2.0  # subtract the prior counts
            mean = alpha / (alpha + beta)
            obs_confidence = 1.0 - math.exp(-0.1 * max(n, 0.0))
            rows.append((name, round(mean, 4), round(obs_confidence, 4)))
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {name: list(ab) for name, ab in self._weights.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, list[float]] = json.load(fh)
        self._weights = {name: (ab[0], ab[1]) for name, ab in raw.items()}
