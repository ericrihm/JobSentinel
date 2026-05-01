"""Bayesian scam scoring with Thompson Sampling for signal weight learning."""

import json
import logging
import math
import os
import random

from sentinel.models import JobPosting, RiskLevel, ScamSignal, SignalCategory, ValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Learned-weight cache (loaded once per process from DB)
# ---------------------------------------------------------------------------

_learned_weights_cache: dict[str, float] | None = None

# Minimum observations before a learned weight overrides the static default.
_MIN_OBSERVATIONS = 10


def _load_learned_weights(db_path: str = "") -> dict[str, float]:
    """Query the DB for all active patterns and return {signal_name: bayesian_weight}.

    The result is cached at module level so the DB is only hit once per
    process.  If the DB is unavailable or empty, returns an empty dict and
    scoring falls back to static weights.
    """
    global _learned_weights_cache
    if _learned_weights_cache is not None:
        return _learned_weights_cache

    weights: dict[str, float] = {}
    try:
        from sentinel.db import SentinelDB  # deferred to keep import lightweight

        db = SentinelDB(path=db_path) if db_path else SentinelDB()
        rows = db.get_patterns(status="active")
        for row in rows:
            obs = row.get("observations", 0)
            if obs < _MIN_OBSERVATIONS:
                continue
            alpha = row.get("alpha", 1.0)
            beta = row.get("beta", 1.0)
            total = alpha + beta
            if total <= 0:
                continue
            # Use pattern_id as the key — it matches signal names in the DB
            pid = row.get("pattern_id", "")
            name = row.get("name", "")
            bayesian_w = alpha / total
            if pid:
                weights[pid] = bayesian_w
            if name and name != pid:
                weights[name] = bayesian_w
        db.close()
    except Exception:
        # DB missing, locked, or schema mismatch — fall back gracefully
        logger.debug("Could not load learned weights from DB; using static defaults", exc_info=True)
        weights = {}

    _learned_weights_cache = weights
    return _learned_weights_cache


def _reset_learned_weights_cache() -> None:
    """Clear the module-level cache (used by tests and after flywheel cycles)."""
    global _learned_weights_cache
    _learned_weights_cache = None


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_signals(
    signals: list[ScamSignal],
    *,
    db_path: str = "",
    use_learned_weights: bool = True,
) -> tuple[float, float]:
    """Compute scam score and confidence from extracted signals.

    Uses weighted Bayesian combination:
    - Positive signals reduce score
    - Red flags and warnings increase score
    - Confidence is based on number and agreement of signals

    When *use_learned_weights* is True (the default), the scorer will try to
    load Bayesian posterior weights from the DB and use them in place of the
    hard-coded signal weights for any pattern that has >= 10 observations.

    Returns (scam_score 0-1, confidence 0-1)
    """
    if not signals:
        return 0.0, 0.0

    _POSITIVE = SignalCategory.POSITIVE

    # Try loading learned weights from the DB
    learned: dict[str, float] = {}
    if use_learned_weights:
        try:
            learned = _load_learned_weights(db_path=db_path)
        except Exception:
            learned = {}

    # Log-odds accumulation: start at log-odds 0 (= 50% prior)
    log_odds = 0.0
    for s in signals:
        # Check for a learned weight override (match on signal name)
        w = learned.get(s.name, s.weight)

        w = max(1e-6, min(1.0 - 1e-6, w))
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


# ---------------------------------------------------------------------------
# Risk classification thresholds (mutable — auto-adjusted by calibration)
# ---------------------------------------------------------------------------

#: Maps risk-level name → upper boundary (exclusive).
#: Score >= scam threshold → SCAM; score in [high, scam) → HIGH; etc.
_RISK_THRESHOLDS: dict[str, float] = {
    "safe": 0.2,        # score < 0.2  → SAFE
    "low": 0.4,         # score < 0.4  → LOW
    "suspicious": 0.6,  # score < 0.6  → SUSPICIOUS
    "high": 0.8,        # score < 0.8  → HIGH
    # score >= 0.8      → SCAM
}

# Maximum allowed shift per auto-adjustment cycle (prevents wild swings)
_MAX_THRESHOLD_DELTA: float = 0.05


def classify_risk(scam_score: float) -> RiskLevel:
    """Map score to risk level using the (adjustable) _RISK_THRESHOLDS table."""
    if scam_score < _RISK_THRESHOLDS["safe"]:
        return RiskLevel.SAFE
    if scam_score < _RISK_THRESHOLDS["low"]:
        return RiskLevel.LOW
    if scam_score < _RISK_THRESHOLDS["suspicious"]:
        return RiskLevel.SUSPICIOUS
    if scam_score < _RISK_THRESHOLDS["high"]:
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


# ---------------------------------------------------------------------------
# Ensemble scoring
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class EnsembleResult:
    """Output of EnsembleScorer.score_ensemble()."""
    primary_score: float
    weighted_avg_score: float
    majority_vote_score: float
    ensemble_score: float
    disagreement: float
    confidence_adjustment: float
    method_scores: dict[str, float] = field(default_factory=dict)


class EnsembleScorer:
    """Combines three independent scoring methods and auto-adjusts weights.

    Methods:
    1. Primary   — existing log-odds Bayesian scorer (weight 0.6 default)
    2. Weighted  — simple weighted average of signal weights / max possible
    3. Majority  — fraction of signals above a threshold

    When disagreement (std-dev across the three scores) > 0.2 the result is
    flagged as "uncertain" and the ensemble score confidence is reduced.

    Per-method accuracy is tracked in ``flywheel_metrics`` and ensemble
    weights auto-adjust toward historically more accurate methods.
    """

    # Default ensemble weights — sum must equal 1.0
    _DEFAULT_WEIGHTS: dict[str, float] = {
        "primary":       0.6,
        "weighted_avg":  0.2,
        "majority_vote": 0.2,
    }

    # Disagreement threshold above which we flag as uncertain
    DISAGREEMENT_THRESHOLD: float = 0.2

    # Majority-vote threshold: signals with weight above this count as "scam vote"
    _MAJORITY_VOTE_SIGNAL_THRESHOLD: float = 0.5

    def __init__(self) -> None:
        # method_name -> (alpha, beta)  Beta posteriors for accuracy tracking
        self._method_posteriors: dict[str, list[float]] = {
            "primary":       [1.0, 1.0],
            "weighted_avg":  [1.0, 1.0],
            "majority_vote": [1.0, 1.0],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_ensemble(
        self,
        db,  # SentinelDB or None — used to load per-method accuracy
        job,  # JobPosting — not used directly, reserved for future use
        signals: list,  # list[ScamSignal]
    ) -> "EnsembleResult":
        """Run all three scoring methods and return an EnsembleResult.

        Args:
            db:      Optional SentinelDB instance for loading per-method weights.
            job:     The JobPosting being scored (reserved for future use).
            signals: The ScamSignal list produced by ``extract_signals(job)``.

        Returns:
            EnsembleResult with all method scores, disagreement, and
            confidence_adjustment (-0.2 when uncertain, 0.0 otherwise).
        """

        # --- Method 1: Primary log-odds Bayesian scorer ---
        primary, _conf = score_signals(signals)

        # --- Method 2: Weighted average (sum of weights / max possible) ---
        weighted_avg = self._score_weighted_avg(signals)

        # --- Method 3: Majority vote (fraction of signals firing above threshold) ---
        majority_vote = self._score_majority_vote(signals)

        method_scores = {
            "primary":       round(primary, 4),
            "weighted_avg":  round(weighted_avg, 4),
            "majority_vote": round(majority_vote, 4),
        }

        # Disagreement: std dev of the three scores
        vals = [primary, weighted_avg, majority_vote]
        mean_val = sum(vals) / 3.0
        variance = sum((v - mean_val) ** 2 for v in vals) / 3.0
        disagreement = round(math.sqrt(variance), 4)

        # Confidence adjustment: penalise high disagreement
        confidence_adjustment = -0.2 if disagreement > self.DISAGREEMENT_THRESHOLD else 0.0

        # Ensemble weights — load auto-adjusted weights from DB if available
        weights = self._get_ensemble_weights(db)

        ensemble_score = (
            weights["primary"]       * primary
            + weights["weighted_avg"]  * weighted_avg
            + weights["majority_vote"] * majority_vote
        )
        ensemble_score = round(min(1.0, max(0.0, ensemble_score)), 4)

        return EnsembleResult(
            primary_score=round(primary, 4),
            weighted_avg_score=round(weighted_avg, 4),
            majority_vote_score=round(majority_vote, 4),
            ensemble_score=ensemble_score,
            disagreement=disagreement,
            confidence_adjustment=confidence_adjustment,
            method_scores=method_scores,
        )

    def update_method_accuracy(
        self, db, method_name: str, was_correct: bool
    ) -> None:
        """Record one accuracy observation for *method_name*.

        Updates both the in-process Beta posterior and the DB flywheel_metrics
        column (via an extra JSON blob in the last cycle row if available).
        Falls back gracefully when db is None.
        """
        if method_name not in self._method_posteriors:
            return
        alpha, beta = self._method_posteriors[method_name]
        if was_correct:
            alpha += 1.0
        else:
            beta += 1.0
        self._method_posteriors[method_name] = [alpha, beta]

        # Persist to DB as a best-effort JSON metadata update
        if db is not None:
            try:
                self._persist_method_accuracy(db)
            except Exception:
                logger.debug("Could not persist method accuracy to DB", exc_info=True)

    def get_method_accuracy(self) -> dict[str, float]:
        """Return the expected accuracy (mean of Beta posterior) per method."""
        return {
            name: ab[0] / (ab[0] + ab[1])
            for name, ab in self._method_posteriors.items()
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_weighted_avg(self, signals: list) -> float:
        """Simple weighted average: sum(weights) / max_possible_weight_sum."""
        from sentinel.models import SignalCategory
        if not signals:
            return 0.0

        scam_weight_sum = 0.0
        max_possible = 0.0
        for s in signals:
            w = max(1e-6, min(1.0, s.weight))
            max_possible += w
            if s.category != SignalCategory.POSITIVE:
                scam_weight_sum += w
            # Positive signals cancel out some scam weight
            else:
                scam_weight_sum -= w * 0.5  # partial cancellation

        if max_possible == 0.0:
            return 0.0
        return max(0.0, min(1.0, scam_weight_sum / max_possible))

    def _score_majority_vote(self, signals: list) -> float:
        """Fraction of signals that vote 'scam' (weight > threshold)."""
        from sentinel.models import SignalCategory
        if not signals:
            return 0.0

        scam_votes = 0
        for s in signals:
            if s.category != SignalCategory.POSITIVE and s.weight >= self._MAJORITY_VOTE_SIGNAL_THRESHOLD:
                scam_votes += 1
        return round(scam_votes / len(signals), 4)

    def _get_ensemble_weights(self, db) -> dict[str, float]:
        """Return ensemble weights, auto-adjusted by historical accuracy if available."""
        try:
            if db is not None:
                return self._compute_adjusted_weights(db)
        except Exception:
            logger.debug("Could not load ensemble weights from DB; using defaults", exc_info=True)
        return dict(self._DEFAULT_WEIGHTS)

    def _compute_adjusted_weights(self, db) -> dict[str, float]:
        """Compute ensemble weights proportional to per-method Beta-posterior accuracy.

        Loads per-method accuracy stored by ``_persist_method_accuracy`` and
        uses softmax-like normalisation to keep weights summing to 1.0.
        Falls back to defaults when no history is available.
        """
        stored = self._load_method_accuracy(db)
        if stored:
            for name, acc in stored.items():
                if name in self._method_posteriors:
                    # Don't override if we have better local data
                    local_obs = sum(self._method_posteriors[name]) - 2.0
                    if local_obs < 5:
                        # Blend stored accuracy into local posterior
                        alpha = acc * 10 + 1.0
                        beta = (1.0 - acc) * 10 + 1.0
                        self._method_posteriors[name] = [alpha, beta]

        accuracies = self.get_method_accuracy()

        # Apply default anchoring: primary gets min 0.4 weight
        raw = {
            "primary":       max(0.4, accuracies.get("primary", 0.6)),
            "weighted_avg":  accuracies.get("weighted_avg", 0.2),
            "majority_vote": accuracies.get("majority_vote", 0.2),
        }
        total = sum(raw.values())
        if total == 0.0:
            return dict(self._DEFAULT_WEIGHTS)
        return {k: round(v / total, 4) for k, v in raw.items()}

    def _persist_method_accuracy(self, db) -> None:
        """Store per-method accuracy as a JSON record in the DB."""
        self.get_method_accuracy()
        try:
            db.conn.execute(
                """
                INSERT OR REPLACE INTO ensemble_method_accuracy
                    (method_name, alpha, beta, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                ("primary", *self._method_posteriors["primary"], _now_iso()),
            )
            db.conn.execute(
                """
                INSERT OR REPLACE INTO ensemble_method_accuracy
                    (method_name, alpha, beta, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                ("weighted_avg", *self._method_posteriors["weighted_avg"], _now_iso()),
            )
            db.conn.execute(
                """
                INSERT OR REPLACE INTO ensemble_method_accuracy
                    (method_name, alpha, beta, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                ("majority_vote", *self._method_posteriors["majority_vote"], _now_iso()),
            )
            db.conn.commit()
        except Exception:
            # Table may not exist yet — handled by DB migration
            pass

    def _load_method_accuracy(self, db) -> dict[str, float]:
        """Load stored per-method accuracy from the ensemble_method_accuracy table."""
        try:
            rows = db.conn.execute(
                "SELECT method_name, alpha, beta FROM ensemble_method_accuracy"
            ).fetchall()
            return {
                row[0]: row[1] / (row[1] + row[2])
                for row in rows
                if (row[1] + row[2]) > 0
            }
        except Exception:
            return {}


def _now_iso() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat()
