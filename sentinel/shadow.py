"""Shadow Scoring / A/B weight testing for safe weight evolution.

Instead of applying optimised weights immediately, candidate weights are
evaluated in shadow mode alongside the current primary weights.  Only when
the shadow weights demonstrably outperform the primary weights over a
configurable number of jobs are they promoted.

Usage in the flywheel:
    shadow = ShadowScorer(db)
    shadow.propose_weights(new_weights)
    ...
    shadow.evaluate(db, n_jobs=50)
    if shadow.should_promote():
        shadow.promote()
    else:
        shadow.reject()
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from sentinel.db import SentinelDB
from sentinel.models import ScamSignal, SignalCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (mirror scorer.py log-odds logic so we can score with arbitrary
# weight overrides without touching the module-level cache)
# ---------------------------------------------------------------------------

def _score_signals_with_weights(
    signals: list[ScamSignal],
    weights: dict[str, float],
) -> tuple[float, float]:
    """Re-implementation of scorer.score_signals using explicit *weights*.

    Returns (scam_score, confidence) exactly as the primary scorer does,
    but uses the provided weight dict instead of the DB-backed learned
    weights.
    """
    if not signals:
        return 0.0, 0.0

    _POSITIVE = SignalCategory.POSITIVE

    log_odds = 0.0
    for s in signals:
        w = weights.get(s.name, s.weight)
        w = max(1e-6, min(1.0 - 1e-6, w))
        if s.category == _POSITIVE:
            log_odds -= math.log((1.0 - w) / w)
        else:
            log_odds += math.log(w / (1.0 - w))

    scam_score = 1.0 / (1.0 + math.exp(-log_odds))

    n_scam = sum(1 for s in signals if s.category != _POSITIVE)
    n_pos = sum(1 for s in signals if s.category == _POSITIVE)
    total = len(signals)
    base_conf = 1.0 - math.exp(-0.3 * total)
    if total > 0:
        majority = max(n_scam, n_pos)
        agreement = majority / total
    else:
        agreement = 1.0

    confidence = round(base_conf * agreement, 4)
    scam_score = round(scam_score, 4)
    return scam_score, confidence


def _load_primary_weights(db: SentinelDB) -> dict[str, float]:
    """Load the current primary (active) weights from the DB patterns table."""
    weights: dict[str, float] = {}
    rows = db.get_patterns(status="active")
    for row in rows:
        obs = row.get("observations", 0)
        if obs < 10:
            continue
        alpha = row.get("alpha", 1.0)
        beta = row.get("beta", 1.0)
        total = alpha + beta
        if total <= 0:
            continue
        pid = row.get("pattern_id", "")
        name = row.get("name", "")
        bayesian_w = alpha / total
        if pid:
            weights[pid] = bayesian_w
        if name and name != pid:
            weights[name] = bayesian_w
    return weights


# ---------------------------------------------------------------------------
# DualScoreResult
# ---------------------------------------------------------------------------

@dataclass
class DualScoreResult:
    """Result of scoring a single job with both primary and shadow weights."""
    primary_score: float = 0.0
    primary_confidence: float = 0.0
    shadow_score: float = 0.0
    shadow_confidence: float = 0.0
    signal_count: int = 0


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Aggregate result of evaluating shadow weights against the baseline."""
    baseline_precision: float = 0.0
    shadow_precision: float = 0.0
    jobs_evaluated: int = 0
    baseline_correct: int = 0
    shadow_correct: int = 0
    improvement: float = 0.0


# ---------------------------------------------------------------------------
# ShadowScorer
# ---------------------------------------------------------------------------

class ShadowScorer:
    """Runs candidate weights in shadow mode alongside the primary scorer.

    Lifecycle:
        1. ``propose_weights(candidate)``  -- starts a shadow run
        2. ``dual_score(signals)``         -- scores a job with both weight sets
        3. ``evaluate(db, n_jobs)``         -- bulk evaluate on recent jobs
        4. ``should_promote()``            -- check if shadow beats primary
        5. ``promote()`` / ``reject()``    -- finalise the run
    """

    # Defaults for promotion criteria
    DEFAULT_MIN_IMPROVEMENT: float = 0.02
    DEFAULT_MIN_JOBS: int = 30

    def __init__(self, db: SentinelDB) -> None:
        self.db = db
        self._candidate_weights: dict[str, float] | None = None
        self._primary_weights: dict[str, float] = {}
        self._run_id: int | None = None
        self._evaluation: EvaluationResult | None = None

        # Check for an existing active shadow run in the DB
        active = db.get_active_shadow_run()
        if active:
            self._run_id = active["id"]
            self._candidate_weights = active.get("candidate_weights", {})
            self._primary_weights = _load_primary_weights(db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True if a shadow run is in progress."""
        return self._candidate_weights is not None and self._run_id is not None

    def propose_weights(self, candidate_weights: dict[str, float]) -> int:
        """Start a new shadow evaluation period with *candidate_weights*.

        If a shadow run is already active it will be rejected first.
        Returns the shadow run id.
        """
        if self.active:
            self.reject()

        self._primary_weights = _load_primary_weights(self.db)
        self._candidate_weights = dict(candidate_weights)
        self._evaluation = None
        self._run_id = self.db.insert_shadow_run(candidate_weights)
        logger.info("Shadow run %d started with %d candidate weights", self._run_id, len(candidate_weights))
        return self._run_id

    def dual_score(self, signals: list[ScamSignal]) -> DualScoreResult:
        """Score a set of signals with both primary and shadow weights.

        Returns a ``DualScoreResult`` containing both scores.  If no shadow
        run is active the shadow fields mirror the primary.
        """
        primary_score, primary_conf = _score_signals_with_weights(signals, self._primary_weights)

        if self._candidate_weights is not None:
            shadow_score, shadow_conf = _score_signals_with_weights(signals, self._candidate_weights)
        else:
            shadow_score, shadow_conf = primary_score, primary_conf

        return DualScoreResult(
            primary_score=primary_score,
            primary_confidence=primary_conf,
            shadow_score=shadow_score,
            shadow_confidence=shadow_conf,
            signal_count=len(signals),
        )

    def evaluate(self, n_jobs: int = 50) -> EvaluationResult:
        """Re-score recent user-reported jobs with both weight sets.

        For each job that has a user report, compares the predicted risk
        against the user verdict and tallies correct predictions for both
        the primary and shadow weight sets.

        Returns an ``EvaluationResult`` and persists the metrics to the
        shadow run record.
        """
        reports = self.db.get_reports(limit=n_jobs)
        if not reports:
            self._evaluation = EvaluationResult()
            return self._evaluation

        baseline_correct = 0
        shadow_correct = 0
        evaluated = 0

        for report in reports:
            url = report.get("url", "")
            if not url:
                continue
            job = self.db.get_job(url)
            if not job:
                continue

            # Reconstruct signals from the stored JSON
            signals_raw = job.get("signals_json", "[]")
            try:
                sig_dicts = json.loads(signals_raw) if isinstance(signals_raw, str) else signals_raw
            except (json.JSONDecodeError, TypeError):
                continue

            signals = _dicts_to_signals(sig_dicts)
            if not signals:
                continue

            is_scam = bool(report.get("is_scam"))
            dual = self.dual_score(signals)

            # A prediction is "correct" when score >= 0.5 matches is_scam
            primary_predicted_scam = dual.primary_score >= 0.5
            shadow_predicted_scam = dual.shadow_score >= 0.5

            if primary_predicted_scam == is_scam:
                baseline_correct += 1
            if shadow_predicted_scam == is_scam:
                shadow_correct += 1

            evaluated += 1

        baseline_precision = baseline_correct / evaluated if evaluated > 0 else 0.0
        shadow_precision = shadow_correct / evaluated if evaluated > 0 else 0.0

        self._evaluation = EvaluationResult(
            baseline_precision=round(baseline_precision, 4),
            shadow_precision=round(shadow_precision, 4),
            jobs_evaluated=evaluated,
            baseline_correct=baseline_correct,
            shadow_correct=shadow_correct,
            improvement=round(shadow_precision - baseline_precision, 4),
        )

        # Persist to DB
        if self._run_id is not None:
            self.db.update_shadow_run(self._run_id, {
                "baseline_precision": self._evaluation.baseline_precision,
                "shadow_precision": self._evaluation.shadow_precision,
                "jobs_evaluated": evaluated,
            })

        logger.info(
            "Shadow evaluation: baseline=%.4f shadow=%.4f improvement=%.4f jobs=%d",
            self._evaluation.baseline_precision,
            self._evaluation.shadow_precision,
            self._evaluation.improvement,
            evaluated,
        )
        return self._evaluation

    def should_promote(
        self,
        min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
        min_jobs: int = DEFAULT_MIN_JOBS,
    ) -> bool:
        """Return True if the shadow weights should replace the primary.

        Criteria:
        - At least *min_jobs* have been evaluated.
        - Shadow precision exceeds baseline by at least *min_improvement*.
        """
        if self._evaluation is None:
            return False
        if self._evaluation.jobs_evaluated < min_jobs:
            return False
        return self._evaluation.improvement >= min_improvement

    def promote(self) -> dict[str, Any]:
        """Write the shadow weights as the new primary weights.

        Updates each pattern's alpha/beta in the DB so the primary scorer
        picks them up on the next cache reload.  Marks the shadow run as
        promoted.
        """
        if not self.active or self._candidate_weights is None:
            return {"promoted": False, "reason": "no active shadow run"}

        promoted_patterns: list[str] = []
        for pattern_id, weight in self._candidate_weights.items():
            # Convert weight to alpha/beta with the same total mass as current
            row = self.db.conn.execute(
                "SELECT * FROM patterns WHERE pattern_id = ? OR name = ?",
                (pattern_id, pattern_id),
            ).fetchone()
            if row is None:
                continue
            pattern = dict(row)
            old_alpha = pattern.get("alpha", 1.0)
            old_beta = pattern.get("beta", 1.0)
            total_mass = old_alpha + old_beta

            # Set new alpha/beta preserving total evidence mass
            new_alpha = weight * total_mass
            new_beta = (1.0 - weight) * total_mass
            pattern["alpha"] = round(new_alpha, 6)
            pattern["beta"] = round(new_beta, 6)
            self.db.save_pattern(pattern)
            promoted_patterns.append(pattern_id)

        if self._run_id is not None:
            self.db.promote_shadow_run(self._run_id)

        # Invalidate the learned-weight cache
        from sentinel import scorer
        scorer._reset_learned_weights_cache()

        result = {
            "promoted": True,
            "run_id": self._run_id,
            "patterns_updated": promoted_patterns,
            "evaluation": {
                "baseline_precision": self._evaluation.baseline_precision if self._evaluation else 0.0,
                "shadow_precision": self._evaluation.shadow_precision if self._evaluation else 0.0,
                "improvement": self._evaluation.improvement if self._evaluation else 0.0,
                "jobs_evaluated": self._evaluation.jobs_evaluated if self._evaluation else 0,
            },
        }

        self._reset()
        logger.info("Shadow weights promoted: %s", promoted_patterns)
        return result

    def reject(self) -> dict[str, Any]:
        """Discard shadow weights without applying them."""
        run_id = self._run_id
        evaluation = self._evaluation

        if run_id is not None:
            self.db.reject_shadow_run(run_id)

        result = {
            "rejected": True,
            "run_id": run_id,
            "evaluation": {
                "baseline_precision": evaluation.baseline_precision if evaluation else 0.0,
                "shadow_precision": evaluation.shadow_precision if evaluation else 0.0,
                "improvement": evaluation.improvement if evaluation else 0.0,
                "jobs_evaluated": evaluation.jobs_evaluated if evaluation else 0,
            },
        }

        self._reset()
        logger.info("Shadow run %s rejected", run_id)
        return result

    def get_status(self) -> dict[str, Any]:
        """Return current shadow scorer status for monitoring."""
        return {
            "active": self.active,
            "run_id": self._run_id,
            "candidate_weight_count": len(self._candidate_weights) if self._candidate_weights else 0,
            "evaluation": {
                "baseline_precision": self._evaluation.baseline_precision if self._evaluation else None,
                "shadow_precision": self._evaluation.shadow_precision if self._evaluation else None,
                "improvement": self._evaluation.improvement if self._evaluation else None,
                "jobs_evaluated": self._evaluation.jobs_evaluated if self._evaluation else 0,
            },
            "history": self.db.get_shadow_history(limit=5),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Clear the in-memory shadow state."""
        self._candidate_weights = None
        self._run_id = None
        self._evaluation = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dicts_to_signals(sig_dicts: list[dict]) -> list[ScamSignal]:
    """Convert stored signal dicts back into ScamSignal objects."""
    signals: list[ScamSignal] = []
    for sd in sig_dicts:
        if not isinstance(sd, dict):
            continue
        name = sd.get("name", "")
        if not name:
            continue
        cat_str = sd.get("category", "warning")
        try:
            category = SignalCategory(cat_str)
        except ValueError:
            category = SignalCategory.WARNING
        signals.append(ScamSignal(
            name=name,
            category=category,
            weight=sd.get("weight", 0.5),
            confidence=sd.get("confidence", 0.5),
            detail=sd.get("detail", ""),
            evidence=sd.get("evidence", ""),
        ))
    return signals
