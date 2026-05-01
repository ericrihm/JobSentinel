"""Autonomous Self-Healing and Self-Iteration Engine for Sentinel.

The AutonomicController is the "immune system" that monitors all subsystems,
detects degradation, and automatically heals without human intervention.

Components:
  CheckpointManager  — snapshot / rollback / diff system state
  RegressionGuard    — CUSUM + EWMA dual monitoring, auto-revert on drop
  SelfIterator       — hill-climbing + simulated annealing improvement loop
  HealthDashboard    — aggregated multi-subsystem health view
  AutonomicController — top-level autonomous loop orchestrating all of the above
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sentinel.db import SentinelDB
from sentinel.flywheel import DetectionFlywheel
from sentinel.innovation import InnovationEngine
from sentinel.shadow import ShadowScorer

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _now_ts() -> float:
    return time.monotonic()


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    """A named snapshot of system state."""
    tag: str
    created_at: str
    signal_weights: dict[str, float]
    pattern_counts: dict[str, int]       # status -> count
    precision_metrics: dict[str, float]  # precision, recall, f1
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Save, restore, and diff system state snapshots.

    Maintains an ordered history of at most *max_checkpoints* entries.
    Each checkpoint captures signal weights, pattern counts, and precision
    metrics so the system can roll back to any prior state with one call.
    """

    def __init__(self, db: SentinelDB, max_checkpoints: int = 20) -> None:
        self.db = db
        self.max_checkpoints = max_checkpoints
        self._history: list[Checkpoint] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, tag: str, flywheel: DetectionFlywheel | None = None) -> Checkpoint:
        """Snapshot current system state and store under *tag*.

        Args:
            tag: Short label such as "pre-weight-update" or "pre-innovation-cycle".
            flywheel: Optional DetectionFlywheel whose weight tracker is captured.
                      When omitted, weights are derived from active DB patterns.

        Returns:
            The newly created Checkpoint.
        """
        # Signal weights
        if flywheel is not None:
            weights = dict(flywheel.weight_tracker.all_weights())
        else:
            weights = self._weights_from_db()

        # Pattern counts by status
        counts: dict[str, int] = {}
        for status in ("active", "candidate", "deprecated"):
            counts[status] = len(self.db.get_patterns(status=status))

        # Precision metrics (computed from DB — fast)
        fw = flywheel or DetectionFlywheel(self.db)
        accuracy = fw.compute_accuracy()
        precision_metrics = {
            "precision": accuracy.get("precision", 0.0),
            "recall": accuracy.get("recall", 0.0),
            "f1": accuracy.get("f1", 0.0),
        }

        cp = Checkpoint(
            tag=tag,
            created_at=_now_iso(),
            signal_weights=weights,
            pattern_counts=counts,
            precision_metrics=precision_metrics,
        )
        self._history.append(cp)
        self._gc()
        logger.debug("Checkpoint saved: tag=%s precision=%.4f", tag, precision_metrics["precision"])
        return cp

    def rollback(self, tag: str, flywheel: DetectionFlywheel | None = None) -> Checkpoint:
        """Restore system state from the most recent checkpoint with *tag*.

        Applies stored weights back to the flywheel's weight tracker and the DB
        patterns table (alpha/beta).

        Args:
            tag: The checkpoint tag to restore.
            flywheel: Optional DetectionFlywheel whose weight tracker is updated.

        Returns:
            The restored Checkpoint.

        Raises:
            KeyError: If no checkpoint with the given tag exists.
        """
        cp = self.get(tag)
        if cp is None:
            raise KeyError(f"No checkpoint found with tag={tag!r}")

        fw = flywheel or DetectionFlywheel(self.db)

        # Apply weights to in-memory tracker
        for signal_name, weight in cp.signal_weights.items():
            # Restore as (alpha, beta) preserving rough evidence mass
            current = fw.weight_tracker.get_posterior(signal_name)
            total_mass = current[0] + current[1]
            new_alpha = weight * total_mass
            new_beta = (1.0 - weight) * total_mass
            fw.weight_tracker._posteriors[signal_name] = [new_alpha, new_beta]

        # Apply to DB patterns
        self._apply_weights_to_db(cp.signal_weights)

        # Invalidate scorer cache
        try:
            from sentinel import scorer
            scorer._reset_learned_weights_cache()
        except Exception:
            pass

        logger.info("Rolled back to checkpoint tag=%s (precision was %.4f)",
                    tag, cp.precision_metrics.get("precision", 0.0))
        return cp

    def get(self, tag: str) -> Checkpoint | None:
        """Return the most recent checkpoint with the given tag, or None."""
        for cp in reversed(self._history):
            if cp.tag == tag:
                return cp
        return None

    def list_checkpoints(self) -> list[dict]:
        """Return summary dicts for all stored checkpoints (newest first)."""
        return [
            {
                "tag": cp.tag,
                "created_at": cp.created_at,
                "precision": cp.precision_metrics.get("precision", 0.0),
                "pattern_counts": cp.pattern_counts,
                "weight_count": len(cp.signal_weights),
            }
            for cp in reversed(self._history)
        ]

    def diff(self, tag_a: str, tag_b: str) -> dict:
        """Diff two checkpoints by tag (most recent of each).

        Returns a dict with:
            - weights_added, weights_removed, weights_changed (by signal name)
            - precision_delta, recall_delta, f1_delta
            - pattern_count_deltas
        """
        cp_a = self.get(tag_a)
        cp_b = self.get(tag_b)
        if cp_a is None:
            raise KeyError(f"No checkpoint with tag={tag_a!r}")
        if cp_b is None:
            raise KeyError(f"No checkpoint with tag={tag_b!r}")

        keys_a = set(cp_a.signal_weights)
        keys_b = set(cp_b.signal_weights)

        added = {k: cp_b.signal_weights[k] for k in keys_b - keys_a}
        removed = {k: cp_a.signal_weights[k] for k in keys_a - keys_b}
        changed: dict[str, dict] = {}
        for k in keys_a & keys_b:
            old = cp_a.signal_weights[k]
            new = cp_b.signal_weights[k]
            if abs(old - new) > 1e-9:
                changed[k] = {"from": old, "to": new, "delta": new - old}

        metric_keys = ("precision", "recall", "f1")
        metric_deltas = {
            f"{m}_delta": round(
                cp_b.precision_metrics.get(m, 0.0) - cp_a.precision_metrics.get(m, 0.0),
                6,
            )
            for m in metric_keys
        }

        count_deltas = {
            status: cp_b.pattern_counts.get(status, 0) - cp_a.pattern_counts.get(status, 0)
            for status in ("active", "candidate", "deprecated")
        }

        return {
            "from_tag": tag_a,
            "to_tag": tag_b,
            "from_at": cp_a.created_at,
            "to_at": cp_b.created_at,
            "weights_added": added,
            "weights_removed": removed,
            "weights_changed": changed,
            **metric_deltas,
            "pattern_count_deltas": count_deltas,
        }

    def gc(self, keep: int | None = None) -> int:
        """Remove old checkpoints, retaining at most *keep* (default max_checkpoints).

        Returns the number of checkpoints removed.
        """
        limit = keep if keep is not None else self.max_checkpoints
        removed = max(0, len(self._history) - limit)
        self._history = self._history[-limit:]
        return removed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gc(self) -> None:
        """Garbage-collect down to max_checkpoints silently."""
        self.gc()

    def _weights_from_db(self) -> dict[str, float]:
        weights: dict[str, float] = {}
        for row in self.db.get_patterns(status="active"):
            alpha = row.get("alpha", 1.0)
            beta = row.get("beta", 1.0)
            total = alpha + beta
            if total > 0:
                weights[row.get("name", "")] = alpha / total
        return weights

    def _apply_weights_to_db(self, weights: dict[str, float]) -> None:
        """Write weight dict back to DB patterns (alpha/beta)."""
        for name, weight in weights.items():
            row = self.db.conn.execute(
                "SELECT * FROM patterns WHERE name = ? OR pattern_id = ?",
                (name, name),
            ).fetchone()
            if row is None:
                continue
            pattern = dict(row)
            total_mass = pattern.get("alpha", 1.0) + pattern.get("beta", 1.0)
            pattern["alpha"] = round(weight * total_mass, 6)
            pattern["beta"] = round((1.0 - weight) * total_mass, 6)
            self.db.save_pattern(pattern)


# ---------------------------------------------------------------------------
# RegressionGuard
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    """Outcome of a regression check after a weight/pattern change."""
    triggered: bool
    metric: str                 # "precision", "recall", or "f1"
    before: float
    after: float
    drop: float
    threshold: float
    cusum_alarm: bool
    ewma_alarm: bool
    reverted: bool
    budget_remaining: int


class _EWMA:
    """Exponentially Weighted Moving Average for gradual drift detection."""

    def __init__(self, alpha: float = 0.1, threshold: float = 0.05) -> None:
        """
        Args:
            alpha: Smoothing factor (0 < alpha < 1); smaller = slower response.
            threshold: Absolute deviation from baseline that triggers an alarm.
        """
        self.alpha = alpha
        self.threshold = threshold
        self._value: float | None = None
        self._baseline: float | None = None

    def update(self, value: float) -> bool:
        """Feed a new observation; return True if alarm is triggered."""
        if self._value is None:
            self._value = value
            self._baseline = value
            return False
        self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        if self._baseline is not None and abs(self._value - self._baseline) > self.threshold:
            return True
        return False

    def set_baseline(self, value: float) -> None:
        self._baseline = value
        self._value = value

    @property
    def current(self) -> float | None:
        return self._value


class RegressionGuard:
    """Dual CUSUM + EWMA monitoring with automatic revert on metric drop.

    After any weight/pattern change the caller should call ``check()`` with
    before/after metrics.  If any metric drops beyond the configured threshold,
    ``check()`` will auto-revert via the CheckpointManager and decrement the
    regression budget.  When the budget is exhausted, all changes are locked.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        flywheel: DetectionFlywheel,
        precision_threshold: float = 0.05,
        recall_threshold: float = 0.05,
        f1_threshold: float = 0.03,
        regression_budget: int = 5,
        cusum_target: float = 0.0,
        cusum_slack: float = 0.1,
        cusum_threshold: float = 4.0,
        ewma_alpha: float = 0.2,
        ewma_threshold: float = 0.05,
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.flywheel = flywheel
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold
        self.f1_threshold = f1_threshold
        self._budget = regression_budget
        self._initial_budget = regression_budget
        self._locked = False

        # Per-metric CUSUM detectors
        from sentinel.flywheel import CUSUMDetector
        self._cusum_precision = CUSUMDetector(
            target=cusum_target, slack=cusum_slack, threshold=cusum_threshold
        )
        self._cusum_f1 = CUSUMDetector(
            target=cusum_target, slack=cusum_slack, threshold=cusum_threshold
        )

        # Per-metric EWMA monitors
        self._ewma_precision = _EWMA(alpha=ewma_alpha, threshold=ewma_threshold)
        self._ewma_f1 = _EWMA(alpha=ewma_alpha, threshold=ewma_threshold)

        self._regression_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def locked(self) -> bool:
        """True when the regression budget is exhausted and changes are locked."""
        return self._locked

    @property
    def budget_remaining(self) -> int:
        return self._budget

    def set_baseline(self, metrics: dict[str, float]) -> None:
        """Set EWMA / CUSUM baselines from *metrics* (precision, recall, f1)."""
        p = metrics.get("precision", 0.5)
        f = metrics.get("f1", 0.5)
        self._ewma_precision.set_baseline(p)
        self._ewma_f1.set_baseline(f)
        # Update CUSUM targets
        from sentinel.flywheel import CUSUMDetector
        self._cusum_precision = CUSUMDetector(
            target=p, slack=self.precision_threshold / 2, threshold=4.0
        )
        self._cusum_f1 = CUSUMDetector(
            target=f, slack=self.f1_threshold / 2, threshold=4.0
        )

    def check(
        self,
        before: dict[str, float],
        after: dict[str, float],
        checkpoint_tag: str = "pre-weight-update",
    ) -> RegressionResult:
        """Compare before/after metrics and auto-revert on regression.

        Args:
            before: Metrics dict before the change (precision, recall, f1).
            after: Metrics dict after the change.
            checkpoint_tag: Tag of the checkpoint to roll back to if regression detected.

        Returns:
            A RegressionResult describing what happened.
        """
        checks = [
            ("precision", self.precision_threshold),
            ("recall", self.recall_threshold),
            ("f1", self.f1_threshold),
        ]

        worst_metric = "precision"
        worst_drop = 0.0
        worst_threshold = self.precision_threshold

        for metric, threshold in checks:
            before_val = before.get(metric, 0.0)
            after_val = after.get(metric, 0.0)
            drop = before_val - after_val
            if drop > worst_drop:
                worst_drop = drop
                worst_metric = metric
                worst_threshold = threshold

        # Update CUSUM / EWMA monitors
        prec_after = after.get("precision", 0.0)
        f1_after = after.get("f1", 0.0)
        cusum_precision_alarm = self._cusum_precision.update(prec_after)
        cusum_f1_alarm = self._cusum_f1.update(f1_after)
        ewma_precision_alarm = self._ewma_precision.update(prec_after)
        ewma_f1_alarm = self._ewma_f1.update(f1_after)

        cusum_alarm = cusum_precision_alarm or cusum_f1_alarm
        ewma_alarm = ewma_precision_alarm or ewma_f1_alarm

        regression_triggered = worst_drop > worst_threshold or cusum_alarm or ewma_alarm
        reverted = False

        if regression_triggered and not self._locked:
            self._budget -= 1
            if self._budget <= 0:
                self._locked = True
                logger.warning("Regression budget exhausted — changes locked")

            # Attempt rollback
            try:
                self.checkpoint_manager.rollback(checkpoint_tag, flywheel=self.flywheel)
                reverted = True
                logger.warning(
                    "Regression detected: %s dropped %.4f (threshold=%.4f). "
                    "Rolled back to checkpoint '%s'. Budget remaining: %d",
                    worst_metric, worst_drop, worst_threshold, checkpoint_tag, self._budget,
                )
            except KeyError:
                logger.warning(
                    "Regression detected but no checkpoint '%s' found. Cannot revert.",
                    checkpoint_tag,
                )

            self._regression_log.append({
                "timestamp": _now_iso(),
                "metric": worst_metric,
                "before": before.get(worst_metric, 0.0),
                "after": after.get(worst_metric, 0.0),
                "drop": worst_drop,
                "cusum_alarm": cusum_alarm,
                "ewma_alarm": ewma_alarm,
                "reverted": reverted,
                "budget_remaining": self._budget,
            })

        return RegressionResult(
            triggered=regression_triggered,
            metric=worst_metric,
            before=before.get(worst_metric, 0.0),
            after=after.get(worst_metric, 0.0),
            drop=worst_drop,
            threshold=worst_threshold,
            cusum_alarm=cusum_alarm,
            ewma_alarm=ewma_alarm,
            reverted=reverted,
            budget_remaining=self._budget,
        )

    def reset_budget(self) -> None:
        """Reset the regression budget (e.g., after a successful improvement cycle)."""
        self._budget = self._initial_budget
        self._locked = False

    def get_log(self) -> list[dict]:
        """Return the full regression event log."""
        return list(self._regression_log)


# ---------------------------------------------------------------------------
# SelfIterator
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """An improvement hypothesis generated from failure analysis."""
    id: str
    description: str
    strategy: str          # maps to InnovationEngine strategy names
    expected_impact: float  # expected precision delta (0-1)
    temperature: float = 1.0  # annealing temperature (higher = bolder)
    result: dict | None = None
    accepted: bool = False
    created_at: str = field(default_factory=_now_iso)


@dataclass
class IterationRecord:
    """Full provenance record for one hill-climbing iteration."""
    iteration: int
    hypothesis: Hypothesis
    before_precision: float
    after_precision: float
    delta: float
    accepted: bool
    reason: str
    timestamp: str = field(default_factory=_now_iso)


class SelfIterator:
    """Hill-climbing + simulated annealing improvement loop.

    Each iteration:
    1. Generate hypotheses from failure analysis (ranked by expected impact).
    2. Execute the top hypothesis via InnovationEngine strategy.
    3. Measure before/after precision.
    4. Accept or revert based on hill-climbing / annealing criteria.
    5. Record full provenance.
    """

    # Annealing schedule: temperature decays by this factor each iteration
    COOLING_RATE: float = 0.95
    # Minimum temperature (prevents annealing from accepting no bad moves)
    MIN_TEMPERATURE: float = 0.01
    # Bold move probability at high temperature
    BOLD_MOVE_THRESHOLD: float = 0.3

    def __init__(
        self,
        db: SentinelDB,
        flywheel: DetectionFlywheel,
        innovation: InnovationEngine,
        checkpoint_manager: CheckpointManager,
        regression_guard: RegressionGuard,
        initial_temperature: float = 1.0,
    ) -> None:
        self.db = db
        self.flywheel = flywheel
        self.innovation = innovation
        self.checkpoint_manager = checkpoint_manager
        self.regression_guard = regression_guard
        self._temperature = initial_temperature
        self._iteration_count = 0
        self._history: list[IterationRecord] = []
        self._best_precision: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    def generate_hypotheses(self) -> list[Hypothesis]:
        """Produce improvement hypotheses from failure analysis.

        Ranks hypotheses by expected impact:
        - High temperature: bold moves (pattern mining, threshold tuning) rank first.
        - Low temperature: conservative moves (weight optimization, FP review) rank first.
        """
        base_hypotheses = [
            Hypothesis(
                id=f"h_{self._iteration_count}_fp_review",
                description="Reduce false positives by downweighting noisy signals",
                strategy="false_positive_review",
                expected_impact=0.03 * (1.0 - self._temperature),
                temperature=self._temperature,
            ),
            Hypothesis(
                id=f"h_{self._iteration_count}_fn_review",
                description="Capture missed scams by reviewing false negatives",
                strategy="false_negative_review",
                expected_impact=0.04 * (1.0 - self._temperature),
                temperature=self._temperature,
            ),
            Hypothesis(
                id=f"h_{self._iteration_count}_weight_opt",
                description="Bayesian weight re-optimization from recent reports",
                strategy="weight_optimization",
                expected_impact=0.05,
                temperature=self._temperature,
            ),
            Hypothesis(
                id=f"h_{self._iteration_count}_pattern_mine",
                description="Mine new scam patterns from user-confirmed reports",
                strategy="pattern_mining",
                expected_impact=0.08 * self._temperature,   # bolder at high temp
                temperature=self._temperature,
            ),
            Hypothesis(
                id=f"h_{self._iteration_count}_threshold",
                description="Tune risk thresholds to recalibrate precision/recall balance",
                strategy="threshold_tuning",
                expected_impact=0.06 * self._temperature,
                temperature=self._temperature,
            ),
            Hypothesis(
                id=f"h_{self._iteration_count}_corr",
                description="Exploit high-lift signal pairs for combo scoring",
                strategy="cross_signal_correlation",
                expected_impact=0.05,
                temperature=self._temperature,
            ),
            Hypothesis(
                id=f"h_{self._iteration_count}_kw",
                description="Expand keyword lists from recent scam language",
                strategy="keyword_expansion",
                expected_impact=0.04,
                temperature=self._temperature,
            ),
        ]
        # Sort: high temperature → bold first; low temperature → conservative first
        base_hypotheses.sort(key=lambda h: h.expected_impact, reverse=True)
        return base_hypotheses

    def run_iteration(self) -> IterationRecord:
        """Execute one hill-climbing iteration.

        Picks the top hypothesis, saves a checkpoint, runs the strategy,
        compares before/after precision, then accepts or reverts.
        """
        self._iteration_count += 1
        hypotheses = self.generate_hypotheses()
        hypothesis = self._pick_hypothesis(hypotheses)

        # Snapshot before state
        before_acc = self.flywheel.compute_accuracy()
        before_precision = before_acc.get("precision", 0.0)

        tag = f"pre-iteration-{self._iteration_count}"
        self.checkpoint_manager.save(tag=tag, flywheel=self.flywheel)

        # Execute strategy via InnovationEngine
        try:
            results = self.innovation.run_cycle(max_strategies=1)
            hypothesis.result = {
                "results": [
                    {
                        "strategy": r.strategy,
                        "success": r.success,
                        "detail": r.detail,
                        "precision_delta": r.precision_delta,
                    }
                    for r in results
                ]
            }
        except Exception as exc:
            hypothesis.result = {"error": str(exc)}
            results = []

        # Measure after state
        after_acc = self.flywheel.compute_accuracy()
        after_precision = after_acc.get("precision", 0.0)
        delta = after_precision - before_precision

        # Acceptance decision (hill-climbing + simulated annealing)
        accepted, reason = self._acceptance_decision(delta)
        hypothesis.accepted = accepted

        if not accepted:
            # Revert to pre-iteration checkpoint
            try:
                self.checkpoint_manager.rollback(tag, flywheel=self.flywheel)
            except KeyError:
                pass
        else:
            if after_precision > self._best_precision:
                self._best_precision = after_precision

        # Cool down temperature
        self._temperature = max(
            self.MIN_TEMPERATURE,
            self._temperature * self.COOLING_RATE,
        )

        record = IterationRecord(
            iteration=self._iteration_count,
            hypothesis=hypothesis,
            before_precision=before_precision,
            after_precision=after_precision,
            delta=delta,
            accepted=accepted,
            reason=reason,
        )
        self._history.append(record)
        logger.info(
            "SelfIterator iteration %d: strategy=%s delta=%.4f accepted=%s temp=%.4f",
            self._iteration_count, hypothesis.strategy, delta, accepted, self._temperature,
        )
        return record

    def run_n_iterations(self, n: int) -> list[IterationRecord]:
        """Run *n* improvement iterations, respecting the regression guard lock."""
        records = []
        for _ in range(n):
            if self.regression_guard.locked:
                logger.warning("SelfIterator: regression budget exhausted — stopping early")
                break
            records.append(self.run_iteration())
        return records

    def get_history(self) -> list[dict]:
        """Return full iteration history with provenance."""
        return [
            {
                "iteration": r.iteration,
                "strategy": r.hypothesis.strategy,
                "expected_impact": r.hypothesis.expected_impact,
                "before_precision": r.before_precision,
                "after_precision": r.after_precision,
                "delta": r.delta,
                "accepted": r.accepted,
                "reason": r.reason,
                "temperature": r.hypothesis.temperature,
                "timestamp": r.timestamp,
            }
            for r in self._history
        ]

    def best_precision(self) -> float:
        """Highest precision achieved across all accepted iterations."""
        return self._best_precision

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_hypothesis(self, hypotheses: list[Hypothesis]) -> Hypothesis:
        """Pick a hypothesis: bold moves at high temperature, greedy at low."""
        if self._temperature > self.BOLD_MOVE_THRESHOLD and len(hypotheses) > 1:
            # Occasionally skip to a bold move (explore)
            if random.random() < self._temperature:
                return hypotheses[0]  # still top, but bold moves sorted higher
        return hypotheses[0]

    def _acceptance_decision(self, delta: float) -> tuple[bool, str]:
        """Apply simulated annealing acceptance criterion.

        - Always accept improvements (delta > 0).
        - Accept mild regressions with probability exp(delta / temperature).
        - Reject large drops (below -0.1 absolute or below annealing probability).
        """
        if delta >= 0.0:
            return True, "improvement"
        # Small regression: accept with annealing probability
        if self._temperature > self.MIN_TEMPERATURE:
            prob = math.exp(delta / self._temperature)
            if random.random() < prob:
                return True, f"accepted_annealing (prob={prob:.4f})"
        return False, f"rejected (delta={delta:.4f}, temp={self._temperature:.4f})"


# ---------------------------------------------------------------------------
# HealthDashboard
# ---------------------------------------------------------------------------

_STATUS_GREEN = "GREEN"
_STATUS_YELLOW = "YELLOW"
_STATUS_RED = "RED"


@dataclass
class SubsystemHealth:
    name: str
    status: str               # GREEN / YELLOW / RED
    score: float              # 0.0 (worst) – 1.0 (best)
    details: dict[str, Any]
    recommendations: list[str]


class HealthDashboard:
    """Aggregate health from all subsystems into a single colour-coded view.

    Also tracks MTBF (mean time between failures) and MTTR (mean time to
    recovery) using a simple rolling event log.
    """

    MTBF_WINDOW = 20   # events to average over

    def __init__(self, db: SentinelDB) -> None:
        self.db = db
        self._failure_events: deque[float] = deque(maxlen=self.MTBF_WINDOW)
        self._recovery_events: deque[float] = deque(maxlen=self.MTBF_WINDOW)
        self._last_failure_ts: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(
        self,
        flywheel: DetectionFlywheel | None = None,
        innovation: InnovationEngine | None = None,
        shadow: ShadowScorer | None = None,
    ) -> dict:
        """Build a full health dashboard snapshot.

        Args:
            flywheel: DetectionFlywheel instance (created if omitted).
            innovation: InnovationEngine instance (created if omitted).
            shadow: ShadowScorer instance (created if omitted).

        Returns:
            Dict with keys: overall_status, overall_score, subsystems (list),
            recommendations, mtbf_seconds, mttr_seconds, checked_at.
        """
        fw = flywheel or DetectionFlywheel(self.db)
        shadow_sc = shadow or ShadowScorer(self.db)

        subsystems: list[SubsystemHealth] = [
            self._check_flywheel(fw),
            self._check_shadow(shadow_sc),
            self._check_patterns(),
            self._check_regression_budget(fw),
        ]

        # Innovation check (optional — may be slow if loading state)
        try:
            innov = innovation or InnovationEngine(db=self.db)
            subsystems.append(self._check_innovation(innov))
        except Exception:
            pass

        scores = [s.score for s in subsystems]
        overall_score = sum(scores) / len(scores) if scores else 0.0

        statuses = [s.status for s in subsystems]
        if _STATUS_RED in statuses:
            overall_status = _STATUS_RED
        elif _STATUS_YELLOW in statuses:
            overall_status = _STATUS_YELLOW
        else:
            overall_status = _STATUS_GREEN

        # Track failure / recovery
        now = _now_ts()
        if overall_status == _STATUS_RED:
            if self._last_failure_ts is None:
                self._failure_events.append(now)
                self._last_failure_ts = now
        else:
            if self._last_failure_ts is not None:
                recovery_duration = now - self._last_failure_ts
                self._recovery_events.append(recovery_duration)
                self._last_failure_ts = None

        # Aggregate recommendations
        all_recs: list[str] = []
        for s in subsystems:
            all_recs.extend(s.recommendations)

        return {
            "overall_status": overall_status,
            "overall_score": round(overall_score, 4),
            "subsystems": [
                {
                    "name": s.name,
                    "status": s.status,
                    "score": round(s.score, 4),
                    "details": s.details,
                    "recommendations": s.recommendations,
                }
                for s in subsystems
            ],
            "recommendations": list(dict.fromkeys(all_recs)),  # deduplicated
            "mtbf_seconds": self._mtbf(),
            "mttr_seconds": self._mttr(),
            "checked_at": _now_iso(),
        }

    def record_failure(self) -> None:
        """Manually record a failure event (for testing / external triggers)."""
        now = _now_ts()
        self._failure_events.append(now)
        self._last_failure_ts = now

    def record_recovery(self) -> None:
        """Manually record a recovery after a failure."""
        if self._last_failure_ts is not None:
            duration = _now_ts() - self._last_failure_ts
            self._recovery_events.append(duration)
            self._last_failure_ts = None

    # ------------------------------------------------------------------
    # Subsystem checks
    # ------------------------------------------------------------------

    def _check_flywheel(self, fw: DetectionFlywheel) -> SubsystemHealth:
        health = fw.get_health()
        precision = health.get("precision", 0.0)
        regression = health.get("regression_alarm", False)
        grade = health.get("grade", "F")

        if regression or precision < 0.4:
            status = _STATUS_RED
            score = max(0.0, precision)
            recs = [
                "Run flywheel.evolve_patterns() to promote/deprecate patterns",
                "Investigate CUSUM regression alarm — recent labelled data may have shifted",
            ]
        elif precision < 0.7:
            status = _STATUS_YELLOW
            score = 0.4 + precision * 0.4
            recs = ["Consider running an innovation cycle to improve precision"]
        else:
            status = _STATUS_GREEN
            score = min(1.0, precision)
            recs = []

        return SubsystemHealth(
            name="flywheel",
            status=status,
            score=score,
            details={
                "precision": precision,
                "recall": health.get("recall", 0.0),
                "f1": health.get("f1", 0.0),
                "grade": grade,
                "regression_alarm": regression,
                "cycle_count": health.get("cycle_count", 0),
                "active_patterns": health.get("active_patterns", 0),
            },
            recommendations=recs,
        )

    def _check_shadow(self, shadow: ShadowScorer) -> SubsystemHealth:
        status_data = shadow.get_status()
        active = status_data.get("active", False)
        jobs_eval = status_data.get("evaluation", {}).get("jobs_evaluated", 0)
        improvement = status_data.get("evaluation", {}).get("improvement") or 0.0

        if active and jobs_eval > 0 and improvement < -0.05:
            status = _STATUS_YELLOW
            score = 0.5
            recs = ["Shadow weights underperforming — consider rejecting the shadow run"]
        else:
            status = _STATUS_GREEN
            score = 1.0
            recs = []

        return SubsystemHealth(
            name="shadow_scorer",
            status=status,
            score=score,
            details={
                "active": active,
                "jobs_evaluated": jobs_eval,
                "improvement": improvement,
                "run_id": status_data.get("run_id"),
            },
            recommendations=recs,
        )

    def _check_patterns(self) -> SubsystemHealth:
        active = len(self.db.get_patterns(status="active"))
        candidate = len(self.db.get_patterns(status="candidate"))
        deprecated = len(self.db.get_patterns(status="deprecated"))
        total = active + candidate + deprecated

        if active == 0:
            status = _STATUS_RED
            score = 0.0
            recs = ["No active patterns — run innovate cycle to generate candidates"]
        elif candidate > active * 2:
            status = _STATUS_YELLOW
            score = 0.6
            recs = [f"{candidate} candidate patterns waiting — run evolve to promote/deprecate"]
        else:
            status = _STATUS_GREEN
            score = 1.0
            recs = []

        return SubsystemHealth(
            name="pattern_store",
            status=status,
            score=score,
            details={
                "active": active,
                "candidate": candidate,
                "deprecated": deprecated,
                "total": total,
            },
            recommendations=recs,
        )

    def _check_innovation(self, innov: InnovationEngine) -> SubsystemHealth:
        report = innov.get_report()
        total_cycles = report.get("total_cycles", 0)
        precision = report.get("precision", 0.0)

        if total_cycles == 0:
            status = _STATUS_YELLOW
            score = 0.5
            recs = ["Innovation engine has never run — execute run_cycle() to initialise"]
        elif precision < 0.5:
            status = _STATUS_YELLOW
            score = 0.4
            recs = ["Innovation precision is low — inspect failed strategy arms"]
        else:
            status = _STATUS_GREEN
            score = min(1.0, 0.5 + precision * 0.5)
            recs = []

        return SubsystemHealth(
            name="innovation",
            status=status,
            score=score,
            details={
                "total_cycles": total_cycles,
                "precision": precision,
                "flywheel_grade": report.get("flywheel_grade", "?"),
            },
            recommendations=recs,
        )

    def _check_regression_budget(self, fw: DetectionFlywheel) -> SubsystemHealth:
        """Lightweight check: just inspect the flywheel's regression alarm."""
        regression = fw.get_health().get("regression_alarm", False)

        if regression:
            status = _STATUS_YELLOW
            score = 0.5
            recs = ["Regression alarm active — verify recent pattern changes"]
        else:
            status = _STATUS_GREEN
            score = 1.0
            recs = []

        return SubsystemHealth(
            name="regression_guard",
            status=status,
            score=score,
            details={"regression_alarm": regression},
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # MTBF / MTTR
    # ------------------------------------------------------------------

    def _mtbf(self) -> float | None:
        """Mean time between failure events (seconds), or None if < 2 events."""
        events = list(self._failure_events)
        if len(events) < 2:
            return None
        gaps = [events[i + 1] - events[i] for i in range(len(events) - 1)]
        return round(sum(gaps) / len(gaps), 2)

    def _mttr(self) -> float | None:
        """Mean time to recovery (seconds), or None if no recovery events."""
        events = list(self._recovery_events)
        if not events:
            return None
        return round(sum(events) / len(events), 2)


# ---------------------------------------------------------------------------
# AutonomicController
# ---------------------------------------------------------------------------

@dataclass
class HealthCycle:
    """Result of one autonomic health check + optional heal cycle."""
    cycle_number: int
    dashboard: dict
    overall_status: str
    healed: bool
    improvement_ran: bool
    backoff_seconds: float
    timestamp: str = field(default_factory=_now_iso)


class AutonomicController:
    """Top-level autonomous loop — the immune system of Sentinel.

    Continuously:
    1. Check health of all subsystems via HealthDashboard.
    2. If YELLOW or RED — trigger healing (flywheel cycle, innovation, regression check).
    3. After failed healing attempts, exponentially back off.
    4. Schedule self-improvement cycles based on system health.
    5. Checkpoint before every change; auto-rollback if RegressionGuard fires.
    """

    BASE_BACKOFF: float = 1.0     # seconds (tests run fast)
    MAX_BACKOFF: float = 3600.0   # 1 hour
    IMPROVEMENT_INTERVAL_HEALTHY: int = 10    # run improvement every N healthy cycles
    IMPROVEMENT_INTERVAL_DEGRADED: int = 3    # more frequent when degraded

    def __init__(
        self,
        db: SentinelDB | None = None,
        flywheel: DetectionFlywheel | None = None,
        innovation: InnovationEngine | None = None,
        shadow: ShadowScorer | None = None,
        max_checkpoints: int = 20,
        regression_budget: int = 5,
        precision_threshold: float = 0.05,
        initial_temperature: float = 1.0,
    ) -> None:
        self.db = db or SentinelDB()
        self.flywheel = flywheel or DetectionFlywheel(self.db)
        # Innovation shares the flywheel instance to avoid double DB connections
        self.innovation = innovation or InnovationEngine(db=self.db)
        self.shadow = shadow or ShadowScorer(self.db)

        self.checkpoint_manager = CheckpointManager(
            self.db, max_checkpoints=max_checkpoints
        )
        self.regression_guard = RegressionGuard(
            checkpoint_manager=self.checkpoint_manager,
            flywheel=self.flywheel,
            precision_threshold=precision_threshold,
            regression_budget=regression_budget,
        )
        self.self_iterator = SelfIterator(
            db=self.db,
            flywheel=self.flywheel,
            innovation=self.innovation,
            checkpoint_manager=self.checkpoint_manager,
            regression_guard=self.regression_guard,
            initial_temperature=initial_temperature,
        )
        self.dashboard = HealthDashboard(self.db)

        self._cycle_count: int = 0
        self._consecutive_failures: int = 0
        self._backoff: float = self.BASE_BACKOFF
        self._cycle_history: list[HealthCycle] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def run_cycle(self) -> HealthCycle:
        """Execute one autonomic cycle.

        1. Take a pre-cycle checkpoint.
        2. Check health dashboard.
        3. If degraded: heal (flywheel cycle + regression check).
        4. If GREEN: maybe schedule an improvement iteration.
        5. Update backoff and persist cycle record.
        """
        self._cycle_count += 1

        # Pre-cycle checkpoint
        pre_metrics = self.flywheel.compute_accuracy()
        self.checkpoint_manager.save(tag="pre-weight-update", flywheel=self.flywheel)
        self.regression_guard.set_baseline(pre_metrics)

        # Health snapshot
        dash = self.dashboard.snapshot(
            flywheel=self.flywheel,
            shadow=self.shadow,
        )
        overall_status = dash["overall_status"]

        healed = False
        improvement_ran = False

        if overall_status in (_STATUS_YELLOW, _STATUS_RED):
            healed = self._heal(pre_metrics)
        elif overall_status == _STATUS_GREEN:
            # Schedule self-improvement based on cycle cadence
            interval = self.IMPROVEMENT_INTERVAL_HEALTHY
            if self._cycle_count % interval == 0:
                improvement_ran = self._improve()

        cycle = HealthCycle(
            cycle_number=self._cycle_count,
            dashboard=dash,
            overall_status=overall_status,
            healed=healed,
            improvement_ran=improvement_ran,
            backoff_seconds=self._backoff,
        )
        self._cycle_history.append(cycle)

        # Update backoff
        if overall_status == _STATUS_RED and not healed:
            self._consecutive_failures += 1
            self._backoff = min(
                self.MAX_BACKOFF,
                self.BASE_BACKOFF * (2 ** self._consecutive_failures),
            )
        else:
            self._consecutive_failures = 0
            self._backoff = self.BASE_BACKOFF

        logger.info(
            "Autonomic cycle %d: status=%s healed=%s improved=%s backoff=%.1fs",
            self._cycle_count, overall_status, healed, improvement_ran, self._backoff,
        )
        return cycle

    def run_n_cycles(self, n: int) -> list[HealthCycle]:
        """Run *n* autonomic cycles and return their results."""
        return [self.run_cycle() for _ in range(n)]

    def get_status(self) -> dict:
        """Return a concise status summary."""
        last = self._cycle_history[-1] if self._cycle_history else None
        return {
            "cycle_count": self._cycle_count,
            "consecutive_failures": self._consecutive_failures,
            "current_backoff_seconds": self._backoff,
            "regression_guard_locked": self.regression_guard.locked,
            "regression_budget_remaining": self.regression_guard.budget_remaining,
            "iterator_temperature": self.self_iterator.temperature,
            "iterator_best_precision": self.self_iterator.best_precision(),
            "last_cycle": {
                "status": last.overall_status if last else None,
                "healed": last.healed if last else None,
                "timestamp": last.timestamp if last else None,
            },
        }

    def get_cycle_history(self) -> list[dict]:
        """Return summary of all past cycles."""
        return [
            {
                "cycle_number": c.cycle_number,
                "overall_status": c.overall_status,
                "healed": c.healed,
                "improvement_ran": c.improvement_ran,
                "backoff_seconds": c.backoff_seconds,
                "timestamp": c.timestamp,
            }
            for c in self._cycle_history
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _heal(self, pre_metrics: dict) -> bool:
        """Run healing actions when subsystems are degraded.

        1. Run the flywheel cycle (evolve patterns, CUSUM regression).
        2. Check regression guard — if triggered, revert automatically.
        3. Return True if healing improved the system.
        """
        if self.regression_guard.locked:
            logger.warning("Autonomic heal skipped: regression budget exhausted")
            return False

        try:
            self.checkpoint_manager.save(tag="pre-innovation-cycle", flywheel=self.flywheel)
            self.flywheel.run_cycle()

            post_metrics = self.flywheel.compute_accuracy()
            reg_result = self.regression_guard.check(
                before=pre_metrics,
                after=post_metrics,
                checkpoint_tag="pre-weight-update",
            )

            if reg_result.triggered and reg_result.reverted:
                logger.info("Heal cycle: regression detected and reverted")
                return False

            # Run one improvement iteration during healing
            interval = self.IMPROVEMENT_INTERVAL_DEGRADED
            if self._cycle_count % interval == 0:
                self._improve()

            return True
        except Exception:
            logger.exception("Heal cycle raised exception (non-fatal)")
            return False

    def _improve(self) -> bool:
        """Run one SelfIterator improvement iteration."""
        if self.regression_guard.locked:
            return False
        try:
            record = self.self_iterator.run_iteration()
            return record.accepted
        except Exception:
            logger.exception("SelfIterator iteration raised exception (non-fatal)")
            return False
