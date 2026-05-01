"""Self-improving detection flywheel: INGEST → SCORE → VALIDATE → LEARN → EVOLVE."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from sentinel.db import SentinelDB
from sentinel.models import ScamPattern, UserReport, ValidationResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Thompson-Sampling weight tracker
# ---------------------------------------------------------------------------

class SignalWeightTracker:
    """Per-signal Bayesian Beta-Binomial posterior tracker.

    Each signal name maps to (alpha, beta).  On a true-positive report alpha
    is incremented; on a false-positive beta is incremented.  The expected
    weight is alpha / (alpha + beta).
    """

    def __init__(self) -> None:
        # signal_name -> [alpha, beta]
        self._posteriors: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, signal_name: str, is_true_positive: bool) -> None:
        """Apply one observation to the posterior for *signal_name*."""
        alpha, beta = self._get(signal_name)
        if is_true_positive:
            alpha += 1.0
        else:
            beta += 1.0
        self._posteriors[signal_name] = [alpha, beta]

    def expected_weight(self, signal_name: str) -> float:
        """E[theta] = alpha / (alpha + beta) for the Beta posterior."""
        alpha, beta = self._get(signal_name)
        return alpha / (alpha + beta)

    def sample(self, signal_name: str) -> float:
        """Thompson sample: draw one value from Beta(alpha, beta).

        Uses the Johnk method so we stay stdlib-only.
        """
        alpha, beta = self._get(signal_name)
        return self._beta_sample(alpha, beta)

    def all_weights(self) -> dict[str, float]:
        return {name: self.expected_weight(name) for name in self._posteriors}

    def get_posterior(self, signal_name: str) -> tuple[float, float]:
        return tuple(self._get(signal_name))  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, signal_name: str) -> list[float]:
        if signal_name not in self._posteriors:
            self._posteriors[signal_name] = [1.0, 1.0]  # flat prior
        return self._posteriors[signal_name]

    @staticmethod
    def _beta_sample(alpha: float, beta: float) -> float:
        """Sample from Beta(alpha, beta) using the Johnk / Cheng method.

        Uses only the `math` module — no numpy/scipy required.
        """
        import random

        # Gamma-based sampler: X = Gamma(alpha) / (Gamma(alpha) + Gamma(beta))
        # We use Marsaglia & Tsang's method for Gamma > 1, and the scale transform
        # for Gamma < 1.
        x = _gamma_sample(alpha)
        y = _gamma_sample(beta)
        total = x + y
        if total == 0.0:
            return 0.5
        return x / total


def _gamma_sample(shape: float) -> float:
    """Sample from Gamma(shape, scale=1) — stdlib only (Marsaglia–Tsang)."""
    import random

    if shape < 1.0:
        # Boost trick: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        return _gamma_sample(shape + 1.0) * (random.random() ** (1.0 / shape))

    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        z = _normal_sample()
        v = 1.0 + c * z
        if v <= 0.0:
            continue
        v = v ** 3
        u = random.random()
        if u < 1.0 - 0.0331 * (z ** 2) ** 2:
            return d * v
        if math.log(u) < 0.5 * z * z + d * (1.0 - v + math.log(v)):
            return d * v


def _normal_sample() -> float:
    """Box-Muller standard normal — stdlib only."""
    import math
    import random

    while True:
        u1 = random.random()
        u2 = random.random()
        if u1 > 0.0:
            return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# CUSUM change-point detection
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """One-sided CUSUM for detecting a *decrease* in precision (regression)."""

    def __init__(self, target: float = 0.0, slack: float = 0.5, threshold: float = 5.0) -> None:
        self.target = target       # reference mean (set after warmup)
        self.slack = slack         # allowable slack (half the tolerated shift)
        self.threshold = threshold # alarm threshold (h)
        self._s_neg: float = 0.0   # lower CUSUM statistic

    def update(self, value: float) -> bool:
        """Return True if a regression alarm is raised."""
        # Normalised deviation: positive = improvement, negative = regression
        deviation = value - (self.target - self.slack)
        self._s_neg = max(0.0, self._s_neg - deviation)
        return self._s_neg >= self.threshold

    def reset(self) -> None:
        self._s_neg = 0.0

    @property
    def statistic(self) -> float:
        return self._s_neg


# ---------------------------------------------------------------------------
# Main flywheel
# ---------------------------------------------------------------------------

class DetectionFlywheel:
    """Self-improving scam detection with Bayesian learning.

    Loop: INGEST → SCORE → VALIDATE → LEARN → EVOLVE

    - LEARN: User reports update signal weights (Thompson Sampling).
    - EVOLVE: Periodic pattern lifecycle management.
      - Promote high-precision patterns (candidate → active).
      - Deprecate low-precision patterns (active → deprecated).
      - CUSUM regression detection on overall accuracy.
    """

    # Thresholds for pattern lifecycle decisions
    PROMOTE_PRECISION_THRESHOLD: float = 0.80
    PROMOTE_MIN_OBSERVATIONS: int = 10
    DEPRECATE_PRECISION_THRESHOLD: float = 0.30
    DEPRECATE_MIN_OBSERVATIONS: int = 20

    def __init__(self, db: SentinelDB | None = None) -> None:
        self.db = db or SentinelDB()
        self.weight_tracker = SignalWeightTracker()
        self._cusum = CUSUMDetector(target=0.0, slack=0.5, threshold=5.0)
        self._cycle_count: int = 0
        # Initialise CUSUM baseline from existing accuracy if available
        self._init_cusum_baseline()

    # ------------------------------------------------------------------
    # LEARN
    # ------------------------------------------------------------------

    def learn_from_report(self, report: UserReport, result: ValidationResult) -> dict:
        """Update signal weights based on a user-submitted verdict.

        For each signal that fired in *result*:
        - If report.is_scam is True (true positive), increment alpha.
        - If report.is_scam is False (false positive), increment beta.
        Returns a summary of what was updated.
        """
        is_tp = report.is_scam
        updated_signals: list[str] = []

        for signal in result.signals:
            self.weight_tracker.update(signal.name, is_true_positive=is_tp)
            self.db.update_pattern_stats(signal.name, is_true_positive=is_tp)
            updated_signals.append(signal.name)

        return {
            "report_url": report.url,
            "is_scam": report.is_scam,
            "signals_updated": updated_signals,
            "signal_count": len(updated_signals),
            "updated_at": _now_iso(),
        }

    # ------------------------------------------------------------------
    # EVOLVE
    # ------------------------------------------------------------------

    def evolve_patterns(self) -> dict:
        """Run one pattern lifecycle pass.

        Returns a dict with counts of promoted, deprecated, and retained patterns.
        """
        promoted: list[str] = []
        deprecated: list[str] = []
        retained: list[str] = []

        # --- Promote candidates ---
        candidates = self.db.get_patterns(status="candidate")
        for row in candidates:
            obs = row.get("observations", 0)
            tp = row.get("true_positives", 0)
            fp = row.get("false_positives", 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if obs >= self.PROMOTE_MIN_OBSERVATIONS and precision >= self.PROMOTE_PRECISION_THRESHOLD:
                self.db.save_pattern({**row, "status": "active"})
                promoted.append(row["pattern_id"])
            else:
                retained.append(row["pattern_id"])

        # --- Deprecate low-performing active patterns ---
        active = self.db.get_patterns(status="active")
        for row in active:
            obs = row.get("observations", 0)
            tp = row.get("true_positives", 0)
            fp = row.get("false_positives", 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            if (
                obs >= self.DEPRECATE_MIN_OBSERVATIONS
                and precision is not None
                and precision < self.DEPRECATE_PRECISION_THRESHOLD
            ):
                self.db.save_pattern({**row, "status": "deprecated"})
                deprecated.append(row["pattern_id"])
            else:
                retained.append(row["pattern_id"])

        return {
            "promoted": promoted,
            "deprecated": deprecated,
            "retained_count": len(retained),
            "evolved_at": _now_iso(),
        }

    # ------------------------------------------------------------------
    # COMPUTE ACCURACY
    # ------------------------------------------------------------------

    def compute_accuracy(self) -> dict:
        """Calculate precision, recall, F1, and accuracy from user reports."""
        reports = self.db.get_reports(limit=10_000)
        total = len(reports)
        if total == 0:
            return {
                "total": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
            }

        tp = sum(1 for r in reports if r.get("is_scam") == 1 and r.get("was_correct") == 1)
        fp = sum(1 for r in reports if r.get("is_scam") == 0 and r.get("was_correct") == 0)
        tn = sum(1 for r in reports if r.get("is_scam") == 0 and r.get("was_correct") == 1)
        fn = sum(1 for r in reports if r.get("is_scam") == 1 and r.get("was_correct") == 0)

        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "total": total,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

    # ------------------------------------------------------------------
    # CUSUM REGRESSION DETECTION
    # ------------------------------------------------------------------

    def detect_regression(self, window: int = 50) -> dict:
        """Run CUSUM over the *window* most recent reports to detect precision regression.

        Returns alarm status, current CUSUM statistic, and the rolling precision
        computed from the window.
        """
        reports = self.db.get_reports(limit=window)
        if len(reports) < 5:
            return {
                "alarm": False,
                "cusum_statistic": 0.0,
                "rolling_precision": None,
                "window": len(reports),
                "message": "Insufficient data (< 5 reports).",
            }

        # Process in chronological order
        reports_sorted = sorted(reports, key=lambda r: r.get("reported_at", ""))

        # Use overall precision as the CUSUM baseline if not yet set
        baseline = self._cusum.target
        if baseline == 0.0:
            all_acc = self.compute_accuracy()
            baseline = all_acc.get("precision", 0.5)
            self._cusum = CUSUMDetector(target=baseline, slack=0.1, threshold=5.0)

        alarm = False
        for r in reports_sorted:
            # Local precision signal: 1 if correctly predicted scam, 0 otherwise
            point = 1.0 if (r.get("is_scam") == 1 and r.get("was_correct") == 1) else 0.0
            alarm = self._cusum.update(point)

        # Rolling precision over the window
        tp_window = sum(1 for r in reports_sorted if r.get("is_scam") == 1 and r.get("was_correct") == 1)
        fp_window = sum(1 for r in reports_sorted if r.get("is_scam") == 0 and r.get("was_correct") == 0)
        rolling_precision = (
            tp_window / (tp_window + fp_window)
            if (tp_window + fp_window) > 0
            else None
        )

        return {
            "alarm": alarm,
            "cusum_statistic": round(self._cusum.statistic, 4),
            "cusum_baseline": round(baseline, 4),
            "rolling_precision": round(rolling_precision, 4) if rolling_precision is not None else None,
            "window": len(reports_sorted),
            "message": "Regression detected — patterns may need retraining." if alarm else "No regression detected.",
        }

    # ------------------------------------------------------------------
    # FULL CYCLE
    # ------------------------------------------------------------------

    def run_cycle(self) -> dict:
        """Execute one complete flywheel cycle and persist metrics."""
        self._cycle_count += 1
        cycle_ts = _now_iso()
        logger.info("Flywheel cycle %d starting", self._cycle_count)

        # Accuracy snapshot
        accuracy = self.compute_accuracy()

        # Pattern evolution
        evolution = self.evolve_patterns()

        # Regression check
        regression = self.detect_regression()

        # Collect updated signal count from all active pattern observations
        active_patterns = self.db.get_patterns(status="active")
        signals_updated = sum(p.get("observations", 0) for p in active_patterns)

        metrics: dict[str, Any] = {
            "cycle_ts": cycle_ts,
            "cycle_number": self._cycle_count,
            "total_analyzed": accuracy.get("total", 0),
            "true_positives": accuracy.get("true_positives", 0),
            "false_positives": accuracy.get("false_positives", 0),
            "precision": accuracy.get("precision", 0.0),
            "recall": accuracy.get("recall", 0.0),
            "f1": accuracy.get("f1", 0.0),
            "accuracy": accuracy.get("accuracy", 0.0),
            "signals_updated": signals_updated,
            "patterns_evolved": len(evolution.get("promoted", [])) + len(evolution.get("deprecated", [])),
            "patterns_promoted": evolution.get("promoted", []),
            "patterns_deprecated": evolution.get("deprecated", []),
            "regression_alarm": regression.get("alarm", False),
            "cusum_statistic": regression.get("cusum_statistic", 0.0),
        }

        self.db.save_flywheel_metrics(metrics)

        logger.info(
            "Flywheel cycle %d complete: precision=%.3f recall=%.3f f1=%.3f "
            "promoted=%d deprecated=%d regression_alarm=%s",
            self._cycle_count,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            len(evolution.get("promoted", [])),
            len(evolution.get("deprecated", [])),
            metrics["regression_alarm"],
        )
        return metrics

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    def get_health(self) -> dict:
        """Return a concise health summary for monitoring."""
        stats = self.db.get_stats()
        accuracy = self.compute_accuracy()
        regression = self.detect_regression()
        active_patterns = len(self.db.get_patterns(status="active"))
        candidate_patterns = len(self.db.get_patterns(status="candidate"))
        deprecated_patterns = len(self.db.get_patterns(status="deprecated"))

        # Simple health grade
        precision = accuracy.get("precision", 0.0)
        if precision >= 0.85:
            grade = "A"
        elif precision >= 0.75:
            grade = "B"
        elif precision >= 0.60:
            grade = "C"
        elif precision >= 0.40:
            grade = "D"
        else:
            grade = "F"

        return {
            "healthy": not regression.get("alarm", False),
            "grade": grade,
            "precision": accuracy.get("precision", 0.0),
            "recall": accuracy.get("recall", 0.0),
            "f1": accuracy.get("f1", 0.0),
            "total_jobs_analyzed": stats.get("total_jobs_analyzed", 0),
            "total_user_reports": stats.get("total_user_reports", 0),
            "active_patterns": active_patterns,
            "candidate_patterns": candidate_patterns,
            "deprecated_patterns": deprecated_patterns,
            "regression_alarm": regression.get("alarm", False),
            "cusum_statistic": regression.get("cusum_statistic", 0.0),
            "cycle_count": self._cycle_count,
            "checked_at": _now_iso(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_cusum_baseline(self) -> None:
        """Set the CUSUM target to the current overall precision from DB."""
        accuracy = self.compute_accuracy()
        baseline = accuracy.get("precision", 0.0)
        if baseline > 0.0:
            self._cusum = CUSUMDetector(target=baseline, slack=0.1, threshold=5.0)
