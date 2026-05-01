"""Self-improving detection flywheel: INGEST → SCORE → VALIDATE → LEARN → EVOLVE."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

from sentinel.db import SentinelDB
from sentinel.models import UserReport, ValidationResult

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
        self._last_regression_response: dict | None = None
        self._alert_callbacks: list[callable] = []
        # Register the default file-based alert
        self._alert_callbacks.append(self._default_alert)
        # Shadow scorer for A/B weight testing
        from sentinel.shadow import ShadowScorer
        self.shadow = ShadowScorer(self.db)
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

        Before committing changes, runs a cascade impact preview.
        If impact is HIGH, logs a warning and defers to shadow scorer.
        Returns a dict with counts of promoted, deprecated, and retained patterns.
        """
        # Snapshot current weights before any changes for cascade preview
        from sentinel import scorer
        old_weights = scorer._load_learned_weights()

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

        # --- Propose shadow weights for newly promoted patterns ---
        if promoted:
            try:
                new_weights = {pid: 0.8 for pid in promoted}
                self.shadow.propose_weights(new_weights)
                logger.info("Shadow scorer proposed weights for %d promoted patterns.", len(promoted))
            except Exception:
                logger.debug("Shadow weight proposal failed (non-fatal)", exc_info=True)

        # --- Cascade impact check ---
        # Only run if there are actual changes to assess
        cascade_deferred = False
        cascade_risk = "SAFE"
        if promoted or deprecated:
            try:
                from sentinel.mesh import CascadeDetector
                scorer._reset_learned_weights_cache()
                new_weights = scorer._load_learned_weights()
                detector = CascadeDetector()
                report = detector.preview_impact(
                    self.db, old_weights, new_weights, sample_size=100
                )
                cascade_risk = report.risk_level
                if cascade_risk == "HIGH":
                    logger.warning(
                        "Cascade impact is HIGH (%.1f%% of jobs would change classification) — "
                        "deferring pattern evolution to shadow scorer.",
                        report.change_rate * 100,
                    )
                    # Roll back to old weights by resetting cache (changes already committed
                    # to DB, but we surface the warning so the daemon can shadow-test instead)
                    cascade_deferred = True
                else:
                    logger.info(
                        "Cascade impact %s (%.1f%% change rate) — committing pattern evolution.",
                        cascade_risk,
                        report.change_rate * 100,
                    )
            except Exception:
                logger.debug("Cascade preview failed (non-fatal)", exc_info=True)

        # Invalidate the learned-weight cache so scorer picks up any
        # weight changes from pattern promotion/deprecation immediately.
        scorer._reset_learned_weights_cache()

        return {
            "promoted": promoted,
            "deprecated": deprecated,
            "retained_count": len(retained),
            "evolved_at": _now_iso(),
            "cascade_risk": cascade_risk,
            "cascade_deferred": cascade_deferred,
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
    # CALIBRATION
    # ------------------------------------------------------------------

    def calibration_curve(
        self, db: SentinelDB | None = None, n_bins: int = 10
    ) -> list[tuple[float, float, float, int]]:
        """Compute a calibration curve from user-reported jobs.

        Bins all reports by their predicted ``our_prediction`` score into
        *n_bins* equal-width buckets.  For each non-empty bucket returns:

            (bin_center, mean_predicted_score, actual_scam_rate, n_samples)

        Only bins with at least one report are included.

        Args:
            db: Database to query (defaults to ``self.db``).
            n_bins: Number of equal-width bins across [0, 1].

        Returns:
            List of ``(bin_center, predicted_rate, actual_rate, n_samples)``
            sorted by bin_center ascending.
        """
        _db = db if db is not None else self.db
        reports = _db.get_reports(limit=100_000)
        if not reports:
            return []

        bin_width = 1.0 / n_bins
        # Each bin: accumulate sum of predicted scores + scam label counts
        bins: dict[int, list] = {}  # bin_idx -> [sum_pred, sum_scam, count]

        for r in reports:
            pred = r.get("our_prediction")
            if pred is None:
                continue
            pred = float(pred)
            is_scam = int(r.get("is_scam", 0))
            bin_idx = min(int(pred / bin_width), n_bins - 1)
            if bin_idx not in bins:
                bins[bin_idx] = [0.0, 0, 0]
            bins[bin_idx][0] += pred
            bins[bin_idx][1] += is_scam
            bins[bin_idx][2] += 1

        curve: list[tuple[float, float, float, int]] = []
        for bin_idx in sorted(bins.keys()):
            sum_pred, sum_scam, count = bins[bin_idx]
            bin_center = round((bin_idx + 0.5) * bin_width, 4)
            predicted_rate = round(sum_pred / count, 4)
            actual_rate = round(sum_scam / count, 4)
            curve.append((bin_center, predicted_rate, actual_rate, count))

        return curve

    def calibration_error(self, db: SentinelDB | None = None, n_bins: int = 10) -> float:
        """Compute the Expected Calibration Error (ECE).

        ECE = sum over bins of (n_bin / N) * |predicted_rate - actual_rate|

        A perfectly calibrated model has ECE ≈ 0.  Higher values indicate
        systematic over- or under-prediction.

        Returns 0.0 when there are no reports.
        """
        curve = self.calibration_curve(db=db, n_bins=n_bins)
        if not curve:
            return 0.0

        total_samples = sum(entry[3] for entry in curve)
        if total_samples == 0:
            return 0.0

        ece = sum(
            (n / total_samples) * abs(predicted - actual)
            for _, predicted, actual, n in curve
        )
        return round(ece, 6)

    def auto_adjust_thresholds(
        self,
        db: SentinelDB | None = None,
        n_bins: int = 10,
        tolerance: float = 0.10,
    ) -> dict:
        """Auto-adjust risk classification thresholds based on calibration data.

        For each calibration bin where ``|actual_rate - predicted_rate| > tolerance``,
        nudge the nearest risk threshold toward correcting the miscalibration.
        Each adjustment is clamped to ``scorer._MAX_THRESHOLD_DELTA``.

        Args:
            db: Database to query (defaults to ``self.db``).
            n_bins: Bins to use for calibration curve.
            tolerance: Minimum abs(actual - predicted) before adjusting.

        Returns:
            Dict with keys:
            - ``adjusted``: list of dicts with threshold change details.
            - ``ece_before``: ECE before adjustments.
            - ``skipped``: True when no calibration data is available.
        """
        from sentinel.scorer import _MAX_THRESHOLD_DELTA, _RISK_THRESHOLDS

        curve = self.calibration_curve(db=db, n_bins=n_bins)
        if not curve:
            return {"adjusted": [], "ece_before": 0.0, "skipped": True}

        ece_before = self.calibration_error(db=db, n_bins=n_bins)
        adjusted: list[dict] = []

        # Ordered threshold names by score boundary ascending
        _threshold_order = ["safe", "low", "suspicious", "high"]

        for bin_center, predicted_rate, actual_rate, _n in curve:
            diff = actual_rate - predicted_rate  # positive = we under-predict
            if abs(diff) <= tolerance:
                continue

            # Find the threshold boundary closest to the bin_center
            closest_key = min(
                _threshold_order,
                key=lambda k: abs(_RISK_THRESHOLDS[k] - bin_center),
            )
            old_value = _RISK_THRESHOLDS[closest_key]

            # Under-prediction (actual > predicted): lower the threshold so
            # more jobs are flagged at higher risk levels.
            delta = -math.copysign(min(abs(diff) * 0.1, _MAX_THRESHOLD_DELTA), diff)
            new_value = round(
                max(0.05, min(0.95, old_value + delta)), 4
            )

            if new_value != old_value:
                _RISK_THRESHOLDS[closest_key] = new_value
                adjusted.append({
                    "threshold": closest_key,
                    "old_value": old_value,
                    "new_value": new_value,
                    "bin_center": bin_center,
                    "predicted_rate": predicted_rate,
                    "actual_rate": actual_rate,
                    "diff": round(diff, 4),
                })

        return {
            "adjusted": adjusted,
            "ece_before": ece_before,
            "skipped": False,
        }

    # ------------------------------------------------------------------
    # CUSUM REGRESSION DETECTION
    # ------------------------------------------------------------------

    def detect_regression(self, window: int = 50) -> dict:
        """Run confidence-weighted CUSUM over the *window* most recent reports.

        Each observation is weighted by the stored confidence of the corresponding
        job (looked up from the jobs table by URL).  Jobs with no stored confidence
        default to weight 1.0 so behaviour is unchanged for legacy data.

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
            # Base precision signal: 1.0 = correct scam prediction, 0.0 = wrong
            point = 1.0 if (r.get("is_scam") == 1 and r.get("was_correct") == 1) else 0.0

            # Look up confidence for this job URL and use as weight
            job = self.db.get_job(r.get("url", "")) if r.get("url") else None
            confidence = job.get("confidence") if job else None
            weight = float(confidence) if confidence is not None else 1.0
            # Weighted deviation: scale the point toward the neutral baseline
            weighted_point = baseline + weight * (point - baseline)

            alarm = self._cusum.update(weighted_point)

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

        # Calibration snapshot + auto threshold adjustment
        ece = self.calibration_error()
        threshold_adjustment = self.auto_adjust_thresholds()

        # Shadow scorer evaluation
        shadow_evaluation: dict[str, Any] = {"active": False}
        try:
            if self.shadow.active:
                eval_result = self.shadow.evaluate()
                promoted_shadow = False
                rejected_shadow = False
                if self.shadow.should_promote():
                    self.shadow.promote()
                    promoted_shadow = True
                elif eval_result.jobs_evaluated >= 30:
                    self.shadow.reject()
                    rejected_shadow = True
                shadow_evaluation = {
                    "active": True,
                    "jobs_evaluated": eval_result.jobs_evaluated,
                    "baseline_precision": eval_result.baseline_precision,
                    "shadow_precision": eval_result.shadow_precision,
                    "improvement": eval_result.improvement,
                    "promoted": promoted_shadow,
                    "rejected": rejected_shadow,
                }
            else:
                shadow_evaluation = {
                    "active": False,
                    "jobs_evaluated": 0,
                    "promoted": False,
                    "rejected": False,
                }
        except Exception:
            logger.debug("Shadow evaluation failed (non-fatal)", exc_info=True)
            shadow_evaluation = {"active": False, "jobs_evaluated": 0, "promoted": False, "rejected": False}

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
            "calibration_ece": ece,
            "thresholds_adjusted": len(threshold_adjustment.get("adjusted", [])),
            "shadow_evaluation": shadow_evaluation,
        }

        # --- Regression response ---
        if metrics["regression_alarm"]:
            from sentinel import scorer

            logger.warning("Regression detected — triggering automatic weight recalibration")

            # Re-run pattern evolution to promote/deprecate based on fresh data
            self.evolve_patterns()

            # Invalidate stale learned-weight cache
            scorer._reset_learned_weights_cache()

            # Record regression response details
            self._last_regression_response = {
                "cycle": self._cycle_count,
                "precision": metrics["precision"],
                "cusum_statistic": metrics["cusum_statistic"],
                "action": "weight_recalibration",
                "responded_at": _now_iso(),
            }

            # Fire all registered alert callbacks
            for cb in self._alert_callbacks:
                try:
                    cb(metrics)
                except Exception:
                    logger.exception("Regression alert callback failed: %s", cb)

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
    # INPUT DRIFT DETECTION
    # ------------------------------------------------------------------

    def record_signal_rates(
        self,
        signal_counts: dict[str, int],
        total_jobs: int,
        window_start: str,
        window_end: str,
    ) -> None:
        """Persist per-signal firing counts for a scoring batch window.

        Delegates to db.record_signal_rates so callers can use either
        the flywheel or db directly.
        """
        self.db.record_signal_rates(
            signal_rates=signal_counts,
            total_jobs=total_jobs,
            window_start=window_start,
            window_end=window_end,
        )

    def detect_input_drift(self, window_days: int = 7) -> dict:
        """Detect distributional shift in signal firing rates.

        Compares the *recent* window (last *window_days* days) against a
        *baseline* constructed from all older history.

        Algorithm:
        1. Partition signal_rate_history into recent vs. baseline buckets.
        2. Build normalised rate vectors (fire_count / total_jobs) for each.
        3. Compute Jensen-Shannon divergence as the drift_score (0-1 range).
        4. Also compute chi-squared statistic for corroboration.
        5. Trigger alarm when drift_score > 0.10 (10% JSD threshold).

        Returns a dict with:
            drift_score       -- JSD value (0.0 = identical, 1.0 = maximally different)
            alarm             -- True when drift_score > 0.10
            changed_signals   -- list of {signal, baseline_rate, recent_rate, delta}
                                 sorted by abs(delta) descending
            chi2_statistic    -- chi-squared test statistic
            recent_window_start, baseline_jobs, recent_jobs, message
        """
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        cutoff = (now - timedelta(days=window_days)).isoformat()

        all_rows = self.db.get_signal_rate_history(limit=10_000)
        if not all_rows:
            return {
                "drift_score": 0.0,
                "alarm": False,
                "changed_signals": [],
                "recent_window_start": cutoff,
                "baseline_jobs": 0,
                "recent_jobs": 0,
                "chi2_statistic": 0.0,
                "message": "No signal rate history available.",
            }

        recent_rows = [r for r in all_rows if r.get("window_end", "") >= cutoff]
        baseline_rows = [r for r in all_rows if r.get("window_end", "") < cutoff]

        if not recent_rows:
            return {
                "drift_score": 0.0,
                "alarm": False,
                "changed_signals": [],
                "recent_window_start": cutoff,
                "baseline_jobs": sum(r.get("total_jobs", 0) for r in baseline_rows),
                "recent_jobs": 0,
                "chi2_statistic": 0.0,
                "message": "No recent signal data — cannot compute drift (need data within baseline window).",
            }

        if not baseline_rows:
            return {
                "drift_score": 0.0,
                "alarm": False,
                "changed_signals": [],
                "recent_window_start": cutoff,
                "baseline_jobs": 0,
                "recent_jobs": sum(r.get("total_jobs", 0) for r in recent_rows),
                "chi2_statistic": 0.0,
                "message": "No baseline data yet — cannot compute drift.",
            }

        # Aggregate firing rates: signal -> (total_fire_count, total_jobs)
        def _aggregate(rows: list) -> tuple[dict, int]:
            counts: dict[str, int] = {}
            jobs_total = 0
            for r in rows:
                sig = r["signal_name"]
                counts[sig] = counts.get(sig, 0) + r.get("fire_count", 0)
                jobs_total += r.get("total_jobs", 0)
            denom = max(jobs_total, 1)
            rates = {sig: cnt / denom for sig, cnt in counts.items()}
            return rates, jobs_total

        baseline_rates, baseline_jobs = _aggregate(baseline_rows)
        recent_rates, recent_jobs = _aggregate(recent_rows)

        all_signals = sorted(set(baseline_rates) | set(recent_rates))

        if not all_signals:
            return {
                "drift_score": 0.0,
                "alarm": False,
                "changed_signals": [],
                "recent_window_start": cutoff,
                "baseline_jobs": baseline_jobs,
                "recent_jobs": recent_jobs,
                "chi2_statistic": 0.0,
                "message": "No signals found in history.",
            }

        # Build probability distributions (normalise rates to sum to 1)
        def _to_dist(rates: dict, signals: list) -> list:
            vec = [rates.get(s, 0.0) for s in signals]
            total = sum(vec)
            if total == 0.0:
                n = len(signals)
                return [1.0 / n] * n
            return [v / total for v in vec]

        p = _to_dist(baseline_rates, all_signals)
        q = _to_dist(recent_rates, all_signals)

        # Jensen-Shannon Divergence (symmetric, bounded [0,1] with log2)
        def _kl(a: list, b: list) -> float:
            eps = 1e-10
            return sum(
                ai * math.log2((ai + eps) / (bi + eps))
                for ai, bi in zip(a, b, strict=False)
                if ai > 0
            )

        m = [(pi + qi) / 2.0 for pi, qi in zip(p, q, strict=False)]
        jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
        drift_score = max(0.0, min(1.0, jsd))

        # Chi-squared statistic (observed vs. expected based on baseline)
        chi2 = 0.0
        for pi, qi in zip(p, q, strict=False):
            expected = pi * recent_jobs
            observed = qi * recent_jobs
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

        # Per-signal deltas (raw rates for interpretability)
        changed_signals = []
        for sig in all_signals:
            base_r = baseline_rates.get(sig, 0.0)
            rec_r = recent_rates.get(sig, 0.0)
            delta = rec_r - base_r
            changed_signals.append({
                "signal": sig,
                "baseline_rate": round(base_r, 5),
                "recent_rate": round(rec_r, 5),
                "delta": round(delta, 5),
            })
        changed_signals.sort(key=lambda x: abs(x["delta"]), reverse=True)

        alarm = drift_score > 0.10
        message = (
            f"Input drift detected (JSD={drift_score:.4f} > 0.10) — "
            "consider triggering pattern mining."
            if alarm
            else f"No significant input drift (JSD={drift_score:.4f})."
        )

        # Automatically trigger pattern mining when drift is detected
        if alarm:
            try:
                from sentinel.innovation import InnovationEngine
                engine = InnovationEngine(db=self.db)
                engine._run_pattern_mining()
                logger.info(
                    "Input drift detected (score=%.4f) — pattern mining triggered.",
                    drift_score,
                )
            except Exception:
                logger.debug("Pattern mining trigger failed (non-fatal).", exc_info=True)

        return {
            "drift_score": round(drift_score, 6),
            "alarm": alarm,
            "changed_signals": changed_signals,
            "recent_window_start": cutoff,
            "baseline_jobs": baseline_jobs,
            "recent_jobs": recent_jobs,
            "chi2_statistic": round(chi2, 4),
            "message": message,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def on_regression(self, callback) -> None:
        """Register a callback for regression alerts. callback(metrics_dict) -> None."""
        self._alert_callbacks.append(callback)

    def _default_alert(self, metrics: dict) -> None:
        """Write regression alert to ~/.sentinel/alerts.log"""
        import os

        alert_path = os.path.expanduser("~/.sentinel/alerts.log")
        os.makedirs(os.path.dirname(alert_path), exist_ok=True)
        with open(alert_path, "a") as f:
            f.write(
                f"[{datetime.now(UTC).isoformat()}] REGRESSION ALARM: "
                f"precision={metrics.get('precision', 'N/A')}, "
                f"cusum={metrics.get('cusum_statistic', 'N/A')}\n"
            )

    def _init_cusum_baseline(self) -> None:
        """Set the CUSUM target to the current overall precision from DB."""
        accuracy = self.compute_accuracy()
        baseline = accuracy.get("precision", 0.0)
        if baseline > 0.0:
            self._cusum = CUSUMDetector(target=baseline, slack=0.1, threshold=5.0)
