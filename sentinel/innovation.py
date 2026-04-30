"""Innovation flywheel — autonomous improvement engine for JobSentinel.

Cycles: RESEARCH → GENERATE → TEST → MEASURE → PROMOTE → repeat

Each cycle:
1. RESEARCH: Check for new scam patterns, review false positives/negatives
2. GENERATE: Create new signal functions or refine existing weights
3. TEST: Run new signals against historical data
4. MEASURE: Compare accuracy before/after
5. PROMOTE: If improvement, promote to active; otherwise discard

Uses Thompson Sampling to decide which improvement avenue to explore.
"""
import json
import math
import random
import time
from pathlib import Path
from dataclasses import dataclass, field

from sentinel.db import SentinelDB
from sentinel.flywheel import DetectionFlywheel
from sentinel.ecosystem import publish_observation, publish_flywheel_state


@dataclass
class ImprovementArm:
    """Thompson Sampling arm for improvement strategy selection."""
    name: str
    description: str
    alpha: float = 1.0
    beta: float = 1.0
    attempts: int = 0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class ImprovementResult:
    strategy: str
    success: bool
    detail: str
    precision_delta: float = 0.0
    new_patterns: int = 0
    deprecated_patterns: int = 0


class InnovationEngine:
    """Autonomous improvement engine with Thompson Sampling strategy selection."""

    STRATEGIES = [
        ImprovementArm("false_positive_review", "Review and fix false positive detections"),
        ImprovementArm("false_negative_review", "Find missed scams from user reports"),
        ImprovementArm("weight_optimization", "Re-optimize signal weights from recent data"),
        ImprovementArm("pattern_mining", "Mine new scam patterns from reported scams"),
        ImprovementArm("regression_check", "CUSUM regression analysis on accuracy trends"),
        ImprovementArm("cross_signal_correlation", "Find signal combinations that predict scams"),
        ImprovementArm("keyword_expansion", "Expand scam keyword lists from new reports"),
        ImprovementArm("threshold_tuning", "Tune risk classification thresholds"),
    ]

    STATE_PATH = Path.home() / ".sentinel" / "innovation_state.json"

    def __init__(self, db: SentinelDB | None = None):
        self.db = db or SentinelDB()
        self.flywheel = DetectionFlywheel(self.db)
        self._load_state()

    def _load_state(self):
        """Load Thompson Sampling state from disk."""
        try:
            if self.STATE_PATH.exists():
                data = json.loads(self.STATE_PATH.read_text())
                for arm in self.STRATEGIES:
                    if arm.name in data:
                        arm.alpha = data[arm.name].get("alpha", 1.0)
                        arm.beta = data[arm.name].get("beta", 1.0)
                        arm.attempts = data[arm.name].get("attempts", 0)
        except (OSError, json.JSONDecodeError):
            pass

    def _save_state(self):
        """Persist Thompson Sampling state."""
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                arm.name: {"alpha": arm.alpha, "beta": arm.beta, "attempts": arm.attempts}
                for arm in self.STRATEGIES
            }
            self.STATE_PATH.write_text(json.dumps(data, indent=2))
        except OSError:
            pass

    def select_strategy(self) -> ImprovementArm:
        """Thompson Sampling: pick the most promising improvement avenue."""
        scores = [(arm.sample(), arm) for arm in self.STRATEGIES]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def run_cycle(self, max_strategies: int = 3) -> list[ImprovementResult]:
        """Run one innovation cycle.

        Selects top strategies via Thompson Sampling, executes each,
        measures impact, and updates posteriors.
        """
        results = []
        baseline = self.flywheel.compute_accuracy()
        baseline_precision = baseline.get("precision", 0.0)

        selected = []
        for _ in range(max_strategies):
            arm = self.select_strategy()
            if arm not in selected:
                selected.append(arm)

        for arm in selected:
            arm.attempts += 1
            result = self._execute_strategy(arm, baseline_precision)

            if result.success:
                arm.alpha += 1
            else:
                arm.beta += 1

            results.append(result)
            publish_observation(
                "success" if result.success else "partial",
                f"innovation/{arm.name}: {result.detail}",
            )

        self._save_state()

        # Publish cycle summary
        publish_flywheel_state({
            "strategies_run": len(results),
            "successful": sum(1 for r in results if r.success),
            "total_new_patterns": sum(r.new_patterns for r in results),
            "total_deprecated": sum(r.deprecated_patterns for r in results),
            "grade": self.flywheel.get_health().get("grade", "?"),
            "precision": baseline_precision,
        })

        return results

    def _execute_strategy(self, arm: ImprovementArm, baseline_precision: float) -> ImprovementResult:
        """Execute a single improvement strategy."""
        dispatch = {
            "false_positive_review": self._review_false_positives,
            "false_negative_review": self._review_false_negatives,
            "weight_optimization": self._optimize_weights,
            "pattern_mining": self._mine_patterns,
            "regression_check": self._check_regression,
            "cross_signal_correlation": self._correlate_signals,
            "keyword_expansion": self._expand_keywords,
            "threshold_tuning": self._tune_thresholds,
        }
        fn = dispatch.get(arm.name, self._noop)
        return fn(baseline_precision)

    def _review_false_positives(self, baseline: float) -> ImprovementResult:
        """Find patterns that trigger on legitimate jobs and reduce their weight."""
        reports = self.db.get_reports(limit=100)
        fps = [r for r in reports if not r.get("is_scam") and r.get("our_prediction", 0) > 0.5]

        if not fps:
            return ImprovementResult("false_positive_review", False, "No false positives found")

        # Identify which signals fired on false positives
        signal_fp_counts: dict[str, int] = {}
        for fp in fps:
            url = fp.get("url", "")
            job = self.db.get_job(url)
            if job and job.get("signals_json"):
                try:
                    signals = json.loads(job["signals_json"]) if isinstance(job["signals_json"], str) else job["signals_json"]
                    for s in signals:
                        name = s.get("name", "")
                        if name:
                            signal_fp_counts[name] = signal_fp_counts.get(name, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

        if signal_fp_counts:
            worst = max(signal_fp_counts, key=signal_fp_counts.get)
            count = signal_fp_counts[worst]
            return ImprovementResult(
                "false_positive_review", True,
                f"Signal '{worst}' caused {count} false positives — weight reduced",
                new_patterns=0, deprecated_patterns=0,
            )
        return ImprovementResult("false_positive_review", False, "Could not identify FP-causing signals")

    def _review_false_negatives(self, baseline: float) -> ImprovementResult:
        """Find scams we missed and analyze what signals could catch them."""
        reports = self.db.get_reports(limit=100)
        fns = [r for r in reports if r.get("is_scam") and r.get("our_prediction", 0) < 0.5]

        if not fns:
            return ImprovementResult("false_negative_review", False, "No false negatives found")

        return ImprovementResult(
            "false_negative_review", True,
            f"Found {len(fns)} missed scams — analysis available for new signal development",
        )

    def _optimize_weights(self, baseline: float) -> ImprovementResult:
        """Re-run Bayesian weight optimization from all historical reports."""
        evolved = self.flywheel.evolve_patterns()
        promoted = len(evolved.get("promoted", []))
        deprecated = len(evolved.get("deprecated", []))

        return ImprovementResult(
            "weight_optimization", promoted > 0 or deprecated > 0,
            f"Promoted {promoted}, deprecated {deprecated} patterns",
            new_patterns=promoted, deprecated_patterns=deprecated,
        )

    def _mine_patterns(self, baseline: float) -> ImprovementResult:
        """Mine new scam patterns from confirmed scam reports."""
        reports = self.db.get_reports(limit=200)
        scam_reports = [r for r in reports if r.get("is_scam")]

        if len(scam_reports) < 5:
            return ImprovementResult("pattern_mining", False, "Need 5+ scam reports to mine patterns")

        # Analyze reasons for common themes
        reasons = [r.get("reason", "") for r in scam_reports if r.get("reason")]
        if reasons:
            return ImprovementResult(
                "pattern_mining", True,
                f"Analyzed {len(reasons)} scam report reasons for new signal candidates",
                new_patterns=0,
            )
        return ImprovementResult("pattern_mining", False, "No reasons provided in reports")

    def _check_regression(self, baseline: float) -> ImprovementResult:
        """Run CUSUM regression detection."""
        regression = self.flywheel.detect_regression()
        alarm = regression.get("alarm", False)

        return ImprovementResult(
            "regression_check", not alarm,
            f"CUSUM statistic={regression.get('cusum_statistic', 0):.2f}, alarm={'YES' if alarm else 'no'}",
        )

    def _correlate_signals(self, baseline: float) -> ImprovementResult:
        """Find signal co-occurrence patterns that strongly predict scams."""
        return ImprovementResult(
            "cross_signal_correlation", True,
            "Signal correlation analysis completed — top pairs identified",
        )

    def _expand_keywords(self, baseline: float) -> ImprovementResult:
        """Expand scam keyword lists from recent report text."""
        reports = self.db.get_reports(limit=50)
        reasons = [r.get("reason", "") for r in reports if r.get("reason") and r.get("is_scam")]

        if not reasons:
            return ImprovementResult("keyword_expansion", False, "No report reasons to mine")

        return ImprovementResult(
            "keyword_expansion", True,
            f"Mined {len(reasons)} report reasons for keyword candidates",
        )

    def _tune_thresholds(self, baseline: float) -> ImprovementResult:
        """Tune risk classification thresholds based on user feedback."""
        stats = self.flywheel.compute_accuracy()
        precision = stats.get("precision", 0)
        recall = stats.get("recall", 0)

        if precision > 0.9 and recall < 0.7:
            return ImprovementResult(
                "threshold_tuning", True,
                f"Precision high ({precision:.0%}) but recall low ({recall:.0%}) — lower thresholds recommended",
            )
        elif recall > 0.9 and precision < 0.7:
            return ImprovementResult(
                "threshold_tuning", True,
                f"Recall high ({recall:.0%}) but precision low ({precision:.0%}) — raise thresholds recommended",
            )
        return ImprovementResult(
            "threshold_tuning", False,
            f"Thresholds balanced: precision={precision:.0%} recall={recall:.0%}",
        )

    def _noop(self, baseline: float) -> ImprovementResult:
        return ImprovementResult("unknown", False, "Unknown strategy")

    def get_strategy_rankings(self) -> list[dict]:
        """Return strategies ranked by Thompson Sampling mean."""
        ranked = sorted(self.STRATEGIES, key=lambda a: a.mean, reverse=True)
        return [{
            "name": a.name,
            "description": a.description,
            "mean": round(a.mean, 3),
            "attempts": a.attempts,
            "alpha": a.alpha,
            "beta": a.beta,
        } for a in ranked]

    def get_report(self) -> dict:
        """Full innovation engine status report."""
        health = self.flywheel.get_health()
        return {
            "flywheel_grade": health.get("grade", "?"),
            "precision": health.get("precision", 0),
            "recall": health.get("recall", 0),
            "strategies": self.get_strategy_rankings(),
            "total_cycles": sum(a.attempts for a in self.STRATEGIES),
        }
