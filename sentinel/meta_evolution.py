"""Meta-evolution engine: makes the flywheel system itself self-improving.

The existing flywheels improve detection weights. This system improves
the flywheels themselves by tracking how well each flywheel learns,
auto-tuning hyperparameters, performing flywheel surgery, and applying
evolutionary pressure to converge on optimal configurations.

Components:
  FitnessLandscape    — maps hyperparameter configs to flywheel performance
  GaussianProcessOptimizer — scipy-free GP for efficient hyperparameter search
  RegressionAnalyzer  — Bayesian structural time-series regime-change detection
  FlywheelSurgeon     — targeted interventions on underperforming flywheels
  MetaEvolutionEngine — top-level orchestrator
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sentinel.db import SentinelDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FlywheelSnapshot:
    """Point-in-time performance measurement of a flywheel."""
    flywheel_name: str
    cycle_number: int
    precision: float
    recall: float
    f1: float
    learning_velocity: float
    cusum_statistic: float
    calibration_ece: float
    timestamp: str = field(default_factory=_now_iso)

    def fitness(self) -> float:
        """Composite fitness: weighted combination of key metrics."""
        return (
            0.40 * self.f1
            + 0.25 * max(0.0, self.learning_velocity * 10 + 0.5)
            + 0.20 * (1.0 - min(self.calibration_ece * 5, 1.0))
            + 0.15 * (1.0 - min(self.cusum_statistic / 10.0, 1.0))
        )


@dataclass
class HyperparamConfig:
    """A single hyperparameter configuration for the flywheel system."""
    config_id: str = field(default_factory=lambda: f"cfg_{uuid.uuid4().hex[:8]}")
    cusum_threshold: float = 5.0
    cusum_slack: float = 0.5
    drift_sensitivity: float = 0.10
    ts_prior_strength: float = 1.0
    shadow_promote_threshold: float = 0.02
    innovation_exploration_rate: float = 0.3
    promote_precision_threshold: float = 0.80
    promote_min_observations: int = 10
    deprecate_precision_threshold: float = 0.30
    deprecate_min_observations: int = 20

    # Fitness tracking
    fitness_scores: list[float] = field(default_factory=list)
    generation: int = 0

    @property
    def mean_fitness(self) -> float:
        if not self.fitness_scores:
            return 0.0
        return statistics.mean(self.fitness_scores)

    @property
    def fitness_variance(self) -> float:
        if len(self.fitness_scores) < 2:
            return 1.0
        return statistics.variance(self.fitness_scores)

    def to_vector(self) -> list[float]:
        """Flatten config to a numeric vector for GP optimization."""
        return [
            self.cusum_threshold,
            self.cusum_slack,
            self.drift_sensitivity,
            self.ts_prior_strength,
            self.shadow_promote_threshold,
            self.innovation_exploration_rate,
            self.promote_precision_threshold,
            float(self.promote_min_observations),
            self.deprecate_precision_threshold,
            float(self.deprecate_min_observations),
        ]

    @classmethod
    def from_vector(cls, vec: list[float], config_id: str = "", generation: int = 0) -> HyperparamConfig:
        return cls(
            config_id=config_id or f"cfg_{uuid.uuid4().hex[:8]}",
            cusum_threshold=max(0.5, vec[0]),
            cusum_slack=max(0.01, min(2.0, vec[1])),
            drift_sensitivity=max(0.01, min(0.50, vec[2])),
            ts_prior_strength=max(0.1, min(10.0, vec[3])),
            shadow_promote_threshold=max(0.001, min(0.20, vec[4])),
            innovation_exploration_rate=max(0.05, min(1.0, vec[5])),
            promote_precision_threshold=max(0.50, min(0.99, vec[6])),
            promote_min_observations=max(3, int(vec[7])),
            deprecate_precision_threshold=max(0.05, min(0.50, vec[8])),
            deprecate_min_observations=max(5, int(vec[9])),
            generation=generation,
        )

    def to_dict(self) -> dict:
        return {
            "config_id": self.config_id,
            "cusum_threshold": self.cusum_threshold,
            "cusum_slack": self.cusum_slack,
            "drift_sensitivity": self.drift_sensitivity,
            "ts_prior_strength": self.ts_prior_strength,
            "shadow_promote_threshold": self.shadow_promote_threshold,
            "innovation_exploration_rate": self.innovation_exploration_rate,
            "promote_precision_threshold": self.promote_precision_threshold,
            "promote_min_observations": self.promote_min_observations,
            "deprecate_precision_threshold": self.deprecate_precision_threshold,
            "deprecate_min_observations": self.deprecate_min_observations,
            "mean_fitness": round(self.mean_fitness, 6),
            "fitness_variance": round(self.fitness_variance, 6),
            "evaluations": len(self.fitness_scores),
            "generation": self.generation,
        }


@dataclass
class LearningVelocityReport:
    """Analysis of how fast a flywheel is learning."""
    flywheel_name: str
    velocity: float  # precision delta per cycle
    acceleration: float  # change in velocity
    trend: str  # "converging", "plateaued", "oscillating", "diverging"
    plateau_cycles: int  # how many cycles at plateau
    oscillation_amplitude: float


@dataclass
class RegimeChange:
    """Detected structural break in a time series."""
    change_point: int  # index in the series
    before_mean: float
    after_mean: float
    effect_size: float
    probability: float  # posterior probability of change
    attribution: str  # what likely caused it


@dataclass
class SurgeryAction:
    """A corrective intervention on a flywheel."""
    flywheel_name: str
    action_type: str
    parameters: dict
    reason: str
    timestamp: str = field(default_factory=_now_iso)


# ---------------------------------------------------------------------------
# Gaussian Process Optimizer (stdlib-only)
# ---------------------------------------------------------------------------

class GaussianProcessOptimizer:
    """Squared-exponential kernel GP for hyperparameter optimization.

    Uses only stdlib math. Implements a simplified GP with RBF kernel
    for surrogate modeling and Expected Improvement acquisition.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 0.01) -> None:
        self.length_scale = length_scale
        self.noise = noise
        self._X: list[list[float]] = []
        self._y: list[float] = []
        self._K_inv: list[list[float]] | None = None

    def add_observation(self, x: list[float], y: float) -> None:
        self._X.append(list(x))
        self._y.append(y)
        self._K_inv = None

    @property
    def n_observations(self) -> int:
        return len(self._X)

    def _rbf_kernel(self, x1: list[float], x2: list[float]) -> float:
        sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2, strict=False))
        return math.exp(-0.5 * sq_dist / (self.length_scale ** 2))

    def _build_K(self) -> list[list[float]]:
        n = len(self._X)
        K = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                K[i][j] = self._rbf_kernel(self._X[i], self._X[j])
                if i == j:
                    K[i][j] += self.noise
        return K

    def _invert_matrix(self, M: list[list[float]]) -> list[list[float]]:
        """Gauss-Jordan matrix inversion for small matrices."""
        n = len(M)
        # Augmented matrix [M | I]
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]

        for col in range(n):
            # Partial pivoting
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            if abs(pivot) < 1e-12:
                # Singular: add jitter
                aug[col][col] += 1e-6
                pivot = aug[col][col]

            for j in range(2 * n):
                aug[col][j] /= pivot

            for i in range(n):
                if i != col:
                    factor = aug[i][col]
                    for j in range(2 * n):
                        aug[i][j] -= factor * aug[col][j]

        return [row[n:] for row in aug]

    def predict(self, x: list[float]) -> tuple[float, float]:
        """Return (mean, variance) prediction at point x."""
        if not self._X:
            return 0.0, 1.0

        if self._K_inv is None:
            K = self._build_K()
            self._K_inv = self._invert_matrix(K)

        k_star = [self._rbf_kernel(x, xi) for xi in self._X]
        k_ss = self._rbf_kernel(x, x) + self.noise

        # mean = k* . K_inv . y
        n = len(self._X)
        alpha = [sum(self._K_inv[i][j] * self._y[j] for j in range(n)) for i in range(n)]
        mean = sum(k_star[i] * alpha[i] for i in range(n))

        # variance = k** - k* . K_inv . k*
        v = [sum(self._K_inv[i][j] * k_star[j] for j in range(n)) for i in range(n)]
        variance = k_ss - sum(k_star[i] * v[i] for i in range(n))
        variance = max(variance, 1e-10)

        return mean, variance

    def expected_improvement(self, x: list[float], best_y: float | None = None) -> float:
        """Compute Expected Improvement acquisition function value."""
        if not self._X:
            return 1.0

        if best_y is None:
            best_y = max(self._y)

        mean, variance = self.predict(x)
        std = math.sqrt(variance)
        if std < 1e-10:
            return 0.0

        z = (mean - best_y) / std
        # Approximate CDF and PDF of standard normal
        ei = std * (z * _norm_cdf(z) + _norm_pdf(z))
        return max(0.0, ei)

    def suggest_next(self, bounds: list[tuple[float, float]], n_candidates: int = 50) -> list[float]:
        """Suggest the next point to evaluate via random candidate EI maximization."""
        best_ei = -1.0
        best_x: list[float] = []

        for _ in range(n_candidates):
            candidate = [lo + random.random() * (hi - lo) for lo, hi in bounds]
            ei = self.expected_improvement(candidate)
            if ei > best_ei:
                best_ei = ei
                best_x = candidate

        return best_x if best_x else [lo + (hi - lo) / 2 for lo, hi in bounds]


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF (Abramowitz & Stegun)."""
    if x < -6.0:
        return 0.0
    if x > 6.0:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = _norm_pdf(x)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - p if x > 0 else p


# ---------------------------------------------------------------------------
# FitnessLandscape
# ---------------------------------------------------------------------------

class FitnessLandscape:
    """Maps hyperparameter configurations to flywheel performance.

    Maintains a history of (config, fitness) pairs and uses a GP
    surrogate to guide hill-climbing.
    """

    # Hyperparameter search bounds: (min, max) for each dimension
    BOUNDS = [
        (1.0, 15.0),    # cusum_threshold
        (0.05, 2.0),    # cusum_slack
        (0.01, 0.50),   # drift_sensitivity
        (0.1, 10.0),    # ts_prior_strength
        (0.001, 0.20),  # shadow_promote_threshold
        (0.05, 1.0),    # innovation_exploration_rate
        (0.50, 0.99),   # promote_precision_threshold
        (3.0, 50.0),    # promote_min_observations
        (0.05, 0.50),   # deprecate_precision_threshold
        (5.0, 100.0),   # deprecate_min_observations
    ]

    def __init__(self) -> None:
        self.gp = GaussianProcessOptimizer(length_scale=2.0, noise=0.05)
        self._history: list[tuple[HyperparamConfig, float]] = []

    def record(self, config: HyperparamConfig, fitness: float) -> None:
        config.fitness_scores.append(fitness)
        self._history.append((config, fitness))
        # Normalize the vector to [0,1] range before feeding to GP
        vec = self._normalize(config.to_vector())
        self.gp.add_observation(vec, fitness)

    def suggest_config(self, generation: int = 0) -> HyperparamConfig:
        """Use GP-EI to suggest the next configuration to evaluate."""
        norm_bounds = [(0.0, 1.0)] * len(self.BOUNDS)
        norm_vec = self.gp.suggest_next(norm_bounds, n_candidates=100)
        raw_vec = self._denormalize(norm_vec)
        return HyperparamConfig.from_vector(raw_vec, generation=generation)

    def best_config(self) -> HyperparamConfig | None:
        if not self._history:
            return None
        return max(self._history, key=lambda pair: pair[1])[0]

    def best_fitness(self) -> float:
        if not self._history:
            return 0.0
        return max(f for _, f in self._history)

    @property
    def evaluation_count(self) -> int:
        return len(self._history)

    def _normalize(self, vec: list[float]) -> list[float]:
        return [
            (v - lo) / (hi - lo) if hi > lo else 0.5
            for v, (lo, hi) in zip(vec, self.BOUNDS, strict=False)
        ]

    def _denormalize(self, nvec: list[float]) -> list[float]:
        return [
            lo + n * (hi - lo) for n, (lo, hi) in zip(nvec, self.BOUNDS, strict=False)
        ]


# ---------------------------------------------------------------------------
# RegressionAnalyzer — Bayesian structural time-series
# ---------------------------------------------------------------------------

class RegressionAnalyzer:
    """Detects regime changes in flywheel metric time series.

    Uses a Bayesian online changepoint detection approach inspired by
    Causal Impact. Computes posterior probability of a structural break
    at each point using cumulative sum statistics and likelihood ratios.
    """

    def __init__(self, hazard_rate: float = 0.05) -> None:
        self.hazard_rate = hazard_rate

    def detect_regime_changes(
        self,
        series: list[float],
        min_segment: int = 3,
    ) -> list[RegimeChange]:
        """Scan a time series for structural breaks.

        Uses a simplified Bayesian changepoint detection:
        for each candidate split point, compute the likelihood ratio
        of two-segment vs. one-segment model and convert to posterior
        probability of change.
        """
        if len(series) < 2 * min_segment:
            return []

        changes: list[RegimeChange] = []
        statistics.mean(series)
        overall_var = max(statistics.variance(series), 1e-10) if len(series) > 1 else 1.0

        for t in range(min_segment, len(series) - min_segment):
            before = series[:t]
            after = series[t:]

            before_mean = statistics.mean(before)
            after_mean = statistics.mean(after)

            before_var = max(statistics.variance(before), 1e-10) if len(before) > 1 else overall_var
            after_var = max(statistics.variance(after), 1e-10) if len(after) > 1 else overall_var

            # Log-likelihood of two-segment model
            ll_two = -0.5 * (
                len(before) * math.log(before_var)
                + len(after) * math.log(after_var)
            )
            # Log-likelihood of one-segment model
            ll_one = -0.5 * len(series) * math.log(overall_var)

            # Bayes factor (log)
            log_bf = ll_two - ll_one
            # Convert to posterior probability via logistic
            prior = self.hazard_rate
            log_prior_odds = math.log(prior / (1 - prior)) if 0 < prior < 1 else 0.0
            log_posterior_odds = log_bf + log_prior_odds
            probability = 1.0 / (1.0 + math.exp(-min(max(log_posterior_odds, -20), 20)))

            effect_size = abs(after_mean - before_mean) / math.sqrt(overall_var)

            if probability > 0.5 and effect_size > 0.3:
                changes.append(RegimeChange(
                    change_point=t,
                    before_mean=round(before_mean, 6),
                    after_mean=round(after_mean, 6),
                    effect_size=round(effect_size, 4),
                    probability=round(probability, 4),
                    attribution="",
                ))

        # Deduplicate: keep only the highest-probability change per cluster
        if not changes:
            return []
        # Use a wider dedup gap to merge nearby detections into one
        return self._deduplicate_changes(changes, min_gap=max(min_segment, len(series) // 4))

    def _deduplicate_changes(
        self,
        changes: list[RegimeChange],
        min_gap: int,
    ) -> list[RegimeChange]:
        """Keep only the highest-probability change within each cluster."""
        changes.sort(key=lambda c: c.change_point)
        deduped: list[RegimeChange] = []
        for ch in changes:
            if deduped and ch.change_point - deduped[-1].change_point < min_gap:
                if ch.probability > deduped[-1].probability:
                    deduped[-1] = ch
            else:
                deduped.append(ch)
        return deduped

    def attribute_change(
        self,
        change: RegimeChange,
        config_changes: list[dict],
    ) -> str:
        """Attempt to attribute a regime change to a specific config change.

        Looks for config changes that occurred near the change point.
        """
        if not config_changes:
            return "unknown"

        for cc in config_changes:
            cc_cycle = cc.get("cycle", 0)
            if abs(cc_cycle - change.change_point) <= 2:
                return f"config_change:{cc.get('parameter', 'unknown')}={cc.get('new_value', '?')}"

        return "organic_shift"


# ---------------------------------------------------------------------------
# FlywheelSurgeon
# ---------------------------------------------------------------------------

class FlywheelSurgeon:
    """Performs targeted interventions on underperforming flywheel components."""

    def __init__(self, db: SentinelDB) -> None:
        self.db = db

    def diagnose(self, snapshot: FlywheelSnapshot) -> list[str]:
        """Identify problems from a flywheel snapshot."""
        issues: list[str] = []
        if snapshot.f1 < 0.4:
            issues.append("low_f1")
        if snapshot.learning_velocity < -0.005:
            issues.append("negative_velocity")
        if snapshot.calibration_ece > 0.15:
            issues.append("high_calibration_error")
        if snapshot.cusum_statistic > 3.0:
            issues.append("cusum_alarm_imminent")
        if abs(snapshot.learning_velocity) < 0.0005 and snapshot.f1 < 0.7:
            issues.append("plateaued_below_target")
        return issues

    def reset_priors(self, signal_names: list[str] | None = None) -> SurgeryAction:
        """Reset Thompson Sampling priors for specified signals to flat Beta(1,1)."""
        patterns = self.db.get_patterns(status="active")
        reset_count = 0
        for p in patterns:
            if signal_names and p["pattern_id"] not in signal_names and p.get("name") not in signal_names:
                continue
            self.db.save_pattern({
                **p,
                "alpha": 1.0,
                "beta": 1.0,
            })
            reset_count += 1

        return SurgeryAction(
            flywheel_name="detection",
            action_type="reset_priors",
            parameters={"signals_reset": reset_count, "filter": signal_names},
            reason=f"Reset {reset_count} signal priors to flat Beta(1,1)",
        )

    def inject_synthetic_data(self, signal_name: str, n_positive: int, n_negative: int) -> SurgeryAction:
        """Inject synthetic observations to bootstrap a signal's posterior."""
        row = None
        for status in ("active", "candidate"):
            patterns = self.db.get_patterns(status=status)
            for p in patterns:
                if p["pattern_id"] == signal_name or p.get("name") == signal_name:
                    row = p
                    break
            if row:
                break

        if row:
            row["alpha"] = row.get("alpha", 1.0) + n_positive
            row["beta"] = row.get("beta", 1.0) + n_negative
            row["observations"] = row.get("observations", 0) + n_positive + n_negative
            row["true_positives"] = row.get("true_positives", 0) + n_positive
            row["false_positives"] = row.get("false_positives", 0) + n_negative
            self.db.save_pattern(row)

        return SurgeryAction(
            flywheel_name="detection",
            action_type="inject_synthetic",
            parameters={
                "signal": signal_name,
                "positive": n_positive,
                "negative": n_negative,
                "found": row is not None,
            },
            reason=f"Injected {n_positive}+ / {n_negative}- synthetic observations for '{signal_name}'",
        )

    def merge_arms(self, arm_names: list[str], merged_name: str) -> SurgeryAction:
        """Merge multiple innovation arms by averaging their posteriors."""
        total_alpha = 0.0
        total_beta = 0.0
        count = 0
        for name in arm_names:
            for status in ("active", "candidate"):
                patterns = self.db.get_patterns(status=status)
                for p in patterns:
                    if p["pattern_id"] == name or p.get("name") == name:
                        total_alpha += p.get("alpha", 1.0)
                        total_beta += p.get("beta", 1.0)
                        count += 1
                        break

        if count > 0:
            avg_alpha = total_alpha / count
            avg_beta = total_beta / count
        else:
            avg_alpha = 1.0
            avg_beta = 1.0

        self.db.save_pattern({
            "pattern_id": merged_name,
            "name": merged_name,
            "description": f"Merged from: {', '.join(arm_names)}",
            "category": "red_flag",
            "regex": "",
            "keywords_json": "[]",
            "alpha": avg_alpha,
            "beta": avg_beta,
            "observations": 0,
            "true_positives": 0,
            "false_positives": 0,
            "status": "candidate",
        })

        return SurgeryAction(
            flywheel_name="detection",
            action_type="merge_arms",
            parameters={
                "merged_from": arm_names,
                "merged_to": merged_name,
                "avg_alpha": round(avg_alpha, 4),
                "avg_beta": round(avg_beta, 4),
            },
            reason=f"Merged {len(arm_names)} arms into '{merged_name}'",
        )

    def split_arm(self, arm_name: str, n_splits: int = 2) -> SurgeryAction:
        """Split an overly-broad arm into n_splits narrower variants."""
        source = None
        for status in ("active", "candidate"):
            patterns = self.db.get_patterns(status=status)
            for p in patterns:
                if p["pattern_id"] == arm_name or p.get("name") == arm_name:
                    source = p
                    break
            if source:
                break

        created: list[str] = []
        if source:
            base_alpha = max(1.0, source.get("alpha", 1.0) / n_splits)
            base_beta = max(1.0, source.get("beta", 1.0) / n_splits)
            for i in range(n_splits):
                split_id = f"{arm_name}_split_{i}"
                self.db.save_pattern({
                    **source,
                    "pattern_id": split_id,
                    "name": f"{source.get('name', arm_name)}_v{i}",
                    "description": f"Split {i+1}/{n_splits} from {arm_name}",
                    "alpha": base_alpha + random.uniform(-0.1, 0.1),
                    "beta": base_beta + random.uniform(-0.1, 0.1),
                    "observations": 0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "status": "candidate",
                })
                created.append(split_id)

        return SurgeryAction(
            flywheel_name="detection",
            action_type="split_arm",
            parameters={
                "source": arm_name,
                "splits": n_splits,
                "created": created,
                "found": source is not None,
            },
            reason=f"Split '{arm_name}' into {n_splits} variants",
        )

    def adjust_reward_function(self, precision_weight: float = 0.6, recall_weight: float = 0.4) -> SurgeryAction:
        """Adjust the balance between precision and recall in the fitness function."""
        return SurgeryAction(
            flywheel_name="detection",
            action_type="adjust_reward",
            parameters={
                "precision_weight": precision_weight,
                "recall_weight": recall_weight,
            },
            reason=f"Adjusted reward function: precision={precision_weight:.2f}, recall={recall_weight:.2f}",
        )

    def perform_surgery(self, snapshot: FlywheelSnapshot) -> list[SurgeryAction]:
        """Auto-select and perform appropriate surgeries based on diagnosis."""
        issues = self.diagnose(snapshot)
        actions: list[SurgeryAction] = []

        if "negative_velocity" in issues:
            actions.append(self.reset_priors())

        if "high_calibration_error" in issues:
            actions.append(self.adjust_reward_function(
                precision_weight=0.7, recall_weight=0.3
            ))

        if "plateaued_below_target" in issues:
            # Get bottom-performing patterns and try splitting them
            patterns = self.db.get_patterns(status="active")
            low_performers = [
                p for p in patterns
                if (p.get("true_positives", 0) + p.get("false_positives", 0)) > 10
                and p.get("true_positives", 0) / max(1, p.get("true_positives", 0) + p.get("false_positives", 0)) < 0.4
            ]
            if low_performers:
                worst = min(low_performers, key=lambda p: p.get("true_positives", 0) / max(1, p.get("true_positives", 0) + p.get("false_positives", 0)))
                actions.append(self.split_arm(worst["pattern_id"]))

        return actions


# ---------------------------------------------------------------------------
# Evolutionary population
# ---------------------------------------------------------------------------

class EvolutionaryPopulation:
    """Maintains a population of hyperparameter configurations
    and applies genetic algorithm operators."""

    def __init__(
        self,
        population_size: int = 10,
        elite_count: int = 2,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
    ) -> None:
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: list[HyperparamConfig] = []
        self.generation: int = 0

    def initialize(self, seed_config: HyperparamConfig | None = None) -> None:
        """Create initial population around a seed config."""
        self.population = []
        base = seed_config or HyperparamConfig()

        # Always include the base config
        self.population.append(copy.deepcopy(base))

        # Fill rest with random perturbations
        for _ in range(self.population_size - 1):
            vec = base.to_vector()
            mutated = self._mutate_vector(vec, strength=0.3)
            cfg = HyperparamConfig.from_vector(mutated, generation=0)
            self.population.append(cfg)

    def evolve(self) -> list[HyperparamConfig]:
        """Run one generation of evolution. Returns the new population."""
        self.generation += 1

        # Sort by fitness (descending)
        ranked = sorted(
            self.population,
            key=lambda c: c.mean_fitness,
            reverse=True,
        )

        new_pop: list[HyperparamConfig] = []

        # Elitism: keep top configs unchanged
        for i in range(min(self.elite_count, len(ranked))):
            elite = copy.deepcopy(ranked[i])
            elite.generation = self.generation
            elite.config_id = f"elite_{self.generation}_{i}"
            new_pop.append(elite)

        # Fill rest via crossover + mutation
        while len(new_pop) < self.population_size:
            if random.random() < self.crossover_rate and len(ranked) >= 2:
                # Tournament selection
                parent1 = self._tournament_select(ranked)
                parent2 = self._tournament_select(ranked)
                child_vec = self._crossover(
                    parent1.to_vector(),
                    parent2.to_vector(),
                )
            else:
                parent = self._tournament_select(ranked)
                child_vec = parent.to_vector()

            if random.random() < self.mutation_rate:
                child_vec = self._mutate_vector(child_vec)

            child = HyperparamConfig.from_vector(
                child_vec,
                generation=self.generation,
            )
            new_pop.append(child)

        self.population = new_pop
        return self.population

    def best_config(self) -> HyperparamConfig | None:
        if not self.population:
            return None
        evaluated = [c for c in self.population if c.fitness_scores]
        if not evaluated:
            return self.population[0]
        return max(evaluated, key=lambda c: c.mean_fitness)

    def diversity_score(self) -> float:
        """Measure population diversity as average pairwise distance."""
        if len(self.population) < 2:
            return 0.0
        vecs = [c.to_vector() for c in self.population]
        total = 0.0
        count = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vecs[i], vecs[j], strict=False)))
                total += dist
                count += 1
        return total / count if count > 0 else 0.0

    def _tournament_select(self, ranked: list[HyperparamConfig], k: int = 3) -> HyperparamConfig:
        """Select the best of k random individuals."""
        candidates = random.sample(ranked, min(k, len(ranked)))
        return max(candidates, key=lambda c: c.mean_fitness)

    def _crossover(self, v1: list[float], v2: list[float]) -> list[float]:
        """Uniform crossover: randomly pick each dimension from either parent."""
        return [v1[i] if random.random() < 0.5 else v2[i] for i in range(len(v1))]

    def _mutate_vector(self, vec: list[float], strength: float = 0.1) -> list[float]:
        """Gaussian mutation on each dimension."""
        bounds = FitnessLandscape.BOUNDS
        mutated = []
        for i, v in enumerate(vec):
            if random.random() < 0.3:  # per-dimension mutation probability
                lo, hi = bounds[i]
                scale = (hi - lo) * strength
                v += random.gauss(0, scale)
                v = max(lo, min(hi, v))
            mutated.append(v)
        return mutated

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "diversity": round(self.diversity_score(), 4),
            "configs": [c.to_dict() for c in self.population],
        }


# ---------------------------------------------------------------------------
# LearningVelocityTracker
# ---------------------------------------------------------------------------

class LearningVelocityTracker:
    """Tracks how fast each flywheel is learning over time."""

    def __init__(self) -> None:
        # flywheel_name -> list of (cycle_number, precision)
        self._history: dict[str, list[tuple[int, float]]] = {}

    def record(self, flywheel_name: str, cycle: int, precision: float) -> None:
        if flywheel_name not in self._history:
            self._history[flywheel_name] = []
        self._history[flywheel_name].append((cycle, precision))
        # Keep last 200 entries
        if len(self._history[flywheel_name]) > 200:
            self._history[flywheel_name] = self._history[flywheel_name][-200:]

    def analyze(self, flywheel_name: str, window: int = 10) -> LearningVelocityReport:
        """Compute velocity, acceleration, and trend classification."""
        history = self._history.get(flywheel_name, [])
        if len(history) < 2:
            return LearningVelocityReport(
                flywheel_name=flywheel_name,
                velocity=0.0,
                acceleration=0.0,
                trend="insufficient_data",
                plateau_cycles=0,
                oscillation_amplitude=0.0,
            )

        recent = [p for _, p in history[-window:]]
        deltas = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

        velocity = statistics.mean(deltas) if deltas else 0.0

        # Acceleration: change in velocity over the window
        if len(deltas) >= 4:
            first_half = statistics.mean(deltas[:len(deltas) // 2])
            second_half = statistics.mean(deltas[len(deltas) // 2:])
            acceleration = second_half - first_half
        else:
            acceleration = 0.0

        # Trend classification
        trend = self._classify_trend(deltas, velocity)

        # Plateau detection: count consecutive cycles with |delta| < threshold
        plateau_cycles = 0
        for d in reversed(deltas):
            if abs(d) < 0.002:
                plateau_cycles += 1
            else:
                break

        # Oscillation amplitude
        oscillation_amplitude = max(recent) - min(recent) if len(recent) >= 3 else 0.0

        return LearningVelocityReport(
            flywheel_name=flywheel_name,
            velocity=round(velocity, 6),
            acceleration=round(acceleration, 6),
            trend=trend,
            plateau_cycles=plateau_cycles,
            oscillation_amplitude=round(oscillation_amplitude, 6),
        )

    def _classify_trend(self, deltas: list[float], velocity: float) -> str:
        if not deltas:
            return "insufficient_data"

        abs_velocity = abs(velocity)

        # Check for oscillation: sign changes in deltas
        sign_changes = sum(
            1 for i in range(1, len(deltas))
            if (deltas[i] > 0) != (deltas[i - 1] > 0)
        )
        oscillation_ratio = sign_changes / max(len(deltas) - 1, 1)

        # Oscillation check first: high sign-change ratio with meaningful amplitude
        # but low net direction (abs_velocity can still be > 0 due to windowing)
        if oscillation_ratio > 0.6:
            max_abs_delta = max(abs(d) for d in deltas)
            if max_abs_delta > 0.005:
                return "oscillating"

        if abs_velocity < 0.001:
            return "plateaued"
        if velocity > 0.001:
            return "converging"
        if velocity < -0.003:
            return "diverging"
        return "plateaued"


# ---------------------------------------------------------------------------
# MetaEvolutionEngine
# ---------------------------------------------------------------------------

# Schema for meta_evolution tables
META_EVOLUTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta_evolution_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flywheel_name TEXT NOT NULL,
    cycle_number INTEGER NOT NULL,
    precision REAL DEFAULT 0.0,
    recall REAL DEFAULT 0.0,
    f1 REAL DEFAULT 0.0,
    learning_velocity REAL DEFAULT 0.0,
    cusum_statistic REAL DEFAULT 0.0,
    calibration_ece REAL DEFAULT 0.0,
    fitness REAL DEFAULT 0.0,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta_evolution_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id TEXT NOT NULL UNIQUE,
    config_json TEXT NOT NULL DEFAULT '{}',
    mean_fitness REAL DEFAULT 0.0,
    evaluations INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta_evolution_surgeries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flywheel_name TEXT NOT NULL,
    action_type TEXT NOT NULL,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    reason TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta_evolution_regime_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    change_point INTEGER NOT NULL,
    before_mean REAL DEFAULT 0.0,
    after_mean REAL DEFAULT 0.0,
    effect_size REAL DEFAULT 0.0,
    probability REAL DEFAULT 0.0,
    attribution TEXT DEFAULT '',
    detected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta_evolution_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_number INTEGER NOT NULL,
    population_json TEXT NOT NULL DEFAULT '{}',
    best_fitness REAL DEFAULT 0.0,
    active_config_id TEXT DEFAULT '',
    generation INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL
);
"""


class MetaEvolutionEngine:
    """Top-level meta-evolution orchestrator.

    Runs after the Cortex phase in the daemon cycle. Observes all flywheel
    outputs, measures learning velocity, tunes hyperparameters via GP
    optimization + evolutionary population, performs surgery when needed,
    and detects regime changes.
    """

    def __init__(self, db: SentinelDB) -> None:
        self.db = db
        self.landscape = FitnessLandscape()
        self.population = EvolutionaryPopulation(population_size=8)
        self.velocity_tracker = LearningVelocityTracker()
        self.regression_analyzer = RegressionAnalyzer()
        self.surgeon = FlywheelSurgeon(db)
        self._cycle_number: int = 0
        self._precision_history: list[float] = []
        self._active_config: HyperparamConfig | None = None

        self._ensure_schema()
        self._load_state()

    def _ensure_schema(self) -> None:
        self.db.conn.executescript(META_EVOLUTION_SCHEMA)
        self.db.conn.commit()

    def _load_state(self) -> None:
        """Restore state from DB."""
        try:
            row = self.db.conn.execute(
                "SELECT * FROM meta_evolution_state ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                row = dict(row)
                self._cycle_number = row.get("cycle_number", 0)
                pop_data = json.loads(row.get("population_json", "{}"))
                if pop_data.get("configs"):
                    self.population.generation = pop_data.get("generation", 0)
                    self.population.population = []
                    for cd in pop_data["configs"]:
                        vec = [
                            cd.get("cusum_threshold", 5.0),
                            cd.get("cusum_slack", 0.5),
                            cd.get("drift_sensitivity", 0.10),
                            cd.get("ts_prior_strength", 1.0),
                            cd.get("shadow_promote_threshold", 0.02),
                            cd.get("innovation_exploration_rate", 0.3),
                            cd.get("promote_precision_threshold", 0.80),
                            float(cd.get("promote_min_observations", 10)),
                            cd.get("deprecate_precision_threshold", 0.30),
                            float(cd.get("deprecate_min_observations", 20)),
                        ]
                        cfg = HyperparamConfig.from_vector(
                            vec,
                            config_id=cd.get("config_id", ""),
                            generation=cd.get("generation", 0),
                        )
                        cfg.fitness_scores = []
                        mf = cd.get("mean_fitness", 0.0)
                        if mf > 0 and cd.get("evaluations", 0) > 0:
                            cfg.fitness_scores = [mf]
                        self.population.population.append(cfg)

                active_id = row.get("active_config_id", "")
                if active_id:
                    for c in self.population.population:
                        if c.config_id == active_id:
                            self._active_config = c
                            break

            # Load precision history from flywheel metrics
            metrics = self.db.get_flywheel_metrics_history(days=365, limit=100)
            self._precision_history = [
                float(m.get("precision", 0.0) or 0.0) for m in reversed(metrics)
            ]
        except Exception:
            logger.debug("Failed to load meta-evolution state — starting fresh.", exc_info=True)

    def _save_state(self) -> None:
        """Persist current state to DB."""
        try:
            pop_data = self.population.to_dict()
            active_id = self._active_config.config_id if self._active_config else ""
            best = self.landscape.best_fitness()

            self.db.conn.execute(
                """
                INSERT INTO meta_evolution_state
                    (cycle_number, population_json, best_fitness, active_config_id, generation, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    self._cycle_number,
                    json.dumps(pop_data),
                    best,
                    active_id,
                    self.population.generation,
                    _now_iso(),
                ),
            )
            self.db.conn.commit()
        except Exception:
            logger.debug("Failed to save meta-evolution state", exc_info=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_cycle(self, metrics: dict) -> dict:
        """Execute one meta-evolution cycle.

        Called after the Cortex phase. Analyzes all flywheel outputs
        and applies meta-level improvements.
        """
        self._cycle_number += 1
        precision = metrics.get("precision", 0.0)
        self._precision_history.append(precision)

        # 1. Take a snapshot
        snapshot = self._take_snapshot(metrics)

        # 2. Track learning velocity
        self.velocity_tracker.record("detection", self._cycle_number, precision)
        velocity_report = self.velocity_tracker.analyze("detection")

        # 3. Evaluate current config fitness
        fitness = snapshot.fitness()
        if self._active_config:
            self.landscape.record(self._active_config, fitness)

        # 4. Detect regime changes
        regime_changes: list[RegimeChange] = []
        if len(self._precision_history) >= 8:
            regime_changes = self.regression_analyzer.detect_regime_changes(
                self._precision_history
            )

        # 5. Flywheel surgery if needed
        surgeries: list[SurgeryAction] = []
        if velocity_report.trend in ("diverging", "oscillating") or snapshot.f1 < 0.3:
            surgeries = self.surgeon.perform_surgery(snapshot)

        # 6. Evolve population (every 5 cycles to allow fitness evaluation)
        evolved = False
        if self._cycle_number % 5 == 0 and len(self.population.population) > 0:
            self.population.evolve()
            evolved = True

        # 7. Select next config to evaluate
        new_config: HyperparamConfig | None = None
        if self._cycle_number % 3 == 0:
            if self.landscape.evaluation_count < 5:
                # Initial exploration: use GP suggestion
                new_config = self.landscape.suggest_config(
                    generation=self.population.generation
                )
            elif self.population.population:
                # Use best from population
                new_config = self.population.best_config()

            if new_config:
                self._active_config = new_config
                self._apply_config(new_config)

        # 8. Persist everything
        self._save_snapshot(snapshot)
        self._save_surgeries(surgeries)
        self._save_regime_changes(regime_changes)
        self._save_state()

        result = {
            "cycle_number": self._cycle_number,
            "fitness": round(fitness, 6),
            "velocity": velocity_report.velocity,
            "velocity_trend": velocity_report.trend,
            "plateau_cycles": velocity_report.plateau_cycles,
            "regime_changes_detected": len(regime_changes),
            "surgeries_performed": len(surgeries),
            "surgery_types": [s.action_type for s in surgeries],
            "population_generation": self.population.generation,
            "population_diversity": round(self.population.diversity_score(), 4),
            "best_fitness": round(self.landscape.best_fitness(), 6),
            "active_config": self._active_config.config_id if self._active_config else None,
            "evolved_population": evolved,
            "gp_observations": self.landscape.gp.n_observations,
        }

        logger.info(
            "MetaEvolution cycle %d: fitness=%.4f velocity=%.6f trend=%s surgeries=%d",
            self._cycle_number, fitness, velocity_report.velocity,
            velocity_report.trend, len(surgeries),
        )

        return result

    # ------------------------------------------------------------------
    # Config application
    # ------------------------------------------------------------------

    def _apply_config(self, config: HyperparamConfig) -> None:
        """Apply a hyperparameter configuration to the live flywheel system.

        Writes to the meta_evolution_configs table so the flywheel and
        innovation engine can pick up the new settings.
        """
        try:
            self.db.conn.execute(
                """
                UPDATE meta_evolution_configs SET is_active = 0
                WHERE is_active = 1
                """,
            )
            self.db.conn.execute(
                """
                INSERT INTO meta_evolution_configs
                    (config_id, config_json, mean_fitness, evaluations, generation, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(config_id) DO UPDATE SET
                    config_json = excluded.config_json,
                    mean_fitness = excluded.mean_fitness,
                    evaluations = excluded.evaluations,
                    is_active = 1,
                    updated_at = excluded.updated_at
                """,
                (
                    config.config_id,
                    json.dumps(config.to_dict()),
                    config.mean_fitness,
                    len(config.fitness_scores),
                    config.generation,
                    _now_iso(),
                    _now_iso(),
                ),
            )
            self.db.conn.commit()
        except Exception:
            logger.debug("Failed to apply config %s", config.config_id, exc_info=True)

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def _take_snapshot(self, metrics: dict) -> FlywheelSnapshot:
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)
        cusum = metrics.get("cusum_statistic", 0.0)
        ece = metrics.get("calibration_ece", 0.0)

        # Compute velocity from recent history
        velocity = 0.0
        if len(self._precision_history) >= 2:
            window = self._precision_history[-10:]
            if len(window) >= 2:
                deltas = [window[i] - window[i - 1] for i in range(1, len(window))]
                velocity = statistics.mean(deltas)

        return FlywheelSnapshot(
            flywheel_name="detection",
            cycle_number=self._cycle_number,
            precision=precision,
            recall=recall,
            f1=f1,
            learning_velocity=velocity,
            cusum_statistic=cusum,
            calibration_ece=ece,
        )

    def _save_snapshot(self, snapshot: FlywheelSnapshot) -> None:
        try:
            self.db.conn.execute(
                """
                INSERT INTO meta_evolution_snapshots
                    (flywheel_name, cycle_number, precision, recall, f1,
                     learning_velocity, cusum_statistic, calibration_ece, fitness, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.flywheel_name,
                    snapshot.cycle_number,
                    snapshot.precision,
                    snapshot.recall,
                    snapshot.f1,
                    snapshot.learning_velocity,
                    snapshot.cusum_statistic,
                    snapshot.calibration_ece,
                    snapshot.fitness(),
                    snapshot.timestamp,
                ),
            )
            self.db.conn.commit()
        except Exception:
            logger.debug("Failed to save snapshot", exc_info=True)

    def _save_surgeries(self, surgeries: list[SurgeryAction]) -> None:
        for s in surgeries:
            try:
                self.db.conn.execute(
                    """
                    INSERT INTO meta_evolution_surgeries
                        (flywheel_name, action_type, parameters_json, reason, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        s.flywheel_name,
                        s.action_type,
                        json.dumps(s.parameters),
                        s.reason,
                        s.timestamp,
                    ),
                )
            except Exception:
                logger.debug("Failed to save surgery %s", s.action_type, exc_info=True)
        if surgeries:
            self.db.conn.commit()

    def _save_regime_changes(self, changes: list[RegimeChange]) -> None:
        for ch in changes:
            try:
                self.db.conn.execute(
                    """
                    INSERT INTO meta_evolution_regime_changes
                        (change_point, before_mean, after_mean, effect_size, probability, attribution, detected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ch.change_point,
                        ch.before_mean,
                        ch.after_mean,
                        ch.effect_size,
                        ch.probability,
                        ch.attribution,
                        _now_iso(),
                    ),
                )
            except Exception:
                logger.debug("Failed to save regime change", exc_info=True)
        if changes:
            self.db.conn.commit()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> dict:
        """Generate a meta-evolution status report."""
        velocity_report = self.velocity_tracker.analyze("detection")
        best = self.landscape.best_config()

        # Recent regime changes from DB
        regime_rows = []
        try:
            rows = self.db.conn.execute(
                "SELECT * FROM meta_evolution_regime_changes ORDER BY id DESC LIMIT 10"
            ).fetchall()
            regime_rows = [dict(r) for r in rows]
        except Exception:
            pass

        # Recent surgeries
        surgery_rows = []
        try:
            rows = self.db.conn.execute(
                "SELECT * FROM meta_evolution_surgeries ORDER BY id DESC LIMIT 10"
            ).fetchall()
            surgery_rows = [dict(r) for r in rows]
        except Exception:
            pass

        return {
            "cycle_number": self._cycle_number,
            "velocity": velocity_report.velocity,
            "velocity_trend": velocity_report.trend,
            "acceleration": velocity_report.acceleration,
            "plateau_cycles": velocity_report.plateau_cycles,
            "oscillation_amplitude": velocity_report.oscillation_amplitude,
            "best_fitness": round(self.landscape.best_fitness(), 6),
            "best_config": best.to_dict() if best else None,
            "active_config": self._active_config.to_dict() if self._active_config else None,
            "population": self.population.to_dict(),
            "gp_observations": self.landscape.gp.n_observations,
            "recent_regime_changes": regime_rows,
            "recent_surgeries": surgery_rows,
            "precision_history_length": len(self._precision_history),
        }

    def get_active_config(self) -> HyperparamConfig | None:
        """Return the currently active hyperparameter configuration."""
        return self._active_config
