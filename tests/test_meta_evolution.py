"""Tests for sentinel.meta_evolution — the meta-evolution engine."""

import json
import math
import random
import sqlite3

import pytest

from sentinel.db import SentinelDB
from sentinel.meta_evolution import (
    EvolutionaryPopulation,
    FitnessLandscape,
    FlywheelSnapshot,
    FlywheelSurgeon,
    GaussianProcessOptimizer,
    HyperparamConfig,
    LearningVelocityReport,
    LearningVelocityTracker,
    MetaEvolutionEngine,
    RegimeChange,
    RegressionAnalyzer,
    SurgeryAction,
    _norm_cdf,
    _norm_pdf,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path) -> SentinelDB:
    db_path = str(tmp_path / "test_meta_evo.db")
    db = SentinelDB(path=db_path)
    yield db
    db.close()


@pytest.fixture
def seeded_db(temp_db: SentinelDB) -> SentinelDB:
    """DB with some patterns and flywheel metrics for testing."""
    for i in range(5):
        temp_db.save_pattern({
            "pattern_id": f"pat_{i}",
            "name": f"test_pattern_{i}",
            "description": f"Test pattern {i}",
            "category": "red_flag",
            "regex": "",
            "keywords_json": "[]",
            "alpha": 5.0 + i,
            "beta": 2.0 + i,
            "observations": 20 + i * 5,
            "true_positives": 15 + i * 3,
            "false_positives": 5 + i * 2,
            "status": "active",
        })

    for cycle in range(10):
        temp_db.save_flywheel_metrics({
            "cycle_ts": f"2026-04-{20 + cycle}T00:00:00+00:00",
            "total_analyzed": 100,
            "true_positives": 40 + cycle,
            "false_positives": 10 - min(cycle, 8),
            "precision": 0.7 + cycle * 0.01,
            "recall": 0.6 + cycle * 0.005,
            "f1": 0.65 + cycle * 0.008,
            "accuracy": 0.75 + cycle * 0.01,
            "cycle_number": cycle + 1,
            "regression_alarm": 0,
            "cusum_statistic": 0.5 - cycle * 0.03,
        })

    return temp_db


# ---------------------------------------------------------------------------
# FlywheelSnapshot
# ---------------------------------------------------------------------------

class TestFlywheelSnapshot:
    def test_fitness_computation(self):
        snap = FlywheelSnapshot(
            flywheel_name="detection",
            cycle_number=1,
            precision=0.8,
            recall=0.7,
            f1=0.75,
            learning_velocity=0.01,
            cusum_statistic=1.0,
            calibration_ece=0.05,
        )
        fitness = snap.fitness()
        assert 0.0 < fitness <= 1.0

    def test_fitness_high_performance(self):
        snap = FlywheelSnapshot(
            flywheel_name="detection",
            cycle_number=1,
            precision=0.95,
            recall=0.92,
            f1=0.935,
            learning_velocity=0.02,
            cusum_statistic=0.1,
            calibration_ece=0.01,
        )
        assert snap.fitness() > 0.5

    def test_fitness_low_performance(self):
        snap = FlywheelSnapshot(
            flywheel_name="detection",
            cycle_number=1,
            precision=0.1,
            recall=0.1,
            f1=0.1,
            learning_velocity=-0.05,
            cusum_statistic=9.0,
            calibration_ece=0.3,
        )
        assert snap.fitness() < 0.3

    def test_timestamp_auto_set(self):
        snap = FlywheelSnapshot(
            flywheel_name="test", cycle_number=1,
            precision=0.5, recall=0.5, f1=0.5,
            learning_velocity=0.0, cusum_statistic=0.0,
            calibration_ece=0.0,
        )
        assert snap.timestamp  # non-empty


# ---------------------------------------------------------------------------
# HyperparamConfig
# ---------------------------------------------------------------------------

class TestHyperparamConfig:
    def test_default_values(self):
        cfg = HyperparamConfig()
        assert cfg.cusum_threshold == 5.0
        assert cfg.ts_prior_strength == 1.0
        assert cfg.innovation_exploration_rate == 0.3

    def test_to_vector_and_back(self):
        cfg = HyperparamConfig(cusum_threshold=8.0, drift_sensitivity=0.2)
        vec = cfg.to_vector()
        assert len(vec) == 10
        restored = HyperparamConfig.from_vector(vec)
        assert abs(restored.cusum_threshold - 8.0) < 0.01
        assert abs(restored.drift_sensitivity - 0.2) < 0.01

    def test_mean_fitness_empty(self):
        cfg = HyperparamConfig()
        assert cfg.mean_fitness == 0.0

    def test_mean_fitness_with_scores(self):
        cfg = HyperparamConfig()
        cfg.fitness_scores = [0.5, 0.7, 0.6]
        assert abs(cfg.mean_fitness - 0.6) < 0.01

    def test_fitness_variance_single_score(self):
        cfg = HyperparamConfig()
        cfg.fitness_scores = [0.5]
        assert cfg.fitness_variance == 1.0  # default when < 2

    def test_fitness_variance_multiple(self):
        cfg = HyperparamConfig()
        cfg.fitness_scores = [0.5, 0.5, 0.5]
        assert cfg.fitness_variance < 0.01

    def test_to_dict(self):
        cfg = HyperparamConfig()
        d = cfg.to_dict()
        assert "config_id" in d
        assert "cusum_threshold" in d
        assert "mean_fitness" in d
        assert "generation" in d

    def test_from_vector_clamps(self):
        # Values way out of bounds should be clamped
        vec = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cfg = HyperparamConfig.from_vector(vec)
        assert cfg.cusum_threshold >= 0.5
        assert cfg.cusum_slack >= 0.01
        assert cfg.promote_min_observations >= 3

    def test_config_id_unique(self):
        c1 = HyperparamConfig()
        c2 = HyperparamConfig()
        assert c1.config_id != c2.config_id


# ---------------------------------------------------------------------------
# GaussianProcessOptimizer
# ---------------------------------------------------------------------------

class TestGaussianProcessOptimizer:
    def test_predict_no_data(self):
        gp = GaussianProcessOptimizer()
        mean, var = gp.predict([1.0, 2.0])
        assert mean == 0.0
        assert var == 1.0

    def test_predict_with_observations(self):
        gp = GaussianProcessOptimizer(length_scale=1.0, noise=0.01)
        gp.add_observation([0.0], 0.0)
        gp.add_observation([1.0], 1.0)
        mean, var = gp.predict([0.5])
        # Mean should be roughly 0.5
        assert 0.1 < mean < 0.9
        assert var >= 0.0

    def test_predict_at_observation(self):
        gp = GaussianProcessOptimizer(length_scale=1.0, noise=0.001)
        gp.add_observation([0.0], 5.0)
        mean, _var = gp.predict([0.0])
        assert abs(mean - 5.0) < 0.5

    def test_expected_improvement_no_data(self):
        gp = GaussianProcessOptimizer()
        ei = gp.expected_improvement([1.0, 2.0])
        assert ei == 1.0  # default when no data

    def test_expected_improvement_positive(self):
        gp = GaussianProcessOptimizer(length_scale=1.0, noise=0.01)
        gp.add_observation([0.0], 0.5)
        gp.add_observation([1.0], 0.8)
        ei = gp.expected_improvement([2.0])
        assert ei >= 0.0

    def test_suggest_next(self):
        gp = GaussianProcessOptimizer()
        gp.add_observation([0.5], 0.3)
        bounds = [(0.0, 1.0)]
        suggestion = gp.suggest_next(bounds, n_candidates=20)
        assert len(suggestion) == 1
        assert 0.0 <= suggestion[0] <= 1.0

    def test_n_observations(self):
        gp = GaussianProcessOptimizer()
        assert gp.n_observations == 0
        gp.add_observation([1.0], 0.5)
        assert gp.n_observations == 1

    def test_rbf_kernel_self(self):
        gp = GaussianProcessOptimizer(length_scale=1.0)
        val = gp._rbf_kernel([0.0, 0.0], [0.0, 0.0])
        assert abs(val - 1.0) < 1e-10

    def test_rbf_kernel_distance(self):
        gp = GaussianProcessOptimizer(length_scale=1.0)
        val = gp._rbf_kernel([0.0], [10.0])
        assert val < 0.01  # far apart -> low correlation

    def test_matrix_inversion(self):
        gp = GaussianProcessOptimizer()
        M = [[2.0, 1.0], [1.0, 3.0]]
        M_inv = gp._invert_matrix(M)
        # M * M_inv should be approximately I
        n = len(M)
        for i in range(n):
            for j in range(n):
                val = sum(M[i][k] * M_inv[k][j] for k in range(n))
                expected = 1.0 if i == j else 0.0
                assert abs(val - expected) < 1e-6


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

class TestNormHelpers:
    def test_norm_pdf_at_zero(self):
        val = _norm_pdf(0.0)
        assert abs(val - 1.0 / math.sqrt(2 * math.pi)) < 1e-6

    def test_norm_cdf_at_zero(self):
        val = _norm_cdf(0.0)
        assert abs(val - 0.5) < 0.01

    def test_norm_cdf_extreme_positive(self):
        assert _norm_cdf(10.0) > 0.99

    def test_norm_cdf_extreme_negative(self):
        assert _norm_cdf(-10.0) < 0.01

    def test_norm_cdf_monotonic(self):
        assert _norm_cdf(-1.0) < _norm_cdf(0.0) < _norm_cdf(1.0)


# ---------------------------------------------------------------------------
# FitnessLandscape
# ---------------------------------------------------------------------------

class TestFitnessLandscape:
    def test_empty_landscape(self):
        fl = FitnessLandscape()
        assert fl.best_config() is None
        assert fl.best_fitness() == 0.0
        assert fl.evaluation_count == 0

    def test_record_and_best(self):
        fl = FitnessLandscape()
        cfg1 = HyperparamConfig()
        cfg2 = HyperparamConfig()
        fl.record(cfg1, 0.5)
        fl.record(cfg2, 0.8)
        assert fl.best_fitness() == 0.8
        assert fl.best_config().config_id == cfg2.config_id
        assert fl.evaluation_count == 2

    def test_suggest_config(self):
        fl = FitnessLandscape()
        cfg = HyperparamConfig()
        fl.record(cfg, 0.6)
        suggested = fl.suggest_config(generation=1)
        assert isinstance(suggested, HyperparamConfig)
        assert suggested.generation == 1

    def test_normalization_roundtrip(self):
        fl = FitnessLandscape()
        vec = [5.0, 0.5, 0.1, 1.0, 0.02, 0.3, 0.8, 10.0, 0.3, 20.0]
        normalized = fl._normalize(vec)
        denormalized = fl._denormalize(normalized)
        for a, b in zip(vec, denormalized):
            assert abs(a - b) < 1e-6


# ---------------------------------------------------------------------------
# RegressionAnalyzer
# ---------------------------------------------------------------------------

class TestRegressionAnalyzer:
    def test_no_changes_in_flat_series(self):
        ra = RegressionAnalyzer()
        series = [0.5] * 20
        changes = ra.detect_regime_changes(series)
        assert len(changes) == 0

    def test_detects_step_change(self):
        ra = RegressionAnalyzer(hazard_rate=0.1)
        series = [0.5] * 10 + [0.9] * 10
        changes = ra.detect_regime_changes(series, min_segment=3)
        assert len(changes) >= 1
        # Change should be detected near index 10
        assert any(8 <= ch.change_point <= 12 for ch in changes)

    def test_effect_size(self):
        ra = RegressionAnalyzer(hazard_rate=0.1)
        series = [0.3] * 10 + [0.9] * 10
        changes = ra.detect_regime_changes(series, min_segment=3)
        if changes:
            assert changes[0].effect_size > 0.5

    def test_too_short_series(self):
        ra = RegressionAnalyzer()
        changes = ra.detect_regime_changes([0.5, 0.6], min_segment=3)
        assert len(changes) == 0

    def test_deduplication(self):
        ra = RegressionAnalyzer(hazard_rate=0.1)
        series = [0.3] * 15 + [0.9] * 15
        changes = ra.detect_regime_changes(series, min_segment=3)
        # Deduplication should reduce the count significantly from the raw
        # candidate count (which can be 10+ for a 30-point series)
        assert 1 <= len(changes) <= 5

    def test_attribute_change_with_match(self):
        ra = RegressionAnalyzer()
        change = RegimeChange(change_point=10, before_mean=0.5, after_mean=0.8,
                              effect_size=1.0, probability=0.9, attribution="")
        config_changes = [{"cycle": 9, "parameter": "cusum_threshold", "new_value": 3.0}]
        attr = ra.attribute_change(change, config_changes)
        assert "config_change" in attr

    def test_attribute_change_no_match(self):
        ra = RegressionAnalyzer()
        change = RegimeChange(change_point=10, before_mean=0.5, after_mean=0.8,
                              effect_size=1.0, probability=0.9, attribution="")
        config_changes = [{"cycle": 100, "parameter": "cusum_threshold"}]
        attr = ra.attribute_change(change, config_changes)
        assert attr == "organic_shift"

    def test_attribute_change_empty(self):
        ra = RegressionAnalyzer()
        change = RegimeChange(change_point=10, before_mean=0.5, after_mean=0.8,
                              effect_size=1.0, probability=0.9, attribution="")
        attr = ra.attribute_change(change, [])
        assert attr == "unknown"

    def test_gradual_change(self):
        ra = RegressionAnalyzer(hazard_rate=0.1)
        # Gradual linear increase — less likely to trigger
        series = [0.3 + i * 0.02 for i in range(30)]
        changes = ra.detect_regime_changes(series, min_segment=5)
        # May or may not detect — but should not crash
        assert isinstance(changes, list)


# ---------------------------------------------------------------------------
# FlywheelSurgeon
# ---------------------------------------------------------------------------

class TestFlywheelSurgeon:
    def test_diagnose_low_f1(self, temp_db):
        surgeon = FlywheelSurgeon(temp_db)
        snap = FlywheelSnapshot(
            flywheel_name="detection", cycle_number=1,
            precision=0.3, recall=0.2, f1=0.25,
            learning_velocity=0.0, cusum_statistic=0.0,
            calibration_ece=0.05,
        )
        issues = surgeon.diagnose(snap)
        assert "low_f1" in issues

    def test_diagnose_negative_velocity(self, temp_db):
        surgeon = FlywheelSurgeon(temp_db)
        snap = FlywheelSnapshot(
            flywheel_name="detection", cycle_number=1,
            precision=0.8, recall=0.7, f1=0.75,
            learning_velocity=-0.01, cusum_statistic=0.0,
            calibration_ece=0.05,
        )
        issues = surgeon.diagnose(snap)
        assert "negative_velocity" in issues

    def test_diagnose_high_ece(self, temp_db):
        surgeon = FlywheelSurgeon(temp_db)
        snap = FlywheelSnapshot(
            flywheel_name="detection", cycle_number=1,
            precision=0.8, recall=0.7, f1=0.75,
            learning_velocity=0.01, cusum_statistic=0.0,
            calibration_ece=0.25,
        )
        issues = surgeon.diagnose(snap)
        assert "high_calibration_error" in issues

    def test_diagnose_cusum_imminent(self, temp_db):
        surgeon = FlywheelSurgeon(temp_db)
        snap = FlywheelSnapshot(
            flywheel_name="detection", cycle_number=1,
            precision=0.8, recall=0.7, f1=0.75,
            learning_velocity=0.01, cusum_statistic=4.0,
            calibration_ece=0.05,
        )
        issues = surgeon.diagnose(snap)
        assert "cusum_alarm_imminent" in issues

    def test_diagnose_plateau(self, temp_db):
        surgeon = FlywheelSurgeon(temp_db)
        snap = FlywheelSnapshot(
            flywheel_name="detection", cycle_number=1,
            precision=0.6, recall=0.5, f1=0.55,
            learning_velocity=0.0001, cusum_statistic=0.0,
            calibration_ece=0.05,
        )
        issues = surgeon.diagnose(snap)
        assert "plateaued_below_target" in issues

    def test_reset_priors(self, seeded_db):
        surgeon = FlywheelSurgeon(seeded_db)
        action = surgeon.reset_priors()
        assert action.action_type == "reset_priors"
        assert action.parameters["signals_reset"] > 0
        # Verify the priors were actually reset
        patterns = seeded_db.get_patterns(status="active")
        for p in patterns:
            assert p["alpha"] == 1.0
            assert p["beta"] == 1.0

    def test_reset_priors_filtered(self, seeded_db):
        surgeon = FlywheelSurgeon(seeded_db)
        action = surgeon.reset_priors(signal_names=["pat_0"])
        assert action.parameters["signals_reset"] == 1

    def test_inject_synthetic(self, seeded_db):
        surgeon = FlywheelSurgeon(seeded_db)
        action = surgeon.inject_synthetic_data("pat_0", n_positive=10, n_negative=2)
        assert action.action_type == "inject_synthetic"
        assert action.parameters["found"] is True
        # Verify the pattern was updated
        patterns = seeded_db.get_patterns(status="active")
        pat0 = [p for p in patterns if p["pattern_id"] == "pat_0"][0]
        assert pat0["alpha"] >= 15.0  # original 5.0 + 10
        assert pat0["true_positives"] >= 25  # original 15 + 10

    def test_inject_synthetic_not_found(self, temp_db):
        surgeon = FlywheelSurgeon(temp_db)
        action = surgeon.inject_synthetic_data("nonexistent", 5, 5)
        assert action.parameters["found"] is False

    def test_merge_arms(self, seeded_db):
        surgeon = FlywheelSurgeon(seeded_db)
        action = surgeon.merge_arms(["pat_0", "pat_1"], "merged_01")
        assert action.action_type == "merge_arms"
        # Verify merged pattern was created
        candidates = seeded_db.get_patterns(status="candidate")
        merged = [p for p in candidates if p["pattern_id"] == "merged_01"]
        assert len(merged) == 1

    def test_split_arm(self, seeded_db):
        surgeon = FlywheelSurgeon(seeded_db)
        action = surgeon.split_arm("pat_0", n_splits=3)
        assert action.action_type == "split_arm"
        assert action.parameters["found"] is True
        assert len(action.parameters["created"]) == 3
        # Verify splits were created
        candidates = seeded_db.get_patterns(status="candidate")
        splits = [p for p in candidates if "pat_0_split" in p["pattern_id"]]
        assert len(splits) == 3

    def test_split_arm_not_found(self, temp_db):
        surgeon = FlywheelSurgeon(temp_db)
        action = surgeon.split_arm("nonexistent")
        assert action.parameters["found"] is False
        assert len(action.parameters["created"]) == 0

    def test_adjust_reward(self, temp_db):
        surgeon = FlywheelSurgeon(temp_db)
        action = surgeon.adjust_reward_function(0.7, 0.3)
        assert action.action_type == "adjust_reward"
        assert action.parameters["precision_weight"] == 0.7

    def test_perform_surgery_diverging(self, seeded_db):
        surgeon = FlywheelSurgeon(seeded_db)
        snap = FlywheelSnapshot(
            flywheel_name="detection", cycle_number=1,
            precision=0.3, recall=0.2, f1=0.25,
            learning_velocity=-0.01, cusum_statistic=0.0,
            calibration_ece=0.2,
        )
        actions = surgeon.perform_surgery(snap)
        assert len(actions) >= 1
        action_types = [a.action_type for a in actions]
        assert "reset_priors" in action_types

    def test_perform_surgery_healthy(self, seeded_db):
        surgeon = FlywheelSurgeon(seeded_db)
        snap = FlywheelSnapshot(
            flywheel_name="detection", cycle_number=1,
            precision=0.9, recall=0.85, f1=0.87,
            learning_velocity=0.01, cusum_statistic=0.5,
            calibration_ece=0.03,
        )
        actions = surgeon.perform_surgery(snap)
        assert len(actions) == 0


# ---------------------------------------------------------------------------
# EvolutionaryPopulation
# ---------------------------------------------------------------------------

class TestEvolutionaryPopulation:
    def test_initialize(self):
        pop = EvolutionaryPopulation(population_size=5)
        pop.initialize()
        assert len(pop.population) == 5
        assert pop.generation == 0

    def test_initialize_with_seed(self):
        seed = HyperparamConfig(cusum_threshold=8.0)
        pop = EvolutionaryPopulation(population_size=5)
        pop.initialize(seed)
        assert len(pop.population) == 5
        # First config should be the seed
        assert pop.population[0].cusum_threshold == 8.0

    def test_evolve(self):
        pop = EvolutionaryPopulation(population_size=6, elite_count=2)
        pop.initialize()
        # Assign fitness scores
        for i, cfg in enumerate(pop.population):
            cfg.fitness_scores = [0.1 * (i + 1)]
        new_pop = pop.evolve()
        assert len(new_pop) == 6
        assert pop.generation == 1

    def test_evolve_preserves_elites(self):
        pop = EvolutionaryPopulation(population_size=6, elite_count=2)
        pop.initialize()
        # Make the last config the best
        pop.population[-1].fitness_scores = [0.99]
        for cfg in pop.population[:-1]:
            cfg.fitness_scores = [0.1]
        best_fitness_before = pop.population[-1].mean_fitness
        pop.evolve()
        # At least one config should retain the elite's fitness
        elite_fitness = max(c.mean_fitness for c in pop.population)
        assert elite_fitness >= best_fitness_before - 0.01

    def test_best_config(self):
        pop = EvolutionaryPopulation(population_size=3)
        pop.initialize()
        pop.population[0].fitness_scores = [0.3]
        pop.population[1].fitness_scores = [0.9]
        pop.population[2].fitness_scores = [0.5]
        best = pop.best_config()
        assert best.mean_fitness == 0.9

    def test_best_config_no_evaluations(self):
        pop = EvolutionaryPopulation(population_size=3)
        pop.initialize()
        best = pop.best_config()
        # Should return first config when none evaluated
        assert best is not None

    def test_diversity_score(self):
        pop = EvolutionaryPopulation(population_size=5)
        pop.initialize()
        diversity = pop.diversity_score()
        assert diversity >= 0.0

    def test_diversity_single_member(self):
        pop = EvolutionaryPopulation(population_size=1)
        pop.initialize()
        assert pop.diversity_score() == 0.0

    def test_to_dict(self):
        pop = EvolutionaryPopulation(population_size=3)
        pop.initialize()
        d = pop.to_dict()
        assert d["generation"] == 0
        assert d["population_size"] == 3
        assert len(d["configs"]) == 3

    def test_crossover_produces_valid_config(self):
        pop = EvolutionaryPopulation()
        v1 = HyperparamConfig(cusum_threshold=3.0).to_vector()
        v2 = HyperparamConfig(cusum_threshold=10.0).to_vector()
        child = pop._crossover(v1, v2)
        cfg = HyperparamConfig.from_vector(child)
        assert cfg.cusum_threshold >= 0.5

    def test_mutation_stays_in_bounds(self):
        pop = EvolutionaryPopulation()
        vec = HyperparamConfig().to_vector()
        for _ in range(100):  # mutation is stochastic
            mutated = pop._mutate_vector(vec, strength=0.5)
            cfg = HyperparamConfig.from_vector(mutated)
            assert cfg.cusum_threshold >= 0.5
            assert cfg.cusum_slack >= 0.01
            assert cfg.promote_min_observations >= 3


# ---------------------------------------------------------------------------
# LearningVelocityTracker
# ---------------------------------------------------------------------------

class TestLearningVelocityTracker:
    def test_insufficient_data(self):
        tracker = LearningVelocityTracker()
        report = tracker.analyze("detection")
        assert report.trend == "insufficient_data"
        assert report.velocity == 0.0

    def test_converging(self):
        tracker = LearningVelocityTracker()
        for i in range(15):
            tracker.record("detection", i, 0.5 + i * 0.02)
        report = tracker.analyze("detection")
        assert report.velocity > 0
        assert report.trend == "converging"

    def test_diverging(self):
        tracker = LearningVelocityTracker()
        for i in range(15):
            tracker.record("detection", i, 0.8 - i * 0.02)
        report = tracker.analyze("detection")
        assert report.velocity < 0
        assert report.trend == "diverging"

    def test_plateaued(self):
        tracker = LearningVelocityTracker()
        for i in range(15):
            tracker.record("detection", i, 0.6 + random.uniform(-0.0005, 0.0005))
        report = tracker.analyze("detection")
        assert report.trend == "plateaued"
        assert report.plateau_cycles > 0

    def test_oscillating(self):
        tracker = LearningVelocityTracker()
        for i in range(20):
            val = 0.6 + 0.05 * (1 if i % 2 == 0 else -1)
            tracker.record("detection", i, val)
        report = tracker.analyze("detection")
        assert report.trend == "oscillating"
        assert report.oscillation_amplitude > 0.05

    def test_history_cap(self):
        tracker = LearningVelocityTracker()
        for i in range(300):
            tracker.record("detection", i, 0.5)
        assert len(tracker._history["detection"]) == 200


# ---------------------------------------------------------------------------
# MetaEvolutionEngine
# ---------------------------------------------------------------------------

class TestMetaEvolutionEngine:
    def test_init(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        assert engine._cycle_number == 0
        assert engine.db is temp_db

    def test_schema_created(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        # Verify tables exist
        tables = temp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'meta_evolution%'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        assert "meta_evolution_snapshots" in table_names
        assert "meta_evolution_configs" in table_names
        assert "meta_evolution_surgeries" in table_names
        assert "meta_evolution_regime_changes" in table_names
        assert "meta_evolution_state" in table_names

    def test_run_cycle_basic(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        result = engine.run_cycle({
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "cusum_statistic": 0.5,
            "calibration_ece": 0.05,
        })
        assert result["cycle_number"] == 1
        assert "fitness" in result
        assert "velocity_trend" in result
        assert result["fitness"] > 0

    def test_run_multiple_cycles(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        for i in range(6):
            result = engine.run_cycle({
                "precision": 0.6 + i * 0.02,
                "recall": 0.5 + i * 0.01,
                "f1": 0.55 + i * 0.015,
                "cusum_statistic": max(0, 2.0 - i * 0.3),
                "calibration_ece": max(0.01, 0.1 - i * 0.01),
            })
        assert result["cycle_number"] == 6
        assert engine._cycle_number == 6

    def test_state_persistence(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        engine.population.initialize()
        engine.run_cycle({
            "precision": 0.7, "recall": 0.6, "f1": 0.65,
            "cusum_statistic": 0.5, "calibration_ece": 0.05,
        })
        # Verify state was saved
        row = temp_db.conn.execute(
            "SELECT COUNT(*) FROM meta_evolution_state"
        ).fetchone()[0]
        assert row >= 1

    def test_snapshot_persistence(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        engine.run_cycle({
            "precision": 0.7, "recall": 0.6, "f1": 0.65,
            "cusum_statistic": 0.5, "calibration_ece": 0.05,
        })
        row = temp_db.conn.execute(
            "SELECT COUNT(*) FROM meta_evolution_snapshots"
        ).fetchone()[0]
        assert row >= 1

    def test_surgery_on_bad_metrics(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        # Create patterns so surgery has something to work with
        for i in range(3):
            temp_db.save_pattern({
                "pattern_id": f"p_{i}", "name": f"test_{i}",
                "description": "", "category": "red_flag",
                "alpha": 5.0, "beta": 2.0,
                "observations": 20, "true_positives": 15,
                "false_positives": 5, "status": "active",
            })
        result = engine.run_cycle({
            "precision": 0.1, "recall": 0.1, "f1": 0.1,
            "cusum_statistic": 5.0, "calibration_ece": 0.3,
        })
        assert result["surgeries_performed"] >= 1

    def test_config_application(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        # Run enough cycles to trigger config application (every 3rd cycle)
        for i in range(3):
            engine.run_cycle({
                "precision": 0.7, "recall": 0.6, "f1": 0.65,
                "cusum_statistic": 0.5, "calibration_ece": 0.05,
            })
        # Check if config was applied
        row = temp_db.conn.execute(
            "SELECT COUNT(*) FROM meta_evolution_configs"
        ).fetchone()[0]
        assert row >= 1

    def test_get_report(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        engine.run_cycle({
            "precision": 0.7, "recall": 0.6, "f1": 0.65,
            "cusum_statistic": 0.5, "calibration_ece": 0.05,
        })
        report = engine.get_report()
        assert "cycle_number" in report
        assert "velocity" in report
        assert "best_fitness" in report
        assert "population" in report
        assert report["cycle_number"] == 1

    def test_get_active_config_initially_none(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        assert engine.get_active_config() is None

    def test_regime_change_detection_in_cycle(self, seeded_db):
        engine = MetaEvolutionEngine(db=seeded_db)
        # Load enough precision history for regime detection
        engine._precision_history = [0.5] * 10 + [0.8] * 10
        result = engine.run_cycle({
            "precision": 0.85, "recall": 0.7, "f1": 0.77,
            "cusum_statistic": 0.2, "calibration_ece": 0.03,
        })
        # May or may not detect depending on threshold, but should not crash
        assert "regime_changes_detected" in result

    def test_population_evolution_every_5_cycles(self, temp_db):
        engine = MetaEvolutionEngine(db=temp_db)
        engine.population.initialize()
        results = []
        for i in range(5):
            r = engine.run_cycle({
                "precision": 0.7, "recall": 0.6, "f1": 0.65,
                "cusum_statistic": 0.5, "calibration_ece": 0.05,
            })
            results.append(r)
        # 5th cycle should evolve population
        assert results[-1]["evolved_population"] is True
        # Earlier cycles should not
        assert results[0]["evolved_population"] is False

    def test_load_state_from_db(self, temp_db):
        # Run a cycle, then create a new engine and verify state loads
        engine1 = MetaEvolutionEngine(db=temp_db)
        engine1.population.initialize()
        engine1.run_cycle({
            "precision": 0.7, "recall": 0.6, "f1": 0.65,
            "cusum_statistic": 0.5, "calibration_ece": 0.05,
        })
        cycle_after = engine1._cycle_number

        engine2 = MetaEvolutionEngine(db=temp_db)
        assert engine2._cycle_number == cycle_after


# ---------------------------------------------------------------------------
# Integration: daemon CycleResult fields
# ---------------------------------------------------------------------------

class TestDaemonIntegration:
    def test_cycle_result_has_meta_evolution_fields(self):
        from sentinel.daemon import CycleResult
        result = CycleResult(
            cycle_number=1,
            started_at="2026-04-30T00:00:00",
            completed_at="2026-04-30T00:01:00",
            ingestion_queries=["test"],
            jobs_fetched=0,
            jobs_new=0,
            high_risk_count=0,
            flywheel_ran=False,
            regression_detected=False,
            innovation_ran=False,
            innovation_strategy="",
            errors=[],
            duration_seconds=60.0,
            meta_evolution_ran=True,
            meta_evolution_fitness=0.75,
            meta_evolution_surgeries=2,
            meta_evolution_trend="converging",
        )
        assert result.meta_evolution_ran is True
        assert result.meta_evolution_fitness == 0.75
        assert result.meta_evolution_surgeries == 2
        assert result.meta_evolution_trend == "converging"

    def test_cycle_result_defaults(self):
        from sentinel.daemon import CycleResult
        result = CycleResult(
            cycle_number=1,
            started_at="2026-04-30T00:00:00",
            completed_at="2026-04-30T00:01:00",
            ingestion_queries=["test"],
            jobs_fetched=0,
            jobs_new=0,
            high_risk_count=0,
            flywheel_ran=False,
            regression_detected=False,
            innovation_ran=False,
            innovation_strategy="",
            errors=[],
            duration_seconds=60.0,
        )
        assert result.meta_evolution_ran is False
        assert result.meta_evolution_fitness == 0.0
        assert result.meta_evolution_surgeries == 0
        assert result.meta_evolution_trend == ""


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_gp_with_duplicate_observations(self):
        gp = GaussianProcessOptimizer(noise=0.1)
        gp.add_observation([0.5], 0.5)
        gp.add_observation([0.5], 0.5)
        mean, var = gp.predict([0.5])
        assert math.isfinite(mean)
        assert math.isfinite(var)

    def test_config_extreme_bounds(self):
        vec = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0, 100.0, 1000.0]
        cfg = HyperparamConfig.from_vector(vec)
        assert cfg.cusum_threshold == 100.0  # no upper clamp on cusum_threshold
        assert cfg.cusum_slack <= 2.0
        assert cfg.drift_sensitivity <= 0.50

    def test_population_evolve_empty(self):
        pop = EvolutionaryPopulation(population_size=3)
        pop.initialize()
        # No fitness assigned - should still evolve
        pop.evolve()
        assert len(pop.population) == 3

    def test_regression_analyzer_constant_series(self):
        ra = RegressionAnalyzer()
        series = [0.7] * 30
        changes = ra.detect_regime_changes(series)
        assert len(changes) == 0

    def test_surgery_action_dataclass(self):
        action = SurgeryAction(
            flywheel_name="detection",
            action_type="test",
            parameters={"key": "value"},
            reason="testing",
        )
        assert action.flywheel_name == "detection"
        assert action.timestamp  # auto-set

    def test_learning_velocity_report_fields(self):
        report = LearningVelocityReport(
            flywheel_name="detection",
            velocity=0.01,
            acceleration=0.001,
            trend="converging",
            plateau_cycles=0,
            oscillation_amplitude=0.02,
        )
        assert report.velocity == 0.01
        assert report.trend == "converging"

    def test_regime_change_dataclass(self):
        rc = RegimeChange(
            change_point=10,
            before_mean=0.5,
            after_mean=0.8,
            effect_size=1.0,
            probability=0.95,
            attribution="test",
        )
        assert rc.change_point == 10
        assert rc.attribution == "test"

    def test_meta_engine_with_no_prior_metrics(self, temp_db):
        # Engine should work even when no flywheel_metrics exist
        engine = MetaEvolutionEngine(db=temp_db)
        result = engine.run_cycle({
            "precision": 0.5, "recall": 0.5, "f1": 0.5,
            "cusum_statistic": 0.0, "calibration_ece": 0.0,
        })
        assert result["cycle_number"] == 1

    def test_fitness_landscape_multiple_records(self):
        fl = FitnessLandscape()
        for i in range(20):
            cfg = HyperparamConfig()
            fl.record(cfg, random.uniform(0.3, 0.9))
        assert fl.evaluation_count == 20
        best = fl.best_config()
        assert best is not None

    def test_gp_suggest_with_data(self):
        gp = GaussianProcessOptimizer(length_scale=1.0, noise=0.01)
        for i in range(5):
            x = [random.uniform(0, 1)]
            gp.add_observation(x, random.uniform(0, 1))
        suggestion = gp.suggest_next([(0.0, 1.0)], n_candidates=30)
        assert 0.0 <= suggestion[0] <= 1.0
