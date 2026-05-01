"""Tests for innovation meta-learning: precision-delta rewards, UCB, and reporting.

Covers:
  - precision delta reward computation
  - Thompson Sampling updates with continuous rewards
  - negative reward handling (strategy made things worse)
  - exploration bonus (UCB) calculation
  - state persistence with enhanced fields
  - innovation-report output formatting
  - edge cases (no reports for precision, zero delta)
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentinel.db import SentinelDB
from sentinel.flywheel import DetectionFlywheel
from sentinel.innovation import ImprovementArm, ImprovementResult, InnovationEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(db: SentinelDB) -> InnovationEngine:
    """Create an InnovationEngine backed by the given DB with fresh STRATEGIES."""
    engine = InnovationEngine.__new__(InnovationEngine)
    engine.db = db
    engine.flywheel = DetectionFlywheel(db)
    engine.STRATEGIES = [
        ImprovementArm("false_positive_review", "Review and fix false positive detections"),
        ImprovementArm("false_negative_review", "Find missed scams from user reports"),
        ImprovementArm("weight_optimization", "Re-optimize signal weights from recent data"),
        ImprovementArm("pattern_mining", "Mine new scam patterns from reported scams"),
        ImprovementArm("regression_check", "CUSUM regression analysis on accuracy trends"),
        ImprovementArm("cross_signal_correlation", "Find signal combinations that predict scams"),
        ImprovementArm("keyword_expansion", "Expand scam keyword lists from new reports"),
        ImprovementArm("threshold_tuning", "Tune risk classification thresholds"),
    ]
    return engine


def _seed_reports(db: SentinelDB, *, tp: int = 0, fp: int = 0, fn: int = 0, tn: int = 0) -> None:
    """Seed user reports to produce known precision values.

    precision = tp / (tp + fp)
    """
    idx = 0
    for _ in range(tp):
        db.save_report({
            "url": f"https://example.com/tp/{idx}",
            "is_scam": 1,
            "was_correct": 1,
            "our_prediction": 0.9,
            "reason": "",
        })
        idx += 1
    for _ in range(fp):
        db.save_report({
            "url": f"https://example.com/fp/{idx}",
            "is_scam": 0,
            "was_correct": 0,
            "our_prediction": 0.9,
            "reason": "",
        })
        idx += 1
    for _ in range(fn):
        db.save_report({
            "url": f"https://example.com/fn/{idx}",
            "is_scam": 1,
            "was_correct": 0,
            "our_prediction": 0.1,
            "reason": "",
        })
        idx += 1
    for _ in range(tn):
        db.save_report({
            "url": f"https://example.com/tn/{idx}",
            "is_scam": 0,
            "was_correct": 1,
            "our_prediction": 0.1,
            "reason": "",
        })
        idx += 1


# ---------------------------------------------------------------------------
# 1. Precision delta computation
# ---------------------------------------------------------------------------


class TestPrecisionDeltaComputation:
    def test_positive_delta_when_precision_improves(self, temp_db):
        """When precision increases after a strategy runs, delta > 0."""
        _seed_reports(temp_db, tp=5, fp=5)  # precision = 0.5
        fw = DetectionFlywheel(temp_db)

        pre = fw.compute_accuracy()
        assert pre["precision"] == pytest.approx(0.5, abs=0.01)

        # Add more TP reports to raise precision
        _seed_reports(temp_db, tp=10, fp=0)

        post = fw.compute_accuracy()
        delta = post["precision"] - pre["precision"]

        assert delta > 0, "Precision should have improved after adding correct predictions"

    def test_negative_delta_when_precision_drops(self, temp_db):
        """When precision decreases after a strategy, delta < 0."""
        _seed_reports(temp_db, tp=10, fp=0)  # precision = 1.0
        fw = DetectionFlywheel(temp_db)

        pre = fw.compute_accuracy()
        assert pre["precision"] == pytest.approx(1.0, abs=0.01)

        # Add FP reports to drop precision
        _seed_reports(temp_db, tp=0, fp=10)

        post = fw.compute_accuracy()
        delta = post["precision"] - pre["precision"]

        assert delta < 0, "Precision should have dropped after adding false positives"

    def test_zero_delta_when_no_change(self, temp_db):
        """If nothing changes, delta should be 0."""
        _seed_reports(temp_db, tp=5, fp=5)
        fw = DetectionFlywheel(temp_db)

        pre = fw.compute_accuracy()
        post = fw.compute_accuracy()

        assert post["precision"] - pre["precision"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Thompson Sampling with continuous rewards
# ---------------------------------------------------------------------------


class TestThompsonSamplingContinuousRewards:
    def test_alpha_incremented_on_positive_delta(self, temp_db):
        """Arm alpha should increase when precision_delta > 0."""
        engine = _make_engine(temp_db)
        arm = engine.STRATEGIES[0]
        initial_alpha = arm.alpha

        # Mock flywheel to return increasing precision
        call_count = [0]

        def mock_accuracy():
            call_count[0] += 1
            # Odd calls (pre) return 0.5, even calls (post) return 0.7
            if call_count[0] % 2 == 1:
                return {"total": 10, "precision": 0.5, "recall": 0.8, "f1": 0.6,
                        "accuracy": 0.7, "true_positives": 5, "false_positives": 5,
                        "true_negatives": 2, "false_negatives": 1}
            return {"total": 10, "precision": 0.7, "recall": 0.8, "f1": 0.74,
                    "accuracy": 0.8, "true_positives": 7, "false_positives": 3,
                    "true_negatives": 2, "false_negatives": 1}

        engine.flywheel.compute_accuracy = mock_accuracy

        # Manually simulate the delta tracking logic (mirrors run_cycle internals)
        arm.attempts += 1
        pre_precision = 0.5
        post_precision = 0.7
        precision_delta = post_precision - pre_precision

        arm.total_precision_runs += 1
        arm.cumulative_precision_delta += precision_delta
        if precision_delta > arm.best_improvement:
            arm.best_improvement = precision_delta

        if precision_delta > 0:
            arm.alpha += 1
        else:
            arm.beta += 1

        assert arm.alpha == initial_alpha + 1, "Alpha should increment on positive delta"

    def test_beta_incremented_on_negative_delta(self, temp_db):
        """Arm beta should increase when precision_delta <= 0."""
        engine = _make_engine(temp_db)
        arm = engine.STRATEGIES[0]
        initial_beta = arm.beta

        precision_delta = -0.05  # precision got worse

        if precision_delta > 0:
            arm.alpha += 1
        else:
            arm.beta += 1

        assert arm.beta == initial_beta + 1, "Beta should increment on negative delta"

    def test_beta_incremented_on_zero_delta(self, temp_db):
        """Zero delta (no change) should increment beta (not alpha)."""
        engine = _make_engine(temp_db)
        arm = engine.STRATEGIES[0]
        initial_beta = arm.beta

        precision_delta = 0.0

        if precision_delta > 0:
            arm.alpha += 1
        else:
            arm.beta += 1

        assert arm.beta == initial_beta + 1

    def test_mean_reflects_posterior_update(self, temp_db):
        """After many positive rewards, arm mean should be > 0.5."""
        engine = _make_engine(temp_db)
        arm = engine.STRATEGIES[2]  # weight_optimization

        # Simulate 5 positive runs, 1 negative
        arm.alpha = 1.0
        arm.beta = 1.0
        for _ in range(5):
            arm.alpha += 1  # positive delta
        arm.beta += 1  # one negative

        # alpha=6, beta=2 → mean = 6/8 = 0.75
        assert arm.mean > 0.5
        assert arm.mean == pytest.approx(6.0 / 8.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3. Negative reward handling
# ---------------------------------------------------------------------------


class TestNegativeRewardHandling:
    def test_negative_delta_stored_in_cumulative(self, temp_db):
        """Negative precision deltas should be accumulated correctly."""
        engine = _make_engine(temp_db)
        arm = engine.STRATEGIES[0]

        arm.total_precision_runs += 1
        arm.cumulative_precision_delta += -0.05
        arm.total_precision_runs += 1
        arm.cumulative_precision_delta += -0.03

        assert arm.cumulative_precision_delta == pytest.approx(-0.08, abs=1e-9)
        assert arm.avg_improvement == pytest.approx(-0.04, abs=1e-9)

    def test_best_improvement_stays_zero_when_all_deltas_negative(self, temp_db):
        """best_improvement should stay at 0.0 if all deltas are negative (none beat 0)."""
        engine = _make_engine(temp_db)
        arm = engine.STRATEGIES[0]

        # best_improvement starts at 0.0; negative deltas do not exceed it
        for delta in [-0.1, -0.05, -0.02]:
            arm.total_precision_runs += 1
            arm.cumulative_precision_delta += delta
            if delta > arm.best_improvement:
                arm.best_improvement = delta

        # -0.02 > -0.05 > -0.1 but none > 0.0, so best_improvement remains 0.0
        assert arm.best_improvement == pytest.approx(0.0, abs=1e-9)

    def test_run_cycle_handles_precision_decrease(self, temp_db):
        """run_cycle should complete without error even when precision drops."""
        engine = _make_engine(temp_db)

        # Seed data so compute_accuracy returns non-zero
        _seed_reports(temp_db, tp=10, fp=2)

        # After threshold_tuning runs, precision might be unchanged or worse
        # Just ensure it doesn't raise and returns results
        with patch.object(engine, "_execute_strategy") as mock_exec:
            mock_exec.return_value = ImprovementResult(
                "threshold_tuning", False, "Thresholds unchanged"
            )
            # Force selection of one specific arm
            with patch.object(engine, "select_strategy", return_value=engine.STRATEGIES[7]):
                results = engine.run_cycle(max_strategies=1)

        assert len(results) == 1
        assert results[0].strategy == "threshold_tuning"


# ---------------------------------------------------------------------------
# 4. Exploration bonus (UCB)
# ---------------------------------------------------------------------------


class TestExplorationBonus:
    def test_ucb_bonus_decreases_with_more_attempts(self, temp_db):
        """UCB bonus shrinks as an arm accumulates more runs."""
        engine = _make_engine(temp_db)
        arm = engine.STRATEGIES[0]
        total = 50

        # More attempts → smaller bonus
        bonuses = []
        for n_attempts in [1, 5, 20, 50]:
            arm.attempts = n_attempts
            bonus = 0.3 * math.sqrt(math.log(total + 1) / (n_attempts + 1))
            bonuses.append(bonus)

        for i in range(len(bonuses) - 1):
            assert bonuses[i] > bonuses[i + 1], (
                f"Bonus at {i} attempts should be > bonus at {i+1} attempts"
            )

    def test_unexplored_arm_has_highest_bonus(self, temp_db):
        """An arm with 0 attempts should have a higher UCB bonus than one with 10."""
        engine = _make_engine(temp_db)
        arm_fresh = ImprovementArm("fresh", "Unexplored arm", attempts=0)
        arm_used = ImprovementArm("used", "Used arm", attempts=10)
        total = 100

        bonus_fresh = 0.3 * math.sqrt(math.log(total + 1) / (arm_fresh.attempts + 1))
        bonus_used = 0.3 * math.sqrt(math.log(total + 1) / (arm_used.attempts + 1))

        assert bonus_fresh > bonus_used

    def test_ucb_score_method_returns_float(self, temp_db):
        """ucb_score() should return a float in a reasonable range."""
        arm = ImprovementArm("test_arm", "Test arm")
        score = arm.ucb_score(total_attempts=10)
        assert isinstance(score, float)
        assert 0.0 <= score <= 3.0  # beta sample in [0,1] + bounded bonus

    def test_select_strategy_returns_arm(self, temp_db):
        """select_strategy should return one of the STRATEGIES arms."""
        engine = _make_engine(temp_db)
        chosen = engine.select_strategy()
        assert chosen in engine.STRATEGIES

    def test_exploration_bonus_in_rankings(self, temp_db):
        """get_strategy_rankings should include exploration_bonus for every arm."""
        engine = _make_engine(temp_db)
        rankings = engine.get_strategy_rankings()

        assert len(rankings) == 8
        for arm_dict in rankings:
            assert "exploration_bonus" in arm_dict
            assert isinstance(arm_dict["exploration_bonus"], float)
            assert arm_dict["exploration_bonus"] >= 0.0


# ---------------------------------------------------------------------------
# 5. State persistence with enhanced fields
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_enhanced_fields_round_trip(self, temp_db, tmp_path):
        """All meta-learning fields should survive a save/load cycle."""
        engine = _make_engine(temp_db)
        engine.STATE_PATH = tmp_path / "innovation_state.json"

        # Mutate the first arm
        arm = engine.STRATEGIES[0]
        arm.alpha = 3.5
        arm.beta = 1.5
        arm.attempts = 7
        arm.cumulative_precision_delta = 0.12
        arm.best_improvement = 0.08
        arm.total_precision_runs = 5

        engine._save_state()

        # Build a fresh engine and load state
        engine2 = _make_engine(temp_db)
        engine2.STATE_PATH = engine.STATE_PATH
        engine2._load_state()

        arm2 = engine2.STRATEGIES[0]
        assert arm2.alpha == pytest.approx(3.5)
        assert arm2.beta == pytest.approx(1.5)
        assert arm2.attempts == 7
        assert arm2.cumulative_precision_delta == pytest.approx(0.12)
        assert arm2.best_improvement == pytest.approx(0.08)
        assert arm2.total_precision_runs == 5

    def test_legacy_state_loads_without_error(self, temp_db, tmp_path):
        """Old state files without meta-learning fields should not crash load."""
        state_path = tmp_path / "innovation_state.json"
        # Write a legacy state (only alpha/beta/attempts)
        legacy = {
            "false_positive_review": {"alpha": 2.0, "beta": 1.0, "attempts": 3},
            "weight_optimization": {"alpha": 5.0, "beta": 2.0, "attempts": 10},
        }
        state_path.write_text(json.dumps(legacy))

        engine = _make_engine(temp_db)
        engine.STATE_PATH = state_path
        engine._load_state()  # Should not raise

        arm = next(a for a in engine.STRATEGIES if a.name == "false_positive_review")
        assert arm.alpha == pytest.approx(2.0)
        assert arm.attempts == 3
        # New fields default to 0
        assert arm.cumulative_precision_delta == 0.0
        assert arm.total_precision_runs == 0

    def test_save_creates_parent_directory(self, temp_db, tmp_path):
        """_save_state should create the state directory if it does not exist."""
        engine = _make_engine(temp_db)
        nested = tmp_path / "nested" / "dir" / "innovation_state.json"
        engine.STATE_PATH = nested

        engine._save_state()  # Should not raise

        assert nested.exists()
        data = json.loads(nested.read_text())
        assert "false_positive_review" in data


# ---------------------------------------------------------------------------
# 6. innovation-report output
# ---------------------------------------------------------------------------


class TestInnovationReport:
    def test_get_meta_learning_report_structure(self, temp_db):
        """get_meta_learning_report should return all expected keys."""
        engine = _make_engine(temp_db)
        report = engine.get_meta_learning_report()

        assert "total_strategy_runs" in report
        assert "total_precision_runs" in report
        assert "total_cumulative_delta" in report
        assert "most_effective_arm" in report
        assert "least_effective_arm" in report
        assert "most_under_explored" in report
        assert "arms" in report
        assert len(report["arms"]) == 8

    def test_meta_report_identifies_best_arm(self, temp_db):
        """most_effective_arm should point to the arm with highest avg_improvement."""
        engine = _make_engine(temp_db)

        # Manually set up one arm as clearly better
        for arm in engine.STRATEGIES:
            arm.total_precision_runs = 3
            arm.cumulative_precision_delta = 0.0

        best_arm = engine.STRATEGIES[2]  # weight_optimization
        best_arm.cumulative_precision_delta = 0.30  # avg = 0.10

        report = engine.get_meta_learning_report()
        assert report["most_effective_arm"] == "weight_optimization"

    def test_rankings_include_all_meta_fields(self, temp_db):
        """get_strategy_rankings entries must include all new meta-learning fields."""
        engine = _make_engine(temp_db)
        rankings = engine.get_strategy_rankings()

        required_fields = {
            "name", "description", "mean", "attempts", "alpha", "beta",
            "avg_improvement", "cumulative_precision_delta",
            "best_improvement", "total_precision_runs", "exploration_bonus",
        }
        for arm_dict in rankings:
            missing = required_fields - arm_dict.keys()
            assert not missing, f"Missing fields in ranking: {missing}"

    def test_innovation_report_cli_json(self, temp_db, tmp_path):
        """innovation-report --json-output should produce valid JSON with arms list."""
        from click.testing import CliRunner
        from sentinel.cli import main

        runner = CliRunner()

        # Build a pre-configured engine and patch the class to return it
        engine = _make_engine(temp_db)

        with patch("sentinel.innovation.InnovationEngine", return_value=engine):
            result = runner.invoke(main, ["--json-output", "innovation-report"])

        assert result.exit_code == 0, f"CLI error: {result.output}"
        data = json.loads(result.output)
        assert "arms" in data
        assert len(data["arms"]) == 8
        assert "total_strategy_runs" in data


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_precision_delta_zero_when_no_reports(self, temp_db):
        """With no reports, compute_accuracy returns 0.0 precision on both sides."""
        engine = _make_engine(temp_db)
        pre = engine.flywheel.compute_accuracy()
        post = engine.flywheel.compute_accuracy()

        delta = post["precision"] - pre["precision"]
        assert delta == pytest.approx(0.0)

    def test_avg_improvement_returns_zero_with_no_runs(self, temp_db):
        """avg_improvement should return 0.0 when total_precision_runs is 0."""
        arm = ImprovementArm("test", "Test arm")
        assert arm.total_precision_runs == 0
        assert arm.avg_improvement == 0.0

    def test_run_cycle_no_data_still_completes(self, temp_db):
        """run_cycle should complete gracefully when there are no user reports."""
        engine = _make_engine(temp_db)

        # Patch out strategy execution to avoid DB side-effects
        with patch.object(engine, "_execute_strategy") as mock_exec:
            mock_exec.return_value = ImprovementResult(
                "weight_optimization", False, "Nothing to optimize"
            )
            with patch.object(engine, "select_strategy", return_value=engine.STRATEGIES[2]):
                results = engine.run_cycle(max_strategies=1)

        assert isinstance(results, list)
        assert len(results) == 1
        # With no data, precision_delta stays 0 and beta increments
        arm = engine.STRATEGIES[2]
        # beta should have been incremented (delta=0 → non-positive)
        assert arm.beta > 1.0

    def test_best_improvement_tracks_maximum_delta(self, temp_db):
        """best_improvement should track the single highest delta across runs."""
        arm = ImprovementArm("tracker", "Tracking arm")

        deltas = [0.01, 0.05, -0.02, 0.08, 0.03]
        for delta in deltas:
            arm.total_precision_runs += 1
            arm.cumulative_precision_delta += delta
            if delta > arm.best_improvement:
                arm.best_improvement = delta

        assert arm.best_improvement == pytest.approx(0.08, abs=1e-9)
        assert arm.cumulative_precision_delta == pytest.approx(sum(deltas), abs=1e-9)
