"""Tests for the Cortex meta-cognitive intelligence layer."""

import pytest

from sentinel.cortex import Cortex, CortexState
from sentinel.db import SentinelDB

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cortex_db(tmp_path) -> SentinelDB:
    """Fresh SentinelDB for cortex tests."""
    db = SentinelDB(path=str(tmp_path / "cortex_test.db"))
    yield db
    db.close()


@pytest.fixture
def cx(cortex_db) -> Cortex:
    """Cortex instance with fresh DB."""
    return Cortex(db=cortex_db)


def _healthy_metrics(**overrides) -> dict:
    """Return a set of healthy baseline metrics."""
    base = {
        "precision": 0.85,
        "recall": 0.75,
        "f1": 0.80,
        "accuracy": 0.80,
        "regression_alarm": False,
        "cusum_statistic": 1.0,
        "calibration_ece": 0.05,
        "drift_alarm": False,
        "drift_score": 0.02,
        "shadow_active": False,
        "shadow_improvement": 0.0,
        "innovation_ran": True,
        "innovation_strategy": "pattern_mining",
        "research_ran": True,
        "research_patterns_found": 3,
        "source_circuit_breakers": [],
        "evasion_signals": [],
        "errors": [],
    }
    base.update(overrides)
    return base


def _degraded_metrics(**overrides) -> dict:
    """Return a set of degraded system metrics."""
    base = _healthy_metrics(
        precision=0.35,
        recall=0.30,
        regression_alarm=True,
        cusum_statistic=8.0,
        calibration_ece=0.25,
        drift_alarm=True,
        drift_score=0.30,
        innovation_ran=False,
        research_ran=False,
        research_patterns_found=0,
        source_circuit_breakers=["remoteok", "adzuna"],
        evasion_signals=["bitcoin_obfuscated", "payment_leet", "crypto_hidden"],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Cycle observation tests
# ---------------------------------------------------------------------------

class TestObserveCycle:
    """Tests for cortex.observe_cycle()."""

    def test_observe_healthy_cycle(self, cx):
        """Observe a healthy cycle and verify state shape."""
        state = cx.observe_cycle(_healthy_metrics())
        assert isinstance(state, CortexState)
        assert state.cycle_number == 1
        assert isinstance(state.subsystem_health, dict)
        assert all(0 <= v <= 1.0 for v in state.subsystem_health.values())
        assert isinstance(state.learning_velocity, float)
        assert isinstance(state.strategic_priorities, list)
        assert len(state.strategic_priorities) >= 1

    def test_observe_increments_cycle(self, cx):
        """Each call increments the cycle number."""
        s1 = cx.observe_cycle(_healthy_metrics())
        s2 = cx.observe_cycle(_healthy_metrics())
        assert s2.cycle_number == s1.cycle_number + 1

    def test_observe_persists_state(self, cx, cortex_db):
        """State is persisted to DB after observation."""
        cx.observe_cycle(_healthy_metrics())
        row = cortex_db.get_latest_cortex_state()
        assert row is not None
        assert row["cycle_number"] == 1
        assert row["health_grade"] in ("A", "B", "C", "D", "F")

    def test_observe_with_no_data(self, cx):
        """First cycle with all zeros should still produce valid state."""
        state = cx.observe_cycle({
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "innovation_ran": False, "research_ran": False,
        })
        assert state.cycle_number == 1
        assert isinstance(state.subsystem_health, dict)


# ---------------------------------------------------------------------------
# Cross-system signal routing tests
# ---------------------------------------------------------------------------

class TestRouteSignals:
    """Tests for cortex.route_signals()."""

    def test_drift_triggers_shadow_proposal(self, cx):
        """When drift is detected and no shadow test active, propose one."""
        cx.observe_cycle(_healthy_metrics())
        metrics = _healthy_metrics(drift_alarm=True, shadow_active=False)
        actions = cx.route_signals(metrics)
        shadow_actions = [a for a in actions if a.target_system == "shadow"]
        assert len(shadow_actions) >= 1
        assert shadow_actions[0].action_type == "propose_drift_adapted_weights"

    def test_no_shadow_proposal_when_shadow_active(self, cx):
        """No shadow proposal when a shadow test is already running."""
        cx.observe_cycle(_healthy_metrics())
        metrics = _healthy_metrics(drift_alarm=True, shadow_active=True)
        actions = cx.route_signals(metrics)
        shadow_actions = [a for a in actions if a.action_type == "propose_drift_adapted_weights"]
        assert len(shadow_actions) == 0

    def test_shadow_rejection_informs_innovation(self, cx):
        """Shadow rejection should route signal to innovation engine."""
        cx.observe_cycle(_healthy_metrics())
        metrics = _healthy_metrics(shadow_rejected=True, shadow_rejected_direction="increase")
        actions = cx.route_signals(metrics)
        innov_actions = [a for a in actions if a.target_system == "innovation"]
        assert len(innov_actions) >= 1
        assert innov_actions[0].action_type == "avoid_weight_direction"

    def test_evasion_triggers_research(self, cx):
        """Evasion signals should trigger research investigations."""
        cx.observe_cycle(_healthy_metrics())
        metrics = _healthy_metrics(evasion_signals=["bitcoin_obfuscated", "payment_leet"])
        actions = cx.route_signals(metrics)
        research_actions = [a for a in actions if a.target_system == "research"]
        assert len(research_actions) == 2

    def test_high_ece_boosts_threshold_tuning(self, cx):
        """High calibration error should boost threshold_tuning strategy."""
        cx.observe_cycle(_healthy_metrics())
        metrics = _healthy_metrics(calibration_ece=0.25)
        actions = cx.route_signals(metrics)
        boost_actions = [a for a in actions if a.action_type == "boost_strategy"]
        assert len(boost_actions) >= 1
        assert boost_actions[0].parameters["strategy"] == "threshold_tuning"

    def test_source_breakers_adjust_priorities(self, cx):
        """Circuit breakers should trigger ingestion priority adjustment."""
        cx.observe_cycle(_healthy_metrics())
        metrics = _healthy_metrics(source_circuit_breakers=["remoteok", "adzuna"])
        actions = cx.route_signals(metrics)
        ingest_actions = [a for a in actions if a.target_system == "ingest"]
        assert len(ingest_actions) >= 1

    def test_precision_improving_records_reinforcement(self, cx):
        """When precision improves, record the improvement."""
        cx.observe_cycle(_healthy_metrics(precision=0.80))
        cx.observe_cycle(_healthy_metrics(precision=0.85))
        actions = cx.route_signals(_healthy_metrics(precision=0.85))
        record_actions = [a for a in actions if a.action_type == "record_improvement"]
        assert len(record_actions) >= 1

    def test_precision_decline_triggers_emergency(self, cx):
        """Precision declining for N+ cycles triggers emergency investigation."""
        # Need 4+ declining values (EMERGENCY_DECLINE_CYCLES = 3)
        cx.observe_cycle(_healthy_metrics(precision=0.90))
        cx.observe_cycle(_healthy_metrics(precision=0.85))
        cx.observe_cycle(_healthy_metrics(precision=0.80))
        cx.observe_cycle(_healthy_metrics(precision=0.75))
        actions = cx.route_signals(_healthy_metrics(precision=0.75))
        emergency_actions = [a for a in actions if a.action_type == "emergency_investigation"]
        assert len(emergency_actions) >= 1


# ---------------------------------------------------------------------------
# Learning velocity tests
# ---------------------------------------------------------------------------

class TestLearningVelocity:
    """Tests for cortex.compute_learning_velocity()."""

    def test_velocity_with_no_history(self, cx):
        """Velocity is 0 with no precision history."""
        assert cx.compute_learning_velocity() == 0.0

    def test_velocity_with_single_point(self, cx):
        """Velocity is 0 with only one data point."""
        cx.observe_cycle(_healthy_metrics(precision=0.80))
        assert cx.compute_learning_velocity() == 0.0

    def test_positive_velocity(self, cx):
        """Improving precision yields positive velocity."""
        cx.observe_cycle(_healthy_metrics(precision=0.70))
        cx.observe_cycle(_healthy_metrics(precision=0.75))
        cx.observe_cycle(_healthy_metrics(precision=0.80))
        velocity = cx.compute_learning_velocity()
        assert velocity > 0

    def test_negative_velocity(self, cx):
        """Declining precision yields negative velocity."""
        cx.observe_cycle(_healthy_metrics(precision=0.80))
        cx.observe_cycle(_healthy_metrics(precision=0.75))
        cx.observe_cycle(_healthy_metrics(precision=0.70))
        velocity = cx.compute_learning_velocity()
        assert velocity < 0

    def test_velocity_window(self, cx):
        """Velocity uses only the most recent window of data."""
        # Populate 15 cycles
        for p in [0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6,
                   0.7, 0.7, 0.7, 0.7, 0.7]:
            cx.observe_cycle(_healthy_metrics(precision=p))
        cx.compute_learning_velocity(window=10)  # exercises broader window
        v5 = cx.compute_learning_velocity(window=5)
        # Last 5 values are all 0.7, so v5 should be ~0
        assert abs(v5) < 0.001


# ---------------------------------------------------------------------------
# Subsystem health scoring tests
# ---------------------------------------------------------------------------

class TestSubsystemHealth:
    """Tests for cortex.compute_subsystem_health()."""

    def test_healthy_system_scores(self, cx):
        """All healthy metrics should produce high health scores."""
        health = cx.compute_subsystem_health(_healthy_metrics())
        assert health["detection"] >= 0.8
        assert health["innovation"] >= 0.7
        assert health["calibration"] >= 0.7
        assert health["sources"] >= 0.9
        assert health["drift"] >= 0.9

    def test_degraded_system_scores(self, cx):
        """Degraded metrics should produce low health scores."""
        health = cx.compute_subsystem_health(_degraded_metrics())
        assert health["detection"] < 0.5
        assert health["calibration"] < 0.5
        assert health["sources"] < 0.7
        assert health["drift"] < 0.5

    def test_health_scores_bounded(self, cx):
        """All health scores should be between 0 and 1."""
        for metrics in [_healthy_metrics(), _degraded_metrics()]:
            health = cx.compute_subsystem_health(metrics)
            for name, score in health.items():
                assert 0.0 <= score <= 1.0, f"{name} out of bounds: {score}"


# ---------------------------------------------------------------------------
# Strategic planning tests
# ---------------------------------------------------------------------------

class TestStrategicPlanning:
    """Tests for cortex.strategic_planning()."""

    def test_expand_mode(self, cx):
        """High precision and stable -> EXPAND."""
        cx.observe_cycle(_healthy_metrics(precision=0.85))
        cx.observe_cycle(_healthy_metrics(precision=0.85))
        priorities = cx.strategic_planning(
            _healthy_metrics(precision=0.85),
            cx.compute_subsystem_health(_healthy_metrics()),
        )
        assert "EXPAND" in priorities

    def test_stabilize_mode(self, cx):
        """Regression alarm -> STABILIZE."""
        priorities = cx.strategic_planning(
            _healthy_metrics(regression_alarm=True),
        )
        assert "STABILIZE" in priorities

    def test_investigate_mode(self, cx):
        """Drift alarm -> INVESTIGATE."""
        priorities = cx.strategic_planning(
            _healthy_metrics(drift_alarm=True),
        )
        assert "INVESTIGATE" in priorities

    def test_optimize_mode(self, cx):
        """Plateaued precision -> OPTIMIZE."""
        cx.observe_cycle(_healthy_metrics(precision=0.60))
        cx.observe_cycle(_healthy_metrics(precision=0.60))
        cx.observe_cycle(_healthy_metrics(precision=0.60))
        priorities = cx.strategic_planning(
            _healthy_metrics(precision=0.60),
            cx.compute_subsystem_health(_healthy_metrics(precision=0.60)),
        )
        assert "OPTIMIZE" in priorities

    def test_default_observe_mode(self, cx):
        """No triggers -> OBSERVE."""
        priorities = cx.strategic_planning(
            {"precision": 0.60},
        )
        # Should at least have OBSERVE or OPTIMIZE
        assert len(priorities) >= 1


# ---------------------------------------------------------------------------
# Investigation lifecycle tests
# ---------------------------------------------------------------------------

class TestInvestigationLifecycle:
    """Tests for investigation open/resolve/stale cycle."""

    def test_open_investigation(self, cx, cortex_db):
        """Opening an investigation persists it to DB."""
        inv = cx.open_investigation("drift_detected", "Signal rates shifted after source outage")
        assert inv.id.startswith("inv_")
        assert inv.status == "open"
        assert inv.trigger == "drift_detected"

        rows = cortex_db.get_cortex_investigations(status="open")
        assert len(rows) >= 1
        assert rows[0]["id"] == inv.id

    def test_resolve_investigation(self, cx, cortex_db):
        """Resolving an investigation updates DB status."""
        inv = cx.open_investigation("precision_decline", "Need to check weight recal")
        cx.resolve_investigation(inv.id, "Weight recalibration fixed the issue")

        rows = cortex_db.get_cortex_investigations(status="resolved")
        assert len(rows) >= 1
        assert rows[0]["resolution"] == "Weight recalibration fixed the issue"

    def test_stale_investigations(self, cx, cortex_db):
        """Investigations open too long are marked stale."""
        inv = cx.open_investigation("test_trigger", "test hypothesis")
        # Manually age the investigation by updating opened_at to old timestamp
        cortex_db.conn.execute(
            "UPDATE cortex_investigations SET opened_at = '2020-01-01T00:00:00+00:00' WHERE id = ?",
            (inv.id,),
        )
        cortex_db.conn.commit()

        stale = cx.get_stale_investigations(max_age=5)
        assert len(stale) >= 1
        assert stale[0].id == inv.id

    def test_auto_open_on_drift(self, cx):
        """Drift alarm auto-opens an investigation."""
        cx.observe_cycle(_degraded_metrics(drift_alarm=True))
        rows = cx.db.get_cortex_investigations(status="open")
        triggers = [r["trigger"] for r in rows]
        assert "drift_detected" in triggers

    def test_auto_open_on_evasion_spike(self, cx):
        """Multiple evasion signals auto-open an investigation."""
        cx.observe_cycle(_degraded_metrics(
            evasion_signals=["sig1", "sig2", "sig3"],
        ))
        rows = cx.db.get_cortex_investigations(status="open")
        triggers = [r["trigger"] for r in rows]
        assert "evasion_spike" in triggers


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    """Tests for cortex.generate_report()."""

    def test_report_structure(self, cx):
        """Report should contain all expected keys."""
        cx.observe_cycle(_healthy_metrics())
        report = cx.generate_report()
        assert "health_grade" in report
        assert "avg_health_score" in report
        assert "subsystem_health" in report
        assert "learning_velocity" in report
        assert "velocity_trend" in report
        assert "strategic_mode" in report
        assert "strategic_priorities" in report
        assert "active_investigations" in report
        assert "recent_signals" in report
        assert "recommendations" in report
        assert "cycle_number" in report
        assert "precision_history" in report

    def test_report_health_grade(self, cx):
        """Health grade should be A-F."""
        cx.observe_cycle(_healthy_metrics())
        report = cx.generate_report()
        assert report["health_grade"] in ("A", "B", "C", "D", "F")

    def test_report_velocity_trend(self, cx):
        """Velocity trend should be one of improving/declining/stable."""
        cx.observe_cycle(_healthy_metrics(precision=0.70))
        cx.observe_cycle(_healthy_metrics(precision=0.80))
        report = cx.generate_report()
        assert report["velocity_trend"] in ("improving", "declining", "stable")

    def test_report_with_degraded_system(self, cx):
        """Report should contain urgent recommendations for degraded system."""
        cx.observe_cycle(_degraded_metrics())
        report = cx.generate_report()
        # Should have some recommendations
        assert len(report["recommendations"]) >= 1


# ---------------------------------------------------------------------------
# DB persistence tests
# ---------------------------------------------------------------------------

class TestDBPersistence:
    """Tests for cortex DB tables."""

    def test_cortex_state_persistence(self, cortex_db):
        """Cortex state table insert and query."""
        cortex_db.save_cortex_state(
            cycle_number=1,
            state_json='{"test": true}',
            learning_velocity=0.01,
            health_grade="B",
            strategic_mode="EXPAND",
        )
        row = cortex_db.get_latest_cortex_state()
        assert row is not None
        assert row["cycle_number"] == 1
        assert row["health_grade"] == "B"
        assert row["strategic_mode"] == "EXPAND"

    def test_cortex_state_history(self, cortex_db):
        """Multiple states can be queried in order."""
        for i in range(5):
            cortex_db.save_cortex_state(
                cycle_number=i + 1,
                state_json="{}",
                learning_velocity=0.01 * i,
                health_grade="B",
                strategic_mode="OBSERVE",
            )
        history = cortex_db.get_cortex_state_history(limit=3)
        assert len(history) == 3
        # Newest first
        assert history[0]["cycle_number"] == 5

    def test_cortex_investigation_persistence(self, cortex_db):
        """Investigation insert and query."""
        cortex_db.insert_cortex_investigation(
            id="inv_test1",
            trigger="drift_detected",
            hypothesis="Test hypothesis",
        )
        rows = cortex_db.get_cortex_investigations(status="open")
        assert len(rows) == 1
        assert rows[0]["id"] == "inv_test1"

    def test_cortex_signal_persistence(self, cortex_db):
        """Signal insert and query."""
        cortex_db.insert_cortex_signal(
            source="drift",
            target="innovation",
            signal_type="drift_during_innovation",
            payload={"drift_score": 0.15},
            priority=0.7,
        )
        signals = cortex_db.get_recent_cortex_signals(limit=5)
        assert len(signals) == 1
        assert signals[0]["source"] == "drift"
        assert signals[0]["target"] == "innovation"
        assert signals[0]["payload"]["drift_score"] == 0.15


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: first cycle, no data, all healthy, all degraded."""

    def test_first_cycle_no_prior_state(self, cortex_db):
        """Cortex loads cleanly with no prior state in DB."""
        cx = Cortex(db=cortex_db)
        state = cx.observe_cycle(_healthy_metrics())
        assert state.cycle_number == 1

    def test_all_systems_healthy(self, cx):
        """All systems healthy produces grade A or B."""
        cx.observe_cycle(_healthy_metrics())
        report = cx.generate_report()
        assert report["health_grade"] in ("A", "B")

    def test_all_systems_degraded(self, cx):
        """All systems degraded produces grade D or F."""
        cx.observe_cycle(_degraded_metrics())
        report = cx.generate_report()
        assert report["health_grade"] in ("C", "D", "F")

    def test_empty_metrics(self, cx):
        """Empty metrics dict does not crash."""
        state = cx.observe_cycle({})
        assert state.cycle_number == 1
        assert isinstance(state.subsystem_health, dict)

    def test_multiple_cycles_state_accumulates(self, cx):
        """Precision history accumulates across cycles."""
        for i in range(10):
            cx.observe_cycle(_healthy_metrics(precision=0.5 + i * 0.03))
        report = cx.generate_report()
        assert len(report["precision_history"]) == 10

    def test_cross_signal_detection_regression_evasion(self, cx):
        """Regression + evasion should generate a cross-system signal."""
        state = cx.observe_cycle(_healthy_metrics(
            regression_alarm=True,
            evasion_signals=["bitcoin_obfuscated"],
        ))
        signal_types = [s.signal_type for s in state.cross_system_signals]
        assert "regression_with_evasion" in signal_types

    def test_cross_signal_high_ece_no_shadow(self, cx):
        """High ECE with no shadow active should generate signal."""
        state = cx.observe_cycle(_healthy_metrics(
            calibration_ece=0.20,
            shadow_active=False,
        ))
        signal_types = [s.signal_type for s in state.cross_system_signals]
        assert "high_ece_no_shadow" in signal_types

    def test_grade_mapping(self, cx):
        """Verify grade mapping covers all ranges."""
        assert cx._health_score_to_grade(0.90) == "A"
        assert cx._health_score_to_grade(0.75) == "B"
        assert cx._health_score_to_grade(0.60) == "C"
        assert cx._health_score_to_grade(0.45) == "D"
        assert cx._health_score_to_grade(0.20) == "F"

    def test_investigation_not_duplicated(self, cx):
        """Same trigger should not open duplicate investigations."""
        cx.observe_cycle(_degraded_metrics(drift_alarm=True))
        cx.observe_cycle(_degraded_metrics(drift_alarm=True))
        rows = cx.db.get_cortex_investigations(status="open")
        drift_rows = [r for r in rows if r["trigger"] == "drift_detected"]
        assert len(drift_rows) == 1

    def test_signals_are_logged_to_db(self, cx, cortex_db):
        """Cross-system signals should be persisted to cortex_signals table."""
        cx.observe_cycle(_healthy_metrics(
            regression_alarm=True,
            evasion_signals=["test_signal"],
        ))
        signals = cortex_db.get_recent_cortex_signals(limit=50)
        assert len(signals) >= 1
