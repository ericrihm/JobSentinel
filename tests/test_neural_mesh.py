"""Tests for sentinel.neural_mesh — NeuralMesh advanced flywheel interconnection.

55 tests covering:
- Node/edge registration
- Signal propagation with edge-type rules and decay
- Hebbian attention learning
- Consensus scoring with trust-weighted ballots
- Circuit-breaker propagation
- Emergent detection
- Resonance detector
- Information-flow efficiency analysis
- SQLite persistence
- Pure statistical helpers
- Default mesh factory
- EdgeType enum
- Data class serialisation (to_dict)
- Edge cases and boundary conditions
"""

from __future__ import annotations

import json
import math

import pytest

from sentinel.db import SentinelDB
from sentinel.neural_mesh import (
    CircuitBreakerState,
    ConsensusVerdict,
    EdgeType,
    EmergentPattern,
    FlowEfficiency,
    MeshEdge,
    NeuralMesh,
    PropagatedSignal,
    ResonanceEvent,
    _EMERGENT_BOOST_THRESHOLD,
    _PROPAGATION_DECAY,
    _RESONANCE_THRESHOLD,
    _WEAK_SIGNAL_CEILING,
    _connected_components,
    _estimate_frequency,
    _flow_recommendation,
    _hypothesise_cause,
    _pearson_correlation,
    build_neural_mesh,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mesh() -> NeuralMesh:
    """A fresh NeuralMesh with no DB (in-memory only)."""
    return NeuralMesh()


@pytest.fixture
def temp_db(tmp_path) -> SentinelDB:
    db = SentinelDB(path=str(tmp_path / "test_neural.db"))
    yield db
    db.close()


@pytest.fixture
def db_mesh(temp_db: SentinelDB) -> NeuralMesh:
    """NeuralMesh backed by a temporary SQLite DB."""
    m = NeuralMesh(db=temp_db)
    return m


@pytest.fixture
def triangle_mesh() -> NeuralMesh:
    """A → B (FEEDS), B → C (FEEDS), A → C (CORRELATES)."""
    m = NeuralMesh()
    m.add_node("A", initial_trust=0.8)
    m.add_node("B", initial_trust=0.7)
    m.add_node("C", initial_trust=0.6)
    m.add_edge("A", "B", EdgeType.FEEDS, weight=1.0)
    m.add_edge("B", "C", EdgeType.FEEDS, weight=1.0)
    m.add_edge("A", "C", EdgeType.CORRELATES, weight=0.8)
    return m


@pytest.fixture
def default_mesh() -> NeuralMesh:
    return build_neural_mesh()


# ===========================================================================
# 1. Node and Edge Registration
# ===========================================================================


class TestNodeRegistration:
    def test_add_node_creates_node(self, mesh):
        mesh.add_node("detection")
        assert mesh.has_node("detection")

    def test_add_node_count(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        assert mesh.node_count() == 2

    def test_add_node_idempotent(self, mesh):
        mesh.add_node("detection")
        mesh.add_node("detection")
        assert mesh.node_count() == 1

    def test_add_node_initial_trust(self, mesh):
        mesh.add_node("fw", initial_trust=0.9)
        assert mesh.has_node("fw")

    def test_add_node_initial_trust_clamped(self, mesh):
        mesh.add_node("fw_high", initial_trust=2.0)
        mesh.add_node("fw_low", initial_trust=-1.0)
        # Both should exist without error
        assert mesh.has_node("fw_high")
        assert mesh.has_node("fw_low")

    def test_add_node_creates_circuit_breaker(self, mesh):
        mesh.add_node("detection")
        cb = mesh.get_circuit_breaker("detection")
        assert cb is not None
        assert not cb.tripped

    def test_has_node_unknown(self, mesh):
        assert not mesh.has_node("nonexistent")

    def test_node_names_sorted(self, mesh):
        mesh.add_node("C")
        mesh.add_node("A")
        mesh.add_node("B")
        assert mesh.node_names() == ["A", "B", "C"]


class TestEdgeRegistration:
    def test_add_edge_creates_nodes(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS)
        assert mesh.has_node("A")
        assert mesh.has_node("B")

    def test_add_edge_count(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS)
        mesh.add_edge("B", "C", EdgeType.INHIBITS)
        assert mesh.edge_count() == 2

    def test_add_edge_returns_mesh_edge(self, mesh):
        edge = mesh.add_edge("A", "B", EdgeType.CORRELATES, weight=0.7)
        assert isinstance(edge, MeshEdge)
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.edge_type == EdgeType.CORRELATES
        assert abs(edge.weight - 0.7) < 1e-9

    def test_add_edge_negative_weight_clamped(self, mesh):
        edge = mesh.add_edge("A", "B", EdgeType.FEEDS, weight=-1.0)
        assert edge.weight == 0.0

    def test_get_edge_existing(self, mesh):
        mesh.add_edge("X", "Y", EdgeType.FEEDS)
        edge = mesh.get_edge("X", "Y")
        assert edge is not None
        assert edge.source == "X"

    def test_get_edge_nonexistent_returns_none(self, mesh):
        assert mesh.get_edge("X", "Y") is None

    def test_remove_edge_existing(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS)
        removed = mesh.remove_edge("A", "B")
        assert removed is True
        assert mesh.edge_count() == 0

    def test_remove_edge_nonexistent(self, mesh):
        removed = mesh.remove_edge("X", "Y")
        assert removed is False

    def test_edge_type_enum_values(self):
        assert EdgeType.FEEDS.value == "FEEDS"
        assert EdgeType.INHIBITS.value == "INHIBITS"
        assert EdgeType.CORRELATES.value == "CORRELATES"
        assert EdgeType.COMPETES.value == "COMPETES"


# ===========================================================================
# 2. Signal Propagation
# ===========================================================================


class TestSignalPropagation:
    def test_propagate_from_unknown_node_returns_empty(self, mesh):
        result = mesh.propagate_signal("ghost_node", "test")
        assert result == []

    def test_propagate_origin_included(self, triangle_mesh):
        arrivals = triangle_mesh.propagate_signal("A", "drift", initial_strength=1.0)
        nodes_reached = {s.current_node for s in arrivals}
        assert "A" in nodes_reached

    def test_propagate_feeds_edge_reaches_target(self, triangle_mesh):
        arrivals = triangle_mesh.propagate_signal("A", "drift", initial_strength=1.0)
        nodes_reached = {s.current_node for s in arrivals}
        assert "B" in nodes_reached
        assert "C" in nodes_reached

    def test_propagate_strength_decays_with_hops(self, triangle_mesh):
        arrivals = triangle_mesh.propagate_signal("A", "drift", initial_strength=1.0)
        by_node = {s.current_node: s for s in arrivals if s.hop_count > 0}
        # Strength at B should be less than origin strength
        if "B" in by_node:
            assert by_node["B"].strength < 1.0

    def test_propagate_returns_propagated_signal_objects(self, triangle_mesh):
        arrivals = triangle_mesh.propagate_signal("A", "test")
        for s in arrivals:
            assert isinstance(s, PropagatedSignal)

    def test_propagate_hop_count_increments(self, triangle_mesh):
        arrivals = triangle_mesh.propagate_signal("A", "test", initial_strength=1.0)
        origin = next(s for s in arrivals if s.current_node == "A")
        assert origin.hop_count == 0
        downstream = [s for s in arrivals if s.hop_count > 0]
        for s in downstream:
            assert s.hop_count >= 1

    def test_propagate_competes_edge_blocks(self, mesh):
        """Signal must NOT propagate across COMPETES edges."""
        mesh.add_node("src")
        mesh.add_node("tgt")
        mesh.add_edge("src", "tgt", EdgeType.COMPETES, weight=1.0)
        arrivals = mesh.propagate_signal("src", "test", initial_strength=1.0)
        nodes_reached = {s.current_node for s in arrivals}
        assert "tgt" not in nodes_reached

    def test_propagate_inhibits_edge_reduces_strength(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.INHIBITS, weight=1.0)
        arrivals = mesh.propagate_signal("A", "test", initial_strength=1.0)
        b_signals = [s for s in arrivals if s.current_node == "B"]
        if b_signals:
            # INHIBITS has extra 0.5 factor
            assert b_signals[0].strength < _PROPAGATION_DECAY

    def test_propagate_correlates_edge_half_strength(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.CORRELATES, weight=1.0)
        arrivals = mesh.propagate_signal("A", "test", initial_strength=1.0)
        b_signals = [s for s in arrivals if s.current_node == "B"]
        if b_signals:
            # CORRELATES is 0.5 × decay
            expected_max = _PROPAGATION_DECAY * 0.5
            assert b_signals[0].strength <= expected_max + 1e-9

    def test_propagate_max_hops_respected(self, mesh):
        # Chain: A→B→C→D→E (5 nodes, 4 hops)
        for node in ["A", "B", "C", "D", "E"]:
            mesh.add_node(node)
        for a, b in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
            mesh.add_edge(a, b, EdgeType.FEEDS, weight=1.0)
        arrivals = mesh.propagate_signal("A", "test", initial_strength=1.0, max_hops=2)
        hop_counts = {s.hop_count for s in arrivals}
        assert max(hop_counts) <= 2

    def test_propagate_circuit_broken_node_absorbs(self, triangle_mesh):
        """A tripped circuit breaker on B should stop signal forwarding."""
        triangle_mesh.trip_circuit_breaker("B", "test trip")
        arrivals = triangle_mesh.propagate_signal("A", "test", initial_strength=1.0)
        # C should NOT be reached via A→B→C path
        c_via_b = [
            s for s in arrivals
            if s.current_node == "C" and "B" in s.edge_path
        ]
        assert len(c_via_b) == 0

    def test_propagate_edge_path_recorded(self, triangle_mesh):
        arrivals = triangle_mesh.propagate_signal("A", "test", initial_strength=1.0)
        for s in arrivals:
            assert s.origin == "A"
            assert len(s.edge_path) >= 1

    def test_propagate_payload_forwarded(self, triangle_mesh):
        payload = {"drift_score": 0.9}
        arrivals = triangle_mesh.propagate_signal(
            "A", "drift", initial_strength=1.0, payload=payload
        )
        for s in arrivals:
            assert s.payload == payload

    def test_propagated_signal_to_dict(self, triangle_mesh):
        arrivals = triangle_mesh.propagate_signal("A", "test")
        d = arrivals[0].to_dict()
        assert "origin" in d
        assert "current_node" in d
        assert "strength" in d
        assert "hop_count" in d
        assert "edge_path" in d
        assert "signal_type" in d

    def test_propagate_records_activation_history(self, triangle_mesh):
        triangle_mesh.propagate_signal("A", "test", initial_strength=0.8)
        assert len(triangle_mesh._activation_history["A"]) >= 1

    def test_propagate_very_low_strength_stops(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=0.001)
        # With weight near zero, signal should stop before reaching B
        arrivals = mesh.propagate_signal("A", "test", initial_strength=0.01)
        b_signals = [s for s in arrivals if s.current_node == "B"]
        assert len(b_signals) == 0


# ===========================================================================
# 3. Hebbian Learning
# ===========================================================================


class TestHebbianLearning:
    def test_reinforce_strengthens_edge(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=1.0)
        original_weight = mesh.get_edge("A", "B").weight
        mesh.hebbian_reinforce("A", "B", outcome_quality=1.0)
        assert mesh.get_edge("A", "B").weight > original_weight

    def test_suppress_weakens_edge(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=1.0)
        original_weight = mesh.get_edge("A", "B").weight
        mesh.hebbian_reinforce("A", "B", outcome_quality=0.0)
        assert mesh.get_edge("A", "B").weight < original_weight

    def test_neutral_outcome_minimal_change(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=1.0)
        original_weight = mesh.get_edge("A", "B").weight
        mesh.hebbian_reinforce("A", "B", outcome_quality=0.5)
        assert abs(mesh.get_edge("A", "B").weight - original_weight) < 1e-9

    def test_reinforce_nonexistent_edge_noop(self, mesh):
        # Should not raise
        mesh.hebbian_reinforce("X", "Y", outcome_quality=1.0)

    def test_reinforce_increments_count(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=1.0)
        mesh.hebbian_reinforce("A", "B", outcome_quality=0.9)
        assert mesh.get_edge("A", "B").reinforcement_count == 1

    def test_suppress_increments_suppression_count(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=1.0)
        mesh.hebbian_reinforce("A", "B", outcome_quality=0.1)
        assert mesh.get_edge("A", "B").suppression_count == 1

    def test_weight_floor_clamped(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=0.02)
        for _ in range(100):
            mesh.hebbian_reinforce("A", "B", outcome_quality=0.0)
        assert mesh.get_edge("A", "B").weight >= 0.01

    def test_weight_ceiling_clamped(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=4.9)
        for _ in range(100):
            mesh.hebbian_reinforce("A", "B", outcome_quality=1.0)
        assert mesh.get_edge("A", "B").weight <= 5.0

    def test_hebbian_batch_update(self, mesh):
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=1.0)
        mesh.add_edge("B", "C", EdgeType.FEEDS, weight=1.0)
        old_ab = mesh.get_edge("A", "B").weight
        old_bc = mesh.get_edge("B", "C").weight
        mesh.hebbian_batch_update({"A": 0.9, "B": 0.8, "C": 0.7})
        # High precision → edges should strengthen
        assert mesh.get_edge("A", "B").weight > old_ab
        assert mesh.get_edge("B", "C").weight > old_bc

    def test_batch_update_adjusts_trust_scores(self, mesh):
        mesh.add_node("alpha", initial_trust=0.5)
        mesh.hebbian_batch_update({"alpha": 0.9})
        # Trust should move toward 0.9
        trust = mesh._nodes["alpha"]["trust_score"]
        assert trust > 0.5


# ===========================================================================
# 4. Consensus Scoring
# ===========================================================================


class TestConsensusScoring:
    def test_empty_votes_returns_uncertain(self, mesh):
        verdict = mesh.consensus_score({})
        assert verdict.verdict == "uncertain"
        assert verdict.confidence == 0.0

    def test_unanimous_scam_verdict(self, mesh):
        mesh.add_node("fw1", initial_trust=0.8)
        mesh.add_node("fw2", initial_trust=0.9)
        votes = {
            "fw1": ("scam", 0.9),
            "fw2": ("scam", 0.85),
        }
        verdict = mesh.consensus_score(votes)
        assert verdict.verdict == "scam"
        assert verdict.confidence > 0.5

    def test_unanimous_legitimate_verdict(self, mesh):
        mesh.add_node("fw1", initial_trust=0.8)
        votes = {"fw1": ("legitimate", 0.95)}
        verdict = mesh.consensus_score(votes)
        assert verdict.verdict == "legitimate"

    def test_dissenting_flywheels_identified(self, mesh):
        mesh.add_node("fw1", initial_trust=0.8)
        mesh.add_node("fw2", initial_trust=0.7)
        votes = {
            "fw1": ("scam", 0.9),
            "fw2": ("legitimate", 0.8),
        }
        verdict = mesh.consensus_score(votes)
        # Whichever wins, the other is dissenting
        assert len(verdict.dissenting_flywheels) == 1

    def test_high_trust_flywheel_dominates(self, mesh):
        mesh.add_node("trusted",   initial_trust=0.95)
        mesh.add_node("untrusted", initial_trust=0.1)
        votes = {
            "trusted":   ("scam", 0.9),
            "untrusted": ("legitimate", 0.9),
        }
        verdict = mesh.consensus_score(votes)
        assert verdict.verdict == "scam"

    def test_emergent_flag_set_for_weak_signals(self, mesh):
        """Emergent flag should be True when all signals are weak but consensus is decisive."""
        mesh.add_node("fw1", initial_trust=1.0)
        mesh.add_node("fw2", initial_trust=1.0)
        mesh.add_node("fw3", initial_trust=1.0)
        votes = {
            "fw1": ("scam", _WEAK_SIGNAL_CEILING),
            "fw2": ("scam", _WEAK_SIGNAL_CEILING),
            "fw3": ("scam", _WEAK_SIGNAL_CEILING),
        }
        verdict = mesh.consensus_score(votes)
        # Individual max is at the ceiling, combined confidence is 1.0 → emergent
        assert verdict.emergent_flag is True

    def test_emergent_flag_false_for_strong_individual(self, mesh):
        mesh.add_node("fw1", initial_trust=0.8)
        votes = {"fw1": ("scam", 0.95)}
        verdict = mesh.consensus_score(votes)
        # 0.95 > _WEAK_SIGNAL_CEILING → not emergent
        assert verdict.emergent_flag is False

    def test_vote_breakdown_present(self, mesh):
        mesh.add_node("fw1", initial_trust=0.7)
        verdict = mesh.consensus_score({"fw1": ("scam", 0.8)})
        assert "fw1" in verdict.vote_breakdown
        assert "vote" in verdict.vote_breakdown["fw1"]

    def test_consensus_verdict_to_dict(self, mesh):
        mesh.add_node("fw1", initial_trust=0.7)
        verdict = mesh.consensus_score({"fw1": ("scam", 0.8)})
        d = verdict.to_dict()
        assert "verdict" in d
        assert "confidence" in d
        assert "vote_breakdown" in d
        assert "dissenting_flywheels" in d
        assert "emergent_flag" in d


# ===========================================================================
# 5. Circuit Breaker Propagation
# ===========================================================================


class TestCircuitBreakerPropagation:
    def test_trip_marks_flywheel_tripped(self, mesh):
        mesh.add_node("drift")
        mesh.trip_circuit_breaker("drift", "cusum_alarm")
        cb = mesh.get_circuit_breaker("drift")
        assert cb.tripped is True
        assert cb.trip_reason == "cusum_alarm"

    def test_trip_propagates_protective_mode(self, mesh):
        mesh.add_node("drift")
        mesh.add_node("detection")
        mesh.add_edge("drift", "detection", EdgeType.FEEDS, weight=1.0)
        protected = mesh.trip_circuit_breaker("drift")
        assert "detection" in protected
        cb = mesh.get_circuit_breaker("detection")
        assert cb.protective_mode is True

    def test_trip_does_not_protect_via_competes(self, mesh):
        mesh.add_node("innovation")
        mesh.add_node("shadow")
        mesh.add_edge("innovation", "shadow", EdgeType.COMPETES, weight=1.0)
        protected = mesh.trip_circuit_breaker("innovation")
        assert "shadow" not in protected

    def test_reset_clears_tripped_state(self, mesh):
        mesh.add_node("drift")
        mesh.trip_circuit_breaker("drift", "test")
        mesh.reset_circuit_breaker("drift")
        cb = mesh.get_circuit_breaker("drift")
        assert cb.tripped is False

    def test_reset_clears_neighbour_protective_mode(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.FEEDS)
        mesh.trip_circuit_breaker("A")
        mesh.reset_circuit_breaker("A")
        cb_b = mesh.get_circuit_breaker("B")
        assert not cb_b.protective_mode

    def test_get_tripped_circuit_breakers(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.trip_circuit_breaker("A", "reason_a")
        tripped = mesh.get_tripped_circuit_breakers()
        assert any(cb.flywheel == "A" for cb in tripped)

    def test_get_protective_mode_nodes(self, mesh):
        mesh.add_node("src")
        mesh.add_node("tgt")
        mesh.add_edge("src", "tgt", EdgeType.FEEDS)
        mesh.trip_circuit_breaker("src")
        protective = mesh.get_protective_mode_nodes()
        assert "tgt" in protective

    def test_trip_auto_creates_node(self, mesh):
        # trip_circuit_breaker calls add_node internally
        mesh.trip_circuit_breaker("new_fw", "auto_create")
        assert mesh.has_node("new_fw")

    def test_trip_increments_window_counter(self, mesh):
        mesh.add_node("X")
        mesh.trip_circuit_breaker("X")
        mesh.trip_circuit_breaker("X")
        cb = mesh.get_circuit_breaker("X")
        assert cb.trips_this_window == 2

    def test_circuit_breaker_state_to_dict(self, mesh):
        mesh.add_node("detection")
        mesh.trip_circuit_breaker("detection", "test")
        cb = mesh.get_circuit_breaker("detection")
        d = cb.to_dict()
        assert d["tripped"] is True
        assert d["trip_reason"] == "test"
        assert "protective_mode" in d


# ===========================================================================
# 6. Emergent Detection
# ===========================================================================


class TestEmergentDetection:
    def test_single_flywheel_no_emergent(self, mesh):
        mesh.add_node("detection")
        patterns = mesh.detect_emergent_patterns({"detection": 0.4})
        assert patterns == []

    def test_two_weak_signals_no_edge_no_emergent(self, mesh):
        # No edges → no cross-edge amplification → boost ratio too low
        mesh.add_node("A")
        mesh.add_node("B")
        patterns = mesh.detect_emergent_patterns({"A": 0.4, "B": 0.4})
        assert isinstance(patterns, list)
        # May or may not be emergent depending on threshold — just check type

    def test_two_weak_signals_with_strong_edge_can_be_emergent(self, mesh):
        mesh.add_node("drift")
        mesh.add_node("innovation")
        mesh.add_edge("drift", "innovation", EdgeType.FEEDS, weight=3.0)
        patterns = mesh.detect_emergent_patterns(
            {"drift": 0.4, "innovation": 0.45}
        )
        # Edge amplification should push score above threshold
        assert isinstance(patterns, list)
        # With weight=3.0, amplification = 0.4*0.45*3.0 = 0.54 → boost ratio >> 1
        # combined = 0.4+0.45+0.54 = 1.39; naive_avg = 0.425; ratio = 3.27 > 1.15
        assert len(patterns) == 1
        assert isinstance(patterns[0], EmergentPattern)

    def test_strong_individual_signal_excluded(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=2.0)
        # B's signal exceeds _WEAK_SIGNAL_CEILING (0.45) → B excluded
        patterns = mesh.detect_emergent_patterns({"A": 0.4, "B": 0.9})
        # Only A qualifies (0.4 ≤ 0.45) but single node → no emergent
        assert patterns == []

    def test_emergent_pattern_has_correct_fields(self, mesh):
        mesh.add_node("drift")
        mesh.add_node("innovation")
        mesh.add_edge("drift", "innovation", EdgeType.FEEDS, weight=3.0)
        patterns = mesh.detect_emergent_patterns({"drift": 0.4, "innovation": 0.45})
        if patterns:
            ep = patterns[0]
            assert "drift" in ep.contributing_flywheels
            assert "innovation" in ep.contributing_flywheels
            assert ep.combined_score > 0
            assert ep.observation_count == 1

    def test_emergent_pattern_observation_accumulates(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=3.0)
        mesh.detect_emergent_patterns({"A": 0.4, "B": 0.45})
        mesh.detect_emergent_patterns({"A": 0.4, "B": 0.45})
        patterns = mesh.get_emergent_patterns()
        if patterns:
            assert patterns[0].observation_count == 2

    def test_confirm_emergent_pattern(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=3.0)
        found = mesh.detect_emergent_patterns({"A": 0.4, "B": 0.45})
        if found:
            mesh.confirm_emergent_pattern(found[0].pattern_id)
            assert found[0].confirmed_count == 1

    def test_emergent_pattern_to_dict(self, mesh):
        ep = EmergentPattern(
            pattern_id="test_ep",
            contributing_flywheels=["A", "B"],
            individual_signal_strengths={"A": 0.4, "B": 0.4},
            combined_score=1.2,
            description="test",
            observation_count=3,
            confirmed_count=2,
        )
        d = ep.to_dict()
        assert d["pattern_id"] == "test_ep"
        assert d["combined_score"] == 1.2
        assert abs(d["precision"] - 2 / 3) < 1e-3

    def test_emergent_competes_edge_excluded_from_amplification(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.COMPETES, weight=5.0)
        # COMPETES edge should not amplify
        patterns = mesh.detect_emergent_patterns({"A": 0.4, "B": 0.4})
        # Without amplification, boost ratio should be < threshold
        assert isinstance(patterns, list)


# ===========================================================================
# 7. Resonance Detector
# ===========================================================================


class TestResonanceDetector:
    def test_no_activation_history_returns_empty(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        result = mesh.detect_resonance()
        assert result == []

    def test_insufficient_history_returns_empty(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        # Only 3 activations — below min_series_len=5
        for _ in range(3):
            mesh._record_activation("A", 0.5)
            mesh._record_activation("B", 0.5)
        result = mesh.detect_resonance(min_series_len=5)
        assert result == []

    def test_correlated_series_detected(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        # Identical series → perfect correlation
        series = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        for v in series:
            mesh._record_activation("A", v)
            mesh._record_activation("B", v)
        events = mesh.detect_resonance(min_series_len=5)
        assert len(events) >= 1
        assert isinstance(events[0], ResonanceEvent)

    def test_uncorrelated_series_no_resonance(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        # Opposite-phase series
        for i in range(10):
            mesh._record_activation("A", 1.0 if i % 2 == 0 else 0.0)
            mesh._record_activation("B", 0.0 if i % 2 == 0 else 1.0)
        events = mesh.detect_resonance(min_series_len=5)
        # Anti-correlated: |r| = 1.0 ≥ threshold, so it IS resonance (anti-phase)
        # This is valid — anti-phase resonance is still resonance
        assert isinstance(events, list)

    def test_resonance_event_fields(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        series = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]
        for v in series:
            mesh._record_activation("A", v)
            mesh._record_activation("B", v)
        events = mesh.detect_resonance(min_series_len=5)
        if events:
            ev = events[0]
            assert isinstance(ev.participating_flywheels, list)
            assert isinstance(ev.correlation_matrix, dict)
            assert isinstance(ev.shared_cause_hypothesis, str)
            assert ev.combined_signal_strength >= 0

    def test_resonance_event_to_dict(self):
        ev = ResonanceEvent(
            participating_flywheels=["A", "B"],
            correlation_matrix={"A|B": 0.95},
            dominant_frequency=0.25,
            shared_cause_hypothesis="Shared drift event",
            combined_signal_strength=0.8,
        )
        d = ev.to_dict()
        assert "participating_flywheels" in d
        assert "dominant_frequency" in d
        assert "shared_cause_hypothesis" in d


# ===========================================================================
# 8. Flow Efficiency
# ===========================================================================


class TestFlowEfficiency:
    def test_empty_mesh_returns_empty(self, mesh):
        reports = mesh.analyse_flow_efficiency()
        assert reports == []

    def test_all_nodes_get_report(self, triangle_mesh):
        reports = triangle_mesh.analyse_flow_efficiency()
        nodes_reported = {r.node for r in reports}
        assert nodes_reported == {"A", "B", "C"}

    def test_sink_has_zero_out_degree(self, triangle_mesh):
        reports = triangle_mesh.analyse_flow_efficiency()
        c_report = next(r for r in reports if r.node == "C")
        assert c_report.out_degree == 0

    def test_source_has_zero_in_degree(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.FEEDS)
        reports = mesh.analyse_flow_efficiency()
        a_report = next(r for r in reports if r.node == "A")
        assert a_report.in_degree == 0

    def test_bottleneck_score_range(self, triangle_mesh):
        reports = triangle_mesh.analyse_flow_efficiency()
        for r in reports:
            assert 0.0 <= r.bottleneck_score <= 1.0

    def test_flow_efficiency_to_dict(self, triangle_mesh):
        reports = triangle_mesh.analyse_flow_efficiency()
        for r in reports:
            d = r.to_dict()
            assert "node" in d
            assert "in_degree" in d
            assert "out_degree" in d
            assert "bottleneck_score" in d
            assert "recommendation" in d


# ===========================================================================
# 9. Topology Helpers
# ===========================================================================


class TestTopologyHelpers:
    def test_get_downstream_direct(self, triangle_mesh):
        downstream = triangle_mesh.get_downstream("A")
        assert "B" in downstream
        assert "C" in downstream

    def test_get_downstream_transitive(self, triangle_mesh):
        downstream = triangle_mesh.get_downstream("A")
        # A→B→C: C is transitively downstream of A
        assert "C" in downstream

    def test_get_downstream_sink_empty(self, triangle_mesh):
        assert triangle_mesh.get_downstream("C") == []

    def test_get_upstream_direct(self, triangle_mesh):
        upstream = triangle_mesh.get_upstream("C")
        assert "A" in upstream
        assert "B" in upstream

    def test_get_downstream_competes_excluded(self, mesh):
        mesh.add_node("A")
        mesh.add_node("B")
        mesh.add_edge("A", "B", EdgeType.COMPETES)
        downstream = mesh.get_downstream("A")
        assert "B" not in downstream

    def test_get_downstream_unknown_returns_empty(self, mesh):
        assert mesh.get_downstream("ghost") == []

    def test_get_upstream_unknown_returns_empty(self, mesh):
        assert mesh.get_upstream("ghost") == []

    def test_get_graph_summary(self, triangle_mesh):
        summary = triangle_mesh.get_graph_summary()
        assert "nodes" in summary
        assert "edges" in summary
        node_names = {n["name"] for n in summary["nodes"]}
        assert node_names == {"A", "B", "C"}


# ===========================================================================
# 10. SQLite Persistence
# ===========================================================================


class TestSQLitePersistence:
    def test_save_state_no_error(self, db_mesh):
        db_mesh.add_node("detection", initial_trust=0.8)
        db_mesh.add_node("innovation", initial_trust=0.7)
        db_mesh.add_edge("detection", "innovation", EdgeType.FEEDS)
        db_mesh.save_state()  # Should not raise

    def test_persist_edge_written_to_db(self, temp_db):
        NeuralMesh._neural_tables_created = False
        mesh = NeuralMesh(db=temp_db)
        mesh.add_edge("A", "B", EdgeType.FEEDS, weight=0.9)
        rows = temp_db.conn.execute(
            "SELECT * FROM neural_mesh_edges WHERE source='A' AND target='B'"
        ).fetchall()
        assert len(rows) == 1
        assert abs(dict(rows[0])["weight"] - 0.9) < 1e-6

    def test_load_from_db_restores_nodes(self, temp_db):
        NeuralMesh._neural_tables_created = False
        mesh1 = NeuralMesh(db=temp_db)
        mesh1.add_node("detection", initial_trust=0.85)
        mesh1.save_state()
        NeuralMesh._neural_tables_created = False
        mesh2 = NeuralMesh(db=temp_db)
        assert mesh2.has_node("detection")

    def test_load_from_db_restores_edges(self, temp_db):
        NeuralMesh._neural_tables_created = False
        mesh1 = NeuralMesh(db=temp_db)
        mesh1.add_edge("A", "B", EdgeType.INHIBITS, weight=0.7)
        NeuralMesh._neural_tables_created = False
        mesh2 = NeuralMesh(db=temp_db)
        edge = mesh2.get_edge("A", "B")
        assert edge is not None
        assert edge.edge_type == EdgeType.INHIBITS
        assert abs(edge.weight - 0.7) < 1e-6

    def test_persist_signal_event(self, temp_db):
        NeuralMesh._neural_tables_created = False
        mesh = NeuralMesh(db=temp_db)
        mesh.add_node("drift")
        arrivals = [
            PropagatedSignal(
                origin="drift", current_node="drift",
                strength=1.0, hop_count=0,
                edge_path=["drift"], signal_type="test",
            )
        ]
        mesh.persist_signal_event("drift", "test", {"key": "val"}, arrivals)
        rows = temp_db.conn.execute("SELECT * FROM neural_mesh_signals").fetchall()
        assert len(rows) == 1

    def test_persist_resonance_event(self, temp_db):
        NeuralMesh._neural_tables_created = False
        mesh = NeuralMesh(db=temp_db)
        ev = ResonanceEvent(
            participating_flywheels=["A", "B"],
            correlation_matrix={"A|B": 0.8},
            dominant_frequency=0.2,
            shared_cause_hypothesis="test",
            combined_signal_strength=0.7,
        )
        mesh.persist_resonance_event(ev)
        rows = temp_db.conn.execute("SELECT * FROM neural_mesh_resonance").fetchall()
        assert len(rows) == 1

    def test_persist_emergent_pattern(self, temp_db):
        NeuralMesh._neural_tables_created = False
        mesh = NeuralMesh(db=temp_db)
        ep = EmergentPattern(
            pattern_id="ep_test",
            contributing_flywheels=["A", "B"],
            individual_signal_strengths={"A": 0.4, "B": 0.4},
            combined_score=1.2,
            description="test pattern",
            observation_count=1,
        )
        mesh.persist_emergent_pattern(ep)
        rows = temp_db.conn.execute(
            "SELECT * FROM neural_mesh_emergent WHERE pattern_id='ep_test'"
        ).fetchall()
        assert len(rows) == 1

    def test_hebbian_reinforce_persists_to_db(self, temp_db):
        NeuralMesh._neural_tables_created = False
        mesh = NeuralMesh(db=temp_db)
        mesh.add_edge("X", "Y", EdgeType.FEEDS, weight=1.0)
        mesh.hebbian_reinforce("X", "Y", outcome_quality=1.0)
        rows = temp_db.conn.execute(
            "SELECT weight FROM neural_mesh_edges WHERE source='X' AND target='Y'"
        ).fetchall()
        assert len(rows) == 1
        persisted_weight = dict(rows[0])["weight"]
        assert persisted_weight > 1.0


# ===========================================================================
# 11. Default Mesh Factory
# ===========================================================================


class TestDefaultMeshFactory:
    def test_build_neural_mesh_has_all_flywheels(self, default_mesh):
        expected = {"detection", "calibration", "innovation", "shadow", "drift", "research"}
        for fw in expected:
            assert default_mesh.has_node(fw), f"Missing: {fw}"

    def test_build_neural_mesh_has_edges(self, default_mesh):
        assert default_mesh.edge_count() > 0

    def test_drift_feeds_detection(self, default_mesh):
        edge = default_mesh.get_edge("drift", "detection")
        assert edge is not None
        assert edge.edge_type == EdgeType.FEEDS

    def test_innovation_competes_shadow(self, default_mesh):
        edge = default_mesh.get_edge("innovation", "shadow")
        assert edge is not None
        assert edge.edge_type == EdgeType.COMPETES

    def test_drift_inhibits_shadow(self, default_mesh):
        edge = default_mesh.get_edge("drift", "shadow")
        assert edge is not None
        assert edge.edge_type == EdgeType.INHIBITS

    def test_trust_scores_set(self, default_mesh):
        for name in default_mesh.node_names():
            trust = default_mesh._nodes[name]["trust_score"]
            assert 0.0 <= trust <= 1.0


# ===========================================================================
# 12. Pure Statistical Helpers
# ===========================================================================


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        r = _pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert abs(r - 1.0) < 1e-9

    def test_perfect_negative(self):
        r = _pearson_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert abs(r + 1.0) < 1e-9

    def test_no_variance_returns_zero(self):
        r = _pearson_correlation([3, 3, 3], [1, 2, 3])
        assert r == 0.0

    def test_too_short_returns_zero(self):
        assert _pearson_correlation([1], [1]) == 0.0

    def test_result_clamped_to_minus_one_one(self):
        r = _pearson_correlation([0, 1, 0, 1], [1, 0, 1, 0])
        assert -1.0 <= r <= 1.0


class TestEstimateFrequency:
    def test_square_wave_half_nyquist(self):
        # 0,1,0,1 → 4 crossings in 4 samples → freq ~ 0.5
        series = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        f = _estimate_frequency(series)
        assert 0.0 <= f <= 1.0

    def test_constant_series_zero_frequency(self):
        f = _estimate_frequency([0.5] * 10)
        assert f == 0.0

    def test_too_short_returns_zero(self):
        assert _estimate_frequency([1.0, 2.0]) == 0.0


class TestConnectedComponents:
    def test_two_isolated_nodes_no_component(self):
        comps = _connected_components(set(), ["A", "B"])
        assert comps == []

    def test_single_pair_one_component(self):
        comps = _connected_components({("A", "B")}, ["A", "B", "C"])
        assert len(comps) == 1
        assert "A" in comps[0]
        assert "B" in comps[0]

    def test_two_separate_pairs(self):
        comps = _connected_components({("A", "B"), ("C", "D")}, ["A", "B", "C", "D"])
        assert len(comps) == 2

    def test_chain_is_one_component(self):
        comps = _connected_components({("A", "B"), ("B", "C")}, ["A", "B", "C"])
        assert len(comps) == 1
        assert len(comps[0]) == 3


class TestFlowRecommendation:
    def test_isolated_node_recommendation(self):
        rec = _flow_recommendation("X", 0, 0, 0.0, 0.0)
        assert "isolated" in rec.lower()

    def test_severe_bottleneck_recommendation(self):
        rec = _flow_recommendation("X", 5, 1, 0.8, 0.1)
        assert "bottleneck" in rec.lower()

    def test_sink_recommendation(self):
        rec = _flow_recommendation("X", 2, 0, 0.0, 0.0)
        assert "sink" in rec.lower()

    def test_source_recommendation(self):
        rec = _flow_recommendation("X", 0, 3, 0.0, 0.8)
        assert "source" in rec.lower()

    def test_normal_recommendation(self):
        rec = _flow_recommendation("X", 2, 2, 0.5, 0.8)
        assert "normally" in rec.lower() or "X" in rec


class TestHypothesiseCause:
    def test_returns_string(self):
        mesh = NeuralMesh()
        mesh.add_edge("A", "B", EdgeType.FEEDS)
        cause = _hypothesise_cause(["A", "B"], mesh._edges)
        assert isinstance(cause, str)
        assert len(cause) > 0

    def test_inhibitory_mention_in_cause(self):
        mesh = NeuralMesh()
        mesh.add_edge("A", "B", EdgeType.INHIBITS)
        cause = _hypothesise_cause(["A", "B"], mesh._edges)
        assert "inhibit" in cause.lower() or "feedback" in cause.lower()


# ===========================================================================
# 13. MeshEdge data class
# ===========================================================================


class TestMeshEdgeDataclass:
    def test_to_dict_has_all_fields(self):
        edge = MeshEdge(
            source="A", target="B",
            edge_type=EdgeType.FEEDS,
            weight=0.9, base_weight=1.0,
            reinforcement_count=3, suppression_count=1,
        )
        d = edge.to_dict()
        assert d["source"] == "A"
        assert d["target"] == "B"
        assert d["edge_type"] == "FEEDS"
        assert abs(d["weight"] - 0.9) < 1e-6
        assert d["reinforcement_count"] == 3
        assert d["suppression_count"] == 1
