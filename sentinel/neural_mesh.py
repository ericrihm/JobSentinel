"""Advanced neural flywheel mesh for JobSentinel.

NeuralMesh — A weighted directed graph with typed edges, signal propagation,
Hebbian attention, consensus scoring, circuit-breaker propagation, emergent
detection, resonance detection, and information-flow efficiency tracking.

All persistent state is stored in SQLite via SentinelDB.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.db import SentinelDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------

class EdgeType(StrEnum):
    FEEDS = "FEEDS"           # source output directly consumed by target
    INHIBITS = "INHIBITS"     # source suppresses / damps target activation
    CORRELATES = "CORRELATES" # co-movement without direct causation
    COMPETES = "COMPETES"     # source and target compete for the same resource


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MeshEdge:
    """A single directed edge in the NeuralMesh graph."""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0          # current edge strength (Hebbian-updated)
    base_weight: float = 1.0     # original / default weight
    reinforcement_count: int = 0  # times this edge led to a good outcome
    suppression_count: int = 0   # times this edge led to a bad outcome

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "weight": round(self.weight, 6),
            "base_weight": round(self.base_weight, 6),
            "reinforcement_count": self.reinforcement_count,
            "suppression_count": self.suppression_count,
        }


@dataclass
class PropagatedSignal:
    """A signal that has propagated through the mesh from a source flywheel."""
    origin: str          # flywheel that first fired
    current_node: str    # current position in the propagation
    strength: float      # 0-1, decays with hop distance
    hop_count: int       # how many edges traversed so far
    edge_path: list[str] # ordered list of nodes visited
    signal_type: str     # "drift", "regression", "evasion", "improvement", etc.
    payload: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "current_node": self.current_node,
            "strength": round(self.strength, 4),
            "hop_count": self.hop_count,
            "edge_path": self.edge_path,
            "signal_type": self.signal_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


@dataclass
class ConsensusVerdict:
    """Result of a multi-flywheel voting round on a detection decision."""
    verdict: str            # "scam" | "legitimate" | "uncertain"
    confidence: float       # weighted confidence 0-1
    vote_breakdown: dict    # flywheel_name → {"vote": str, "trust_weight": float}
    dissenting_flywheels: list[str]
    emergent_flag: bool     # True if emerged from combination of weak signals
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "vote_breakdown": self.vote_breakdown,
            "dissenting_flywheels": self.dissenting_flywheels,
            "emergent_flag": self.emergent_flag,
            "timestamp": self.timestamp,
        }


@dataclass
class ResonanceEvent:
    """Two or more flywheels oscillating in a correlated pattern."""
    participating_flywheels: list[str]
    correlation_matrix: dict        # pair → pearson_r
    dominant_frequency: float       # estimated oscillation frequency (cycles⁻¹)
    shared_cause_hypothesis: str    # inferred common driver
    combined_signal_strength: float # amplified signal from resonance
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "participating_flywheels": self.participating_flywheels,
            "correlation_matrix": {k: round(v, 4) for k, v in self.correlation_matrix.items()},
            "dominant_frequency": round(self.dominant_frequency, 4),
            "shared_cause_hypothesis": self.shared_cause_hypothesis,
            "combined_signal_strength": round(self.combined_signal_strength, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class FlowEfficiency:
    """Information-flow efficiency report for a single node."""
    node: str
    in_degree: int
    out_degree: int
    avg_incoming_weight: float
    avg_outgoing_weight: float
    betweenness_proxy: float    # fraction of all shortest paths through this node
    bottleneck_score: float     # 0-1; 1 = severe bottleneck
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "avg_incoming_weight": round(self.avg_incoming_weight, 4),
            "avg_outgoing_weight": round(self.avg_outgoing_weight, 4),
            "betweenness_proxy": round(self.betweenness_proxy, 4),
            "bottleneck_score": round(self.bottleneck_score, 4),
            "recommendation": self.recommendation,
        }


@dataclass
class EmergentPattern:
    """A scam pattern detectable only through cross-flywheel signal combination."""
    pattern_id: str
    contributing_flywheels: list[str]
    individual_signal_strengths: dict  # flywheel → float (0-1)
    combined_score: float              # > sum of parts when emergent
    description: str
    observation_count: int = 0
    confirmed_count: int = 0
    timestamp: str = field(default_factory=_now_iso)

    @property
    def precision(self) -> float:
        return self.confirmed_count / self.observation_count if self.observation_count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "contributing_flywheels": self.contributing_flywheels,
            "individual_signal_strengths": {k: round(v, 4) for k, v in self.individual_signal_strengths.items()},
            "combined_score": round(self.combined_score, 4),
            "description": self.description,
            "observation_count": self.observation_count,
            "confirmed_count": self.confirmed_count,
            "precision": round(self.precision, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class CircuitBreakerState:
    """Protective state that a flywheel node can enter."""
    flywheel: str
    tripped: bool = False
    trip_reason: str = ""
    trips_this_window: int = 0
    last_trip_ts: str = ""
    protective_mode: bool = False   # downstream nodes entering pre-emptive protection

    def to_dict(self) -> dict:
        return {
            "flywheel": self.flywheel,
            "tripped": self.tripped,
            "trip_reason": self.trip_reason,
            "trips_this_window": self.trips_this_window,
            "last_trip_ts": self.last_trip_ts,
            "protective_mode": self.protective_mode,
        }


# ---------------------------------------------------------------------------
# NeuralMesh
# ---------------------------------------------------------------------------

# Signal decay factor per hop (so 1 hop: strength * DECAY, 2 hops: * DECAY², …)
_PROPAGATION_DECAY = 0.6

# Minimum signal strength to keep propagating
_MIN_PROPAGATION_STRENGTH = 0.05

# Hebbian learning rate
_HEBBIAN_RATE = 0.05

# Resonance correlation threshold
_RESONANCE_THRESHOLD = 0.65

# Emergent detection: max individual signal strength that still qualifies as "weak"
_WEAK_SIGNAL_CEILING = 0.45

# Emergent detection: min combined score boost to call it emergent
_EMERGENT_BOOST_THRESHOLD = 1.15  # combined ≥ 1.15 × weighted average


class NeuralMesh:
    """Advanced flywheel interconnection system.

    Graph model
    -----------
    * Nodes  — flywheel names (str).
    * Edges  — MeshEdge with typed relationship and learned weight.

    Key capabilities
    ----------------
    1. Signal propagation — fires from a source, decays with hop distance.
    2. Attention (Hebbian learning) — edges that led to good outcomes strengthen.
    3. Consensus scoring — trust-weighted multi-flywheel voting.
    4. Circuit-breaker propagation — tripped nodes warn connected neighbours.
    5. Emergent detection — combination of weak signals from multiple flywheels.
    6. Resonance detection — correlated oscillations across flywheels.
    7. Information-flow efficiency — bottleneck identification.
    8. SQLite persistence for all state.
    """

    def __init__(self, db: SentinelDB | None = None) -> None:
        self._db = db
        # node_name → CircuitBreakerState
        self._circuit_breakers: dict[str, CircuitBreakerState] = {}
        # (source, target) → MeshEdge
        self._edges: dict[tuple[str, str], MeshEdge] = {}
        # node_name → {in_edges, out_edges, trust_score, precision_history}
        self._nodes: dict[str, dict] = {}
        # node_name → recent activation magnitudes (for resonance)
        self._activation_history: dict[str, list[float]] = {}
        # emergent patterns observed so far
        self._emergent_patterns: dict[str, EmergentPattern] = {}

        if db is not None:
            self._load_from_db()

    # ------------------------------------------------------------------
    # Node / edge registration
    # ------------------------------------------------------------------

    def add_node(self, name: str, initial_trust: float = 0.5) -> None:
        """Register a flywheel node."""
        if name not in self._nodes:
            self._nodes[name] = {
                "trust_score": max(0.0, min(1.0, initial_trust)),
                "precision_history": [],
                "activation_count": 0,
            }
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreakerState(flywheel=name)
        if name not in self._activation_history:
            self._activation_history[name] = []

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType = EdgeType.FEEDS,
        weight: float = 1.0,
    ) -> MeshEdge:
        """Add a directed edge. Nodes are auto-created if missing."""
        self.add_node(source)
        self.add_node(target)
        edge = MeshEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=max(0.0, weight),
            base_weight=max(0.0, weight),
        )
        self._edges[(source, target)] = edge
        if self._db is not None:
            self._persist_edge(edge)
        return edge

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge. Returns True if it existed."""
        key = (source, target)
        if key in self._edges:
            del self._edges[key]
            return True
        return False

    def get_edge(self, source: str, target: str) -> MeshEdge | None:
        return self._edges.get((source, target))

    def has_node(self, name: str) -> bool:
        return name in self._nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def node_names(self) -> list[str]:
        return sorted(self._nodes.keys())

    # ------------------------------------------------------------------
    # 1. Signal Propagation
    # ------------------------------------------------------------------

    def propagate_signal(
        self,
        origin: str,
        signal_type: str,
        initial_strength: float = 1.0,
        payload: dict | None = None,
        max_hops: int = 5,
    ) -> list[PropagatedSignal]:
        """Fire a signal from *origin* and let it travel through the graph.

        Rules
        -----
        * FEEDS edges propagate the signal forward (strength × decay).
        * INHIBITS edges invert the signal direction and dampen by 50%.
        * CORRELATES edges propagate at reduced strength (× 0.5).
        * COMPETES edges do not propagate the signal.
        * Signal dies when strength < _MIN_PROPAGATION_STRENGTH or hops > max_hops.
        * Circuit-broken nodes absorb the signal (don't forward it).
        """
        if origin not in self._nodes:
            return []

        payload = payload or {}
        arrivals: list[PropagatedSignal] = []
        visited_in_path: set[tuple[str, int]] = set()  # (node, hop)

        # BFS queue: (current_node, strength, hop_count, path_so_far)
        queue: list[tuple[str, float, int, list[str]]] = [
            (origin, initial_strength, 0, [origin])
        ]

        while queue:
            current, strength, hops, path = queue.pop(0)

            # Record arrival at this node
            sig = PropagatedSignal(
                origin=origin,
                current_node=current,
                strength=strength,
                hop_count=hops,
                edge_path=list(path),
                signal_type=signal_type,
                payload=payload,
            )
            arrivals.append(sig)

            # Update activation history for resonance tracking
            self._record_activation(current, strength)

            # Circuit breaker: absorb and don't forward
            cb = self._circuit_breakers.get(current)
            if cb and cb.tripped and current != origin:
                continue

            if hops >= max_hops:
                continue

            # Propagate along outgoing edges
            for (src, tgt), edge in self._edges.items():
                if src != current:
                    continue
                if (tgt, hops + 1) in visited_in_path:
                    continue

                new_strength = self._compute_propagated_strength(strength, edge)
                if new_strength < _MIN_PROPAGATION_STRENGTH:
                    continue

                visited_in_path.add((tgt, hops + 1))
                queue.append((tgt, new_strength, hops + 1, path + [tgt]))

        return arrivals

    def _compute_propagated_strength(self, incoming: float, edge: MeshEdge) -> float:
        """Apply edge-type-specific decay to an incoming signal strength."""
        w = edge.weight
        if edge.edge_type == EdgeType.FEEDS:
            return incoming * _PROPAGATION_DECAY * w
        elif edge.edge_type == EdgeType.INHIBITS or edge.edge_type == EdgeType.CORRELATES:
            return incoming * _PROPAGATION_DECAY * w * 0.5
        else:  # COMPETES — no propagation
            return 0.0

    def _record_activation(self, node: str, strength: float) -> None:
        """Append an activation magnitude to the node's history."""
        hist = self._activation_history.setdefault(node, [])
        hist.append(strength)
        # Keep last 50 activations
        if len(hist) > 50:
            hist[:] = hist[-50:]
        if node in self._nodes:
            self._nodes[node]["activation_count"] = self._nodes[node].get("activation_count", 0) + 1

    # ------------------------------------------------------------------
    # 2. Hebbian Attention (edge weight learning)
    # ------------------------------------------------------------------

    def hebbian_reinforce(self, source: str, target: str, outcome_quality: float) -> None:
        """Strengthen or weaken an edge based on whether it led to a good outcome.

        outcome_quality: 1.0 = perfect outcome, 0.0 = worst outcome, 0.5 = neutral.
        Uses Hebbian-style update: Δw = η × (outcome − 0.5) × current_weight.
        Weight is clamped to [0.01, 5.0].
        """
        edge = self._edges.get((source, target))
        if edge is None:
            return

        delta = _HEBBIAN_RATE * (outcome_quality - 0.5) * edge.weight
        edge.weight = max(0.01, min(5.0, edge.weight + delta))

        if outcome_quality >= 0.6:
            edge.reinforcement_count += 1
        elif outcome_quality <= 0.4:
            edge.suppression_count += 1

        if self._db is not None:
            self._persist_edge(edge)

    def hebbian_batch_update(
        self,
        flywheel_outcomes: dict[str, float],
    ) -> None:
        """Update all edges connecting nodes in *flywheel_outcomes*.

        flywheel_outcomes: {flywheel_name: precision_score}
        For each edge (A→B), the outcome quality is the geometric mean of
        A's and B's precision scores.
        """
        for (src, tgt), _edge in self._edges.items():
            if src in flywheel_outcomes and tgt in flywheel_outcomes:
                combined = math.sqrt(flywheel_outcomes[src] * flywheel_outcomes[tgt])
                self.hebbian_reinforce(src, tgt, combined)

        # Also update node trust scores
        for name, precision in flywheel_outcomes.items():
            if name in self._nodes:
                old_trust = self._nodes[name]["trust_score"]
                # Exponential moving average: α = 0.2
                new_trust = 0.8 * old_trust + 0.2 * precision
                self._nodes[name]["trust_score"] = max(0.0, min(1.0, new_trust))
                precision_hist = self._nodes[name].setdefault("precision_history", [])
                precision_hist.append(precision)
                if len(precision_hist) > 50:
                    precision_hist[:] = precision_hist[-50:]

    # ------------------------------------------------------------------
    # 3. Consensus Scoring
    # ------------------------------------------------------------------

    def consensus_score(
        self,
        flywheel_votes: dict[str, tuple[str, float]],
    ) -> ConsensusVerdict:
        """Produce a trust-weighted consensus verdict.

        Args:
            flywheel_votes: {flywheel_name: (vote_label, raw_confidence)}
                            vote_label should be "scam", "legitimate", or "uncertain".

        Returns:
            ConsensusVerdict with weighted majority vote and dissent list.
        """
        if not flywheel_votes:
            return ConsensusVerdict(
                verdict="uncertain",
                confidence=0.0,
                vote_breakdown={},
                dissenting_flywheels=[],
                emergent_flag=False,
            )

        # Accumulate weighted votes per class
        weighted_votes: dict[str, float] = {}
        total_weight = 0.0
        vote_breakdown: dict[str, dict] = {}

        for fw_name, (vote, raw_conf) in flywheel_votes.items():
            trust = self._nodes.get(fw_name, {}).get("trust_score", 0.5)
            ballot_weight = trust * raw_conf
            weighted_votes[vote] = weighted_votes.get(vote, 0.0) + ballot_weight
            total_weight += ballot_weight
            vote_breakdown[fw_name] = {
                "vote": vote,
                "raw_confidence": round(raw_conf, 4),
                "trust_weight": round(trust, 4),
                "ballot_weight": round(ballot_weight, 4),
            }

        if total_weight == 0.0:
            verdict = "uncertain"
            confidence = 0.0
        else:
            # Pick the winner
            verdict = max(weighted_votes, key=lambda v: weighted_votes[v])
            winner_weight = weighted_votes[verdict]
            confidence = winner_weight / total_weight

        # Identify dissenting flywheels
        dissenting = [
            fw for fw, info in vote_breakdown.items()
            if info["vote"] != verdict
        ]

        # Check for emergent pattern: no single flywheel is highly confident,
        # yet the combined vote is decisive
        individual_max = max(
            info["raw_confidence"] for info in vote_breakdown.values()
        )
        emergent = (
            individual_max <= _WEAK_SIGNAL_CEILING
            and confidence >= 0.6
        )

        return ConsensusVerdict(
            verdict=verdict,
            confidence=confidence,
            vote_breakdown=vote_breakdown,
            dissenting_flywheels=dissenting,
            emergent_flag=emergent,
        )

    # ------------------------------------------------------------------
    # 4. Circuit-Breaker Propagation
    # ------------------------------------------------------------------

    def trip_circuit_breaker(self, flywheel: str, reason: str = "") -> list[str]:
        """Trip a node's circuit breaker and warn connected neighbours.

        Returns the list of flywheels that entered protective mode.
        """
        self.add_node(flywheel)
        cb = self._circuit_breakers[flywheel]
        cb.tripped = True
        cb.trip_reason = reason
        cb.trips_this_window += 1
        cb.last_trip_ts = _now_iso()

        # Propagate protective mode to direct neighbours via FEEDS and CORRELATES edges
        protected: list[str] = []
        for (src, tgt), edge in self._edges.items():
            if src == flywheel and edge.edge_type in (EdgeType.FEEDS, EdgeType.CORRELATES):
                neighbour_cb = self._circuit_breakers.setdefault(
                    tgt, CircuitBreakerState(flywheel=tgt)
                )
                if not neighbour_cb.tripped:
                    neighbour_cb.protective_mode = True
                    protected.append(tgt)

        logger.warning(
            "NeuralMesh: circuit breaker tripped on '%s' (%s). "
            "Protective mode on: %s",
            flywheel, reason, protected,
        )
        return protected

    def reset_circuit_breaker(self, flywheel: str) -> None:
        """Reset a tripped circuit breaker and clear neighbour protective mode."""
        if flywheel in self._circuit_breakers:
            cb = self._circuit_breakers[flywheel]
            cb.tripped = False
            cb.protective_mode = False
            # Clear protective mode on neighbours that were set by this node
            for (src, tgt), _edge in self._edges.items():
                if src == flywheel:
                    neighbour_cb = self._circuit_breakers.get(tgt)
                    if neighbour_cb and neighbour_cb.protective_mode:
                        neighbour_cb.protective_mode = False

    def get_circuit_breaker(self, flywheel: str) -> CircuitBreakerState | None:
        return self._circuit_breakers.get(flywheel)

    def get_all_circuit_breakers(self) -> list[CircuitBreakerState]:
        return list(self._circuit_breakers.values())

    def get_tripped_circuit_breakers(self) -> list[CircuitBreakerState]:
        return [cb for cb in self._circuit_breakers.values() if cb.tripped]

    def get_protective_mode_nodes(self) -> list[str]:
        return [name for name, cb in self._circuit_breakers.items() if cb.protective_mode]

    # ------------------------------------------------------------------
    # 5. Emergent Detection
    # ------------------------------------------------------------------

    def detect_emergent_patterns(
        self,
        flywheel_signals: dict[str, float],
        threshold: float = 0.3,
    ) -> list[EmergentPattern]:
        """Identify scam patterns that emerge only from multi-flywheel combination.

        A pattern is emergent if:
        - No single flywheel's signal exceeds _WEAK_SIGNAL_CEILING.
        - The weighted combined score exceeds threshold × _EMERGENT_BOOST_THRESHOLD.

        Args:
            flywheel_signals: {flywheel_name: signal_strength (0-1)}
            threshold: minimum individual signal to consider a flywheel "participating".

        Returns:
            List of EmergentPattern objects (new or updated from internal registry).
        """
        # Filter to participating flywheels that have a weak but nonzero signal
        participants = {
            fw: strength
            for fw, strength in flywheel_signals.items()
            if threshold <= strength <= _WEAK_SIGNAL_CEILING
        }

        if len(participants) < 2:
            return []

        # Compute edge-weighted combined score
        # For each pair of participating flywheels that share an edge, the edge
        # weight amplifies their combined signal.
        combined_score = 0.0
        edge_amplifications: list[float] = []

        for fw_a, sig_a in participants.items():
            for fw_b, sig_b in participants.items():
                if fw_a >= fw_b:
                    continue
                edge = self._edges.get((fw_a, fw_b)) or self._edges.get((fw_b, fw_a))
                if edge and edge.edge_type != EdgeType.COMPETES:
                    amp = sig_a * sig_b * edge.weight
                    edge_amplifications.append(amp)

        if not edge_amplifications:
            # No edges between participants — fall back to sum
            combined_score = sum(participants.values())
        else:
            # Combine direct signals + cross-edge amplification
            direct_sum = sum(participants.values())
            amplification = sum(edge_amplifications)
            combined_score = direct_sum + amplification

        # Compare against what we'd expect from individual signals alone
        naive_avg = sum(participants.values()) / len(participants)
        boost_ratio = combined_score / max(naive_avg, 1e-9)

        if boost_ratio < _EMERGENT_BOOST_THRESHOLD:
            return []

        # Build or update the emergent pattern
        fw_key = "_".join(sorted(participants.keys()))
        pattern_id = f"emergent_{fw_key}"

        if pattern_id in self._emergent_patterns:
            ep = self._emergent_patterns[pattern_id]
            ep.observation_count += 1
            ep.combined_score = combined_score
            ep.individual_signal_strengths = dict(participants)
        else:
            ep = EmergentPattern(
                pattern_id=pattern_id,
                contributing_flywheels=sorted(participants.keys()),
                individual_signal_strengths=dict(participants),
                combined_score=combined_score,
                description=(
                    f"Emergent scam pattern from {len(participants)} flywheels "
                    f"({', '.join(sorted(participants.keys()))}). "
                    f"Combined score {combined_score:.3f} exceeds naive expectation "
                    f"by {boost_ratio:.2f}×."
                ),
                observation_count=1,
            )
            self._emergent_patterns[pattern_id] = ep

        return [ep]

    def confirm_emergent_pattern(self, pattern_id: str) -> None:
        """Record a confirmed true-positive for an emergent pattern."""
        if pattern_id in self._emergent_patterns:
            self._emergent_patterns[pattern_id].confirmed_count += 1

    def get_emergent_patterns(self) -> list[EmergentPattern]:
        return list(self._emergent_patterns.values())

    # ------------------------------------------------------------------
    # 6. Resonance Detector
    # ------------------------------------------------------------------

    def detect_resonance(
        self,
        min_series_len: int = 5,
    ) -> list[ResonanceEvent]:
        """Identify flywheels oscillating in correlated patterns.

        Algorithm:
        1. For each pair of flywheels with sufficient activation history, compute
           Pearson correlation of their activation magnitude series.
        2. If |r| >= _RESONANCE_THRESHOLD, the pair is resonating.
        3. Build resonance groups (connected components of resonating pairs).
        4. Estimate dominant frequency using zero-crossing rate.
        5. Hypothesise a shared cause from the group members and edge types.
        """
        # Build series map — only nodes with sufficient history
        series_map: dict[str, list[float]] = {
            name: list(hist)
            for name, hist in self._activation_history.items()
            if len(hist) >= min_series_len
        }
        if len(series_map) < 2:
            return []

        names = sorted(series_map.keys())
        # Pairwise correlations
        resonating_pairs: set[tuple[str, str]] = set()
        corr_matrix: dict[str, float] = {}

        for i, a in enumerate(names):
            for b in names[i + 1:]:
                r = _pearson_correlation(series_map[a], series_map[b])
                key = f"{a}|{b}"
                corr_matrix[key] = r
                if abs(r) >= _RESONANCE_THRESHOLD:
                    resonating_pairs.add((a, b))

        if not resonating_pairs:
            return []

        # Build connected components (resonance groups)
        groups = _connected_components(resonating_pairs, names)

        events: list[ResonanceEvent] = []
        for group in groups:
            if len(group) < 2:
                continue

            # Sub-matrix for this group
            group_corr: dict[str, float] = {}
            for a in group:
                for b in group:
                    if a < b:
                        k = f"{a}|{b}"
                        if k in corr_matrix:
                            group_corr[k] = corr_matrix[k]

            # Estimate dominant frequency from zero-crossing rate
            # Take the longest common series prefix
            min_len = min(len(series_map[g]) for g in group)
            avg_series = [
                sum(series_map[g][i] for g in group) / len(group)
                for i in range(min_len)
            ]
            freq = _estimate_frequency(avg_series)

            # Shared cause hypothesis
            cause = _hypothesise_cause(group, self._edges)

            # Combined signal = sqrt(avg_corr) * avg_activation
            avg_activation = sum(
                statistics.mean(series_map[g]) for g in group
            ) / len(group)
            avg_abs_corr = (
                sum(abs(v) for v in group_corr.values()) / max(len(group_corr), 1)
            )
            combined_strength = math.sqrt(avg_abs_corr) * avg_activation

            events.append(ResonanceEvent(
                participating_flywheels=sorted(group),
                correlation_matrix=group_corr,
                dominant_frequency=freq,
                shared_cause_hypothesis=cause,
                combined_signal_strength=combined_strength,
            ))

        return events

    # ------------------------------------------------------------------
    # 7. Information-Flow Efficiency
    # ------------------------------------------------------------------

    def analyse_flow_efficiency(self) -> list[FlowEfficiency]:
        """Measure how effectively information propagates through each node.

        Bottleneck score is high when:
        - High in-degree but low out-degree (accumulator).
        - Low avg outgoing weight (signal attenuation).
        - High betweenness proxy (all paths must pass through it).
        """
        reports: list[FlowEfficiency] = []

        if not self._nodes:
            return reports

        total_paths = max(self._count_all_simple_paths(), 1)

        for name in sorted(self._nodes.keys()):
            incoming = [e for (s, t), e in self._edges.items() if t == name]
            outgoing = [e for (s, t), e in self._edges.items() if s == name]

            in_deg = len(incoming)
            out_deg = len(outgoing)
            avg_in_w = (
                sum(e.weight for e in incoming) / in_deg if in_deg > 0 else 0.0
            )
            avg_out_w = (
                sum(e.weight for e in outgoing) / out_deg if out_deg > 0 else 0.0
            )

            # Proxy betweenness: fraction of direct edges that pass through this node
            paths_through = sum(
                1 for (s, t) in self._edges
                if s == name or t == name
            )
            betweenness = paths_through / (total_paths * 2 + 1)

            # Bottleneck: penalise high-in/low-out and low outgoing weight
            bottleneck = 0.0
            if in_deg > 0:
                ratio_penalty = max(0.0, (in_deg - out_deg) / max(in_deg, 1))
                weight_penalty = max(0.0, 1.0 - avg_out_w) if out_deg > 0 else 0.5
                bottleneck = min(1.0, (ratio_penalty + weight_penalty + betweenness) / 3)

            rec = _flow_recommendation(name, in_deg, out_deg, bottleneck, avg_out_w)

            reports.append(FlowEfficiency(
                node=name,
                in_degree=in_deg,
                out_degree=out_deg,
                avg_incoming_weight=avg_in_w,
                avg_outgoing_weight=avg_out_w,
                betweenness_proxy=betweenness,
                bottleneck_score=bottleneck,
                recommendation=rec,
            ))

        return reports

    def _count_all_simple_paths(self) -> int:
        """Count total directed simple paths (proxy for betweenness denominator)."""
        count = 0
        for start in self._nodes:
            visited: set[str] = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                count += 1
                for (s, t) in self._edges:
                    if s == node and t not in visited:
                        stack.append(t)
        return count

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def get_downstream(self, name: str, include_weights: bool = False) -> list[str]:
        """Return all transitively downstream node names (BFS)."""
        if name not in self._nodes:
            return []
        visited: set[str] = set()
        queue = [
            tgt for (src, tgt), e in self._edges.items()
            if src == name and e.edge_type != EdgeType.COMPETES
        ]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for (s, t), e in self._edges.items():
                if s == node and t not in visited and e.edge_type != EdgeType.COMPETES:
                    queue.append(t)
        return sorted(visited)

    def get_upstream(self, name: str) -> list[str]:
        """Return all transitively upstream node names (BFS)."""
        if name not in self._nodes:
            return []
        visited: set[str] = set()
        queue = [src for (src, tgt) in self._edges if tgt == name]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for (s, t) in self._edges:
                if t == node and s not in visited:
                    queue.append(s)
        return sorted(visited)

    def get_graph_summary(self) -> dict:
        """Return a serialisable summary of the mesh topology and state."""
        return {
            "nodes": [
                {
                    "name": name,
                    "trust_score": round(info["trust_score"], 4),
                    "activation_count": info.get("activation_count", 0),
                    "circuit_breaker": self._circuit_breakers.get(name, CircuitBreakerState(name)).to_dict(),
                }
                for name, info in self._nodes.items()
            ],
            "edges": [e.to_dict() for e in self._edges.values()],
            "emergent_patterns": len(self._emergent_patterns),
        }

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _persist_edge(self, edge: MeshEdge) -> None:
        """Upsert the edge into the DB's neural_mesh_edges table (created lazily)."""
        if self._db is None:
            return
        try:
            self._ensure_neural_tables()
            self._db.conn.execute(
                """
                INSERT INTO neural_mesh_edges
                    (source, target, edge_type, weight, base_weight,
                     reinforcement_count, suppression_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, target) DO UPDATE SET
                    edge_type           = excluded.edge_type,
                    weight              = excluded.weight,
                    base_weight         = excluded.base_weight,
                    reinforcement_count = excluded.reinforcement_count,
                    suppression_count   = excluded.suppression_count,
                    updated_at          = excluded.updated_at
                """,
                (
                    edge.source, edge.target, edge.edge_type.value,
                    edge.weight, edge.base_weight,
                    edge.reinforcement_count, edge.suppression_count,
                    _now_iso(),
                ),
            )
            self._db.conn.commit()
        except Exception:
            logger.debug("NeuralMesh: failed to persist edge", exc_info=True)

    def _persist_node(self, name: str) -> None:
        if self._db is None:
            return
        try:
            self._ensure_neural_tables()
            info = self._nodes[name]
            self._db.conn.execute(
                """
                INSERT INTO neural_mesh_nodes (name, trust_score, activation_count, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    trust_score      = excluded.trust_score,
                    activation_count = excluded.activation_count,
                    updated_at       = excluded.updated_at
                """,
                (
                    name,
                    info["trust_score"],
                    info.get("activation_count", 0),
                    _now_iso(),
                ),
            )
            self._db.conn.commit()
        except Exception:
            logger.debug("NeuralMesh: failed to persist node", exc_info=True)

    def save_state(self) -> None:
        """Persist all nodes and edges to SQLite."""
        if self._db is None:
            return
        self._ensure_neural_tables()
        for name in self._nodes:
            self._persist_node(name)
        for edge in self._edges.values():
            self._persist_edge(edge)

    def _load_from_db(self) -> None:
        """Load nodes and edges from SQLite (if tables exist)."""
        if self._db is None:
            return
        try:
            self._ensure_neural_tables()
            rows = self._db.conn.execute(
                "SELECT * FROM neural_mesh_nodes"
            ).fetchall()
            for row in rows:
                d = dict(row)
                name = d["name"]
                self._nodes[name] = {
                    "trust_score": d.get("trust_score", 0.5),
                    "precision_history": [],
                    "activation_count": d.get("activation_count", 0),
                }
                self._circuit_breakers[name] = CircuitBreakerState(flywheel=name)
                self._activation_history[name] = []

            edge_rows = self._db.conn.execute(
                "SELECT * FROM neural_mesh_edges"
            ).fetchall()
            for row in edge_rows:
                d = dict(row)
                src, tgt = d["source"], d["target"]
                try:
                    etype = EdgeType(d.get("edge_type", "FEEDS"))
                except ValueError:
                    etype = EdgeType.FEEDS
                edge = MeshEdge(
                    source=src, target=tgt,
                    edge_type=etype,
                    weight=d.get("weight", 1.0),
                    base_weight=d.get("base_weight", 1.0),
                    reinforcement_count=d.get("reinforcement_count", 0),
                    suppression_count=d.get("suppression_count", 0),
                )
                self._edges[(src, tgt)] = edge
                self.add_node(src)
                self.add_node(tgt)
        except sqlite3.OperationalError:
            pass  # Tables not yet created

    _neural_tables_created: bool = False

    def _ensure_neural_tables(self) -> None:
        if self._db is None or NeuralMesh._neural_tables_created:
            return
        self._db.conn.executescript("""
            CREATE TABLE IF NOT EXISTS neural_mesh_nodes (
                name TEXT PRIMARY KEY,
                trust_score REAL DEFAULT 0.5,
                activation_count INTEGER DEFAULT 0,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS neural_mesh_edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'FEEDS',
                weight REAL DEFAULT 1.0,
                base_weight REAL DEFAULT 1.0,
                reinforcement_count INTEGER DEFAULT 0,
                suppression_count INTEGER DEFAULT 0,
                updated_at TEXT,
                PRIMARY KEY (source, target)
            );

            CREATE TABLE IF NOT EXISTS neural_mesh_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                payload_json TEXT DEFAULT '{}',
                arrivals_json TEXT DEFAULT '[]',
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS neural_mesh_resonance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flywheels_json TEXT NOT NULL,
                correlation_json TEXT NOT NULL,
                dominant_frequency REAL DEFAULT 0.0,
                cause TEXT DEFAULT '',
                combined_strength REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS neural_mesh_emergent (
                pattern_id TEXT PRIMARY KEY,
                flywheels_json TEXT NOT NULL,
                combined_score REAL DEFAULT 0.0,
                observation_count INTEGER DEFAULT 0,
                confirmed_count INTEGER DEFAULT 0,
                description TEXT DEFAULT '',
                updated_at TEXT
            );
        """)
        self._db.conn.commit()
        NeuralMesh._neural_tables_created = True

    def persist_signal_event(
        self,
        origin: str,
        signal_type: str,
        payload: dict,
        arrivals: list[PropagatedSignal],
    ) -> None:
        """Persist a propagated signal event to SQLite."""
        if self._db is None:
            return
        try:
            self._ensure_neural_tables()
            self._db.conn.execute(
                """
                INSERT INTO neural_mesh_signals
                    (origin, signal_type, payload_json, arrivals_json, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    origin,
                    signal_type,
                    json.dumps(payload),
                    json.dumps([a.to_dict() for a in arrivals]),
                    _now_iso(),
                ),
            )
            self._db.conn.commit()
        except Exception:
            logger.debug("NeuralMesh: failed to persist signal event", exc_info=True)

    def persist_resonance_event(self, event: ResonanceEvent) -> None:
        """Persist a resonance event to SQLite."""
        if self._db is None:
            return
        try:
            self._ensure_neural_tables()
            self._db.conn.execute(
                """
                INSERT INTO neural_mesh_resonance
                    (flywheels_json, correlation_json, dominant_frequency,
                     cause, combined_strength, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    json.dumps(event.participating_flywheels),
                    json.dumps(event.correlation_matrix),
                    event.dominant_frequency,
                    event.shared_cause_hypothesis,
                    event.combined_signal_strength,
                    event.timestamp,
                ),
            )
            self._db.conn.commit()
        except Exception:
            logger.debug("NeuralMesh: failed to persist resonance event", exc_info=True)

    def persist_emergent_pattern(self, ep: EmergentPattern) -> None:
        """Upsert an emergent pattern into SQLite."""
        if self._db is None:
            return
        try:
            self._ensure_neural_tables()
            self._db.conn.execute(
                """
                INSERT INTO neural_mesh_emergent
                    (pattern_id, flywheels_json, combined_score,
                     observation_count, confirmed_count, description, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    combined_score    = excluded.combined_score,
                    observation_count = excluded.observation_count,
                    confirmed_count   = excluded.confirmed_count,
                    updated_at        = excluded.updated_at
                """,
                (
                    ep.pattern_id,
                    json.dumps(ep.contributing_flywheels),
                    ep.combined_score,
                    ep.observation_count,
                    ep.confirmed_count,
                    ep.description,
                    _now_iso(),
                ),
            )
            self._db.conn.commit()
        except Exception:
            logger.debug("NeuralMesh: failed to persist emergent pattern", exc_info=True)


# ---------------------------------------------------------------------------
# Default Sentinel mesh factory
# ---------------------------------------------------------------------------

def build_neural_mesh(db: SentinelDB | None = None) -> NeuralMesh:
    """Return a NeuralMesh pre-wired with Sentinel's flywheel topology."""
    mesh = NeuralMesh(db=db)

    # Core topology
    mesh.add_node("detection",   initial_trust=0.8)
    mesh.add_node("calibration", initial_trust=0.75)
    mesh.add_node("innovation",  initial_trust=0.7)
    mesh.add_node("shadow",      initial_trust=0.7)
    mesh.add_node("drift",       initial_trust=0.65)
    mesh.add_node("research",    initial_trust=0.6)

    # research → detection/innovation (knowledge injection)
    mesh.add_edge("research",    "detection",   EdgeType.FEEDS,      weight=0.8)
    mesh.add_edge("research",    "innovation",  EdgeType.FEEDS,      weight=0.9)

    # drift → detection/innovation (alarm signals)
    mesh.add_edge("drift",       "detection",   EdgeType.FEEDS,      weight=1.0)
    mesh.add_edge("drift",       "innovation",  EdgeType.FEEDS,      weight=0.7)

    # calibration → detection (threshold adjustments)
    mesh.add_edge("calibration", "detection",   EdgeType.FEEDS,      weight=0.9)

    # shadow → detection (weight promotion)
    mesh.add_edge("shadow",      "detection",   EdgeType.FEEDS,      weight=0.85)

    # innovation ↔ detection feedback loop
    mesh.add_edge("innovation",  "detection",   EdgeType.FEEDS,      weight=0.8)
    mesh.add_edge("detection",   "calibration", EdgeType.FEEDS,      weight=0.75)
    mesh.add_edge("detection",   "innovation",  EdgeType.CORRELATES, weight=0.5)
    mesh.add_edge("detection",   "drift",       EdgeType.CORRELATES, weight=0.6)

    # shadow ↔ innovation correlation
    mesh.add_edge("shadow",      "innovation",  EdgeType.CORRELATES, weight=0.4)

    # innovation and shadow compete for the same exploration budget
    mesh.add_edge("innovation",  "shadow",      EdgeType.COMPETES,   weight=0.3)
    mesh.add_edge("shadow",      "innovation",  EdgeType.COMPETES,   weight=0.3)

    # drift inhibits shadow (don't promote weights during instability)
    mesh.add_edge("drift",       "shadow",      EdgeType.INHIBITS,   weight=0.6)

    return mesh


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Pearson r for two equal-length float lists. Returns 0.0 on degenerate input."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[:n], y[:n]
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=False))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    denom = denom_x * denom_y
    return 0.0 if denom == 0.0 else max(-1.0, min(1.0, num / denom))


def _estimate_frequency(series: list[float]) -> float:
    """Estimate oscillation frequency (cycles per sample) via zero-crossing rate."""
    if len(series) < 3:
        return 0.0
    mean_val = sum(series) / len(series)
    centred = [v - mean_val for v in series]
    crossings = sum(
        1 for i in range(1, len(centred))
        if centred[i - 1] * centred[i] < 0
    )
    # Each full cycle has 2 zero crossings
    return crossings / (2 * max(len(series) - 1, 1))


def _connected_components(
    pairs: set[tuple[str, str]],
    all_nodes: list[str],
) -> list[list[str]]:
    """Return connected components from a set of undirected pairs."""
    adjacency: dict[str, set[str]] = {n: set() for n in all_nodes}
    for a, b in pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)

    visited: set[str] = set()
    components: list[list[str]] = []

    for node in all_nodes:
        if node in visited or not adjacency[node]:
            continue
        component: list[str] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            component.append(n)
            stack.extend(adjacency[n] - visited)
        if len(component) >= 2:
            components.append(sorted(component))

    return components


def _hypothesise_cause(
    group: list[str],
    edges: dict[tuple[str, str], MeshEdge],
) -> str:
    """Infer a probable shared cause from the resonating group's topology."""
    # Find nodes with highest in-degree within the group
    in_degree: dict[str, int] = {n: 0 for n in group}
    for (src, tgt), _edge in edges.items():
        if src in group and tgt in group:
            in_degree[tgt] += 1

    likely_driver = max(in_degree, key=lambda n: in_degree[n]) if in_degree else group[0]
    involved = ", ".join(g for g in group if g != likely_driver)

    inhibit_edges = [
        f"{s}→{t}" for (s, t), e in edges.items()
        if s in group and t in group and e.edge_type == EdgeType.INHIBITS
    ]

    if inhibit_edges:
        return (
            f"Shared oscillation driven by '{likely_driver}' with inhibitory "
            f"feedback through {inhibit_edges}. Participants: {involved}."
        )
    return (
        f"Correlated activation pattern driven by '{likely_driver}'. "
        f"Likely common input signal or coordinated scam campaign affecting: {involved}."
    )


def _flow_recommendation(
    name: str,
    in_deg: int,
    out_deg: int,
    bottleneck: float,
    avg_out_w: float,
) -> str:
    if in_deg == 0 and out_deg == 0:
        return f"'{name}' is isolated — connect it to the mesh for signal flow."
    if bottleneck >= 0.7:
        return (
            f"'{name}' is a severe bottleneck (score={bottleneck:.2f}). "
            "Consider adding bypass edges or increasing outgoing weights."
        )
    if out_deg == 0:
        return f"'{name}' is a sink — signals terminate here. Intentional?"
    if in_deg == 0:
        return f"'{name}' is a source node — ensure it fires reliably."
    if avg_out_w < 0.3:
        return f"'{name}' has weak outgoing weights — signals attenuate quickly. Reinforce via Hebbian updates."
    return f"'{name}' is flowing normally."
