"""Meta-cognitive Cortex intelligence layer for JobSentinel.

The Cortex observes all subsystem outputs, detects cross-system patterns,
routes signals between systems, makes strategic resource allocation decisions,
and tracks overall learning velocity and health.

Subsystems observed:
  - DetectionFlywheel (CUSUM regression, pattern evolution)
  - InnovationEngine (Thompson Sampling, 8+ strategy arms)
  - ShadowScorer (A/B weight testing)
  - ResearchEngine (autonomous research)
  - Source Quality Tracking
  - Calibration/Confidence system
  - Input Drift Detection
  - Adversarial Evasion Detection
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sentinel.db import SentinelDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CrossSignal:
    """A routed signal from one subsystem to another."""
    source_system: str
    target_system: str
    signal_type: str
    payload: dict
    priority: float


@dataclass
class Investigation:
    """An open investigation triggered by cross-system anomalies."""
    id: str
    trigger: str  # e.g., "drift_detected", "shadow_rejected", "evasion_spike"
    hypothesis: str
    actions_taken: list[str]
    status: str  # "open", "resolved", "stale"
    cycles_open: int
    resolution: str | None = None


@dataclass
class CortexState:
    """Snapshot of the Cortex's understanding of the system."""
    cycle_number: int
    subsystem_health: dict[str, float]  # 0-1 health score per subsystem
    learning_velocity: float  # rate of precision improvement per cycle
    active_investigations: list[Investigation]
    cross_system_signals: list[CrossSignal]
    strategic_priorities: list[str]


# ---------------------------------------------------------------------------
# Action — what the cortex instructs subsystems to do
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """An action dispatched by the Cortex to a subsystem."""
    target_system: str
    action_type: str
    parameters: dict
    reason: str


# ---------------------------------------------------------------------------
# Cortex
# ---------------------------------------------------------------------------

class Cortex:
    """Meta-cognitive layer orchestrating all flywheel subsystems.

    Lightweight: observes and routes, does not duplicate subsystem logic.
    All cross-system signals are logged and auditable.
    Strategic planning uses simple interpretable rules, not ML.
    """

    # Precision decline cycles before emergency investigation
    EMERGENCY_DECLINE_CYCLES = 3
    # Max age for investigations before marking stale
    MAX_INVESTIGATION_AGE = 5

    def __init__(self, db: SentinelDB) -> None:
        self.db = db
        self._state: CortexState | None = None
        self._cycle_number: int = 0
        self._precision_history: list[float] = []
        self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load cortex state from DB."""
        try:
            row = self.db.get_latest_cortex_state()
            if row:
                self._cycle_number = row.get("cycle_number", 0)
                state_json = row.get("state_json", "{}")
                state_data = json.loads(state_json) if isinstance(state_json, str) else state_json
                self._precision_history = state_data.get("precision_history", [])
        except Exception:
            logger.debug("No existing cortex state found — starting fresh.")

    def _save_state(self, state: CortexState) -> None:
        """Persist cortex state to DB."""
        state_data = {
            "precision_history": self._precision_history[-50:],  # keep last 50
            "subsystem_health": state.subsystem_health,
            "strategic_priorities": state.strategic_priorities,
        }
        health_scores = state.subsystem_health
        avg_health = sum(health_scores.values()) / max(len(health_scores), 1)
        grade = self._health_score_to_grade(avg_health)
        mode = state.strategic_priorities[0] if state.strategic_priorities else "OBSERVE"

        self.db.save_cortex_state(
            cycle_number=state.cycle_number,
            state_json=json.dumps(state_data),
            learning_velocity=state.learning_velocity,
            health_grade=grade,
            strategic_mode=mode,
        )

    # ------------------------------------------------------------------
    # Main entry point — called each daemon cycle
    # ------------------------------------------------------------------

    def observe_cycle(self, metrics: dict) -> CortexState:
        """Ingest metrics from all subsystems and produce a CortexState.

        Args:
            metrics: Dict with keys from each subsystem phase:
                - precision, recall, f1, accuracy (flywheel)
                - regression_alarm, cusum_statistic (flywheel)
                - calibration_ece (flywheel)
                - drift_alarm, drift_score (drift detection)
                - shadow_active, shadow_improvement (shadow scorer)
                - innovation_strategy, innovation_ran (innovation)
                - research_ran, research_patterns_found (research)
                - source_circuit_breakers (source quality)
                - evasion_signals (adversarial detection)
                - errors (list of error strings)
        """
        self._cycle_number += 1

        # Track precision history
        precision = metrics.get("precision", 0.0)
        self._precision_history.append(precision)

        # Compute subsystem health
        subsystem_health = self.compute_subsystem_health(metrics)

        # Compute learning velocity
        velocity = self.compute_learning_velocity()

        # Detect cross-system patterns and generate signals
        signals = self._detect_cross_system_patterns(metrics)

        # Load active investigations from DB
        active_investigations = self._load_active_investigations()

        # Age investigations
        for inv in active_investigations:
            inv.cycles_open += 1

        # Check for stale investigations
        for inv in active_investigations:
            if inv.status == "open" and inv.cycles_open > self.MAX_INVESTIGATION_AGE:
                inv.status = "stale"
                self.db.update_cortex_investigation(inv.id, {
                    "status": "stale",
                })

        # Strategic planning
        priorities = self.strategic_planning(metrics, subsystem_health, velocity)

        # Auto-open investigations based on metrics
        self._auto_open_investigations(metrics, active_investigations)

        state = CortexState(
            cycle_number=self._cycle_number,
            subsystem_health=subsystem_health,
            learning_velocity=velocity,
            active_investigations=active_investigations,
            cross_system_signals=signals,
            strategic_priorities=priorities,
        )

        self._state = state
        self._save_state(state)

        # Persist signals
        for sig in signals:
            self.db.insert_cortex_signal(
                source=sig.source_system,
                target=sig.target_system,
                signal_type=sig.signal_type,
                payload=sig.payload,
                priority=sig.priority,
            )

        logger.info(
            "Cortex cycle %d: health=%s, velocity=%.4f, signals=%d, investigations=%d, mode=%s",
            self._cycle_number,
            {k: f"{v:.2f}" for k, v in subsystem_health.items()},
            velocity,
            len(signals),
            len(active_investigations),
            priorities[0] if priorities else "OBSERVE",
        )

        return state

    # ------------------------------------------------------------------
    # Cross-system signal routing
    # ------------------------------------------------------------------

    def route_signals(self, metrics: dict) -> list[Action]:
        """Route cross-system signals into concrete actions.

        Rules:
        1. Drift detected AND shadow has no active test -> propose shadow test
        2. Shadow rejected weights -> inform innovation to avoid that direction
        3. Evasion detected on signal X -> trigger research on evasion tactics
        4. Calibration error increasing -> increase innovation budget for threshold_tuning
        5. Source quality dropping -> adjust ingest priorities
        6. Precision improving -> record what changed and reinforce it
        7. Precision declining for >3 cycles -> trigger emergency investigation
        """
        actions: list[Action] = []

        # Rule 1: drift + no shadow -> propose shadow test
        drift_alarm = metrics.get("drift_alarm", False)
        shadow_active = metrics.get("shadow_active", False)
        if drift_alarm and not shadow_active:
            actions.append(Action(
                target_system="shadow",
                action_type="propose_drift_adapted_weights",
                parameters={"drift_score": metrics.get("drift_score", 0.0)},
                reason="Input drift detected with no active shadow test — "
                       "propose drift-adapted weights for A/B testing.",
            ))

        # Rule 2: shadow rejected -> inform innovation
        shadow_rejected = metrics.get("shadow_rejected", False)
        if shadow_rejected:
            rejected_direction = metrics.get("shadow_rejected_direction", "unknown")
            actions.append(Action(
                target_system="innovation",
                action_type="avoid_weight_direction",
                parameters={"rejected_direction": rejected_direction},
                reason="Shadow test rejected weights — innovation should avoid "
                       "this weight direction in future explorations.",
            ))

        # Rule 3: evasion detected -> trigger research
        evasion_signals = metrics.get("evasion_signals", [])
        if evasion_signals:
            for sig_name in evasion_signals[:3]:  # cap at 3
                actions.append(Action(
                    target_system="research",
                    action_type="investigate_evasion",
                    parameters={"signal_name": sig_name},
                    reason=f"Adversarial evasion detected on signal '{sig_name}' — "
                           "research engine should investigate evasion tactics.",
                ))

        # Rule 4: calibration error increasing -> boost threshold_tuning
        ece = metrics.get("calibration_ece", 0.0)
        if ece > 0.15:
            actions.append(Action(
                target_system="innovation",
                action_type="boost_strategy",
                parameters={"strategy": "threshold_tuning", "ece": ece},
                reason=f"Calibration error is high (ECE={ece:.3f}) — "
                       "boost threshold_tuning innovation arm.",
            ))

        # Rule 5: source quality dropping
        source_breakers = metrics.get("source_circuit_breakers", [])
        if source_breakers:
            actions.append(Action(
                target_system="ingest",
                action_type="adjust_priorities",
                parameters={"broken_sources": source_breakers},
                reason=f"Circuit breaker tripped for {len(source_breakers)} source(s) — "
                       "adjust ingestion priorities.",
            ))

        # Rule 6: precision improving -> reinforce
        if len(self._precision_history) >= 2:
            recent = self._precision_history[-1]
            prev = self._precision_history[-2]
            if recent > prev + 0.005:
                actions.append(Action(
                    target_system="cortex",
                    action_type="record_improvement",
                    parameters={
                        "precision_from": prev,
                        "precision_to": recent,
                        "delta": round(recent - prev, 6),
                    },
                    reason="Precision improving — recording change for reinforcement.",
                ))

        # Rule 7: precision declining for >N cycles -> emergency investigation
        if self._is_precision_declining(self.EMERGENCY_DECLINE_CYCLES):
            actions.append(Action(
                target_system="cortex",
                action_type="emergency_investigation",
                parameters={
                    "recent_precision": self._precision_history[-self.EMERGENCY_DECLINE_CYCLES:],
                },
                reason=f"Precision declining for {self.EMERGENCY_DECLINE_CYCLES}+ cycles — "
                       "opening emergency investigation.",
            ))

        # Log all actions as signals
        for action in actions:
            self.db.insert_cortex_signal(
                source="cortex",
                target=action.target_system,
                signal_type=action.action_type,
                payload=action.parameters,
                priority=0.8 if "emergency" in action.action_type else 0.5,
            )

        return actions

    # ------------------------------------------------------------------
    # Learning velocity
    # ------------------------------------------------------------------

    def compute_learning_velocity(self, window: int = 10) -> float:
        """Compute precision delta per cycle over the given window.

        Returns the average precision change per cycle. Positive means
        improving, negative means degrading.
        """
        history = self._precision_history
        if len(history) < 2:
            return 0.0

        window_data = history[-window:]
        if len(window_data) < 2:
            return 0.0

        # Average per-step delta
        deltas = [
            window_data[i] - window_data[i - 1]
            for i in range(1, len(window_data))
        ]
        return sum(deltas) / len(deltas)

    # ------------------------------------------------------------------
    # Subsystem health scoring
    # ------------------------------------------------------------------

    def compute_subsystem_health(self, metrics: dict) -> dict[str, float]:
        """Score each subsystem 0-1 based on its current metrics."""
        health: dict[str, float] = {}

        # Detection: based on precision trend
        precision = metrics.get("precision", 0.0)
        health["detection"] = min(precision / 0.9, 1.0)  # 0.9 precision = 1.0 health

        # Innovation: arm diversity + improvement rate
        innovation_ran = metrics.get("innovation_ran", False)
        if innovation_ran:
            # Score based on strategy diversity (did it explore?)
            health["innovation"] = 0.8  # Base score for running
        else:
            health["innovation"] = 0.3  # Penalise for not running

        # Shadow: based on promotion rate and time-to-decision
        shadow_active = metrics.get("shadow_active", False)
        shadow_improvement = metrics.get("shadow_improvement", 0.0)
        if shadow_active:
            health["shadow"] = 0.7 + min(shadow_improvement * 2, 0.3)
        else:
            health["shadow"] = 0.5  # Neutral when idle

        # Calibration: ECE trend
        ece = metrics.get("calibration_ece", 0.0)
        health["calibration"] = max(0.0, 1.0 - ece * 5)  # ECE 0.2 = health 0.0

        # Sources: circuit breaker states + yield rates
        breakers = metrics.get("source_circuit_breakers", [])
        if not breakers:
            health["sources"] = 1.0
        else:
            # Degrade proportionally to broken sources
            health["sources"] = max(0.0, 1.0 - len(breakers) * 0.2)

        # Drift: stability score
        drift_alarm = metrics.get("drift_alarm", False)
        drift_score = metrics.get("drift_score", 0.0)
        if drift_alarm:
            health["drift"] = max(0.0, 1.0 - drift_score * 5)
        else:
            health["drift"] = 1.0

        # Research: based on whether it ran and found patterns
        research_ran = metrics.get("research_ran", False)
        research_patterns = metrics.get("research_patterns_found", 0)
        if research_ran:
            health["research"] = 0.6 + min(research_patterns * 0.1, 0.4)
        else:
            health["research"] = 0.4

        return health

    # ------------------------------------------------------------------
    # Strategic planning
    # ------------------------------------------------------------------

    def strategic_planning(
        self,
        metrics: dict | None = None,
        subsystem_health: dict[str, float] | None = None,
        velocity: float | None = None,
    ) -> list[str]:
        """Determine high-level strategic priorities.

        Modes:
        - EXPAND: precision >0.8 and stable -> explore new fraud types
        - STABILIZE: precision declining -> focus on regression fix
        - INVESTIGATE: anomalies detected -> deep dive
        - OPTIMIZE: precision plateaued -> try ensemble/new approaches
        """
        if metrics is None:
            metrics = {}
        if subsystem_health is None:
            subsystem_health = {}
        if velocity is None:
            velocity = self.compute_learning_velocity()

        priorities: list[str] = []
        precision = metrics.get("precision", 0.0)
        regression_alarm = metrics.get("regression_alarm", False)
        drift_alarm = metrics.get("drift_alarm", False)

        # STABILIZE takes highest priority when things are going wrong
        if regression_alarm or self._is_precision_declining(self.EMERGENCY_DECLINE_CYCLES):
            priorities.append("STABILIZE")

        # INVESTIGATE when anomalies are detected
        if drift_alarm or metrics.get("evasion_signals"):
            priorities.append("INVESTIGATE")

        # EXPAND when precision is high and stable
        if precision > 0.8 and velocity >= -0.001:
            priorities.append("EXPAND")

        # OPTIMIZE when precision has plateaued
        if abs(velocity) < 0.001 and precision < 0.8:
            priorities.append("OPTIMIZE")

        # Default: OBSERVE
        if not priorities:
            priorities.append("OBSERVE")

        return priorities

    # ------------------------------------------------------------------
    # Investigations
    # ------------------------------------------------------------------

    def open_investigation(self, trigger: str, hypothesis: str) -> Investigation:
        """Open a new investigation and persist to DB."""
        inv_id = f"inv_{uuid.uuid4().hex[:8]}"
        inv = Investigation(
            id=inv_id,
            trigger=trigger,
            hypothesis=hypothesis,
            actions_taken=[],
            status="open",
            cycles_open=0,
        )
        self.db.insert_cortex_investigation(
            id=inv_id,
            trigger=trigger,
            hypothesis=hypothesis,
        )
        return inv

    def resolve_investigation(self, inv_id: str, resolution: str) -> None:
        """Mark an investigation as resolved with a resolution note."""
        self.db.update_cortex_investigation(inv_id, {
            "status": "resolved",
            "resolution": resolution,
            "resolved_at": _now_iso(),
        })

    def get_stale_investigations(self, max_age: int = 5) -> list[Investigation]:
        """Return investigations that have been open longer than max_age cycles."""
        rows = self.db.get_cortex_investigations(status="open")
        stale = []
        for row in rows:
            actions_raw = row.get("actions_json", "[]")
            try:
                actions = json.loads(actions_raw) if isinstance(actions_raw, str) else actions_raw
            except (json.JSONDecodeError, TypeError):
                actions = []
            inv = Investigation(
                id=row["id"],
                trigger=row.get("trigger", ""),
                hypothesis=row.get("hypothesis", ""),
                actions_taken=actions,
                status=row.get("status", "open"),
                cycles_open=self._compute_cycles_open(row.get("opened_at", "")),
                resolution=row.get("resolution"),
            )
            if inv.cycles_open > max_age:
                stale.append(inv)
        return stale

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> dict:
        """Generate a comprehensive cortex report.

        Includes:
        - Overall system health grade (A-F)
        - Learning velocity with trend arrow
        - Active investigations
        - Cross-system signal log
        - Strategic mode and priorities
        - Recommendations
        """
        state = self._state

        # Gather data
        if state:
            health_scores = state.subsystem_health
            velocity = state.learning_velocity
            priorities = state.strategic_priorities
            investigations = state.active_investigations
        else:
            health_scores = {}
            velocity = self.compute_learning_velocity()
            priorities = self.strategic_planning()
            investigations = self._load_active_investigations()

        # Overall health grade
        avg_health = sum(health_scores.values()) / max(len(health_scores), 1) if health_scores else 0.0
        grade = self._health_score_to_grade(avg_health)

        # Trend arrow
        if velocity > 0.005:
            trend = "improving"
        elif velocity < -0.005:
            trend = "declining"
        else:
            trend = "stable"

        # Recommendations
        recommendations = self._generate_recommendations(
            health_scores, velocity, priorities, investigations
        )

        # Recent signal log from DB
        recent_signals = self.db.get_recent_cortex_signals(limit=20)

        return {
            "health_grade": grade,
            "avg_health_score": round(avg_health, 3),
            "subsystem_health": {k: round(v, 3) for k, v in health_scores.items()},
            "learning_velocity": round(velocity, 6),
            "velocity_trend": trend,
            "strategic_mode": priorities[0] if priorities else "OBSERVE",
            "strategic_priorities": priorities,
            "active_investigations": [
                {
                    "id": inv.id,
                    "trigger": inv.trigger,
                    "hypothesis": inv.hypothesis,
                    "status": inv.status,
                    "cycles_open": inv.cycles_open,
                    "actions_taken": inv.actions_taken,
                    "resolution": inv.resolution,
                }
                for inv in investigations
            ],
            "recent_signals": recent_signals,
            "recommendations": recommendations,
            "cycle_number": self._cycle_number,
            "precision_history": self._precision_history[-20:],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_cross_system_patterns(self, metrics: dict) -> list[CrossSignal]:
        """Detect cross-system patterns from the current metrics snapshot."""
        signals: list[CrossSignal] = []

        # Drift -> Innovation correlation
        if metrics.get("drift_alarm") and metrics.get("innovation_ran"):
            signals.append(CrossSignal(
                source_system="drift",
                target_system="innovation",
                signal_type="drift_during_innovation",
                payload={
                    "drift_score": metrics.get("drift_score", 0.0),
                    "innovation_strategy": metrics.get("innovation_strategy", ""),
                },
                priority=0.7,
            ))

        # Regression + evasion correlation
        if metrics.get("regression_alarm") and metrics.get("evasion_signals"):
            signals.append(CrossSignal(
                source_system="detection",
                target_system="adversarial",
                signal_type="regression_with_evasion",
                payload={
                    "cusum_statistic": metrics.get("cusum_statistic", 0.0),
                    "evasion_count": len(metrics.get("evasion_signals", [])),
                },
                priority=0.9,
            ))

        # Calibration + shadow correlation
        ece = metrics.get("calibration_ece", 0.0)
        shadow_active = metrics.get("shadow_active", False)
        if ece > 0.15 and not shadow_active:
            signals.append(CrossSignal(
                source_system="calibration",
                target_system="shadow",
                signal_type="high_ece_no_shadow",
                payload={"ece": ece},
                priority=0.6,
            ))

        return signals

    def _auto_open_investigations(
        self,
        metrics: dict,
        active: list[Investigation],
    ) -> None:
        """Auto-open investigations based on metric triggers."""
        active_triggers = {inv.trigger for inv in active if inv.status == "open"}

        # Precision declining for N+ cycles
        if (
            self._is_precision_declining(self.EMERGENCY_DECLINE_CYCLES)
            and "precision_decline" not in active_triggers
        ):
            self.open_investigation(
                trigger="precision_decline",
                hypothesis="Precision has been declining for multiple cycles — "
                           "possible regression, drift, or evasion.",
            )

        # Drift detected
        if metrics.get("drift_alarm") and "drift_detected" not in active_triggers:
            self.open_investigation(
                trigger="drift_detected",
                hypothesis="Input drift detected — signal firing rates have shifted "
                           "significantly from baseline.",
            )

        # Evasion spike
        evasion_signals = metrics.get("evasion_signals", [])
        if len(evasion_signals) >= 3 and "evasion_spike" not in active_triggers:
            self.open_investigation(
                trigger="evasion_spike",
                hypothesis=f"Multiple evasion signals detected ({len(evasion_signals)}) — "
                           "possible coordinated adversarial activity.",
            )

    def _load_active_investigations(self) -> list[Investigation]:
        """Load open and stale investigations from DB."""
        investigations: list[Investigation] = []
        for status in ("open", "stale"):
            rows = self.db.get_cortex_investigations(status=status)
            for row in rows:
                actions_raw = row.get("actions_json", "[]")
                try:
                    actions = json.loads(actions_raw) if isinstance(actions_raw, str) else actions_raw
                except (json.JSONDecodeError, TypeError):
                    actions = []
                investigations.append(Investigation(
                    id=row["id"],
                    trigger=row.get("trigger", ""),
                    hypothesis=row.get("hypothesis", ""),
                    actions_taken=actions,
                    status=row.get("status", "open"),
                    cycles_open=self._compute_cycles_open(row.get("opened_at", "")),
                    resolution=row.get("resolution"),
                ))
        return investigations

    def _is_precision_declining(self, n_cycles: int) -> bool:
        """Return True if precision has been declining for the last n_cycles."""
        if len(self._precision_history) < n_cycles + 1:
            return False
        recent = self._precision_history[-(n_cycles + 1):]
        return all(recent[i] < recent[i - 1] for i in range(1, len(recent)))

    @staticmethod
    def _health_score_to_grade(score: float) -> str:
        """Convert a 0-1 health score to A-F letter grade."""
        if score >= 0.85:
            return "A"
        elif score >= 0.70:
            return "B"
        elif score >= 0.55:
            return "C"
        elif score >= 0.40:
            return "D"
        else:
            return "F"

    def _compute_cycles_open(self, opened_at: str) -> int:
        """Estimate how many cycles an investigation has been open.

        Falls back to counting from the DB state cycle numbers if
        timestamps are not precise enough.
        """
        if not opened_at:
            return 0
        try:
            opened = datetime.fromisoformat(opened_at)
            now = datetime.now(UTC)
            # Estimate: 1 cycle per hour (daemon default interval)
            hours_open = (now - opened).total_seconds() / 3600
            return max(0, int(hours_open))
        except (ValueError, TypeError):
            return 0

    def _generate_recommendations(
        self,
        health: dict[str, float],
        velocity: float,
        priorities: list[str],
        investigations: list[Investigation],
    ) -> list[str]:
        """Generate actionable recommendations based on current state."""
        recs: list[str] = []

        # Low health subsystems
        for name, score in health.items():
            if score < 0.4:
                recs.append(f"URGENT: {name} health is critically low ({score:.0%}) — needs immediate attention.")
            elif score < 0.6:
                recs.append(f"WARNING: {name} health is degraded ({score:.0%}) — monitor closely.")

        # Velocity issues
        if velocity < -0.01:
            recs.append("Precision is declining rapidly — consider pausing innovation experiments.")
        elif velocity > 0.01:
            recs.append("Precision is improving well — consider expanding to new fraud categories.")

        # Investigation overload
        open_count = sum(1 for inv in investigations if inv.status == "open")
        if open_count > 3:
            recs.append(f"Too many open investigations ({open_count}) — prioritise and resolve some.")

        # Stale investigations
        stale_count = sum(1 for inv in investigations if inv.status == "stale")
        if stale_count > 0:
            recs.append(f"{stale_count} stale investigation(s) — review and close or escalate.")

        # Mode-specific
        if "STABILIZE" in priorities:
            recs.append("System is in STABILIZE mode — focus on regression fixes, not new features.")
        if "EXPAND" in priorities and not recs:
            recs.append("System is stable and performant — good time to explore new detection areas.")

        if not recs:
            recs.append("All systems nominal — no action required.")

        return recs

    @property
    def state(self) -> CortexState | None:
        """Return the last computed state, or None if no cycle has run."""
        return self._state
