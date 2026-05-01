"""Cross-flywheel mesh and cascade impact detection for JobSentinel.

FlywheelMesh  — models the dependency graph between all flywheel systems.
CascadeDetector — detects, previews, and records cross-flywheel ripple effects.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentinel.db import SentinelDB

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CascadeReport:
    """Result of a weight-change impact preview."""
    jobs_sampled: int
    classifications_changed: int
    change_rate: float
    risk_level: str          # SAFE / MODERATE / HIGH
    score_delta_mean: float
    score_delta_std: float
    promoted_count: int      # jobs that moved to a higher-risk bucket
    demoted_count: int       # jobs that moved to a lower-risk bucket

    def to_dict(self) -> dict:
        return {
            "jobs_sampled": self.jobs_sampled,
            "classifications_changed": self.classifications_changed,
            "change_rate": round(self.change_rate, 4),
            "risk_level": self.risk_level,
            "score_delta_mean": round(self.score_delta_mean, 4),
            "score_delta_std": round(self.score_delta_std, 4),
            "promoted_count": self.promoted_count,
            "demoted_count": self.demoted_count,
        }


@dataclass
class CascadeRecord:
    """Persisted record of a change event and its downstream impact."""
    trigger: str
    change_type: str
    before_metrics: dict
    after_metrics: dict
    magnitude: float
    affected_flywheels: list[str]
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "change_type": self.change_type,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "magnitude": round(self.magnitude, 6),
            "affected_flywheels": self.affected_flywheels,
            "timestamp": self.timestamp,
        }


@dataclass
class RippleEffect:
    """A detected correlated change propagating across flywheels."""
    trigger_flywheel: str
    affected_flywheel: str
    correlation: float
    lag_cycles: int
    direction: str  # "amplifying" | "dampening"

    def to_dict(self) -> dict:
        return {
            "trigger_flywheel": self.trigger_flywheel,
            "affected_flywheel": self.affected_flywheel,
            "correlation": round(self.correlation, 4),
            "lag_cycles": self.lag_cycles,
            "direction": self.direction,
        }


# ---------------------------------------------------------------------------
# FlywheelMesh
# ---------------------------------------------------------------------------

class FlywheelMesh:
    """Directed dependency graph of Sentinel's flywheel subsystems.

    Nodes are flywheel names (strings).
    Edges represent data/signal flow: source → target means *source* output
    is consumed by *target*.

    Example topology (registered on first use):
        detection  → calibration, innovation, shadow
        innovation → detection, drift
        drift      → detection, innovation
        shadow     → detection
        calibration→ detection
        research   → innovation, detection
    """

    def __init__(self) -> None:
        # node_name → {"dependencies": [str], "outputs": [str]}
        self._nodes: dict[str, dict] = {}
        # (source, target) → edge_type
        self._edges: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_flywheel(
        self,
        name: str,
        dependencies: list[str] | None = None,
        outputs: list[str] | None = None,
        edge_type: str = "data",
    ) -> None:
        """Declare a flywheel node and its connections.

        *dependencies* is the list of flywheels this node reads from.
        *outputs* is the list of flywheels that consume this node's output.
        Both directions are recorded as directed edges.
        """
        deps = dependencies or []
        outs = outputs or []

        self._nodes[name] = {
            "dependencies": list(deps),
            "outputs": list(outs),
        }

        # dependency edges: dep → name
        for dep in deps:
            self._edges[(dep, name)] = edge_type
            # ensure the dep node exists
            if dep not in self._nodes:
                self._nodes[dep] = {"dependencies": [], "outputs": []}
            if name not in self._nodes[dep]["outputs"]:
                self._nodes[dep]["outputs"].append(name)

        # output edges: name → out
        for out in outs:
            self._edges[(name, out)] = edge_type
            if out not in self._nodes:
                self._nodes[out] = {"dependencies": [], "outputs": []}
            if name not in self._nodes[out]["dependencies"]:
                self._nodes[out]["dependencies"].append(name)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_dependency_graph(self) -> dict:
        """Return the full mesh topology as a plain dict."""
        return {
            "nodes": list(self._nodes.keys()),
            "edges": [
                {"source": src, "target": tgt, "type": etype}
                for (src, tgt), etype in self._edges.items()
            ],
            "adjacency": {
                name: dict(info)
                for name, info in self._nodes.items()
            },
        }

    def get_downstream(self, flywheel_name: str) -> list[str]:
        """Return all flywheels (direct + transitive) that depend on *flywheel_name*."""
        if flywheel_name not in self._nodes:
            return []
        visited: set[str] = set()
        queue = list(self._nodes[flywheel_name].get("outputs", []))
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(self._nodes.get(node, {}).get("outputs", []))
        return sorted(visited)

    def get_upstream(self, flywheel_name: str) -> list[str]:
        """Return all flywheels (direct + transitive) that *flywheel_name* depends on."""
        if flywheel_name not in self._nodes:
            return []
        visited: set[str] = set()
        queue = list(self._nodes[flywheel_name].get("dependencies", []))
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(self._nodes.get(node, {}).get("dependencies", []))
        return sorted(visited)

    def get_direct_dependencies(self, flywheel_name: str) -> list[str]:
        """Return immediate upstream nodes only."""
        return list(self._nodes.get(flywheel_name, {}).get("dependencies", []))

    def get_direct_outputs(self, flywheel_name: str) -> list[str]:
        """Return immediate downstream nodes only."""
        return list(self._nodes.get(flywheel_name, {}).get("outputs", []))

    def has_node(self, name: str) -> bool:
        return name in self._nodes

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    # ------------------------------------------------------------------
    # ASCII rendering
    # ------------------------------------------------------------------

    def render_ascii(self) -> str:
        """Return a text-based representation of the mesh topology.

        Format:
            [node]  -->  [downstream1], [downstream2]
        Nodes with no outputs are shown with "  (sink)".
        """
        if not self._nodes:
            return "(empty mesh — no flywheels registered)"

        lines: list[str] = ["Flywheel Dependency Mesh", "=" * 40]

        for name in sorted(self._nodes.keys()):
            info = self._nodes[name]
            outs = info.get("outputs", [])
            deps = info.get("dependencies", [])

            if outs:
                targets = ", ".join(f"[{o}]" for o in sorted(outs))
                lines.append(f"  [{name}]  -->  {targets}")
            else:
                lines.append(f"  [{name}]  (sink)")

            if deps:
                upstreams = ", ".join(f"[{d}]" for d in sorted(deps))
                lines.append(f"    ^ depends on: {upstreams}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sentinel default mesh factory
# ---------------------------------------------------------------------------

def build_default_mesh() -> FlywheelMesh:
    """Return a FlywheelMesh pre-populated with Sentinel's known topology."""
    mesh = FlywheelMesh()

    # detection flywheel: core scoring; calibration adjusts its thresholds;
    # innovation + drift feed back new patterns and signals
    mesh.register_flywheel(
        "detection",
        dependencies=["calibration", "innovation", "drift", "shadow", "research"],
        outputs=[],
        edge_type="feedback",
    )
    mesh.register_flywheel(
        "calibration",
        dependencies=["detection"],
        outputs=["detection"],
        edge_type="threshold_adjustment",
    )
    mesh.register_flywheel(
        "innovation",
        dependencies=["detection", "drift", "research"],
        outputs=["detection"],
        edge_type="pattern_candidate",
    )
    mesh.register_flywheel(
        "shadow",
        dependencies=["detection"],
        outputs=["detection"],
        edge_type="weight_promotion",
    )
    mesh.register_flywheel(
        "drift",
        dependencies=["detection"],
        outputs=["detection", "innovation"],
        edge_type="signal_alarm",
    )
    mesh.register_flywheel(
        "research",
        dependencies=[],
        outputs=["detection", "innovation"],
        edge_type="knowledge_injection",
    )

    return mesh


# ---------------------------------------------------------------------------
# CascadeDetector
# ---------------------------------------------------------------------------

_RISK_BUCKETS = ["safe", "low", "suspicious", "high", "scam"]
_BUCKET_BOUNDS = [0.2, 0.4, 0.6, 0.8, 1.0]   # upper exclusive except last


def _score_to_bucket(score: float) -> str:
    """Map a scam_score to its risk bucket name."""
    for i, bound in enumerate(_BUCKET_BOUNDS):
        if score < bound:
            return _RISK_BUCKETS[i]
    return _RISK_BUCKETS[-1]


def _bucket_rank(bucket: str) -> int:
    try:
        return _RISK_BUCKETS.index(bucket)
    except ValueError:
        return 0


def _apply_weight_delta(
    base_score: float,
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    signal_names: list[str],
) -> float:
    """Estimate the score under *new_weights* by proportional delta.

    We compute the weighted-average multiplier from the weight changes of
    signals present in this job, then scale the base score accordingly,
    clamping to [0, 1].
    """
    if not signal_names or not (old_weights or new_weights):
        return base_score

    delta_sum = 0.0
    count = 0
    for sig in signal_names:
        old_w = old_weights.get(sig, 0.5)
        new_w = new_weights.get(sig, old_w)  # default: no change
        if old_w > 0:
            delta_sum += (new_w - old_w) / old_w
            count += 1

    if count == 0:
        return base_score

    avg_rel_delta = delta_sum / count
    new_score = base_score * (1.0 + avg_rel_delta * 0.5)  # 0.5 damping factor
    return max(0.0, min(1.0, new_score))


class CascadeDetector:
    """Detects, previews, and records cross-flywheel cascade effects."""

    # Impact thresholds (fraction of jobs whose risk classification changes)
    SAFE_THRESHOLD = 0.05        # < 5%  → SAFE
    MODERATE_THRESHOLD = 0.15   # < 15% → MODERATE, else HIGH

    def __init__(self, mesh: FlywheelMesh | None = None) -> None:
        self.mesh = mesh or build_default_mesh()

    # ------------------------------------------------------------------
    # preview_impact
    # ------------------------------------------------------------------

    def preview_impact(
        self,
        db: "SentinelDB",
        old_weights: dict[str, float],
        new_weights: dict[str, float],
        sample_size: int = 100,
    ) -> CascadeReport:
        """Re-score a sample of recent jobs under old vs new weights.

        Returns a CascadeReport characterising how many jobs would change
        risk classification and by how much.
        """
        from sentinel.scorer import classify_risk  # deferred import

        jobs = db.get_recent_jobs_for_sampling(limit=sample_size)
        if not jobs:
            return CascadeReport(
                jobs_sampled=0,
                classifications_changed=0,
                change_rate=0.0,
                risk_level="SAFE",
                score_delta_mean=0.0,
                score_delta_std=0.0,
                promoted_count=0,
                demoted_count=0,
            )

        classifications_changed = 0
        promoted = 0
        demoted = 0
        deltas: list[float] = []

        for job in jobs:
            base_score = float(job.get("scam_score") or 0.0)
            signals_raw = job.get("signals_json", "[]")
            try:
                signals_list = (
                    json.loads(signals_raw) if isinstance(signals_raw, str)
                    else signals_raw
                )
            except (json.JSONDecodeError, TypeError):
                signals_list = []

            # Collect signal names present in this job
            sig_names: list[str] = []
            for s in signals_list:
                if isinstance(s, dict):
                    name = s.get("name", "")
                elif isinstance(s, str):
                    name = s
                else:
                    name = ""
                if name:
                    sig_names.append(name)

            old_score = base_score
            new_score = _apply_weight_delta(base_score, old_weights, new_weights, sig_names)

            old_bucket = _score_to_bucket(old_score)
            new_bucket = _score_to_bucket(new_score)

            delta = new_score - old_score
            deltas.append(delta)

            if old_bucket != new_bucket:
                classifications_changed += 1
                old_rank = _bucket_rank(old_bucket)
                new_rank = _bucket_rank(new_bucket)
                if new_rank > old_rank:
                    promoted += 1
                else:
                    demoted += 1

        n = len(jobs)
        change_rate = classifications_changed / n if n > 0 else 0.0
        delta_mean = statistics.mean(deltas) if deltas else 0.0
        delta_std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

        if change_rate < self.SAFE_THRESHOLD:
            impact = "SAFE"
        elif change_rate < self.MODERATE_THRESHOLD:
            impact = "MODERATE"
        else:
            impact = "HIGH"

        return CascadeReport(
            jobs_sampled=n,
            classifications_changed=classifications_changed,
            change_rate=change_rate,
            risk_level=impact,
            score_delta_mean=delta_mean,
            score_delta_std=delta_std,
            promoted_count=promoted,
            demoted_count=demoted,
        )

    # ------------------------------------------------------------------
    # track_cascade
    # ------------------------------------------------------------------

    def track_cascade(
        self,
        db: "SentinelDB",
        change_event: str,
        before_metrics: dict,
        after_metrics: dict,
    ) -> CascadeRecord:
        """Record a change event and compute magnitude + affected flywheels.

        Magnitude = max abs change across precision/recall/f1 in metrics dicts.
        Affected flywheels are derived from the mesh's downstream graph for
        the flywheel that triggered the event.
        """
        # Extract flywheel name from change_event prefix (e.g. "detection:evolve")
        trigger_fw = change_event.split(":")[0] if ":" in change_event else change_event

        # Compute magnitude from before/after metric diffs
        magnitude = 0.0
        for key in ("precision", "recall", "f1", "accuracy", "cusum_statistic"):
            before_val = float(before_metrics.get(key, 0.0) or 0.0)
            after_val = float(after_metrics.get(key, 0.0) or 0.0)
            magnitude = max(magnitude, abs(after_val - before_val))

        affected = self.mesh.get_downstream(trigger_fw)

        record = CascadeRecord(
            trigger=change_event,
            change_type=change_event.split(":")[-1] if ":" in change_event else "unknown",
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            magnitude=magnitude,
            affected_flywheels=affected,
        )

        # Persist to DB
        try:
            db.insert_cascade_event(
                trigger=change_event,
                change_type=record.change_type,
                impact_json=json.dumps(record.to_dict()),
            )
        except Exception:
            logger.debug("Failed to persist cascade record", exc_info=True)

        return record

    # ------------------------------------------------------------------
    # detect_ripple_effects
    # ------------------------------------------------------------------

    def detect_ripple_effects(
        self,
        db: "SentinelDB",
        lookback_cycles: int = 5,
    ) -> list[RippleEffect]:
        """Detect correlated changes across flywheel metric histories.

        Algorithm:
        1. Pull the last *lookback_cycles* flywheel_metrics rows.
        2. For each pair of flywheels, compute cross-correlation on their
           precision time-series with lag 0 and lag 1.
        3. If |correlation| > 0.5, record a RippleEffect.

        Currently we have one rich time-series: flywheel_metrics (detection).
        For other flywheels we proxy via cascade_events counts per cycle.
        """
        ripples: list[RippleEffect] = []

        # Fetch recent flywheel metric history
        metric_rows = db.get_flywheel_metrics_history(days=365, limit=max(lookback_cycles * 2, 20))
        if len(metric_rows) < 3:
            return ripples

        # Build per-flywheel series (proxy: use precision as the KPI series)
        # For detection flywheel: directly from flywheel_metrics
        detection_series = [float(r.get("precision", 0.0) or 0.0) for r in reversed(metric_rows)]
        calibration_series = [float(r.get("thresholds_adjusted", r.get("patterns_evolved", 0)) or 0.0)
                              for r in reversed(metric_rows)]
        innovation_series = [float(r.get("patterns_evolved", 0) or 0.0)
                             for r in reversed(metric_rows)]

        series_map = {
            "detection": detection_series,
            "calibration": calibration_series,
            "innovation": innovation_series,
        }

        # Supplement with cascade event counts per metric bucket (best-effort)
        try:
            cascade_history = db.get_cascade_history(limit=50)
            if cascade_history:
                # Build a time-indexed count series (simplified)
                event_counts = [0] * len(metric_rows)
                for i, row in enumerate(reversed(metric_rows)):
                    ts = row.get("cycle_ts", "")
                    event_counts[i] = sum(
                        1 for ce in cascade_history
                        if ce.get("timestamp", "") <= ts
                    )
                series_map["shadow"] = [float(v) for v in event_counts]
        except Exception:
            pass

        # Pairwise cross-correlations
        flywheel_pairs = [
            ("detection", "calibration"),
            ("detection", "innovation"),
            ("innovation", "detection"),
            ("calibration", "detection"),
            ("detection", "shadow"),
            ("innovation", "calibration"),
        ]

        for source, target in flywheel_pairs:
            src_series = series_map.get(source)
            tgt_series = series_map.get(target)
            if not src_series or not tgt_series:
                continue

            # Lag 0
            corr_lag0 = _pearson_correlation(src_series, tgt_series)
            if abs(corr_lag0) >= 0.5:
                direction = "amplifying" if corr_lag0 > 0 else "dampening"
                ripples.append(RippleEffect(
                    trigger_flywheel=source,
                    affected_flywheel=target,
                    correlation=corr_lag0,
                    lag_cycles=0,
                    direction=direction,
                ))

            # Lag 1 (source leads target by 1 cycle)
            if len(src_series) > 1 and len(tgt_series) > 1:
                corr_lag1 = _pearson_correlation(src_series[:-1], tgt_series[1:])
                if abs(corr_lag1) >= 0.5:
                    direction = "amplifying" if corr_lag1 > 0 else "dampening"
                    ripples.append(RippleEffect(
                        trigger_flywheel=source,
                        affected_flywheel=target,
                        correlation=corr_lag1,
                        lag_cycles=1,
                        direction=direction,
                    ))

        return ripples


# ---------------------------------------------------------------------------
# Statistical helper
# ---------------------------------------------------------------------------

def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Return Pearson r for two equal-length float lists.

    Returns 0.0 if there is no variance in either series or if lists are
    too short (< 2 elements).
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    x = x[:n]
    y = y[:n]

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    denom = denom_x * denom_y
    if denom == 0.0:
        return 0.0

    return max(-1.0, min(1.0, num / denom))
