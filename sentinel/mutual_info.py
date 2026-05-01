"""Mutual Information (MI) based feature selection for Sentinel scam signals.

Uses information theory (stdlib math only) to rank signals by their actual
predictive value, detect redundancy, and track how signal informativeness
changes over time.

Classes
-------
MutualInformationCalculator — Compute & rank MI between each signal and labels.
ConditionalMI               — Conditional MI and minimum sufficient signal set.
InformationGainTracker      — Track per-signal MI over time; alert on decay.

All computation is pure Python + stdlib (math, collections, statistics).
Data is read from sentinel.db (SentinelDB) or supplied directly as lists.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import mean, stdev
from typing import Iterable


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------


def _entropy(counts: Iterable[int]) -> float:
    """Shannon entropy H(X) in bits from raw counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def _joint_entropy(pairs: list[tuple]) -> float:
    """H(X, Y) from a list of (x, y) pairs."""
    counts: Counter = Counter(pairs)
    return _entropy(counts.values())


def _conditional_entropy(pairs: list[tuple]) -> float:
    """H(Y | X) from a list of (x, y) pairs."""
    # H(Y|X) = H(X,Y) - H(X)
    x_counts: Counter = Counter(x for x, _ in pairs)
    h_joint = _joint_entropy(pairs)
    h_x = _entropy(x_counts.values())
    return max(0.0, h_joint - h_x)


def _mutual_information(x_vals: list, y_vals: list) -> float:
    """MI(X; Y) = H(Y) - H(Y|X) from aligned value lists."""
    if len(x_vals) != len(y_vals) or not x_vals:
        return 0.0
    y_counts: Counter = Counter(y_vals)
    h_y = _entropy(y_counts.values())
    h_y_given_x = _conditional_entropy(list(zip(x_vals, y_vals)))
    return max(0.0, h_y - h_y_given_x)


def _conditional_mutual_information(
    x_vals: list, y_vals: list, z_vals: list
) -> float:
    """CMI(X; Y | Z) = H(X | Z) - H(X | Y, Z)."""
    if not x_vals:
        return 0.0
    # H(X | Z)
    h_x_given_z = _conditional_entropy(list(zip(z_vals, x_vals)))
    # H(X | Y, Z) — treat (y, z) as a compound variable
    yz = [(y, z) for y, z in zip(y_vals, z_vals)]
    h_x_given_yz = _conditional_entropy(list(zip(yz, x_vals)))
    return max(0.0, h_x_given_z - h_x_given_yz)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _binarise_score(score: float, threshold: float = 0.5) -> int:
    """Convert a continuous scam score to a binary label (1=scam, 0=legit)."""
    return 1 if score >= threshold else 0


def _signal_present(job_signals: list[dict], signal_name: str) -> int:
    """Return 1 if signal_name appears in a job's signal list, else 0."""
    for s in job_signals:
        if s.get("name") == signal_name:
            return 1
    return 0


# ---------------------------------------------------------------------------
# SignalMIResult
# ---------------------------------------------------------------------------


@dataclass
class SignalMIResult:
    """MI analysis result for a single signal.

    Attributes
    ----------
    signal_name:        Name of the signal.
    mi_bits:            MI(signal, label) in bits.
    mi_normalised:      MI normalised by H(Y) so it is in [0, 1].
    fire_rate:          Fraction of jobs where signal fired (prevalence).
    precision_when_fired: P(scam=1 | signal fired) — observed precision.
    rank:               Rank among all signals (1 = highest MI).
    is_redundant:       True if the signal adds little unique info beyond others.
    redundant_with:     Name of the signal this one is redundant with, or "".
    """

    signal_name: str
    mi_bits: float
    mi_normalised: float
    fire_rate: float
    precision_when_fired: float
    rank: int = 0
    is_redundant: bool = False
    redundant_with: str = ""


# ---------------------------------------------------------------------------
# MutualInformationCalculator
# ---------------------------------------------------------------------------


class MutualInformationCalculator:
    """Compute and rank MI between each signal and the scam/legit label.

    MI(signal, label) = how much knowing this signal reduces uncertainty about
    whether a job is a scam.  Higher MI → more informative signal.

    Parameters
    ----------
    scam_threshold: Score above which a job is labelled "scam" (default 0.5).
    redundancy_threshold: MI-overlap above which two signals are "redundant"
                          (default 0.8 = 80% shared information).
    """

    def __init__(
        self,
        scam_threshold: float = 0.5,
        redundancy_threshold: float = 0.80,
    ) -> None:
        self.scam_threshold = scam_threshold
        self.redundancy_threshold = redundancy_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_from_db(self, db: object) -> list[SignalMIResult]:
        """Load job records from *db* and compute MI for every signal found.

        Parameters
        ----------
        db: SentinelDB instance (duck-typed; we use ``get_recent_jobs_for_sampling``
            and ``get_patterns``).

        Returns
        -------
        List of SignalMIResult sorted by MI descending (highest first).
        """
        from sentinel.db import SentinelDB

        assert isinstance(db, SentinelDB)

        jobs = db.get_recent_jobs_for_sampling(limit=2000)
        if not jobs:
            return []

        # Build (label, signals_list) pairs
        import json
        labelled: list[tuple[int, list[dict]]] = []
        for job in jobs:
            score = job.get("scam_score") or 0.0
            label = _binarise_score(score, self.scam_threshold)
            raw = job.get("signals_json", "[]")
            try:
                signals = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                signals = []
            labelled.append((label, signals if isinstance(signals, list) else []))

        return self._compute(labelled)

    def compute_from_records(
        self,
        records: list[tuple[int, list[str]]],
    ) -> list[SignalMIResult]:
        """Compute MI from pre-processed records.

        Parameters
        ----------
        records: List of (label, [signal_name, ...]) pairs where label is
                 1 (scam) or 0 (legit) and the signal list contains signal
                 names that fired for that job.

        Returns
        -------
        List of SignalMIResult sorted by MI descending.
        """
        # Convert to internal format: (label, [{name: s} for s in names])
        labelled = [
            (label, [{"name": s} for s in signal_names])
            for label, signal_names in records
        ]
        return self._compute(labelled)

    def rank_signals(self, results: list[SignalMIResult]) -> list[SignalMIResult]:
        """Re-rank and number a list of results (sorts by MI descending)."""
        ranked = sorted(results, key=lambda r: r.mi_bits, reverse=True)
        for i, r in enumerate(ranked):
            r.rank = i + 1
        return ranked

    def prune_recommendations(
        self, results: list[SignalMIResult], low_mi_threshold: float = 0.005
    ) -> list[str]:
        """Return signal names recommended for pruning.

        A signal is recommended for pruning when EITHER:
        - Its MI < *low_mi_threshold* bits (essentially uninformative), OR
        - It is flagged as redundant with a higher-ranked signal.
        """
        prune: list[str] = []
        for r in results:
            if r.mi_bits < low_mi_threshold or r.is_redundant:
                prune.append(r.signal_name)
        return prune

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute(
        self, labelled: list[tuple[int, list[dict]]]
    ) -> list[SignalMIResult]:
        if not labelled:
            return []

        labels = [lbl for lbl, _ in labelled]
        h_label = _entropy(Counter(labels).values())

        # Discover all unique signal names
        all_signal_names: set[str] = set()
        for _, signals in labelled:
            for s in signals:
                name = s.get("name", "")
                if name:
                    all_signal_names.add(name)

        if not all_signal_names:
            return []

        results: list[SignalMIResult] = []
        for signal_name in all_signal_names:
            x_vals = [_signal_present(sigs, signal_name) for _, sigs in labelled]
            mi_bits = _mutual_information(x_vals, labels)
            mi_norm = mi_bits / h_label if h_label > 0 else 0.0

            fired_count = sum(x_vals)
            fire_rate = fired_count / len(x_vals) if x_vals else 0.0

            # P(scam | signal fired)
            if fired_count > 0:
                scam_when_fired = sum(
                    lbl for xv, (lbl, _) in zip(x_vals, labelled) if xv == 1
                )
                precision_when_fired = scam_when_fired / fired_count
            else:
                precision_when_fired = 0.0

            results.append(
                SignalMIResult(
                    signal_name=signal_name,
                    mi_bits=round(mi_bits, 6),
                    mi_normalised=round(mi_norm, 4),
                    fire_rate=round(fire_rate, 4),
                    precision_when_fired=round(precision_when_fired, 4),
                )
            )

        # Rank
        results = self.rank_signals(results)

        # Mark redundant pairs
        self._mark_redundant(results, labelled, labels, h_label)

        return results

    def _mark_redundant(
        self,
        results: list[SignalMIResult],
        labelled: list[tuple[int, list[dict]]],
        labels: list[int],
        h_label: float,
    ) -> None:
        """Flag lower-ranked signals as redundant when they share > threshold MI."""
        # Build signal presence vectors
        vectors: dict[str, list[int]] = {}
        for r in results:
            vectors[r.signal_name] = [
                _signal_present(sigs, r.signal_name) for _, sigs in labelled
            ]

        # Compare pairs (only check lower-ranked vs higher-ranked)
        for i, r_low in enumerate(results):
            for r_high in results[:i]:
                if r_low.is_redundant:
                    break
                # CMI(low_signal; label | high_signal) — if near 0, low adds nothing
                cmi = _conditional_mutual_information(
                    vectors[r_low.signal_name],
                    labels,
                    vectors[r_high.signal_name],
                )
                # Normalise by H(label)
                cmi_norm = cmi / h_label if h_label > 0 else 0.0
                # If the remaining unique info is tiny fraction of original MI
                if r_low.mi_bits > 0 and cmi / r_low.mi_bits < (1 - self.redundancy_threshold):
                    r_low.is_redundant = True
                    r_low.redundant_with = r_high.signal_name


# ---------------------------------------------------------------------------
# ConditionalMI
# ---------------------------------------------------------------------------


@dataclass
class MinimalSignalSet:
    """Result of ConditionalMI.find_minimal_set.

    Attributes
    ----------
    selected_signals:   Names of signals in the minimum sufficient set.
    removed_signals:    Signals not selected (safely prunable).
    retained_mi_bits:   Total MI captured by the selected set.
    total_mi_bits:      Total MI if all signals were kept.
    information_retained_pct: retained_mi_bits / total_mi_bits * 100.
    dependency_edges:   List of (signal_a, signal_b, shared_mi_bits) pairs.
    """

    selected_signals: list[str]
    removed_signals: list[str]
    retained_mi_bits: float
    total_mi_bits: float
    information_retained_pct: float
    dependency_edges: list[tuple[str, str, float]] = field(default_factory=list)


class ConditionalMI:
    """Compute conditional MI and find the minimum set of signals.

    Uses a greedy forward selection algorithm:
    1. Start with the signal of highest MI.
    2. At each step, add the signal with the highest CMI(signal; label | selected).
    3. Stop when adding more signals yields < min_gain_bits of new information.

    Parameters
    ----------
    min_gain_bits: Minimum CMI gain to justify adding another signal (default 0.001).
    max_loss_pct:  Maximum tolerated MI loss when pruning (default 2.0 = 2%).
    """

    def __init__(
        self,
        min_gain_bits: float = 0.001,
        max_loss_pct: float = 2.0,
    ) -> None:
        self.min_gain_bits = min_gain_bits
        self.max_loss_pct = max_loss_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_minimal_set(
        self,
        records: list[tuple[int, list[str]]],
    ) -> MinimalSignalSet:
        """Find the smallest set of signals that captures > (100 - max_loss_pct)% MI.

        Parameters
        ----------
        records: List of (label, [signal_names...]) pairs.

        Returns
        -------
        MinimalSignalSet dataclass.
        """
        if not records:
            return MinimalSignalSet([], [], 0.0, 0.0, 0.0)

        labels = [lbl for lbl, _ in records]
        h_label = _entropy(Counter(labels).values())

        # Build presence vectors for all signals
        all_signals: set[str] = set()
        for _, names in records:
            all_signals.update(names)

        if not all_signals:
            return MinimalSignalSet([], [], 0.0, 0.0, 0.0)

        vectors: dict[str, list[int]] = {
            sig: [1 if sig in names else 0 for _, names in records]
            for sig in all_signals
        }

        # Individual MI for each signal
        mi_map: dict[str, float] = {
            sig: _mutual_information(vec, labels)
            for sig, vec in vectors.items()
        }
        total_mi = max(mi_map.values()) if mi_map else 0.0

        # Greedy forward selection
        selected: list[str] = []
        remaining = set(all_signals)

        while remaining:
            best_sig = None
            best_gain = 0.0
            for sig in remaining:
                if not selected:
                    gain = mi_map[sig]
                else:
                    # CMI(sig; label | selected_so_far)
                    # Use joint compound variable of selected signals as conditioning
                    z_vals = self._compound(selected, vectors, len(labels))
                    gain = _conditional_mutual_information(vectors[sig], labels, z_vals)
                if gain > best_gain:
                    best_gain = gain
                    best_sig = sig

            if best_sig is None or best_gain < self.min_gain_bits:
                break

            selected.append(best_sig)
            remaining.discard(best_sig)

            # Check if we've captured enough MI
            retained = self._retained_mi(selected, vectors, labels, h_label)
            if h_label > 0 and retained / h_label >= (1.0 - self.max_loss_pct / 100.0):
                break

        removed = [s for s in all_signals if s not in selected]
        retained_mi = self._retained_mi(selected, vectors, labels, h_label)
        pct = (retained_mi / total_mi * 100.0) if total_mi > 0 else 100.0

        edges = self._build_dependency_edges(list(all_signals), vectors, labels)

        return MinimalSignalSet(
            selected_signals=selected,
            removed_signals=removed,
            retained_mi_bits=round(retained_mi, 6),
            total_mi_bits=round(total_mi, 6),
            information_retained_pct=round(pct, 2),
            dependency_edges=edges,
        )

    def build_signal_dependency_graph(
        self, records: list[tuple[int, list[str]]]
    ) -> list[tuple[str, str, float]]:
        """Return edges (signal_a, signal_b, shared_mi_bits) for pairs with CMI > 0.

        Shared MI is approximated as the reduction in CMI vs raw MI.
        """
        if not records:
            return []
        labels = [lbl for lbl, _ in records]
        all_signals: set[str] = set()
        for _, names in records:
            all_signals.update(names)

        vectors: dict[str, list[int]] = {
            sig: [1 if sig in names else 0 for _, names in records]
            for sig in all_signals
        }
        return self._build_dependency_edges(list(all_signals), vectors, labels)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compound(signal_names: list[str], vectors: dict[str, list[int]], n: int) -> list[tuple]:
        """Create a compound variable by zipping the presence vectors of selected signals."""
        if not signal_names:
            return [() for _ in range(n)]
        return list(zip(*(vectors[s] for s in signal_names)))

    @staticmethod
    def _retained_mi(
        selected: list[str],
        vectors: dict[str, list[int]],
        labels: list[int],
        h_label: float,
    ) -> float:
        """Estimate the total MI retained by the selected set (upper bound via chain rule)."""
        if not selected or h_label == 0:
            return 0.0
        # Treat the joint presence vector as a compound feature
        joint = [
            tuple(vectors[s][i] for s in selected) for i in range(len(labels))
        ]
        return _mutual_information(joint, labels)

    @staticmethod
    def _build_dependency_edges(
        signals: list[str],
        vectors: dict[str, list[int]],
        labels: list[int],
    ) -> list[tuple[str, str, float]]:
        """Build dependency edges between signal pairs with substantial MI overlap."""
        edges: list[tuple[str, str, float]] = []
        for i, a in enumerate(signals):
            for b in signals[i + 1:]:
                mi_a = _mutual_information(vectors[a], labels)
                cmi_a_given_b = _conditional_mutual_information(vectors[a], labels, vectors[b])
                shared = mi_a - cmi_a_given_b
                if shared > 0.001:
                    edges.append((a, b, round(shared, 6)))
        return sorted(edges, key=lambda e: e[2], reverse=True)


# ---------------------------------------------------------------------------
# InformationGainTracker
# ---------------------------------------------------------------------------


@dataclass
class MISnapshot:
    """A point-in-time MI measurement for a single signal."""

    signal_name: str
    mi_bits: float
    timestamp: str
    window_label: str = ""


@dataclass
class MIDecayAlert:
    """Alert emitted when a signal's MI drops significantly."""

    signal_name: str
    baseline_mi: float
    recent_mi: float
    drop_pct: float
    severity: str  # "warning" | "critical"
    message: str


class InformationGainTracker:
    """Track per-signal MI over time and alert when a signal becomes less informative.

    Scammers may learn to avoid a specific keyword/pattern.  When they do, the
    signal fires less on true scams, so its MI with the scam label drops.

    Parameters
    ----------
    alert_threshold_pct: Drop percentage that triggers a "warning" (default 25%).
    critical_threshold_pct: Drop percentage that triggers "critical" (default 50%).
    min_snapshots:          Minimum snapshots needed before alerting (default 3).
    """

    ALERT_THRESHOLD_PCT = 25.0
    CRITICAL_THRESHOLD_PCT = 50.0
    MIN_SNAPSHOTS = 3

    def __init__(
        self,
        alert_threshold_pct: float = ALERT_THRESHOLD_PCT,
        critical_threshold_pct: float = CRITICAL_THRESHOLD_PCT,
        min_snapshots: int = MIN_SNAPSHOTS,
    ) -> None:
        self.alert_threshold_pct = alert_threshold_pct
        self.critical_threshold_pct = critical_threshold_pct
        self.min_snapshots = min_snapshots
        # signal_name -> list of MISnapshot (chronological)
        self._history: dict[str, list[MISnapshot]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_snapshot(
        self,
        records: list[tuple[int, list[str]]],
        window_label: str = "",
    ) -> list[MISnapshot]:
        """Compute MI for the current window and record it.

        Parameters
        ----------
        records:      List of (label, [signal_names]) for the current window.
        window_label: Human-readable label (e.g. "2026-W18").

        Returns
        -------
        List of MISnapshot objects recorded this cycle.
        """
        if not records:
            return []

        labels = [lbl for lbl, _ in records]
        all_signals: set[str] = set()
        for _, names in records:
            all_signals.update(names)

        ts = datetime.now(UTC).isoformat()
        snapshots: list[MISnapshot] = []

        for sig in all_signals:
            x_vals = [1 if sig in names else 0 for _, names in records]
            mi = _mutual_information(x_vals, labels)
            snap = MISnapshot(
                signal_name=sig,
                mi_bits=round(mi, 6),
                timestamp=ts,
                window_label=window_label,
            )
            self._history[sig].append(snap)
            snapshots.append(snap)

        return snapshots

    def check_for_decay(self) -> list[MIDecayAlert]:
        """Check all tracked signals for significant MI decay.

        Compares the mean MI of the older half of snapshots (baseline) with
        the mean MI of the newer half (recent).

        Returns
        -------
        List of MIDecayAlert objects (empty if all signals healthy).
        """
        alerts: list[MIDecayAlert] = []

        for sig_name, snapshots in self._history.items():
            if len(snapshots) < self.min_snapshots:
                continue

            half = len(snapshots) // 2
            baseline_vals = [s.mi_bits for s in snapshots[:half]]
            recent_vals = [s.mi_bits for s in snapshots[half:]]

            baseline_mi = mean(baseline_vals)
            recent_mi = mean(recent_vals)

            if baseline_mi <= 0:
                continue

            drop_pct = (baseline_mi - recent_mi) / baseline_mi * 100.0

            if drop_pct >= self.critical_threshold_pct:
                severity = "critical"
            elif drop_pct >= self.alert_threshold_pct:
                severity = "warning"
            else:
                continue

            alerts.append(
                MIDecayAlert(
                    signal_name=sig_name,
                    baseline_mi=round(baseline_mi, 6),
                    recent_mi=round(recent_mi, 6),
                    drop_pct=round(drop_pct, 2),
                    severity=severity,
                    message=(
                        f"Signal '{sig_name}' MI dropped {drop_pct:.1f}% "
                        f"({baseline_mi:.4f} → {recent_mi:.4f} bits) — "
                        f"scammers may be evading this signal."
                    ),
                )
            )

        # Sort by severity (critical first) then by drop_pct
        alerts.sort(key=lambda a: (a.severity != "critical", -a.drop_pct))
        return alerts

    def get_trend(self, signal_name: str) -> list[MISnapshot]:
        """Return MI snapshots for *signal_name* in chronological order."""
        return list(self._history.get(signal_name, []))

    def all_tracked_signals(self) -> list[str]:
        """Return names of all signals that have at least one snapshot."""
        return list(self._history)

    def signal_stats(self, signal_name: str) -> dict:
        """Return descriptive stats for a signal's MI history.

        Returns
        -------
        dict with keys: signal_name, n_snapshots, mean_mi, min_mi, max_mi,
                        std_mi, trend ("+", "-", or "~"), latest_mi.
        """
        snaps = self._history.get(signal_name, [])
        if not snaps:
            return {"signal_name": signal_name, "n_snapshots": 0}

        vals = [s.mi_bits for s in snaps]
        trend = "~"
        if len(vals) >= 2:
            if vals[-1] > vals[0] * 1.05:
                trend = "+"
            elif vals[-1] < vals[0] * 0.95:
                trend = "-"

        return {
            "signal_name": signal_name,
            "n_snapshots": len(vals),
            "mean_mi": round(mean(vals), 6),
            "min_mi": round(min(vals), 6),
            "max_mi": round(max(vals), 6),
            "std_mi": round(stdev(vals) if len(vals) > 1 else 0.0, 6),
            "trend": trend,
            "latest_mi": round(vals[-1], 6),
        }

    def persist_to_db(self, db: object, window_label: str = "") -> None:
        """Write the current MI snapshot history to the DB signal_decay_history table.

        Uses the existing ``insert_signal_rate`` DB method, storing MI as fire_rate.

        Parameters
        ----------
        db:           SentinelDB instance.
        window_label: Used as window_start identifier.
        """
        from sentinel.db import SentinelDB
        assert isinstance(db, SentinelDB)

        for sig_name, snapshots in self._history.items():
            if not snapshots:
                continue
            latest = snapshots[-1]
            db.insert_signal_rate(
                signal_name=sig_name,
                window_start=window_label or latest.timestamp,
                fire_rate=latest.mi_bits,
            )

    def load_from_db(self, db: object) -> None:
        """Reload MI history from the DB's signal_decay_history table.

        Interprets each row's ``fire_rate`` column as a stored MI value
        (as written by ``persist_to_db``).

        Parameters
        ----------
        db: SentinelDB instance.
        """
        from sentinel.db import SentinelDB
        assert isinstance(db, SentinelDB)

        rows = db.get_signal_decay()
        for row in rows:
            sig = row.get("signal_name", "")
            if not sig:
                continue
            snap = MISnapshot(
                signal_name=sig,
                mi_bits=float(row.get("fire_rate", 0.0)),
                timestamp=row.get("recorded_at", ""),
                window_label=row.get("window_start", ""),
            )
            self._history[sig].append(snap)
