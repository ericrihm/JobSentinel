"""Counterfactual Analysis Engine — diagnoses false negatives and false positives.

For each misclassification, this module answers:
  "What would have caught this? What went wrong?"

Public API:
  CounterfactualEngine   — enumerate signal gaps, rank by impact, find minimum flip set
  FailureAnalyzer        — categorise failure modes and prioritise fixes
  SignalGapFinder        — compare caught vs. missed scams to propose new signals
  WeightTuner            — gradient-free weight optimisation over historical data
  RootCauseTracer        — build diagnosis trees for false negatives

All stdlib — no external dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Re-use the scorer's log-odds accumulator directly so results are identical.
# ---------------------------------------------------------------------------

def _score_signals_pure(
    signals: list,  # list[ScamSignal]
    weight_overrides: dict[str, float] | None = None,
) -> float:
    """Compute scam score (0-1) from signals without touching the DB.

    Mirrors scorer.score_signals log-odds logic but accepts optional per-signal
    weight overrides so counterfactual re-scoring is cheap and deterministic.
    """
    from sentinel.models import SignalCategory

    if not signals:
        return 0.0

    overrides = weight_overrides or {}
    log_odds = 0.0
    for s in signals:
        w = overrides.get(s.name, s.weight)
        w = max(1e-6, min(1.0 - 1e-6, w))
        if s.category == SignalCategory.POSITIVE:
            log_odds -= math.log((1.0 - w) / w)
        else:
            log_odds += math.log(w / (1.0 - w))
    return 1.0 / (1.0 + math.exp(-log_odds))


def _classification_threshold() -> float:
    """Score at or above this is classified as SCAM (mirrors scorer._RISK_THRESHOLDS)."""
    try:
        from sentinel.scorer import _RISK_THRESHOLDS
        return _RISK_THRESHOLDS.get("high", 0.8)
    except Exception:
        return 0.8


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

class FailureMode(Enum):
    EVASION = "evasion"       # scammer deliberately avoided triggering signals
    GAP = "gap"               # no signal exists for this pattern
    THRESHOLD = "threshold"   # signals fired but weights too low
    NOVEL = "novel"           # completely new scam type, no related signals


@dataclass
class CounterfactualResult:
    """Impact of adding one signal to a misclassified job."""
    signal_name: str
    original_score: float
    counterfactual_score: float
    score_delta: float          # counterfactual - original
    would_flip: bool            # True if this single signal flips classification
    signal_weight: float        # weight of the hypothetical signal


@dataclass
class MinimumInterventionSet:
    """Smallest set of signals that flips the classification."""
    signals: list[str]
    original_score: float
    flipped_score: float
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = len(self.signals)


@dataclass
class FailureRecord:
    """A single misclassification with its failure mode."""
    job_url: str
    true_label: str            # "scam" | "legitimate"
    predicted_score: float
    failure_mode: FailureMode
    fired_signals: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""


@dataclass
class SignalProposal:
    """A proposed new signal inferred from missed-scam patterns."""
    name: str
    pattern_description: str
    example_texts: list[str]
    missed_scam_count: int      # how many missed scams show this pattern
    estimated_precision: float  # 0-1
    estimated_recall_lift: float
    category: str = "red_flag"


@dataclass
class WeightAdjustment:
    """A recommended per-signal weight change."""
    signal_name: str
    current_weight: float
    proposed_weight: float
    delta: float = field(init=False)
    correctly_classified_gain: int = 0  # extra correct predictions on historical data
    direction: str = field(init=False)  # "increase" | "decrease"

    def __post_init__(self) -> None:
        self.delta = round(self.proposed_weight - self.current_weight, 6)
        self.direction = "increase" if self.delta > 0 else "decrease"


@dataclass
class DiagnosisNode:
    """One step in a root-cause diagnosis tree."""
    step: str                   # "text_normalisation" | "signal_extraction" | "weight" | "threshold"
    finding: str
    passed: bool
    children: list["DiagnosisNode"] = field(default_factory=list)


@dataclass
class DiagnosisTree:
    """Complete root-cause tree for one false negative."""
    job_url: str
    root: DiagnosisNode
    failure_mode: FailureMode
    summary: str


# ---------------------------------------------------------------------------
# CounterfactualEngine
# ---------------------------------------------------------------------------

class CounterfactualEngine:
    """Enumerates signal gaps and computes counterfactual impact for misclassified jobs.

    Usage::

        engine = CounterfactualEngine()
        results = engine.rank_counterfactuals(job, fired_signals, candidate_signals)
        mis = engine.minimum_intervention_set(job, fired_signals, candidate_signals)
    """

    # Default weight to use for a hypothetical new signal
    DEFAULT_HYPOTHETICAL_WEIGHT: float = 0.75

    def __init__(
        self,
        scam_threshold: float | None = None,
        hypothetical_weight: float = DEFAULT_HYPOTHETICAL_WEIGHT,
    ) -> None:
        self._threshold = scam_threshold if scam_threshold is not None else _classification_threshold()
        self._hypo_weight = hypothetical_weight

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def is_scam(self, score: float) -> bool:
        """Return True if *score* is at or above the scam classification threshold."""
        return score >= self._threshold

    def rank_counterfactuals(
        self,
        fired_signals: list,           # list[ScamSignal] — signals that actually fired
        candidate_signal_names: list[str],
        *,
        hypothetical_weight: float | None = None,
    ) -> list[CounterfactualResult]:
        """For each candidate signal that did NOT fire, compute impact if it had.

        Returns results sorted by |score_delta| descending — highest-impact gaps first.

        Args:
            fired_signals:          Signals that actually fired on the job.
            candidate_signal_names: Names of signals to probe (those that did NOT fire).
            hypothetical_weight:    Weight to assign the hypothetical signal.
                                    Defaults to self._hypo_weight.
        """
        from sentinel.models import ScamSignal, SignalCategory

        hw = hypothetical_weight if hypothetical_weight is not None else self._hypo_weight
        base_score = _score_signals_pure(fired_signals)
        results: list[CounterfactualResult] = []

        for name in candidate_signal_names:
            hypo = ScamSignal(
                name=name,
                category=SignalCategory.RED_FLAG,
                weight=hw,
            )
            cf_score = _score_signals_pure(fired_signals + [hypo])
            delta = cf_score - base_score
            results.append(
                CounterfactualResult(
                    signal_name=name,
                    original_score=round(base_score, 4),
                    counterfactual_score=round(cf_score, 4),
                    score_delta=round(delta, 4),
                    would_flip=(not self.is_scam(base_score)) and self.is_scam(cf_score),
                    signal_weight=hw,
                )
            )

        results.sort(key=lambda r: abs(r.score_delta), reverse=True)
        return results

    def minimum_intervention_set(
        self,
        fired_signals: list,           # list[ScamSignal]
        candidate_signal_names: list[str],
        *,
        hypothetical_weight: float | None = None,
        max_set_size: int = 5,
    ) -> MinimumInterventionSet | None:
        """Find the smallest subset of candidate signals that flips classification.

        Uses greedy search: at each step add the candidate with the highest score lift
        until classification flips or max_set_size is reached.

        Returns None if no combination within max_set_size flips the classification.
        """
        from sentinel.models import ScamSignal, SignalCategory

        hw = hypothetical_weight if hypothetical_weight is not None else self._hypo_weight
        base_score = _score_signals_pure(fired_signals)

        # If already scam → nothing to flip for a false-positive direction
        # If already safe → greedy add until we cross the threshold
        if self.is_scam(base_score):
            # For false-positive analysis: find signals whose removal flips it
            return self._min_removal_set(fired_signals, max_set_size)

        augmented = list(fired_signals)
        chosen: list[str] = []
        remaining = list(candidate_signal_names)

        for _ in range(max_set_size):
            if not remaining:
                break

            best_name: str | None = None
            best_score: float = _score_signals_pure(augmented)
            best_hypo = None

            for name in remaining:
                hypo = ScamSignal(name=name, category=SignalCategory.RED_FLAG, weight=hw)
                score = _score_signals_pure(augmented + [hypo])
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_hypo = hypo

            if best_name is None:
                break

            chosen.append(best_name)
            augmented.append(best_hypo)
            remaining.remove(best_name)

            if self.is_scam(best_score):
                return MinimumInterventionSet(
                    signals=chosen,
                    original_score=round(base_score, 4),
                    flipped_score=round(best_score, 4),
                )

        return None  # could not flip within max_set_size

    def score_with_weight_override(
        self,
        signals: list,
        overrides: dict[str, float],
    ) -> float:
        """Re-score signals with per-signal weight overrides (for WeightTuner)."""
        return _score_signals_pure(signals, weight_overrides=overrides)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _min_removal_set(
        self, fired_signals: list, max_set_size: int
    ) -> MinimumInterventionSet | None:
        """For false-positives: find the fewest signals to remove to flip to non-scam."""
        from sentinel.models import SignalCategory

        base_score = _score_signals_pure(fired_signals)
        if not self.is_scam(base_score):
            return None

        # Only consider non-positive signals (positive signals already pull score down)
        removable = [
            s for s in fired_signals if s.category != SignalCategory.POSITIVE
        ]
        removable.sort(key=lambda s: s.weight, reverse=True)

        removed_names: list[str] = []
        current = list(fired_signals)

        for sig in removable[:max_set_size]:
            current = [s for s in current if s.name != sig.name]
            removed_names.append(sig.name)
            new_score = _score_signals_pure(current)
            if not self.is_scam(new_score):
                return MinimumInterventionSet(
                    signals=removed_names,
                    original_score=round(base_score, 4),
                    flipped_score=round(new_score, 4),
                )

        return None


# ---------------------------------------------------------------------------
# FailureAnalyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """Categorise failure modes and maintain a priority queue of fixes.

    Failure modes:
      EVASION   — text obfuscation defeated signal regex
      GAP       — no signal covers the scam tactic at all
      THRESHOLD — signals fired but combined weight was too low
      NOVEL     — scam uses no recognisable patterns

    Priority = frequency × impact (fraction of these failures that matter most).
    """

    # How many signals is "suspiciously low" for a scam?
    _LOW_SIGNAL_COUNT = 2
    # Score above which a job should have been caught
    _EXPECTED_SCAM_THRESHOLD = 0.6

    def __init__(self) -> None:
        self._records: list[FailureRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_false_negative(
        self,
        job_url: str,
        predicted_score: float,
        fired_signals: list,   # list[ScamSignal]
        *,
        job_text: str = "",
        near_misses: list[str] | None = None,
        notes: str = "",
    ) -> FailureRecord:
        """Classify and store one false-negative failure.

        Args:
            job_url:        URL of the missed scam.
            predicted_score: Our score at the time of the miss.
            fired_signals:  Signals that actually fired.
            job_text:       Raw description text (used to probe for evasion).
            near_misses:    Signal names that nearly fired (partial regex matches).
            notes:          Free-form notes.
        """
        mode = self._classify_failure_mode(
            predicted_score, fired_signals, job_text, near_misses or []
        )
        signal_names = [s.name for s in fired_signals]
        record = FailureRecord(
            job_url=job_url,
            true_label="scam",
            predicted_score=predicted_score,
            failure_mode=mode,
            fired_signals=signal_names,
            notes=notes,
        )
        self._records.append(record)
        return record

    def record_false_positive(
        self,
        job_url: str,
        predicted_score: float,
        fired_signals: list,   # list[ScamSignal]
        notes: str = "",
    ) -> FailureRecord:
        """Classify and store one false-positive failure."""
        # FPs are almost always THRESHOLD issues (weights too high)
        # or NOVEL (legitimate posting patterns that look scammy)
        mode = self._classify_fp_mode(predicted_score, fired_signals)
        signal_names = [s.name for s in fired_signals]
        record = FailureRecord(
            job_url=job_url,
            true_label="legitimate",
            predicted_score=predicted_score,
            failure_mode=mode,
            fired_signals=signal_names,
            notes=notes,
        )
        self._records.append(record)
        return record

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def failure_mode_distribution(self) -> dict[str, int]:
        """Return count of each failure mode across all recorded failures."""
        return dict(Counter(r.failure_mode.value for r in self._records))

    def priority_queue(self) -> list[tuple[FailureMode, int, float]]:
        """Return failure modes ranked by frequency × impact.

        Impact is approximated by the average score gap (|predicted - expected|).

        Returns list of (FailureMode, count, priority_score) sorted desc.
        """
        mode_counts: dict[FailureMode, int] = Counter(  # type: ignore[assignment]
            r.failure_mode for r in self._records
        )
        mode_gaps: dict[FailureMode, list[float]] = defaultdict(list)

        for r in self._records:
            expected = 1.0 if r.true_label == "scam" else 0.0
            mode_gaps[r.failure_mode].append(abs(r.predicted_score - expected))

        results: list[tuple[FailureMode, int, float]] = []
        total = max(len(self._records), 1)

        for mode, count in mode_counts.items():
            avg_gap = sum(mode_gaps[mode]) / len(mode_gaps[mode])
            frequency = count / total
            priority = round(frequency * avg_gap, 4)
            results.append((mode, count, priority))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def failure_trends(self, window: int = 10) -> dict[str, list[int]]:
        """Return failure mode counts in rolling windows of *window* records.

        Returns {mode_name: [count_window0, count_window1, ...]} for the last
        two full windows (useful to spot if a mode is growing).
        """
        if len(self._records) < window:
            return self.failure_mode_distribution()

        recent = self._records[-window:]
        previous = self._records[-(2 * window):-window] if len(self._records) >= 2 * window else []

        recent_counts = Counter(r.failure_mode.value for r in recent)
        prev_counts = Counter(r.failure_mode.value for r in previous)

        all_modes = set(recent_counts) | set(prev_counts)
        return {
            mode: [prev_counts.get(mode, 0), recent_counts.get(mode, 0)]
            for mode in sorted(all_modes)
        }

    def get_records(self, mode: FailureMode | None = None) -> list[FailureRecord]:
        """Return all records, optionally filtered by failure mode."""
        if mode is None:
            return list(self._records)
        return [r for r in self._records if r.failure_mode == mode]

    def summary(self) -> dict:
        """Return a compact summary dict for dashboards."""
        dist = self.failure_mode_distribution()
        pq = self.priority_queue()
        top_mode = pq[0][0].value if pq else None
        return {
            "total_failures": len(self._records),
            "distribution": dist,
            "top_priority_mode": top_mode,
            "priority_queue": [
                {"mode": m.value, "count": c, "priority": p}
                for m, c, p in pq
            ],
        }

    # ------------------------------------------------------------------
    # Internal classification helpers
    # ------------------------------------------------------------------

    def _classify_failure_mode(
        self,
        predicted_score: float,
        fired_signals: list,
        job_text: str,
        near_misses: list[str],
    ) -> FailureMode:
        """Heuristic: classify the failure mode for a false negative."""
        n_signals = len(fired_signals)

        # Near-misses → likely evasion
        if near_misses:
            return FailureMode.EVASION

        # Signals fired but score was too low → threshold problem
        if n_signals >= self._LOW_SIGNAL_COUNT and predicted_score >= 0.3:
            return FailureMode.THRESHOLD

        # No signals fired at all
        if n_signals == 0:
            # Check for obfuscation markers in text
            if job_text and self._looks_obfuscated(job_text):
                return FailureMode.EVASION
            return FailureMode.NOVEL

        # Few signals, low score → gap
        return FailureMode.GAP

    def _classify_fp_mode(self, predicted_score: float, fired_signals: list) -> FailureMode:
        """Heuristic: classify the failure mode for a false positive."""
        n_signals = len(fired_signals)
        if n_signals == 0:
            return FailureMode.NOVEL
        if predicted_score >= 0.9:
            return FailureMode.THRESHOLD
        return FailureMode.GAP

    @staticmethod
    def _looks_obfuscated(text: str) -> bool:
        """Very lightweight check for zero-width / homoglyph evasion in text."""
        # Zero-width characters often used to break keyword detection
        # U+200B zero-width space through U+200F right-to-left mark,
        # U+2060 word joiner, U+FEFF zero-width no-break space
        zero_width = re.search(r"[​-‏⁠﻿]", text)
        # Homoglyph substitution — Cyrillic-in-Latin context
        cyrillic = re.search(r"[а-яёА-ЯЁ]", text)
        return bool(zero_width or cyrillic)


# ---------------------------------------------------------------------------
# SignalGapFinder
# ---------------------------------------------------------------------------

class SignalGapFinder:
    """Finds patterns present in missed scams but absent from detected scams.

    Workflow:
      1. Feed it lists of fired-signal sets for caught scams and missed scams.
      2. Call `find_gaps()` to get the differential signal analysis.
      3. Call `generate_proposals()` to get actionable signal proposals.
    """

    def __init__(self) -> None:
        # signal_name → count of times it appeared in caught scams
        self._caught_signal_counts: Counter = Counter()
        # signal_name → count of times it appeared in missed scams
        self._missed_signal_counts: Counter = Counter()
        # Free-form text fragments from missed scams (for pattern mining)
        self._missed_texts: list[str] = []
        self._n_caught: int = 0
        self._n_missed: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_caught_scam(self, signal_names: list[str]) -> None:
        """Record signals that fired on a correctly detected scam."""
        self._n_caught += 1
        self._caught_signal_counts.update(signal_names)

    def add_missed_scam(self, signal_names: list[str], text_fragment: str = "") -> None:
        """Record signals that fired on a missed scam (false negative)."""
        self._n_missed += 1
        self._missed_signal_counts.update(signal_names)
        if text_fragment:
            self._missed_texts.append(text_fragment)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def find_gaps(self) -> list[dict]:
        """Return signals that are over-represented in missed vs. caught scams.

        Each entry: {signal_name, missed_rate, caught_rate, gap_ratio}.
        Sorted by gap_ratio descending.
        """
        if self._n_caught == 0 and self._n_missed == 0:
            return []

        all_signals = set(self._caught_signal_counts) | set(self._missed_signal_counts)
        gaps: list[dict] = []

        for sig in all_signals:
            missed_rate = self._missed_signal_counts.get(sig, 0) / max(self._n_missed, 1)
            caught_rate = self._caught_signal_counts.get(sig, 0) / max(self._n_caught, 1)
            # Gap ratio: how much more common in missed than caught
            gap_ratio = missed_rate / max(caught_rate, 0.01)
            gaps.append(
                {
                    "signal_name": sig,
                    "missed_rate": round(missed_rate, 4),
                    "caught_rate": round(caught_rate, 4),
                    "gap_ratio": round(gap_ratio, 4),
                    "missed_count": self._missed_signal_counts.get(sig, 0),
                    "caught_count": self._caught_signal_counts.get(sig, 0),
                }
            )

        gaps.sort(key=lambda g: g["gap_ratio"], reverse=True)
        return gaps

    def missing_signals(self, min_missed_count: int = 2) -> list[dict]:
        """Return signals that appear exclusively or heavily in missed scams.

        A signal is "missing" (in need of higher weight or a companion signal)
        if its missed_rate is significantly higher than its caught_rate.
        """
        return [
            g for g in self.find_gaps()
            if g["missed_count"] >= min_missed_count
            and g["caught_rate"] < g["missed_rate"] * 0.5
        ]

    def generate_proposals(
        self,
        *,
        top_n: int = 10,
        min_missed_count: int = 2,
    ) -> list[SignalProposal]:
        """Generate SignalProposal objects for patterns we should add.

        Also mines self._missed_texts for recurring n-grams that could become
        new signal regex patterns.
        """
        proposals: list[SignalProposal] = []

        # Proposals from gap analysis
        for gap in self.missing_signals(min_missed_count=min_missed_count)[:top_n]:
            name = gap["signal_name"]
            missed_count = gap["missed_count"]
            # Estimate precision: if caught_rate is near zero, precision is uncertain but promising
            caught_rate = gap["caught_rate"]
            estimated_precision = max(0.5, 1.0 - caught_rate) if caught_rate < 0.5 else 0.5
            recall_lift = round(gap["missed_rate"] * (1.0 - caught_rate), 4)

            proposals.append(
                SignalProposal(
                    name=f"enhanced_{name}",
                    pattern_description=(
                        f"Signal '{name}' fires on {missed_count} missed scams "
                        f"but only {gap['caught_count']} caught scams. "
                        f"Consider increasing its weight or adding a companion signal."
                    ),
                    example_texts=[],
                    missed_scam_count=missed_count,
                    estimated_precision=round(estimated_precision, 4),
                    estimated_recall_lift=recall_lift,
                    category="red_flag",
                )
            )

        # Mine recurring n-grams from missed-scam texts
        ngram_proposals = self._mine_ngram_proposals(min_count=min_missed_count)
        proposals.extend(ngram_proposals[:max(0, top_n - len(proposals))])

        return proposals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mine_ngram_proposals(self, min_count: int = 2) -> list[SignalProposal]:
        """Extract recurring 3-5 word phrases from missed-scam text fragments."""
        if not self._missed_texts:
            return []

        ngram_counts: Counter = Counter()
        for text in self._missed_texts:
            words = re.findall(r"\b[a-z]{3,}\b", text.lower())
            for n in (3, 4, 5):
                for i in range(len(words) - n + 1):
                    gram = " ".join(words[i : i + n])
                    ngram_counts[gram] += 1

        proposals: list[SignalProposal] = []
        for gram, count in ngram_counts.most_common(20):
            if count < min_count:
                break
            proposals.append(
                SignalProposal(
                    name=f"ngram_{gram[:30].replace(' ', '_')}",
                    pattern_description=f"Recurring phrase in {count} missed scams: '{gram}'",
                    example_texts=[gram],
                    missed_scam_count=count,
                    estimated_precision=0.6,
                    estimated_recall_lift=round(count / max(self._n_missed, 1), 4),
                    category="red_flag",
                )
            )
        return proposals


# ---------------------------------------------------------------------------
# WeightTuner
# ---------------------------------------------------------------------------

class WeightTuner:
    """Gradient-free weight optimiser over historical classification data.

    Uses finite-difference gradients (forward differences) to estimate how
    each signal's weight affects overall accuracy on the historical dataset.

    Constraints:
      - No weight moves more than MAX_DELTA_PER_CYCLE per call to tune().
      - Weights stay in [MIN_WEIGHT, MAX_WEIGHT].
    """

    MAX_DELTA_PER_CYCLE: float = 0.1
    MIN_WEIGHT: float = 0.05
    MAX_WEIGHT: float = 0.99
    FD_EPSILON: float = 0.01  # finite-difference step size

    def __init__(
        self,
        *,
        max_delta: float = MAX_DELTA_PER_CYCLE,
        learning_rate: float = 0.5,
    ) -> None:
        self._max_delta = max_delta
        self._lr = learning_rate

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        historical_data: list[dict],
        current_weights: dict[str, float],
        *,
        scam_threshold: float | None = None,
    ) -> list[WeightAdjustment]:
        """Compute weight adjustments that reduce misclassification.

        Args:
            historical_data: List of dicts with keys:
                               "signals": list[ScamSignal]
                               "true_scam": bool
            current_weights: {signal_name: weight} — current signal weights.
            scam_threshold:  Score at or above which we classify as scam.

        Returns:
            List of WeightAdjustment sorted by |correctly_classified_gain| desc.
        """
        if not historical_data or not current_weights:
            return []

        threshold = scam_threshold if scam_threshold is not None else _classification_threshold()

        # Baseline accuracy
        baseline_correct = self._count_correct(historical_data, current_weights, threshold)

        adjustments: list[WeightAdjustment] = []

        for signal_name, current_w in current_weights.items():
            # Forward finite difference: +epsilon
            w_up = min(self.MAX_WEIGHT, current_w + self.FD_EPSILON)
            correct_up = self._count_correct(
                historical_data, {**current_weights, signal_name: w_up}, threshold
            )
            grad_up = (correct_up - baseline_correct) / self.FD_EPSILON

            # Backward finite difference: -epsilon
            w_down = max(self.MIN_WEIGHT, current_w - self.FD_EPSILON)
            correct_down = self._count_correct(
                historical_data, {**current_weights, signal_name: w_down}, threshold
            )
            grad_down = (correct_down - baseline_correct) / self.FD_EPSILON

            # Choose direction with higher gradient
            if grad_up >= grad_down and grad_up > 0:
                raw_delta = self._lr * grad_up * self.FD_EPSILON
                direction_delta = min(raw_delta, self._max_delta)
            elif grad_down > grad_up and grad_down > 0:
                raw_delta = self._lr * grad_down * self.FD_EPSILON
                direction_delta = -min(raw_delta, self._max_delta)
            else:
                direction_delta = 0.0

            if abs(direction_delta) < 1e-8:
                continue  # no improvement from moving this weight

            proposed_w = max(
                self.MIN_WEIGHT,
                min(self.MAX_WEIGHT, current_w + direction_delta),
            )
            # Count gain from applying this single change
            gain = self._count_correct(
                historical_data, {**current_weights, signal_name: proposed_w}, threshold
            ) - baseline_correct

            if gain == 0 and abs(direction_delta) < 1e-6:
                continue

            adjustments.append(
                WeightAdjustment(
                    signal_name=signal_name,
                    current_weight=round(current_w, 6),
                    proposed_weight=round(proposed_w, 6),
                    correctly_classified_gain=gain,
                )
            )

        adjustments.sort(key=lambda a: abs(a.correctly_classified_gain), reverse=True)
        return adjustments

    def apply_adjustments(
        self,
        current_weights: dict[str, float],
        adjustments: list[WeightAdjustment],
        *,
        max_apply: int | None = None,
    ) -> dict[str, float]:
        """Return a new weight dict with the top adjustments applied.

        Args:
            current_weights: The baseline weights dict.
            adjustments:     Adjustments to apply (should be sorted by gain already).
            max_apply:       If set, only apply this many adjustments.
        """
        new_weights = dict(current_weights)
        to_apply = adjustments[:max_apply] if max_apply else adjustments
        for adj in to_apply:
            new_weights[adj.signal_name] = adj.proposed_weight
        return new_weights

    def report(
        self,
        adjustments: list[WeightAdjustment],
        *,
        top_n: int = 10,
    ) -> list[str]:
        """Human-readable report lines for the top adjustments."""
        lines: list[str] = []
        for adj in adjustments[:top_n]:
            verb = "increase" if adj.delta > 0 else "decrease"
            lines.append(
                f"Adjusting '{adj.signal_name}' weight from {adj.current_weight:.3f} "
                f"to {adj.proposed_weight:.3f} ({verb} by {abs(adj.delta):.3f}) "
                f"would correctly classify {adj.correctly_classified_gain:+d} more records."
            )
        return lines

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_correct(
        self,
        historical_data: list[dict],
        weights: dict[str, float],
        threshold: float,
    ) -> int:
        """Count how many historical examples are correctly classified under *weights*."""
        correct = 0
        for item in historical_data:
            signals = item.get("signals", [])
            true_scam = item.get("true_scam", False)
            score = _score_signals_pure(signals, weight_overrides=weights)
            predicted_scam = score >= threshold
            if predicted_scam == true_scam:
                correct += 1
        return correct


# ---------------------------------------------------------------------------
# RootCauseTracer
# ---------------------------------------------------------------------------

class RootCauseTracer:
    """Build a diagnosis tree for each false-negative failure.

    Checks each stage of the pipeline in order:
      1. Text normalisation  — was the text mangled before signals ran?
      2. Signal extraction   — did the relevant signals exist and could they run?
      3. Weight adequacy     — were the weights high enough to score above threshold?
      4. Threshold placement — is the classification threshold itself too high?
    """

    # Minimum weight to be considered "adequate"
    _ADEQUATE_WEIGHT: float = 0.6

    def __init__(self, scam_threshold: float | None = None) -> None:
        self._threshold = scam_threshold if scam_threshold is not None else _classification_threshold()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace(
        self,
        job_url: str,
        job_text: str,
        fired_signals: list,             # list[ScamSignal] that actually fired
        expected_signals: list[str],     # signal names we'd expect to fire on a scam
        predicted_score: float,
    ) -> DiagnosisTree:
        """Build a root-cause tree for a false negative.

        Args:
            job_url:          URL of the missed job.
            job_text:         Raw text that was analysed.
            fired_signals:    Signals that actually fired.
            expected_signals: Signal names that should have fired (ground truth / expert opinion).
            predicted_score:  The score we gave the job.

        Returns:
            DiagnosisTree with a cascading set of DiagnosisNodes.
        """
        root = DiagnosisNode(
            step="pipeline",
            finding="Root-cause analysis for false negative",
            passed=False,
        )

        # Stage 1: text normalisation
        norm_node = self._check_normalisation(job_text)
        root.children.append(norm_node)

        # Stage 2: signal extraction
        fired_names = {s.name for s in fired_signals}
        missing_signals = [s for s in expected_signals if s not in fired_names]
        extraction_node = self._check_signal_extraction(missing_signals, fired_names)
        root.children.append(extraction_node)

        # Stage 3: weight adequacy
        weight_node = self._check_weights(fired_signals)
        root.children.append(weight_node)

        # Stage 4: threshold placement
        threshold_node = self._check_threshold(predicted_score)
        root.children.append(threshold_node)

        # Determine dominant failure mode
        failure_mode = self._infer_failure_mode(
            norm_node, extraction_node, weight_node, threshold_node, missing_signals
        )

        summary = self._build_summary(
            failure_mode, missing_signals, fired_signals, predicted_score
        )

        return DiagnosisTree(
            job_url=job_url,
            root=root,
            failure_mode=failure_mode,
            summary=summary,
        )

    def batch_trace(
        self,
        cases: list[dict],
    ) -> list[DiagnosisTree]:
        """Trace multiple false negatives and return a list of trees.

        Each dict in *cases* must have keys:
          job_url, job_text, fired_signals, expected_signals, predicted_score
        """
        return [
            self.trace(
                job_url=c["job_url"],
                job_text=c.get("job_text", ""),
                fired_signals=c.get("fired_signals", []),
                expected_signals=c.get("expected_signals", []),
                predicted_score=c.get("predicted_score", 0.0),
            )
            for c in cases
        ]

    def failure_mode_counts(self, trees: list[DiagnosisTree]) -> dict[str, int]:
        """Aggregate failure mode counts across a batch of trees."""
        return dict(Counter(t.failure_mode.value for t in trees))

    # ------------------------------------------------------------------
    # Stage checks
    # ------------------------------------------------------------------

    def _check_normalisation(self, job_text: str) -> DiagnosisNode:
        """Check for signs that text normalisation may have introduced noise."""
        issues: list[str] = []

        if not job_text:
            return DiagnosisNode(
                step="text_normalisation",
                finding="No text provided — cannot check normalisation.",
                passed=True,  # can't diagnose without text
            )

        # Check for excessive whitespace (collapsed tokens)
        if re.search(r"\s{5,}", job_text):
            issues.append("Excessive whitespace detected (possible token collapse).")

        # Check for unicode normalisation artifacts
        if re.search(r"[�]", job_text):
            issues.append("Replacement characters found (encoding issues).")

        # Obfuscation markers
        if re.search(u'[\u200b-\u200f\u2060\ufeff]', job_text):
            issues.append("Zero-width characters detected (possible evasion).")

        passed = len(issues) == 0
        finding = "Text normalisation OK." if passed else "; ".join(issues)

        node = DiagnosisNode(step="text_normalisation", finding=finding, passed=passed)
        return node

    def _check_signal_extraction(
        self, missing_signals: list[str], fired_names: set[str]
    ) -> DiagnosisNode:
        """Check whether expected signals were extracted."""
        if not missing_signals:
            node = DiagnosisNode(
                step="signal_extraction",
                finding=f"All expected signals fired. Fired: {sorted(fired_names)}",
                passed=True,
            )
            return node

        node = DiagnosisNode(
            step="signal_extraction",
            finding=f"Missing signals: {missing_signals}. Fired: {sorted(fired_names)}",
            passed=False,
        )
        # Add one child per missing signal as a diagnostic hint
        for sig_name in missing_signals:
            hint = self._extraction_hint(sig_name)
            node.children.append(
                DiagnosisNode(
                    step=f"signal:{sig_name}",
                    finding=hint,
                    passed=False,
                )
            )
        return node

    def _check_weights(self, fired_signals: list) -> DiagnosisNode:
        """Check whether signal weights are sufficient."""
        low_weight_signals = [
            s for s in fired_signals if s.weight < self._ADEQUATE_WEIGHT
        ]
        if not low_weight_signals:
            node = DiagnosisNode(
                step="weight_adequacy",
                finding="All fired signal weights are adequate.",
                passed=True,
            )
            return node

        names = [f"{s.name}={s.weight:.3f}" for s in low_weight_signals]
        node = DiagnosisNode(
            step="weight_adequacy",
            finding=f"Low-weight signals that may have suppressed score: {names}",
            passed=False,
        )
        return node

    def _check_threshold(self, predicted_score: float) -> DiagnosisNode:
        """Check whether the threshold is the issue."""
        gap = self._threshold - predicted_score
        if gap > 0.2:
            finding = (
                f"Score {predicted_score:.3f} is far below threshold {self._threshold:.3f} "
                f"(gap={gap:.3f}). Threshold is not the primary issue."
            )
            passed = True
        elif 0 < gap <= 0.2:
            finding = (
                f"Score {predicted_score:.3f} is close to threshold {self._threshold:.3f} "
                f"(gap={gap:.3f}). A slight threshold reduction would have caught this."
            )
            passed = False
        else:
            finding = f"Score {predicted_score:.3f} already above threshold {self._threshold:.3f}."
            passed = True

        return DiagnosisNode(step="threshold_placement", finding=finding, passed=passed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extraction_hint(signal_name: str) -> str:
        """Return a diagnostic hint for why a signal might not have fired."""
        hints = {
            "upfront_payment": "Check if upfront-payment regex covers obfuscated variants.",
            "guaranteed_income": "Check for Unicode substitutions in 'guaranteed'.",
            "personal_info_request": "Verify SSN regex catches abbreviation variants.",
            "suspicious_email_domain": "Ensure email extraction handles obfuscated '@'.",
            "no_company_linkedin": "Verify company URL is being parsed correctly.",
        }
        for key, hint in hints.items():
            if key in signal_name.lower():
                return hint
        return (
            f"Signal '{signal_name}' did not fire. "
            "Verify the regex/extractor covers this job's text patterns."
        )

    def _infer_failure_mode(
        self,
        norm_node: DiagnosisNode,
        extraction_node: DiagnosisNode,
        weight_node: DiagnosisNode,
        threshold_node: DiagnosisNode,
        missing_signals: list[str],
    ) -> FailureMode:
        """Infer the dominant failure mode from the diagnosis nodes."""
        if not norm_node.passed:
            return FailureMode.EVASION
        if not extraction_node.passed:
            if missing_signals:
                return FailureMode.GAP
            return FailureMode.EVASION
        if not weight_node.passed:
            return FailureMode.THRESHOLD
        if not threshold_node.passed:
            return FailureMode.THRESHOLD
        return FailureMode.NOVEL

    @staticmethod
    def _build_summary(
        failure_mode: FailureMode,
        missing_signals: list[str],
        fired_signals: list,
        predicted_score: float,
    ) -> str:
        mode_descriptions = {
            FailureMode.EVASION: "Scammer likely used text obfuscation to evade signal extraction.",
            FailureMode.GAP: f"No signals cover the scam tactic. Missing: {missing_signals}.",
            FailureMode.THRESHOLD: (
                f"Signals fired ({[s.name for s in fired_signals]}) but combined weight "
                f"produced score {predicted_score:.3f} — below the classification threshold."
            ),
            FailureMode.NOVEL: "Completely new scam type with no recognisable signals.",
        }
        return mode_descriptions.get(failure_mode, "Unknown failure mode.")
