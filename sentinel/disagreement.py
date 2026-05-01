"""Ensemble Disagreement Learning for Sentinel.

Three components:
- DisagreementDetector  — find cases where scoring subsystems strongly disagree
- ActiveLearningSelector — pool-based active learning: select most informative
                           unlabeled jobs to present for human labeling
- ConsensusBuilder      — weighted voting + stacking across all subsystems
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score bundle — a named dict of subsystem → score
# ---------------------------------------------------------------------------

SubsystemScores = dict[str, float]  # e.g. {"primary": 0.8, "shadow": 0.2, ...}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy(scores: list[float]) -> float:
    """Shannon entropy of a score distribution (binary buckets at 0.5).

    We treat each score as p(scam) and compute the average binary entropy
    H(p) = -p*log2(p) - (1-p)*log2(1-p).  Higher entropy → more uncertainty.
    """
    if not scores:
        return 0.0
    total = 0.0
    for p in scores:
        p = max(1e-9, min(1.0 - 1e-9, p))
        total += -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    return total / len(scores)


def _std_dev(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    return statistics.stdev(scores)


def _mean(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# DisagreementCase — returned by DisagreementDetector
# ---------------------------------------------------------------------------

@dataclass
class DisagreementCase:
    """A single job where subsystems strongly disagree."""
    job_id: str                         # URL or identifier
    subsystem_scores: SubsystemScores   # per-subsystem score
    score_spread: float                 # max - min across subsystems
    entropy: float                      # Shannon entropy of score distribution
    information_value: float            # composite rank metric (higher = more informative)
    disagreeing_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    # Each tuple: (subsystem_a, subsystem_b, |score_a - score_b|)


# ---------------------------------------------------------------------------
# DisagreementDetector
# ---------------------------------------------------------------------------

DISAGREEMENT_THRESHOLD = 0.3   # min score difference to count as "strong" disagreement


class DisagreementDetector:
    """Detect cases where scoring subsystems strongly disagree.

    Subsystems:
        - primary       (log-odds Bayesian scorer)
        - shadow        (candidate weight shadow scorer)
        - fraud_triangle
        - benford
        - linguistic    (linguistic forensics / LLM detector)

    Any subset of subsystems may be provided; at least two are required to
    compute disagreement.
    """

    def __init__(self, threshold: float = DISAGREEMENT_THRESHOLD) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def detect(
        self,
        job_id: str,
        subsystem_scores: SubsystemScores,
    ) -> DisagreementCase | None:
        """Return a DisagreementCase if subsystems disagree above threshold.

        Returns None if fewer than 2 subsystems provided or disagreement is
        below the configured threshold.
        """
        scores = list(subsystem_scores.values())
        names = list(subsystem_scores.keys())

        if len(scores) < 2:
            return None

        spread = max(scores) - min(scores)
        if spread <= self.threshold:
            return None

        pairs = self._disagreeing_pairs(names, subsystem_scores)
        ent = _entropy(scores)
        info_value = self._information_value(spread, ent, len(scores))

        return DisagreementCase(
            job_id=job_id,
            subsystem_scores=dict(subsystem_scores),
            score_spread=round(spread, 4),
            entropy=round(ent, 4),
            information_value=round(info_value, 4),
            disagreeing_pairs=pairs,
        )

    def detect_batch(
        self,
        jobs: list[tuple[str, SubsystemScores]],
    ) -> list[DisagreementCase]:
        """Detect disagreement across a batch of jobs.

        Args:
            jobs: list of (job_id, subsystem_scores) tuples.

        Returns cases sorted by information_value descending (most
        informative first).
        """
        cases: list[DisagreementCase] = []
        for job_id, scores in jobs:
            case = self.detect(job_id, scores)
            if case is not None:
                cases.append(case)
        cases.sort(key=lambda c: c.information_value, reverse=True)
        return cases

    def rank_by_information_value(
        self, cases: list[DisagreementCase]
    ) -> list[DisagreementCase]:
        """Re-rank an existing list of DisagreementCases."""
        return sorted(cases, key=lambda c: c.information_value, reverse=True)

    def summarise(self, cases: list[DisagreementCase]) -> dict[str, Any]:
        """Aggregate statistics across a list of disagreement cases."""
        if not cases:
            return {"count": 0, "mean_spread": 0.0, "mean_entropy": 0.0}
        spreads = [c.score_spread for c in cases]
        entropies = [c.entropy for c in cases]
        return {
            "count": len(cases),
            "mean_spread": round(_mean(spreads), 4),
            "max_spread": round(max(spreads), 4),
            "mean_entropy": round(_mean(entropies), 4),
            "top_job_id": cases[0].job_id if cases else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _disagreeing_pairs(
        self,
        names: list[str],
        scores: SubsystemScores,
    ) -> list[tuple[str, str, float]]:
        """All pairs where |score_a - score_b| >= threshold."""
        pairs = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                diff = abs(scores[a] - scores[b])
                if diff >= self.threshold:
                    pairs.append((a, b, round(diff, 4)))
        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs

    def _information_value(
        self, spread: float, entropy: float, n_subsystems: int
    ) -> float:
        """Composite information value: higher = more worth investigating.

        Combines:
        - spread (max - min): raw disagreement magnitude
        - entropy: uncertainty of the score distribution
        - n_subsystems: more systems = more evidence
        """
        # Weight: spread matters most, entropy second, slight bonus for coverage
        subsystem_bonus = math.log1p(n_subsystems) / math.log1p(10)
        return spread * 0.6 + entropy * 0.3 + subsystem_bonus * 0.1


# ---------------------------------------------------------------------------
# ActiveLearningSelector
# ---------------------------------------------------------------------------

@dataclass
class LabelingCandidate:
    """A job selected by the active learner to be presented for labeling."""
    job_id: str
    score: float             # ensemble / best-guess score
    uncertainty: float       # how close to decision boundary (0 = certain, 1 = max uncertainty)
    committee_disagreement: float  # std-dev of subsystem scores
    expected_model_change: float   # estimated weight update magnitude if labeled
    selection_reason: str = ""     # "uncertainty" | "committee" | "model_change"
    subsystem_scores: SubsystemScores = field(default_factory=dict)


class ActiveLearningSelector:
    """Pool-based active learning: select the most informative unlabeled jobs.

    Three sampling strategies are combined (configurable weights):
    1. Uncertainty sampling  — jobs closest to the decision boundary (score ≈ 0.5)
    2. Query-by-committee   — jobs where sub-scorers disagree most (high std-dev)
    3. Expected model change — estimated weight update magnitude if labeled

    The labeling budget tracks how many queries have been made in the current cycle.
    """

    BOUNDARY = 0.5  # decision boundary for uncertainty sampling

    def __init__(
        self,
        budget_per_cycle: int = 10,
        uncertainty_weight: float = 0.4,
        committee_weight: float = 0.4,
        model_change_weight: float = 0.2,
    ) -> None:
        if abs(uncertainty_weight + committee_weight + model_change_weight - 1.0) > 1e-6:
            raise ValueError("Strategy weights must sum to 1.0")
        self.budget_per_cycle = budget_per_cycle
        self._uncertainty_w = uncertainty_weight
        self._committee_w = committee_weight
        self._model_change_w = model_change_weight

        self._queries_this_cycle: int = 0
        self._labeled_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def queries_remaining(self) -> int:
        return max(0, self.budget_per_cycle - self._queries_this_cycle)

    @property
    def budget_exhausted(self) -> bool:
        return self._queries_this_cycle >= self.budget_per_cycle

    def reset_cycle(self) -> None:
        """Begin a new labeling cycle (resets query counter)."""
        self._queries_this_cycle = 0

    def mark_labeled(self, job_id: str) -> None:
        """Record that a job has been labeled (won't be selected again)."""
        self._labeled_ids.add(job_id)
        self._queries_this_cycle += 1

    def select(
        self,
        pool: list[tuple[str, SubsystemScores]],
        n: int | None = None,
    ) -> list[LabelingCandidate]:
        """Select the most informative jobs to label from *pool*.

        Args:
            pool: list of (job_id, subsystem_scores) for all unlabeled jobs.
            n: number of candidates to return. Defaults to ``queries_remaining``.

        Returns candidates sorted by composite informativeness (highest first).
        """
        if n is None:
            n = self.queries_remaining
        if n <= 0:
            return []

        candidates = []
        for job_id, sub_scores in pool:
            if job_id in self._labeled_ids:
                continue
            candidate = self._score_candidate(job_id, sub_scores)
            candidates.append(candidate)

        candidates.sort(key=lambda c: self._composite_score(c), reverse=True)
        return candidates[:n]

    def uncertainty_sample(
        self, pool: list[tuple[str, float]], n: int = 5
    ) -> list[tuple[str, float]]:
        """Return jobs closest to decision boundary (score ≈ 0.5).

        Args:
            pool: list of (job_id, ensemble_score).
            n: number of candidates to return.
        """
        filtered = [(jid, s) for jid, s in pool if jid not in self._labeled_ids]
        filtered.sort(key=lambda t: abs(t[1] - self.BOUNDARY))
        return filtered[:n]

    def committee_sample(
        self, pool: list[tuple[str, SubsystemScores]], n: int = 5
    ) -> list[tuple[str, float]]:
        """Return jobs with highest inter-subsystem disagreement (std-dev).

        Returns list of (job_id, committee_disagreement).
        """
        scored = []
        for job_id, sub_scores in pool:
            if job_id in self._labeled_ids:
                continue
            scores_vals = list(sub_scores.values())
            if len(scores_vals) < 2:
                continue
            scored.append((job_id, _std_dev(scores_vals)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:n]

    def expected_model_change(
        self, job_id: str, score: float, n_signals: int = 5
    ) -> float:
        """Estimate how much model weights would change if *job_id* were labeled.

        Formula: change ≈ 2 * uncertainty * sqrt(n_signals)
        (Higher uncertainty + more signals = bigger potential gradient step.)
        """
        uncertainty = 1.0 - 2.0 * abs(score - self.BOUNDARY)
        uncertainty = max(0.0, uncertainty)
        return round(uncertainty * math.sqrt(max(1, n_signals)), 4)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_candidate(
        self, job_id: str, sub_scores: SubsystemScores
    ) -> LabelingCandidate:
        scores_vals = list(sub_scores.values())
        ensemble_score = _mean(scores_vals) if scores_vals else 0.5

        uncertainty = max(0.0, 1.0 - 2.0 * abs(ensemble_score - self.BOUNDARY))
        committee_disagreement = _std_dev(scores_vals) if len(scores_vals) >= 2 else 0.0
        emc = self.expected_model_change(job_id, ensemble_score, len(scores_vals))

        (
            self._uncertainty_w * uncertainty
            + self._committee_w * committee_disagreement
            + self._model_change_w * emc
        )
        # Determine primary reason
        reasons = {
            "uncertainty": self._uncertainty_w * uncertainty,
            "committee": self._committee_w * committee_disagreement,
            "model_change": self._model_change_w * emc,
        }
        reason = max(reasons, key=lambda k: reasons[k])

        return LabelingCandidate(
            job_id=job_id,
            score=round(ensemble_score, 4),
            uncertainty=round(uncertainty, 4),
            committee_disagreement=round(committee_disagreement, 4),
            expected_model_change=round(emc, 4),
            selection_reason=reason,
            subsystem_scores=dict(sub_scores),
        )

    def _composite_score(self, c: LabelingCandidate) -> float:
        return (
            self._uncertainty_w * c.uncertainty
            + self._committee_w * c.committee_disagreement
            + self._model_change_w * c.expected_model_change
        )


# ---------------------------------------------------------------------------
# ConsensusBuilder
# ---------------------------------------------------------------------------

@dataclass
class ConsensusResult:
    """Output of ConsensusBuilder.build_consensus()."""
    consensus_score: float              # final weighted score
    confidence: float                   # 0–1
    agreement_ratio: float              # fraction of subsystems on the same side
    subsystem_scores: SubsystemScores
    meta_weights: dict[str, float]      # learned weights per subsystem
    breakdown: str                      # human-readable, e.g. "3/5 say scam"
    stacking_score: float | None = None  # logistic regression stacking score


class ConsensusBuilder:
    """Weighted voting across scoring subsystems with learned meta-weights.

    Meta-weights are learned from historical accuracy: subsystems that have
    been more accurate receive higher weight.  A simple logistic regression
    stacking layer combines subsystem scores into a final prediction.

    Initialise with equal weights; call ``update_accuracy`` after each labeled
    example to update the Bayesian posteriors.
    """

    # Logistic regression coefficient for stacking: each subsystem contributes
    # proportionally to its weight.  We use a simple dot-product + sigmoid
    # rather than fitting a full LR model (stdlib only).
    STACKING_BIAS: float = 0.0          # log-odds bias (0 = balanced prior)

    def __init__(self, subsystem_names: list[str] | None = None) -> None:
        # Default subsystems mirror the fraud_handbook / shadow pipeline
        if subsystem_names is None:
            subsystem_names = [
                "primary",
                "shadow",
                "fraud_triangle",
                "benford",
                "linguistic",
            ]
        # Beta posteriors: equal priors → equal starting weights
        self._posteriors: dict[str, list[float]] = {
            name: [1.0, 1.0] for name in subsystem_names
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_consensus(
        self,
        subsystem_scores: SubsystemScores,
        *,
        use_stacking: bool = True,
    ) -> ConsensusResult:
        """Combine subsystem scores into a consensus prediction.

        Args:
            subsystem_scores: {subsystem_name: score_0_to_1}.
            use_stacking: whether to also compute a stacking score.

        Returns ConsensusResult with weighted score, confidence, and breakdown.
        """
        weights = self.get_meta_weights()
        # Only use subsystems that are present in both scores and weights
        active = {k: v for k, v in subsystem_scores.items() if k in weights}

        if not active:
            # Fallback: simple mean
            scores_vals = list(subsystem_scores.values())
            mean = _mean(scores_vals) if scores_vals else 0.0
            return ConsensusResult(
                consensus_score=round(mean, 4),
                confidence=0.0,
                agreement_ratio=0.0,
                subsystem_scores=subsystem_scores,
                meta_weights={},
                breakdown="no known subsystems",
            )

        # Weighted sum
        weight_total = sum(weights[k] for k in active)
        if weight_total == 0.0:
            consensus_score = _mean(list(active.values()))
        else:
            consensus_score = sum(
                weights[k] * v for k, v in active.items()
            ) / weight_total

        consensus_score = round(max(0.0, min(1.0, consensus_score)), 4)

        # Agreement: how many subsystems agree with the consensus side
        threshold = 0.5
        n_scam = sum(1 for v in active.values() if v >= threshold)
        n_legit = sum(1 for v in active.values() if v < threshold)
        total = len(active)
        majority = max(n_scam, n_legit)
        agreement_ratio = round(majority / total, 4) if total > 0 else 0.0

        # Confidence: grows with agreement + number of subsystems
        base_conf = 1.0 - math.exp(-0.5 * total)
        confidence = round(base_conf * agreement_ratio, 4)

        # Breakdown string
        side = "scam" if consensus_score >= threshold else "legit"
        breakdown = (
            f"{majority}/{total} subsystems say {side}; "
            f"consensus={consensus_score:.2f}"
        )

        # Stacking: logistic regression over weighted subsystem scores
        stacking_score = None
        if use_stacking:
            stacking_score = self._stacking_score(active, weights)

        return ConsensusResult(
            consensus_score=consensus_score,
            confidence=confidence,
            agreement_ratio=agreement_ratio,
            subsystem_scores=dict(subsystem_scores),
            meta_weights={k: round(weights[k], 4) for k in active},
            breakdown=breakdown,
            stacking_score=stacking_score,
        )

    def update_accuracy(self, subsystem_name: str, was_correct: bool) -> None:
        """Update the Beta posterior for *subsystem_name*.

        Call this after a labeled example to teach the meta-learner which
        subsystems are more accurate.
        """
        if subsystem_name not in self._posteriors:
            self._posteriors[subsystem_name] = [1.0, 1.0]
        alpha, beta = self._posteriors[subsystem_name]
        if was_correct:
            alpha += 1.0
        else:
            beta += 1.0
        self._posteriors[subsystem_name] = [alpha, beta]

    def get_meta_weights(self) -> dict[str, float]:
        """Return softmax-normalised meta-weights from Beta posteriors."""
        raw = {
            name: ab[0] / (ab[0] + ab[1])
            for name, ab in self._posteriors.items()
        }
        total = sum(raw.values())
        if total == 0.0:
            n = len(raw)
            return {name: 1.0 / n for name in raw}
        return {name: v / total for name, v in raw.items()}

    def get_accuracy_summary(self) -> dict[str, dict[str, float]]:
        """Return per-subsystem accuracy mean and observation count."""
        result = {}
        for name, (alpha, beta) in self._posteriors.items():
            obs = alpha + beta - 2.0   # subtract prior
            result[name] = {
                "accuracy": round(alpha / (alpha + beta), 4),
                "observations": max(0.0, obs),
            }
        return result

    def register_subsystem(self, name: str) -> None:
        """Register a new subsystem with a uniform prior."""
        if name not in self._posteriors:
            self._posteriors[name] = [1.0, 1.0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stacking_score(
        self,
        active_scores: SubsystemScores,
        weights: dict[str, float],
    ) -> float:
        """Simple stacking: weighted dot-product through sigmoid.

        Each subsystem's score is mapped to log-odds, weighted by meta-weight,
        summed, then transformed back via sigmoid.
        """
        log_odds = self.STACKING_BIAS
        for name, score in active_scores.items():
            w = weights.get(name, 0.0)
            p = max(1e-6, min(1.0 - 1e-6, score))
            # Subsystem log-odds, scaled by its meta-weight
            log_odds += w * math.log(p / (1.0 - p))
        stacking_score = 1.0 / (1.0 + math.exp(-log_odds))
        return round(stacking_score, 4)
