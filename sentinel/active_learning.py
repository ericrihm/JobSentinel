"""Active learning module for intelligent human review selection.

Uses Query-by-Committee (QBC) disagreement and margin confidence to surface
the jobs where human review adds the most information to the learning loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReviewCandidate:
    """A job annotated with its active-learning informativeness scores."""
    url: str
    title: str
    company: str
    scam_score: float
    informativeness: float
    disagreement: float
    margin: float


def _method_scores(job: dict[str, Any]) -> tuple[float, float, float]:
    """Return (bayesian, weighted_mean, max_signal) scores for a job dict.

    Derives three independent estimates from the signals stored in the job
    record so we can compute QBC disagreement without re-hitting the scorer
    when a pre-computed scam_score is already available.

    When signals are absent we fall back to perturbations of scam_score so
    the disagreement is zero (maximally confident prediction) rather than
    artificially inflated.
    """
    base = float(job.get("scam_score") or 0.0)
    signals = job.get("signals") or []

    if not signals:
        return base, base, base

    weights = [float(s.get("weight", 0.5)) for s in signals]
    categories = [s.get("category", "red_flag") for s in signals]

    # Method 1 – bayesian: use stored scam_score (it's the log-odds output)
    bayesian = base

    # Method 2 – weighted_mean: signed sum of weights / max possible
    scam_w = 0.0
    max_w = 0.0
    for w, cat in zip(weights, categories):
        w = max(1e-6, min(1.0, w))
        max_w += w
        if cat != "positive":
            scam_w += w
        else:
            scam_w -= w * 0.5
    weighted_mean = max(0.0, min(1.0, scam_w / max_w)) if max_w > 0 else base

    # Method 3 – max_signal: fraction of signals that are high-weight scam signals
    scam_votes = sum(
        1 for w, cat in zip(weights, categories)
        if cat != "positive" and w >= 0.5
    )
    max_signal = round(scam_votes / len(signals), 4)

    return bayesian, weighted_mean, max_signal


def _disagreement(scores: tuple[float, float, float]) -> float:
    """Standard deviation of the three method scores — higher = more uncertain."""
    mean = sum(scores) / 3.0
    variance = sum((s - mean) ** 2 for s in scores) / 3.0
    return round(math.sqrt(variance), 6)


def _margin(scam_score: float) -> float:
    """Distance from the decision boundary at 0.5.

    Returns 1.0 when the score is exactly 0.5 (maximally informative) and
    0.0 when it is 0.0 or 1.0 (model is certain either way).
    """
    return round(1.0 - abs(scam_score - 0.5) * 2.0, 6)


class ActiveLearner:
    """Selects which jobs will yield the most signal from human review.

    Combines Query-by-Committee disagreement with margin confidence:
    - High disagreement  → the three scoring methods disagree → uncertain case
    - High margin        → score is near 0.5 → model is on the fence
    - Informativeness    → 0.6 * disagreement + 0.4 * margin
    """

    DISAGREEMENT_WEIGHT: float = 0.6
    MARGIN_WEIGHT: float = 0.4

    def rank_for_review(
        self,
        jobs: list[dict[str, Any]],
        top_n: int = 20,
    ) -> list[ReviewCandidate]:
        """Score each job by informativeness and return the top_n.

        Args:
            jobs:  List of job dicts (as returned by SentinelDB queries).
                   Each must have at minimum: url, scam_score.
            top_n: Maximum number of candidates to return.

        Returns:
            List of ReviewCandidate, sorted by informativeness descending.
        """
        candidates: list[ReviewCandidate] = []

        for job in jobs:
            scam_score = float(job.get("scam_score") or 0.0)
            scores = _method_scores(job)
            d = _disagreement(scores)
            m = _margin(scam_score)
            informativeness = round(
                self.DISAGREEMENT_WEIGHT * d + self.MARGIN_WEIGHT * m, 6
            )

            candidates.append(ReviewCandidate(
                url=str(job.get("url") or ""),
                title=str(job.get("title") or ""),
                company=str(job.get("company") or ""),
                scam_score=round(scam_score, 4),
                informativeness=informativeness,
                disagreement=d,
                margin=m,
            ))

        candidates.sort(key=lambda c: c.informativeness, reverse=True)
        return candidates[:top_n]


def select_review_batch(
    db,
    batch_size: int = 20,
) -> list[ReviewCandidate]:
    """Load recent unreviewed jobs from DB and return the most informative ones.

    Pulls up to 4 * batch_size recent unreviewed jobs from the DB and
    ranks them so that the top batch_size most informative are returned.
    The 4x pool ensures we have enough candidates even when most jobs are
    easy (score near 0 or 1) to still surface genuinely uncertain cases.

    Args:
        db:         SentinelDB instance.
        batch_size: How many jobs to return.

    Returns:
        List of ReviewCandidate, sorted by informativeness descending.
    """
    pool_size = batch_size * 4
    rows = db.conn.execute(
        """
        SELECT url, title, company, scam_score, confidence, signals_json
        FROM jobs
        WHERE scam_score IS NOT NULL
          AND (user_reported IS NULL OR user_reported = 0)
        ORDER BY analyzed_at DESC
        LIMIT ?
        """,
        (pool_size,),
    ).fetchall()

    import json

    jobs: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        try:
            d["signals"] = json.loads(d.get("signals_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            d["signals"] = []
        jobs.append(d)

    learner = ActiveLearner()
    return learner.rank_for_review(jobs, top_n=batch_size)
