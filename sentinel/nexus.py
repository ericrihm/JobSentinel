"""Nexus — Unified Intelligence Orchestrator for Sentinel.

The Nexus is the single entry point that coordinates ALL detection subsystems
for maximum scam-detection power.  It degrades gracefully: any subsystem that
fails to import or raises at runtime is silently skipped and its slot in the
final score is omitted.

Public surface
--------------
NexusReport     — unified result dataclass
Nexus           — deep_analyze(job) -> NexusReport
NexusLearner    — learning loop that updates every subsystem on feedback
NexusDashboard  — system-wide health and accuracy tracking
NexusEvolver    — one full autonomous improvement cycle via evolve()
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional

from sentinel.models import JobPosting, ScamSignal, SignalCategory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional subsystem imports — all wrapped in try/except
# ---------------------------------------------------------------------------

try:
    from sentinel.signals import extract_signals
    _HAS_SIGNALS = True
except Exception:
    _HAS_SIGNALS = False
    logger.debug("nexus: signals module unavailable")

try:
    from sentinel.scorer import score_signals, classify_risk, build_result
    _HAS_SCORER = True
except Exception:
    _HAS_SCORER = False
    logger.debug("nexus: scorer module unavailable")

try:
    from sentinel.fraud_handbook import (
        FraudTriangleScorer,
        BenfordAnalyzer,
        LinguisticForensics,
        extract_fraud_handbook_signals,
    )
    _HAS_FRAUD = True
except Exception:
    _HAS_FRAUD = False
    logger.debug("nexus: fraud_handbook module unavailable")

try:
    from sentinel.llm_detect import LLMDetector
    _HAS_LLM = True
except Exception:
    _HAS_LLM = False
    logger.debug("nexus: llm_detect module unavailable")

try:
    from sentinel.stylometry import StyleExtractor, OperatorLinker
    _HAS_STYLO = True
except Exception:
    _HAS_STYLO = False
    logger.debug("nexus: stylometry module unavailable")

try:
    from sentinel.economics import validate_economics
    _HAS_ECON = True
except Exception:
    _HAS_ECON = False
    logger.debug("nexus: economics module unavailable")

try:
    from sentinel.graph import TextSimilarityIndex
    _HAS_GRAPH = True
except Exception:
    _HAS_GRAPH = False
    logger.debug("nexus: graph module unavailable")

try:
    from sentinel.robustness import RobustnessScorer
    _HAS_ROBUST = True
except Exception:
    _HAS_ROBUST = False
    logger.debug("nexus: robustness module unavailable")

try:
    from sentinel.adversarial import EvasionDetector, TextNormalizer
    _HAS_ADVERSARIAL = True
except Exception:
    _HAS_ADVERSARIAL = False
    logger.debug("nexus: adversarial module unavailable")

try:
    from sentinel.disagreement import DisagreementDetector, ConsensusBuilder
    _HAS_DISAGREE = True
except Exception:
    _HAS_DISAGREE = False
    logger.debug("nexus: disagreement module unavailable")

try:
    from sentinel.counterfactual import CounterfactualEngine
    _HAS_COUNTER = True
except Exception:
    _HAS_COUNTER = False
    logger.debug("nexus: counterfactual module unavailable")

try:
    from sentinel.research import ResearchEngine
    _HAS_RESEARCH = True
except Exception:
    _HAS_RESEARCH = False
    logger.debug("nexus: research module unavailable")

try:
    from sentinel.innovation import InnovationEngine
    _HAS_INNOVATION = True
except Exception:
    _HAS_INNOVATION = False
    logger.debug("nexus: innovation module unavailable")

try:
    from sentinel.temporal import TemporalTracker
    _HAS_TEMPORAL = True
except Exception:
    _HAS_TEMPORAL = False
    logger.debug("nexus: temporal module unavailable")

try:
    from sentinel.db import SentinelDB
    _HAS_DB = True
except Exception:
    _HAS_DB = False
    logger.debug("nexus: db module unavailable")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _full_text(job: JobPosting) -> str:
    parts = [job.title, job.company, job.description, job.location]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Risk level constants — aligns with NexusReport.risk_level string
# ---------------------------------------------------------------------------

_RISK_THRESHOLDS = [
    (0.20, "SAFE"),
    (0.40, "CAUTION"),
    (0.60, "WARNING"),
    (0.80, "DANGER"),
    (1.01, "CRITICAL"),
]


def _score_to_risk(score: float) -> str:
    for threshold, label in _RISK_THRESHOLDS:
        if score < threshold:
            return label
    return "CRITICAL"


# ---------------------------------------------------------------------------
# NexusReport
# ---------------------------------------------------------------------------

@dataclass
class NexusReport:
    """Unified analysis report from all Nexus subsystems."""

    # Core result
    overall_score: float = 0.0
    confidence: float = 0.0
    risk_level: str = "SAFE"

    # Per-subsystem contributions
    subsystem_scores: dict[str, float] = field(default_factory=dict)

    # Signal intelligence
    signals_fired: list[ScamSignal] = field(default_factory=list)

    # Human-readable outputs
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)

    # Cross-posting
    similar_postings: list[str] = field(default_factory=list)

    # Stylometry
    operator_fingerprint: Optional[str] = None

    # Economic flags
    economic_flags: list[str] = field(default_factory=list)

    # Adversarial / evasion
    evasion_detected: bool = False

    # LLM generation probability
    llm_generated_probability: float = 0.0

    # Counterfactual
    counterfactual_insights: list[str] = field(default_factory=list)

    # Meta
    subsystems_run: list[str] = field(default_factory=list)
    analysis_time_ms: float = 0.0
    analyzed_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "subsystem_scores": {k: round(v, 4) for k, v in self.subsystem_scores.items()},
            "signals_fired_count": len(self.signals_fired),
            "key_findings": self.key_findings,
            "recommendations": self.recommendations,
            "disagreements": self.disagreements,
            "similar_postings": self.similar_postings,
            "operator_fingerprint": self.operator_fingerprint,
            "economic_flags": self.economic_flags,
            "evasion_detected": self.evasion_detected,
            "llm_generated_probability": round(self.llm_generated_probability, 4),
            "counterfactual_insights": self.counterfactual_insights,
            "subsystems_run": self.subsystems_run,
            "analysis_time_ms": round(self.analysis_time_ms, 1),
            "analyzed_at": self.analyzed_at,
        }


# ---------------------------------------------------------------------------
# Default subsystem meta-weights (how much each contributes to overall score)
# These are updated by NexusLearner based on per-subsystem accuracy.
# ---------------------------------------------------------------------------

_DEFAULT_META_WEIGHTS: dict[str, float] = {
    "signals":        0.30,
    "fraud_triangle": 0.12,
    "benford":        0.04,
    "linguistic":     0.08,
    "llm_detect":     0.10,
    "stylometry":     0.06,
    "economics":      0.10,
    "graph":          0.10,
    "robustness":     0.05,
    "adversarial":    0.05,
}


# ---------------------------------------------------------------------------
# Nexus — the main orchestrator
# ---------------------------------------------------------------------------

class Nexus:
    """Single entry point that coordinates ALL analysis subsystems.

    Each subsystem is optional (try/except) so the system degrades
    gracefully if a module is missing or raises.

    Parameters
    ----------
    db:
        Optional SentinelDB instance. If None, a fresh DB is created when
        needed (and only if the DB module is available).
    meta_weights:
        Per-subsystem contribution weights to the overall score.
        Defaults to _DEFAULT_META_WEIGHTS and can be updated by NexusLearner.
    similarity_index:
        Optional TextSimilarityIndex to check for near-duplicate postings.
        Shared across calls so the corpus grows over time.
    operator_linker:
        Optional OperatorLinker for stylometric fingerprinting.
    """

    def __init__(
        self,
        db: Any = None,
        meta_weights: dict[str, float] | None = None,
        similarity_index: Any = None,
        operator_linker: Any = None,
    ) -> None:
        self._db = db
        self._meta_weights: dict[str, float] = dict(
            meta_weights if meta_weights is not None else _DEFAULT_META_WEIGHTS
        )
        self._similarity_index = similarity_index  # TextSimilarityIndex
        self._operator_linker = operator_linker    # OperatorLinker
        self._analysis_count = 0

    # ------------------------------------------------------------------
    # Main analysis entry point
    # ------------------------------------------------------------------

    def deep_analyze(self, job: JobPosting) -> NexusReport:
        """Run ALL analysis subsystems and produce a unified NexusReport.

        Subsystems run (each is optional, graceful on failure):
          a. Signal extraction          (signals.py)
          b. Bayesian scoring           (scorer.py)
          c. Fraud Triangle scoring     (fraud_handbook.py)
          d. Benford's Law on salary    (fraud_handbook.py)
          e. Linguistic forensics       (fraud_handbook.py)
          f. LLM content detection      (llm_detect.py)
          g. Stylometric fingerprinting (stylometry.py)
          h. Economic validation        (economics.py)
          i. Text similarity/near-dup   (graph.py)
          j. Robustness testing         (robustness.py)
          k. Adversarial evasion        (adversarial.py)
          l. Disagreement detection     (disagreement.py)
        """
        t0 = time.monotonic()
        self._analysis_count += 1

        report = NexusReport()
        subsystem_scores: dict[str, float] = {}
        all_signals: list[ScamSignal] = []
        subsystems_run: list[str] = []

        text = _full_text(job)

        # ── a. Signal extraction ────────────────────────────────────────
        signals: list[ScamSignal] = []
        if _HAS_SIGNALS:
            try:
                signals = extract_signals(job)
                all_signals.extend(signals)
                subsystems_run.append("signals")
            except Exception as exc:
                logger.debug("nexus: signals failed: %s", exc)

        # ── b. Bayesian scoring ─────────────────────────────────────────
        if _HAS_SCORER and signals:
            try:
                # score_signals returns (scam_score, confidence) tuple
                bay_result = score_signals(signals)
                bay_score = bay_result[0] if isinstance(bay_result, tuple) else float(bay_result)
                subsystem_scores["signals"] = _clamp(bay_score)
            except Exception as exc:
                logger.debug("nexus: scorer failed: %s", exc)

        # ── c. Fraud Triangle ───────────────────────────────────────────
        if _HAS_FRAUD:
            try:
                ft = FraudTriangleScorer()
                ft_result = ft.score(job)
                ft_score = _clamp(ft_result.get("composite_score", 0.0))
                subsystem_scores["fraud_triangle"] = ft_score
                subsystems_run.append("fraud_triangle")
                ft_sig = ft.to_signal(job)
                if ft_sig:
                    all_signals.append(ft_sig)
            except Exception as exc:
                logger.debug("nexus: fraud_triangle failed: %s", exc)

        # ── d. Benford's Law ────────────────────────────────────────────
        if _HAS_FRAUD:
            try:
                ba = BenfordAnalyzer()
                salary_vals: list[float] = []
                if job.salary_min > 0:
                    salary_vals.append(job.salary_min)
                if job.salary_max > 0:
                    salary_vals.append(job.salary_max)
                if salary_vals:
                    bf_result = ba.analyze(salary_vals)
                    bf_score = _clamp(bf_result.get("anomaly_score", 0.0))
                    subsystem_scores["benford"] = bf_score
                    if "benford" not in subsystems_run:
                        subsystems_run.append("benford")
                    if bf_result.get("is_anomalous"):
                        report.key_findings.append(
                            f"Benford's Law anomaly on salary figures (score={bf_score:.2f})"
                        )
            except Exception as exc:
                logger.debug("nexus: benford failed: %s", exc)

        # ── e. Linguistic forensics ─────────────────────────────────────
        if _HAS_FRAUD and text:
            try:
                lf = LinguisticForensics()
                lf_result = lf.analyze(text)
                lf_score = _clamp(lf_result.get("deception_score", 0.0))
                subsystem_scores["linguistic"] = lf_score
                if "linguistic" not in subsystems_run:
                    subsystems_run.append("linguistic")
                lf_sig = lf.to_signal(job)
                if lf_sig:
                    all_signals.append(lf_sig)
            except Exception as exc:
                logger.debug("nexus: linguistic forensics failed: %s", exc)

        # ── f. LLM-generated content detection ──────────────────────────
        if _HAS_LLM and text:
            try:
                llm_det = LLMDetector()
                llm_result = llm_det.detect(text)
                llm_prob = _clamp(llm_result.probability if hasattr(llm_result, "probability") else 0.0)
                report.llm_generated_probability = llm_prob
                subsystem_scores["llm_detect"] = llm_prob
                subsystems_run.append("llm_detect")
                if llm_prob > 0.7:
                    report.key_findings.append(
                        f"High probability of LLM-generated content ({llm_prob:.0%})"
                    )
            except Exception as exc:
                logger.debug("nexus: llm_detect failed: %s", exc)

        # ── g. Stylometric fingerprinting ───────────────────────────────
        if _HAS_STYLO and text:
            try:
                extractor = StyleExtractor()
                fp = extractor.extract(text)

                if self._operator_linker is None:
                    self._operator_linker = OperatorLinker()

                posting_id = job.url or f"job_{hash(text) & 0xFFFFFF}"
                link_result = self._operator_linker.link(posting_id, fp)
                if link_result.is_match and link_result.operator_id:
                    report.operator_fingerprint = link_result.operator_id
                    stylo_score = _clamp(link_result.confidence)
                    subsystem_scores["stylometry"] = stylo_score
                    report.key_findings.append(
                        f"Stylometric match to known operator '{link_result.operator_id}' "
                        f"(confidence={link_result.confidence:.0%})"
                    )
                else:
                    subsystem_scores["stylometry"] = 0.0
                    # Register for future matching
                    self._operator_linker.add_fingerprint(posting_id, fp)

                subsystems_run.append("stylometry")
            except Exception as exc:
                logger.debug("nexus: stylometry failed: %s", exc)

        # ── h. Economic validation ──────────────────────────────────────
        if _HAS_ECON:
            try:
                econ = validate_economics(job)
                econ_signals = econ.all_signals
                all_signals.extend(econ_signals)

                econ_score = 0.0
                if econ_signals:
                    scam_sigs = [s for s in econ_signals if s.category != SignalCategory.POSITIVE]
                    econ_score = _clamp(len(scam_sigs) / max(len(econ_signals), 1))

                subsystem_scores["economics"] = econ_score
                subsystems_run.append("economics")

                for sig in econ_signals:
                    if sig.detail:
                        report.economic_flags.append(sig.detail)
            except Exception as exc:
                logger.debug("nexus: economics failed: %s", exc)

        # ── i. Text similarity / near-duplicate check ───────────────────
        if _HAS_GRAPH and text:
            try:
                if self._similarity_index is None:
                    self._similarity_index = TextSimilarityIndex()

                posting_id = job.url or f"job_{hash(text) & 0xFFFFFF}"
                sim_score, similar_ids = self._run_similarity_check(
                    posting_id, text
                )
                subsystem_scores["graph"] = sim_score
                report.similar_postings = similar_ids
                subsystems_run.append("graph")

                if similar_ids:
                    report.key_findings.append(
                        f"Found {len(similar_ids)} near-duplicate posting(s)"
                    )
            except Exception as exc:
                logger.debug("nexus: graph failed: %s", exc)

        # ── j. Robustness testing ───────────────────────────────────────
        if _HAS_ROBUST and _HAS_SCORER and all_signals and text:
            try:
                def _score_fn(t: str) -> float:
                    # Re-extract signals on perturbed text
                    perturbed = JobPosting(description=t, title=job.title, company=job.company)
                    if _HAS_SIGNALS:
                        sigs = extract_signals(perturbed)
                    else:
                        sigs = []
                    return score_signals(sigs) if sigs else 0.0

                rs = RobustnessScorer(scoring_fn=_score_fn, n_perturbations=10)
                rb_report = rs.score(text)
                subsystem_scores["robustness"] = rb_report.fragility_score
                subsystems_run.append("robustness")

                if rb_report.is_fragile:
                    report.key_findings.append(
                        f"Classification is FRAGILE (fragility={rb_report.fragility_score:.2f}); "
                        f"perturbations shift score {rb_report.min_score:.2f}–{rb_report.max_score:.2f}"
                    )
                    report.recommendations.extend(rb_report.suggested_improvements or [])
            except Exception as exc:
                logger.debug("nexus: robustness failed: %s", exc)

        # ── k. Adversarial evasion detection ────────────────────────────
        if _HAS_ADVERSARIAL and text:
            try:
                evasion_det = EvasionDetector()
                normalizer = TextNormalizer()
                normalized = normalizer.normalize(text)
                evasion_results = evasion_det.detect_evasion_attempts(text, normalized)
                evasion_detected = len(evasion_results) > 0
                report.evasion_detected = evasion_detected
                evasion_score = _clamp(len(evasion_results) * 0.25)
                subsystem_scores["adversarial"] = evasion_score
                subsystems_run.append("adversarial")

                if evasion_detected:
                    evasion_names = [r.get("type", "obfuscation") for r in evasion_results[:3]]
                    report.key_findings.append(
                        f"Adversarial evasion detected: {', '.join(evasion_names)}"
                    )
            except Exception as exc:
                logger.debug("nexus: adversarial failed: %s", exc)

        # ── l. Disagreement detection ───────────────────────────────────
        if _HAS_DISAGREE and len(subsystem_scores) >= 2:
            try:
                dd = DisagreementDetector()
                job_id = job.url or f"job_{hash(text) & 0xFFFFFF}"
                dis_case = dd.detect(job_id, subsystem_scores)
                if dis_case:
                    for pair in dis_case.disagreeing_pairs:
                        sa, sb, delta = pair
                        report.disagreements.append(
                            f"{sa} vs {sb}: delta={delta:.2f}"
                        )
            except Exception as exc:
                logger.debug("nexus: disagreement failed: %s", exc)

        # ── Compute overall score (weighted consensus) ──────────────────
        overall, confidence = self._compute_overall(subsystem_scores)
        report.overall_score = overall
        report.confidence = confidence
        report.risk_level = _score_to_risk(overall)
        report.subsystem_scores = subsystem_scores
        report.signals_fired = all_signals
        report.subsystems_run = subsystems_run

        # ── Counterfactual insights ─────────────────────────────────────
        if _HAS_COUNTER and all_signals and overall < 0.5:
            try:
                ce = CounterfactualEngine()
                known_names = [s.name for s in all_signals]
                # Generate a short list of candidate near-miss signals
                candidate_names = self._candidate_signal_names(known_names)
                cf_results = ce.rank_counterfactuals(
                    all_signals, candidate_names
                )
                for cr in cf_results[:3]:
                    delta = getattr(cr, "score_delta", 0.0)
                    name = getattr(cr, "signal_name", "")
                    if name and abs(delta) >= 0.05:
                        report.counterfactual_insights.append(
                            f"Signal '{name}' almost fired (would add {delta:+.2f})"
                        )
            except Exception as exc:
                logger.debug("nexus: counterfactual failed: %s", exc)

        # ── Build recommendations ───────────────────────────────────────
        self._build_recommendations(report)

        # ── Timing ─────────────────────────────────────────────────────
        report.analysis_time_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "Nexus analysis complete: score=%.3f, risk=%s, subsystems=%d, signals=%d, time=%.1fms",
            overall, report.risk_level, len(subsystems_run),
            len(all_signals), report.analysis_time_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_similarity_check(
        self, posting_id: str, text: str
    ) -> tuple[float, list[str]]:
        """Add posting to index and return (sim_score, similar_ids)."""
        idx = self._similarity_index
        idx.add(posting_id, text)
        near_dups = idx.find_near_duplicates(posting_id, threshold=0.7)
        similar_ids = [nd for nd in near_dups if nd != posting_id]
        score = _clamp(min(len(similar_ids) * 0.25, 1.0))
        return score, similar_ids

    def _compute_overall(
        self, subsystem_scores: dict[str, float]
    ) -> tuple[float, float]:
        """Compute weighted consensus overall score and confidence.

        Returns (overall_score, confidence) both in [0, 1].
        """
        if not subsystem_scores:
            return 0.0, 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for name, score in subsystem_scores.items():
            w = self._meta_weights.get(name, 0.05)
            weighted_sum += w * score
            total_weight += w

        if total_weight == 0:
            return 0.0, 0.0

        overall = _clamp(weighted_sum / total_weight)

        # Confidence: proportion of available subsystems that ran
        available_keys = set(_DEFAULT_META_WEIGHTS.keys())
        ran_keys = set(subsystem_scores.keys())
        coverage = len(ran_keys & available_keys) / max(len(available_keys), 1)

        # Also factor in agreement: lower std dev -> higher confidence
        scores_list = list(subsystem_scores.values())
        if len(scores_list) >= 2:
            std = statistics.stdev(scores_list)
            agreement = _clamp(1.0 - std * 2)
        else:
            agreement = 0.5

        confidence = _clamp(coverage * 0.6 + agreement * 0.4)
        return overall, confidence

    def _candidate_signal_names(self, fired_names: list[str]) -> list[str]:
        """Return a short list of well-known signals that did NOT fire."""
        all_known = [
            "upfront_payment", "personal_info_request", "guaranteed_income",
            "crypto_payment", "suspicious_email", "urgency_language",
            "no_qualifications_required", "vague_description", "grammar_issues",
            "suspicious_links", "interview_bypass", "mlm_language",
            "reshipping_scam", "data_harvesting", "salary_anomaly",
            "ghost_job_indicators", "repost_pattern", "new_recruiter_account",
            "high_posting_velocity", "llm_generated_content",
        ]
        fired_set = set(fired_names)
        return [n for n in all_known if n not in fired_set][:10]

    def _build_recommendations(self, report: NexusReport) -> None:
        """Append actionable recommendations based on report state."""
        score = report.overall_score
        if score >= 0.8:
            report.recommendations.append(
                "DO NOT apply — strong evidence this is a scam."
            )
            report.recommendations.append(
                "Report the posting to LinkedIn and the FTC (reportfraud.ftc.gov)."
            )
        elif score >= 0.6:
            report.recommendations.append(
                "Exercise extreme caution — multiple fraud indicators detected."
            )
            report.recommendations.append(
                "Verify the company independently before sharing any personal information."
            )
        elif score >= 0.4:
            report.recommendations.append(
                "Proceed with caution — some suspicious patterns detected."
            )
            report.recommendations.append(
                "Research the company on LinkedIn, Glassdoor, and via WHOIS before engaging."
            )
        else:
            # Covers 0.0 – 0.4 (low and safe scores)
            report.recommendations.append(
                "Posting appears mostly legitimate but review flagged items before applying."
            )

        if report.evasion_detected:
            report.recommendations.append(
                "Obfuscation techniques detected — scammer may be actively evading detection."
            )
        if report.similar_postings:
            report.recommendations.append(
                f"This posting closely resembles {len(report.similar_postings)} other(s) — "
                "possible template scam operation."
            )
        if report.operator_fingerprint:
            report.recommendations.append(
                f"Writing style matches known scam operator '{report.operator_fingerprint}'."
            )
        if report.llm_generated_probability > 0.7:
            report.recommendations.append(
                "Description appears AI-generated — scammers increasingly use LLMs to create postings."
            )

    # ------------------------------------------------------------------
    # Meta-weight management
    # ------------------------------------------------------------------

    def get_meta_weights(self) -> dict[str, float]:
        return dict(self._meta_weights)

    def set_meta_weights(self, weights: dict[str, float]) -> None:
        self._meta_weights.update(weights)

    def subsystem_availability(self) -> dict[str, bool]:
        """Return which subsystems are importable."""
        return {
            "signals":        _HAS_SIGNALS,
            "scorer":         _HAS_SCORER,
            "fraud_triangle": _HAS_FRAUD,
            "benford":        _HAS_FRAUD,
            "linguistic":     _HAS_FRAUD,
            "llm_detect":     _HAS_LLM,
            "stylometry":     _HAS_STYLO,
            "economics":      _HAS_ECON,
            "graph":          _HAS_GRAPH,
            "robustness":     _HAS_ROBUST,
            "adversarial":    _HAS_ADVERSARIAL,
            "disagreement":   _HAS_DISAGREE,
            "counterfactual": _HAS_COUNTER,
            "research":       _HAS_RESEARCH,
            "innovation":     _HAS_INNOVATION,
            "temporal":       _HAS_TEMPORAL,
            "db":             _HAS_DB,
        }


# ---------------------------------------------------------------------------
# NexusLearner — learning loop
# ---------------------------------------------------------------------------

class NexusLearner:
    """Update ALL relevant subsystems after user feedback.

    Feedback contract:
        is_scam=True  → the job was confirmed as a scam
        is_scam=False → the job was confirmed as legitimate

    The learner:
    1. Updates signal weights in the scorer (Bayesian update per signal).
    2. Tracks per-subsystem accuracy and adjusts Nexus meta-weights.
    3. Records feedback in the DB (if available).
    """

    def __init__(self, nexus: Nexus, db: Any = None) -> None:
        self._nexus = nexus
        self._db = db
        # Per-subsystem accuracy trackers: {name: [correct_bools]}
        self._accuracy: dict[str, list[bool]] = {}

    # ------------------------------------------------------------------
    # Primary feedback method
    # ------------------------------------------------------------------

    def learn(
        self,
        job: JobPosting,
        report: NexusReport,
        is_scam: bool,
    ) -> dict[str, Any]:
        """Ingest user feedback and propagate updates to all subsystems.

        Returns a summary dict of what was updated.
        """
        updates: dict[str, Any] = {"subsystems_updated": [], "meta_weight_changes": {}}

        # ── 1. Update signal weights (Bayesian) ─────────────────────────
        if _HAS_SCORER:
            try:
                from sentinel.scorer import SignalWeightTracker
                tracker = SignalWeightTracker()
                for sig in report.signals_fired:
                    # A signal is "correct" if it fired AND the job is a scam
                    # (or didn't fire AND job is legitimate)
                    was_correct = is_scam and sig.category != SignalCategory.POSITIVE
                    was_correct = was_correct or (
                        not is_scam and sig.category == SignalCategory.POSITIVE
                    )
                    tracker.update(sig.name, was_correct=was_correct)
                updates["subsystems_updated"].append("signal_weights")
            except Exception as exc:
                logger.debug("nexus learner: signal weight update failed: %s", exc)

        # ── 2. Update LLM detector corpus ───────────────────────────────
        if _HAS_LLM and is_scam and report.llm_generated_probability > 0.5:
            try:
                from sentinel.llm_detect import StyleFingerprinter
                fingerprinter = StyleFingerprinter()
                text = _full_text(job)
                if text:
                    label = "scam_llm" if is_scam else "legitimate"
                    fingerprinter.add_to_corpus(text, label)
                    updates["subsystems_updated"].append("llm_corpus")
            except Exception as exc:
                logger.debug("nexus learner: llm corpus update failed: %s", exc)

        # ── 3. Update stylometry operator library ───────────────────────
        if _HAS_STYLO and is_scam and report.operator_fingerprint:
            try:
                if self._nexus._operator_linker is not None:
                    extractor = StyleExtractor()
                    text = _full_text(job)
                    if text:
                        fp = extractor.extract(text)
                        posting_id = job.url or f"job_{hash(text) & 0xFFFFFF}"
                        self._nexus._operator_linker.add_fingerprint(
                            posting_id, fp,
                            operator_id=report.operator_fingerprint,
                        )
                        updates["subsystems_updated"].append("stylometry_library")
            except Exception as exc:
                logger.debug("nexus learner: stylometry update failed: %s", exc)

        # ── 4. Track per-subsystem accuracy and update meta-weights ─────
        predicted_scam = report.overall_score >= 0.5
        was_correct_overall = predicted_scam == is_scam

        for subsystem, score in report.subsystem_scores.items():
            predicted = score >= 0.5
            correct = predicted == is_scam
            if subsystem not in self._accuracy:
                self._accuracy[subsystem] = []
            self._accuracy[subsystem].append(correct)

        # Recompute meta-weights from rolling accuracy
        new_weights = self._recompute_meta_weights()
        if new_weights:
            old_weights = self._nexus.get_meta_weights()
            self._nexus.set_meta_weights(new_weights)
            for k, v in new_weights.items():
                old = old_weights.get(k, 0.0)
                if abs(v - old) > 0.005:
                    updates["meta_weight_changes"][k] = {"old": round(old, 4), "new": round(v, 4)}

        # ── 5. Persist to DB ─────────────────────────────────────────────
        if _HAS_DB and self._db is not None:
            try:
                self._db.save_flywheel_metrics({
                    "timestamp": _now_iso(),
                    "feedback_is_scam": is_scam,
                    "prediction_score": report.overall_score,
                    "was_correct": was_correct_overall,
                    "signals_count": len(report.signals_fired),
                    "subsystems_run": ",".join(report.subsystems_run),
                })
                updates["subsystems_updated"].append("db_feedback")
            except Exception as exc:
                logger.debug("nexus learner: db persist failed: %s", exc)

        updates["was_correct"] = was_correct_overall
        updates["predicted_score"] = report.overall_score
        updates["ground_truth_scam"] = is_scam

        logger.info(
            "NexusLearner.learn: correct=%s, score=%.3f, updated=%s",
            was_correct_overall, report.overall_score,
            updates["subsystems_updated"],
        )
        return updates

    # ------------------------------------------------------------------
    # Internal: recompute meta-weights from accuracy history
    # ------------------------------------------------------------------

    def _recompute_meta_weights(self) -> dict[str, float]:
        """Compute new meta-weights proportional to per-subsystem accuracy.

        Uses a rolling window of the last 50 observations.
        Falls back to default weights for subsystems with < 5 observations.
        """
        MIN_OBS = 5
        WINDOW = 50

        accuracy_scores: dict[str, float] = {}
        for name, history in self._accuracy.items():
            recent = history[-WINDOW:]
            if len(recent) >= MIN_OBS:
                accuracy_scores[name] = sum(recent) / len(recent)

        if not accuracy_scores:
            return {}

        # Baseline: default weights, then scale by relative accuracy
        base = dict(_DEFAULT_META_WEIGHTS)
        total_acc = sum(accuracy_scores.values())
        if total_acc == 0:
            return {}

        new_weights: dict[str, float] = {}
        for name, default_w in base.items():
            if name in accuracy_scores:
                # Scale weight by accuracy relative to average accuracy
                avg_acc = total_acc / len(accuracy_scores)
                scale = accuracy_scores[name] / avg_acc if avg_acc > 0 else 1.0
                # Dampen changes: blend 70% old + 30% new
                new_w = default_w * (0.7 + 0.3 * scale)
                new_weights[name] = _clamp(new_w, 0.01, 0.5)
            else:
                new_weights[name] = default_w

        # Re-normalise so weights sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        return new_weights

    def accuracy_summary(self) -> dict[str, float]:
        """Return current rolling accuracy per subsystem."""
        return {
            name: sum(hist[-50:]) / len(hist[-50:])
            for name, hist in self._accuracy.items()
            if hist
        }

    def most_accurate_subsystem(self) -> str | None:
        """Return name of the most accurate subsystem (or None)."""
        acc = self.accuracy_summary()
        if not acc:
            return None
        return max(acc, key=lambda k: acc[k])


# ---------------------------------------------------------------------------
# NexusDashboard — system-wide health and accuracy tracking
# ---------------------------------------------------------------------------

@dataclass
class SubsystemStatus:
    name: str
    available: bool
    accuracy: float = 0.0
    observations: int = 0
    avg_score: float = 0.0
    meta_weight: float = 0.0


@dataclass
class DashboardSnapshot:
    timestamp: str
    total_analyses: int
    subsystem_statuses: list[SubsystemStatus]
    top_signals: list[str]
    system_health_score: float
    accuracy_trend: list[float]
    recommendations: list[str]


class NexusDashboard:
    """System-wide health, accuracy, and signal tracking.

    Parameters
    ----------
    nexus:
        The Nexus instance to monitor.
    learner:
        Optional NexusLearner for accuracy data.
    db:
        Optional SentinelDB for historical metrics.
    window:
        Rolling window size for accuracy trend (default 30).
    """

    def __init__(
        self,
        nexus: Nexus,
        learner: NexusLearner | None = None,
        db: Any = None,
        window: int = 30,
    ) -> None:
        self._nexus = nexus
        self._learner = learner
        self._db = db
        self._window = window
        self._recent_reports: list[NexusReport] = []
        self._correct_predictions: list[bool] = []

    def record_result(self, report: NexusReport, ground_truth_scam: bool | None = None) -> None:
        """Record a completed analysis for tracking."""
        self._recent_reports.append(report)
        # Enforce window size — keep only the most recent `_window` entries
        if len(self._recent_reports) > self._window:
            self._recent_reports = self._recent_reports[-self._window:]
        if ground_truth_scam is not None:
            predicted = report.overall_score >= 0.5
            self._correct_predictions.append(predicted == ground_truth_scam)

    def snapshot(self) -> DashboardSnapshot:
        """Build a full health snapshot."""
        availability = self._nexus.subsystem_availability()
        meta_weights = self._nexus.get_meta_weights()
        acc_summary = self._learner.accuracy_summary() if self._learner else {}

        # Per-subsystem statuses
        statuses: list[SubsystemStatus] = []
        for name, available in availability.items():
            # Compute avg score from recent reports
            scores = [
                r.subsystem_scores.get(name, 0.0)
                for r in self._recent_reports
                if name in r.subsystem_scores
            ]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            statuses.append(SubsystemStatus(
                name=name,
                available=available,
                accuracy=acc_summary.get(name, 0.0),
                observations=len(scores),
                avg_score=round(avg_score, 4),
                meta_weight=round(meta_weights.get(name, 0.0), 4),
            ))

        # Top signals across recent analyses
        from collections import Counter
        signal_counter: Counter = Counter()
        for r in self._recent_reports:
            for s in r.signals_fired:
                signal_counter[s.name] += 1
        top_signals = [name for name, _ in signal_counter.most_common(10)]

        # Accuracy trend (rolling)
        recent_correct = self._correct_predictions[-self._window:]
        if recent_correct:
            chunk_size = max(1, len(recent_correct) // 5)
            chunks = [
                recent_correct[i:i + chunk_size]
                for i in range(0, len(recent_correct), chunk_size)
            ]
            accuracy_trend = [
                round(sum(c) / len(c), 3) for c in chunks if c
            ]
        else:
            accuracy_trend = []

        # System health: fraction of subsystems available
        n_available = sum(1 for v in availability.values() if v)
        health_score = round(n_available / max(len(availability), 1), 3)

        # Recommendations
        recs: list[str] = []
        unavailable = [n for n, v in availability.items() if not v]
        if unavailable:
            recs.append(
                f"Subsystems unavailable: {', '.join(unavailable[:5])}. "
                "Check imports and dependencies."
            )
        if accuracy_trend and len(accuracy_trend) >= 2:
            if accuracy_trend[-1] < accuracy_trend[-2] - 0.05:
                recs.append("Accuracy declining — consider running NexusEvolver.evolve().")
        if not self._recent_reports:
            recs.append("No analyses recorded yet — run Nexus.deep_analyze() to populate.")

        return DashboardSnapshot(
            timestamp=_now_iso(),
            total_analyses=self._nexus._analysis_count,
            subsystem_statuses=statuses,
            top_signals=top_signals,
            system_health_score=health_score,
            accuracy_trend=accuracy_trend,
            recommendations=recs,
        )

    def most_informative_subsystems(self, top_n: int = 5) -> list[str]:
        """Return the top N subsystems by meta-weight * accuracy."""
        acc = self._learner.accuracy_summary() if self._learner else {}
        weights = self._nexus.get_meta_weights()
        scores = {
            name: weights.get(name, 0.0) * acc.get(name, 0.5)
            for name in weights
        }
        return sorted(scores, key=lambda k: scores[k], reverse=True)[:top_n]

    def most_active_signals(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return (signal_name, count) for the most-fired signals in recent analyses."""
        from collections import Counter
        c: Counter = Counter()
        for r in self._recent_reports:
            for s in r.signals_fired:
                c[s.name] += 1
        return c.most_common(top_n)


# ---------------------------------------------------------------------------
# NexusEvolver — autonomous improvement orchestration
# ---------------------------------------------------------------------------

@dataclass
class EvolveResult:
    """Result from one full autonomous improvement cycle."""
    cycle: int
    timestamp: str
    steps_completed: list[str]
    counterfactual_insights: list[str]
    research_topics_added: int
    innovation_ran: bool
    regression_clean: bool
    meta_weight_adjustments: dict[str, float]
    summary: str


class NexusEvolver:
    """Orchestrate autonomous improvement across ALL subsystems.

    A single call to ``evolve()`` performs one full improvement cycle:
      1. Counterfactual analysis on recent misses (counterfactual.py)
      2. Feed gap analysis to research engine (research.py)
      3. Run innovation cycle on weakest areas (innovation.py)
      4. Check regression guard after changes
      5. Update temporal tracking (temporal.py)

    Parameters
    ----------
    nexus:
        The Nexus instance to evolve.
    learner:
        NexusLearner providing accuracy data.
    dashboard:
        NexusDashboard for system health data.
    db:
        Optional SentinelDB.
    """

    def __init__(
        self,
        nexus: Nexus,
        learner: NexusLearner | None = None,
        dashboard: NexusDashboard | None = None,
        db: Any = None,
    ) -> None:
        self._nexus = nexus
        self._learner = learner
        self._dashboard = dashboard
        self._db = db
        self._cycle = 0

    def evolve(self, recent_reports: list[NexusReport] | None = None) -> EvolveResult:
        """Run one full autonomous improvement cycle.

        Parameters
        ----------
        recent_reports:
            NexusReport objects from recent analyses.  If None, uses reports
            from the dashboard's rolling window.
        """
        t0 = time.monotonic()
        self._cycle += 1
        steps: list[str] = []
        cf_insights: list[str] = []
        research_added = 0
        innovation_ran = False
        regression_clean = True
        meta_adjustments: dict[str, float] = {}

        reports = recent_reports or (
            self._dashboard._recent_reports if self._dashboard else []
        )

        # ── Step 1: Counterfactual analysis on misses ───────────────────
        if _HAS_COUNTER and reports:
            try:
                ce = CounterfactualEngine()
                for r in reports[-20:]:  # last 20 reports
                    if r.overall_score < 0.4 and r.signals_fired:
                        # These might be false negatives — analyze what almost fired
                        candidates = self._nexus._candidate_signal_names(
                            [s.name for s in r.signals_fired]
                        )
                        cf_results = ce.rank_counterfactuals(r.signals_fired, candidates)
                        for cr in cf_results[:2]:
                            delta = getattr(cr, "score_delta", 0.0)
                            name = getattr(cr, "signal_name", "")
                            if name and abs(delta) >= 0.08:
                                insight = f"Signal '{name}' could add {delta:+.2f} to score"
                                if insight not in cf_insights:
                                    cf_insights.append(insight)
                steps.append("counterfactual_analysis")
            except Exception as exc:
                logger.debug("nexus evolver: counterfactual failed: %s", exc)

        # ── Step 2: Feed gap analysis to research engine ────────────────
        if _HAS_RESEARCH and cf_insights:
            try:
                from sentinel.research import ResearchEngine, ResearchTopic
                re_engine = ResearchEngine(db=self._db)
                for insight in cf_insights[:3]:
                    topic = ResearchTopic(
                        name=f"gap_from_nexus: {insight[:60]}",
                        query=insight,
                        priority=0.8,
                    )
                    re_engine.add_topic(topic)
                    research_added += 1
                steps.append("research_gap_feeding")
            except Exception as exc:
                logger.debug("nexus evolver: research failed: %s", exc)

        # ── Step 3: Innovation on weakest subsystems ────────────────────
        if _HAS_INNOVATION and self._learner:
            try:
                from sentinel.innovation import InnovationEngine
                acc = self._learner.accuracy_summary()
                weakest = min(acc, key=lambda k: acc[k]) if acc else None
                if weakest:
                    ie = InnovationEngine(db=self._db)
                    ie.run_cycle(focus_area=weakest)
                    innovation_ran = True
                    steps.append(f"innovation_on_{weakest}")
            except Exception as exc:
                logger.debug("nexus evolver: innovation failed: %s", exc)

        # ── Step 4: Regression guard ────────────────────────────────────
        if self._learner:
            try:
                acc = self._learner.accuracy_summary()
                if acc:
                    avg_acc = sum(acc.values()) / len(acc)
                    regression_clean = avg_acc >= 0.6
                    if not regression_clean:
                        logger.warning(
                            "NexusEvolver: regression guard triggered — avg accuracy=%.2f", avg_acc
                        )
                steps.append("regression_guard")
            except Exception as exc:
                logger.debug("nexus evolver: regression guard failed: %s", exc)

        # ── Step 5: Temporal tracking ────────────────────────────────────
        if _HAS_TEMPORAL and self._db:
            try:
                from sentinel.temporal import TemporalTracker
                tt = TemporalTracker(db=self._db)
                tt.record_cycle(cycle=self._cycle, timestamp=_now_iso())
                steps.append("temporal_tracking")
            except Exception as exc:
                logger.debug("nexus evolver: temporal failed: %s", exc)

        # ── Collect meta-weight adjustments ─────────────────────────────
        if self._learner:
            new_w = self._learner._recompute_meta_weights()
            old_w = self._nexus.get_meta_weights()
            for k, v in new_w.items():
                diff = v - old_w.get(k, 0.0)
                if abs(diff) > 0.005:
                    meta_adjustments[k] = round(diff, 4)

        elapsed_ms = (time.monotonic() - t0) * 1000
        summary = (
            f"Cycle {self._cycle}: {len(steps)} steps in {elapsed_ms:.0f}ms. "
            f"Counterfactual insights={len(cf_insights)}, "
            f"research_added={research_added}, "
            f"innovation={'yes' if innovation_ran else 'no'}, "
            f"regression={'clean' if regression_clean else 'ALERT'}."
        )

        logger.info("NexusEvolver.evolve: %s", summary)

        return EvolveResult(
            cycle=self._cycle,
            timestamp=_now_iso(),
            steps_completed=steps,
            counterfactual_insights=cf_insights,
            research_topics_added=research_added,
            innovation_ran=innovation_ran,
            regression_clean=regression_clean,
            meta_weight_adjustments=meta_adjustments,
            summary=summary,
        )

    def run_cycles(self, n: int, reports: list[NexusReport] | None = None) -> list[EvolveResult]:
        """Run *n* evolution cycles sequentially."""
        return [self.evolve(recent_reports=reports) for _ in range(n)]
