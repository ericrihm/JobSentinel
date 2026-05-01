"""Input perturbation robustness testing for Sentinel scam detection.

Inspired by SmoothLLM (2025-2026 research).  Tests whether Sentinel's scam
classification is robust by slightly modifying inputs and checking whether the
score changes dramatically.

Classes
-------
PerturbationEngine   — Generate N perturbed variants of a text input.
RobustnessScorer     — Score original + perturbations, measure variance.
AdversarialProber    — Find minimal changes that flip a classification.
RobustnessReport     — Dataclass capturing the full analysis result.
"""

from __future__ import annotations

import math
import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Callable

# ---------------------------------------------------------------------------
# Synonym table (inline, stdlib-only)
# ---------------------------------------------------------------------------

# Maps common job-ad words to semantically similar alternatives.
# Kept intentionally modest so substitutions stay plausible.
_SYNONYMS: dict[str, list[str]] = {
    "guaranteed": ["assured", "certain", "promised", "secure"],
    "earn": ["make", "gain", "receive", "get"],
    "income": ["salary", "pay", "compensation", "wages"],
    "immediately": ["right away", "at once", "promptly", "now"],
    "apply": ["submit", "register", "enroll", "sign up"],
    "opportunity": ["chance", "opening", "position", "role"],
    "required": ["needed", "mandatory", "necessary", "essential"],
    "experience": ["background", "expertise", "knowledge", "skills"],
    "remote": ["work from home", "virtual", "telecommute", "distributed"],
    "fee": ["cost", "charge", "payment", "deposit"],
    "contact": ["reach", "email", "message", "call"],
    "hiring": ["recruiting", "seeking", "looking for", "accepting applications"],
    "company": ["firm", "organization", "business", "employer"],
    "position": ["role", "job", "opening", "vacancy"],
    "team": ["group", "department", "staff", "crew"],
    "skills": ["abilities", "competencies", "qualifications", "expertise"],
    "salary": ["pay", "compensation", "wages", "remuneration"],
    "benefits": ["perks", "advantages", "offerings", "package"],
    "join": ["become part of", "work with", "partner with", "come aboard"],
    "grow": ["develop", "expand", "advance", "progress"],
}

_SYNONYM_KEYS = list(_SYNONYMS)

# ---------------------------------------------------------------------------
# PerturbationEngine
# ---------------------------------------------------------------------------


class PerturbationEngine:
    """Generate perturbed variants of a text string.

    Perturbation types
    ------------------
    char_swap       — swap two adjacent characters (1-3 % of chars)
    char_insert     — insert a random letter (1-3 % of chars)
    char_delete     — delete a character (1-3 % of chars)
    synonym_sub     — replace a word with a synonym
    word_reorder    — swap two adjacent words in a sentence
    whitespace      — add/remove spaces around punctuation / words
    capitalisation  — change case of individual words
    punctuation     — add/remove terminal punctuation marks
    """

    # Fraction of characters to perturb in character-level operations.
    CHAR_PERTURB_MIN = 0.01
    CHAR_PERTURB_MAX = 0.03

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, text: str, n: int = 20, strategies: list[str] | None = None) -> list[str]:
        """Return *n* perturbed variants of *text*.

        Parameters
        ----------
        text:       Input text to perturb.
        n:          Number of variants to generate.
        strategies: Subset of perturbation strategy names to use.
                    Defaults to all available strategies.

        Returns
        -------
        List of perturbed strings (may duplicate if text is very short).
        """
        if not text:
            return [text] * n

        available: list[Callable[[str], str]] = self._strategy_map(strategies)
        if not available:
            return [text] * n

        variants: list[str] = []
        for _ in range(n):
            strategy = self._rng.choice(available)
            variants.append(strategy(text))
        return variants

    def generate_targeted(self, text: str, n: int, strategy: str) -> list[str]:
        """Generate *n* variants using a single named strategy."""
        fn_map = {name: fn for name, fn in zip(self._strategy_names(), self._all_strategies())}
        fn = fn_map.get(strategy)
        if fn is None:
            raise ValueError(f"Unknown strategy: {strategy!r}. Available: {list(fn_map)}")
        return [fn(text) for _ in range(n)]

    def strategy_names(self) -> list[str]:
        """Return names of all available perturbation strategies."""
        return list(self._strategy_names())

    # ------------------------------------------------------------------
    # Character-level perturbations
    # ------------------------------------------------------------------

    def char_swap(self, text: str) -> str:
        """Swap pairs of adjacent characters at random positions."""
        chars = list(text)
        n_swaps = max(1, int(len(chars) * self._perturb_rate()))
        for _ in range(n_swaps):
            idx = self._rng.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    def char_insert(self, text: str) -> str:
        """Insert random lowercase letters at random positions."""
        chars = list(text)
        n_inserts = max(1, int(len(chars) * self._perturb_rate()))
        for _ in range(n_inserts):
            idx = self._rng.randint(0, len(chars))
            chars.insert(idx, self._rng.choice(string.ascii_lowercase))
        return "".join(chars)

    def char_delete(self, text: str) -> str:
        """Delete characters at random positions (at least 1 remains)."""
        if len(text) <= 1:
            return text
        chars = list(text)
        n_deletes = max(1, int(len(chars) * self._perturb_rate()))
        n_deletes = min(n_deletes, len(chars) - 1)
        indices = sorted(self._rng.sample(range(len(chars)), n_deletes), reverse=True)
        for idx in indices:
            del chars[idx]
        return "".join(chars)

    # ------------------------------------------------------------------
    # Word-level perturbations
    # ------------------------------------------------------------------

    def synonym_sub(self, text: str) -> str:
        """Replace a word with a synonym where one is available."""
        words = text.split()
        lower_words = [w.lower().strip(string.punctuation) for w in words]

        # Find candidate positions that have synonyms
        candidates = [
            i for i, lw in enumerate(lower_words) if lw in _SYNONYMS
        ]
        if not candidates:
            return text

        idx = self._rng.choice(candidates)
        base_word = lower_words[idx]
        synonym = self._rng.choice(_SYNONYMS[base_word])

        # Preserve original capitalisation pattern
        original = words[idx].rstrip(string.punctuation)
        if original.isupper():
            synonym = synonym.upper()
        elif original[0].isupper():
            synonym = synonym.capitalize()

        words[idx] = synonym + words[idx][len(original):]  # keep trailing punctuation
        return " ".join(words)

    def word_reorder(self, text: str) -> str:
        """Swap two adjacent words within the same sentence."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        modified = []
        swapped = False
        for sent in sentences:
            words = sent.split()
            if len(words) >= 2 and not swapped:
                idx = self._rng.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
                swapped = True
            modified.append(" ".join(words))
        return " ".join(modified)

    # ------------------------------------------------------------------
    # Formatting perturbations
    # ------------------------------------------------------------------

    def whitespace_perturb(self, text: str) -> str:
        """Add or remove whitespace around punctuation / between words."""
        choice = self._rng.randint(0, 2)
        if choice == 0:
            # Add extra space before a random punctuation mark
            text = re.sub(r"([,;:])", r" \1", text, count=1)
        elif choice == 1:
            # Remove space after a comma (once)
            text = re.sub(r",\s+", ",", text, count=1)
        else:
            # Double a space somewhere
            text = re.sub(r"  ", "   ", text, count=1)
        return text

    def capitalisation_perturb(self, text: str) -> str:
        """Change capitalisation of a random word."""
        words = text.split()
        if not words:
            return text
        idx = self._rng.randint(0, len(words) - 1)
        w = words[idx]
        choice = self._rng.randint(0, 2)
        if choice == 0:
            words[idx] = w.upper()
        elif choice == 1:
            words[idx] = w.lower()
        else:
            words[idx] = w.capitalize()
        return " ".join(words)

    def punctuation_perturb(self, text: str) -> str:
        """Add or remove terminal/inline punctuation."""
        choice = self._rng.randint(0, 2)
        if choice == 0:
            # Add exclamation mark at the end
            text = text.rstrip() + "!"
        elif choice == 1:
            # Remove a comma
            text = text.replace(",", "", 1)
        else:
            # Add a period mid-sentence
            words = text.split()
            if len(words) > 4:
                idx = self._rng.randint(2, len(words) - 2)
                words[idx] = words[idx] + "."
            text = " ".join(words)
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perturb_rate(self) -> float:
        return self._rng.uniform(self.CHAR_PERTURB_MIN, self.CHAR_PERTURB_MAX)

    def _all_strategies(self) -> list[Callable[[str], str]]:
        return [
            self.char_swap,
            self.char_insert,
            self.char_delete,
            self.synonym_sub,
            self.word_reorder,
            self.whitespace_perturb,
            self.capitalisation_perturb,
            self.punctuation_perturb,
        ]

    def _strategy_names(self) -> list[str]:
        return [
            "char_swap",
            "char_insert",
            "char_delete",
            "synonym_sub",
            "word_reorder",
            "whitespace_perturb",
            "capitalisation_perturb",
            "punctuation_perturb",
        ]

    def _strategy_map(self, strategies: list[str] | None) -> list[Callable[[str], str]]:
        name_to_fn = dict(zip(self._strategy_names(), self._all_strategies()))
        if strategies is None:
            return self._all_strategies()
        result = []
        for s in strategies:
            fn = name_to_fn.get(s)
            if fn is not None:
                result.append(fn)
        return result or self._all_strategies()


# ---------------------------------------------------------------------------
# RobustnessReport
# ---------------------------------------------------------------------------


@dataclass
class RobustnessReport:
    """Full robustness analysis result for a single job posting.

    Attributes
    ----------
    original_score:         Scam score for the unmodified text (0–1).
    mean_perturbed_score:   Mean score across all perturbed variants.
    score_std:              Standard deviation of perturbed scores.
    min_score:              Minimum score observed across perturbations.
    max_score:              Maximum score observed across perturbations.
    fragility_score:        0–1; higher = more fragile classification.
    is_fragile:             True when fragility exceeds the configured threshold.
    weakest_signal:         Signal name most easily evaded, or empty string.
    weakest_signal_impact:  Score delta caused by removing the weakest signal.
    suggested_improvements: Human-readable improvement hints.
    perturbed_scores:       Raw list of all perturbed scores.
    adversarial_examples:   List of (perturbed_text, score) flips found.
    human_review_requested: True when fragility warrants escalation.
    summary:                One-sentence human-readable summary.
    """

    original_score: float
    mean_perturbed_score: float
    score_std: float
    min_score: float
    max_score: float
    fragility_score: float
    is_fragile: bool
    weakest_signal: str
    weakest_signal_impact: float
    suggested_improvements: list[str]
    perturbed_scores: list[float]
    adversarial_examples: list[tuple[str, float]]
    human_review_requested: bool
    summary: str = field(default="")

    def __post_init__(self) -> None:
        if not self.summary:
            self.summary = self._build_summary()

    def _build_summary(self) -> str:
        label = f"This job scored {self.original_score:.2f}"
        if self.perturbed_scores:
            label += (
                f" but with minor changes scores range from "
                f"{self.min_score:.2f} to {self.max_score:.2f}"
            )
        if self.is_fragile:
            label += " — classification is FRAGILE"
        return label


# ---------------------------------------------------------------------------
# RobustnessScorer
# ---------------------------------------------------------------------------


class RobustnessScorer:
    """Score original + N perturbations and measure classification stability.

    Parameters
    ----------
    scoring_fn:         Callable(text: str) -> float returning a scam score 0–1.
    n_perturbations:    Number of perturbed variants to generate (default 20).
    fragility_threshold: Score std above which the classification is fragile
                         (default 0.15).
    flip_threshold:     Score difference considered a classification flip
                         (default 0.4).
    seed:               Optional random seed for reproducibility.
    """

    FRAGILITY_THRESHOLD = 0.15
    FLIP_THRESHOLD = 0.40

    def __init__(
        self,
        scoring_fn: Callable[[str], float],
        n_perturbations: int = 20,
        fragility_threshold: float = FRAGILITY_THRESHOLD,
        flip_threshold: float = FLIP_THRESHOLD,
        seed: int | None = None,
    ) -> None:
        self.scoring_fn = scoring_fn
        self.n_perturbations = n_perturbations
        self.fragility_threshold = fragility_threshold
        self.flip_threshold = flip_threshold
        self._engine = PerturbationEngine(seed=seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        text: str,
        strategies: list[str] | None = None,
    ) -> RobustnessReport:
        """Analyse robustness for *text*.

        Parameters
        ----------
        text:       The raw job posting text to evaluate.
        strategies: Optional subset of perturbation strategies.

        Returns
        -------
        RobustnessReport with full analysis.
        """
        original_score = self.scoring_fn(text)

        variants = self._engine.generate(text, n=self.n_perturbations, strategies=strategies)
        perturbed_scores = [self.scoring_fn(v) for v in variants]

        if not perturbed_scores:
            return self._empty_report(original_score)

        mu = mean(perturbed_scores)
        sd = stdev(perturbed_scores) if len(perturbed_scores) > 1 else 0.0
        lo = min(perturbed_scores)
        hi = max(perturbed_scores)

        fragility = self._compute_fragility(sd, lo, hi)
        is_fragile = sd > self.fragility_threshold

        # Collect adversarial examples (flips)
        adversarial: list[tuple[str, float]] = []
        for variant, ps in zip(variants, perturbed_scores):
            if abs(ps - original_score) >= self.flip_threshold:
                adversarial.append((variant, ps))

        improvements = self._suggest_improvements(fragility, sd, adversarial)

        return RobustnessReport(
            original_score=round(original_score, 4),
            mean_perturbed_score=round(mu, 4),
            score_std=round(sd, 4),
            min_score=round(lo, 4),
            max_score=round(hi, 4),
            fragility_score=round(fragility, 4),
            is_fragile=is_fragile,
            weakest_signal="",  # populated by AdversarialProber
            weakest_signal_impact=0.0,
            suggested_improvements=improvements,
            perturbed_scores=[round(s, 4) for s in perturbed_scores],
            adversarial_examples=adversarial,
            human_review_requested=is_fragile or bool(adversarial),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fragility(sd: float, lo: float, hi: float) -> float:
        """Combine std and score range into a 0-1 fragility score."""
        score_range = hi - lo
        # Weighted average: std drives half, range drives half
        raw = 0.5 * min(sd / 0.5, 1.0) + 0.5 * min(score_range / 0.8, 1.0)
        return min(raw, 1.0)

    @staticmethod
    def _suggest_improvements(
        fragility: float,
        sd: float,
        adversarial: list[tuple[str, float]],
    ) -> list[str]:
        hints: list[str] = []
        if sd > 0.15:
            hints.append(
                "High score variance suggests over-reliance on single keywords; "
                "consider ensemble signals."
            )
        if adversarial:
            hints.append(
                f"{len(adversarial)} perturbation(s) flipped the classification; "
                "strengthen signal weighting or add redundancy."
            )
        if fragility > 0.7:
            hints.append(
                "Extremely fragile — add contextual signals that are harder to perturb "
                "(e.g. company age, recruiter credibility)."
            )
        if not hints:
            hints.append("Classification appears robust to minor perturbations.")
        return hints

    @staticmethod
    def _empty_report(original_score: float) -> RobustnessReport:
        return RobustnessReport(
            original_score=round(original_score, 4),
            mean_perturbed_score=round(original_score, 4),
            score_std=0.0,
            min_score=round(original_score, 4),
            max_score=round(original_score, 4),
            fragility_score=0.0,
            is_fragile=False,
            weakest_signal="",
            weakest_signal_impact=0.0,
            suggested_improvements=["Classification appears robust to minor perturbations."],
            perturbed_scores=[],
            adversarial_examples=[],
            human_review_requested=False,
        )


# ---------------------------------------------------------------------------
# AdversarialProber
# ---------------------------------------------------------------------------


class AdversarialProber:
    """Systematically find minimal changes that flip a classification.

    The prober works by ablating signals one at a time from the description
    text and measuring the resulting score change.  It identifies:

    - Adversarial examples (smallest change that flips scam→legit or vice versa)
    - Single-point-of-failure signals (removing one word drops score drastically)
    - A full ranking of signals by their individual impact

    Parameters
    ----------
    scoring_fn:         Callable(text: str) -> float.
    flip_threshold:     Minimum score delta to count as a flip (default 0.3).
    """

    FLIP_THRESHOLD = 0.30

    # Known high-weight scam keywords that are prime flip candidates.
    _PROBE_KEYWORDS: list[str] = [
        "guaranteed",
        "bitcoin",
        "payment",
        "fee",
        "social security",
        "SSN",
        "upfront",
        "wire transfer",
        "no experience",
        "immediately",
        "earn",
        "registration fee",
        "training fee",
        "gift card",
        "crypto",
        "western union",
        "moneygram",
        "cashapp",
        "venmo",
        "no interview",
        "apply now",
        "limited spots",
    ]

    def __init__(
        self,
        scoring_fn: Callable[[str], float],
        flip_threshold: float = FLIP_THRESHOLD,
    ) -> None:
        self.scoring_fn = scoring_fn
        self.flip_threshold = flip_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(self, text: str) -> dict:
        """Probe *text* for adversarial examples.

        Returns
        -------
        dict with keys:
            original_score      — float
            signal_impacts      — list of (keyword, score_after_removal, delta)
            weakest_signal      — keyword with largest score drop on removal
            weakest_signal_impact — magnitude of that drop
            adversarial_flips   — list of (keyword, score_after_removal) that flip
            single_point_failures — keywords where removal alone flips classification
            report_line         — human-readable one-liner
        """
        original_score = self.scoring_fn(text)
        signal_impacts: list[tuple[str, float, float]] = []

        for keyword in self._PROBE_KEYWORDS:
            # Case-insensitive removal of the keyword phrase
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            if not pattern.search(text):
                continue
            ablated = pattern.sub("", text)
            ablated = re.sub(r"  +", " ", ablated).strip()
            new_score = self.scoring_fn(ablated)
            delta = original_score - new_score
            signal_impacts.append((keyword, round(new_score, 4), round(delta, 4)))

        # Sort by delta descending (biggest impact first)
        signal_impacts.sort(key=lambda t: t[2], reverse=True)

        adversarial_flips = [
            (kw, ns) for kw, ns, delta in signal_impacts if delta >= self.flip_threshold
        ]
        single_point_failures = [
            (kw, ns) for kw, ns, delta in signal_impacts if delta >= self.flip_threshold * 0.8
        ]

        weakest_signal = ""
        weakest_impact = 0.0
        if signal_impacts:
            weakest_signal, _, weakest_impact = signal_impacts[0]

        report_line = self._build_report_line(
            original_score, weakest_signal, weakest_impact, adversarial_flips
        )

        return {
            "original_score": round(original_score, 4),
            "signal_impacts": signal_impacts,
            "weakest_signal": weakest_signal,
            "weakest_signal_impact": round(weakest_impact, 4),
            "adversarial_flips": adversarial_flips,
            "single_point_failures": single_point_failures,
            "report_line": report_line,
        }

    def find_minimal_flip(self, text: str) -> tuple[str, float] | None:
        """Return (keyword_removed, new_score) for the smallest flip, or None."""
        result = self.probe(text)
        if result["adversarial_flips"]:
            return result["adversarial_flips"][-1]  # smallest delta that still flips
        return None

    def rank_signals_by_impact(self, text: str) -> list[tuple[str, float]]:
        """Return [(keyword, impact_delta)] sorted from highest impact to lowest."""
        result = self.probe(text)
        return [(kw, delta) for kw, _ns, delta in result["signal_impacts"]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report_line(
        original_score: float,
        weakest_signal: str,
        weakest_impact: float,
        adversarial_flips: list[tuple[str, float]],
    ) -> str:
        if not weakest_signal:
            return f"Original score {original_score:.2f}: no probe keywords found in text."
        line = (
            f"Removing '{weakest_signal}' drops score from "
            f"{original_score:.2f} to {original_score - weakest_impact:.2f} "
            f"(delta {weakest_impact:.2f})"
        )
        if weakest_impact >= AdversarialProber.FLIP_THRESHOLD:
            line += " — single-point-of-failure"
        if adversarial_flips:
            line += f"; {len(adversarial_flips)} keyword(s) individually flip the classification"
        return line


# ---------------------------------------------------------------------------
# Convenience: combined analysis
# ---------------------------------------------------------------------------


def analyse_robustness(
    text: str,
    scoring_fn: Callable[[str], float],
    *,
    n_perturbations: int = 20,
    fragility_threshold: float = RobustnessScorer.FRAGILITY_THRESHOLD,
    flip_threshold: float = RobustnessScorer.FLIP_THRESHOLD,
    seed: int | None = None,
) -> RobustnessReport:
    """One-call interface: run RobustnessScorer + AdversarialProber together.

    Returns a RobustnessReport enriched with the weakest_signal found by the
    AdversarialProber.
    """
    scorer = RobustnessScorer(
        scoring_fn=scoring_fn,
        n_perturbations=n_perturbations,
        fragility_threshold=fragility_threshold,
        flip_threshold=flip_threshold,
        seed=seed,
    )
    prober = AdversarialProber(scoring_fn=scoring_fn, flip_threshold=flip_threshold)

    report = scorer.score(text)
    probe_result = prober.probe(text)

    # Enrich report with prober findings
    report.weakest_signal = probe_result["weakest_signal"]
    report.weakest_signal_impact = probe_result["weakest_signal_impact"]
    if probe_result["report_line"]:
        report.suggested_improvements.insert(0, probe_result["report_line"])

    return report
