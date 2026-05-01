"""LLM-Generated Content Detection for Sentinel.

Detect AI-generated job postings (ChatGPT/Claude-written scam descriptions).

Two components:
- LLMDetector       — perplexity proxy, burstiness, n-gram patterns, phrase regex
- StyleFingerprinter — stylometric feature extraction + Mahalanobis distance
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common LLM-tell phrases (regex)
# ---------------------------------------------------------------------------

_LLM_PHRASES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Conversational openers
        r"\bI'?d be happy to\b",
        r"\bI'?m here to help\b",
        r"\bfeel free to (reach out|contact|ask)\b",
        r"\bdon'?t hesitate to\b",
        # Meta-commentary
        r"\bit'?s important to note\b",
        r"\bit'?s worth (noting|mentioning)\b",
        r"\bplease note that\b",
        r"\bit should be noted\b",
        # Filler / hedge phrases
        r"\bin today'?s (fast.?paced|rapidly changing|ever.?evolving|dynamic)\b",
        r"\bin the (modern|current|contemporary) (workplace|business|world|landscape)\b",
        r"\bas (a|an) (dynamic|passionate|dedicated|driven|motivated|results.?driven)\b",
        r"\bwe are (looking for|seeking) (a|an) (dynamic|passionate|talented|motivated)\b",
        r"\bexciting opportunity\b",
        r"\bjoin our (dynamic|growing|passionate|innovative) team\b",
        r"\bfast.?paced (environment|company|startup|team)\b",
        r"\bwork.?life balance\b",
        r"\bcompetitive (salary|compensation|benefits|pay|package)\b",
        # Overly balanced constructions
        r"\bon the one hand.{0,50}on the other hand\b",
        r"\bpros and cons\b",
        r"\badvantages and disadvantages\b",
        # Generic closing
        r"\bwe look forward to (hearing from you|your application|meeting you)\b",
        r"\bsuccessful candidate (will|must|should)\b",
        r"\bequal opportunity employer\b",
        r"\bsend your (resume|cv|application) to\b",
        # Transition clichés
        r"\bfurthermore,?\b",
        r"\badditionally,?\b",
        r"\bin conclusion,?\b",
        r"\bto summarize,?\b",
        r"\bin summary,?\b",
        r"\boverall,?\b",
    ]
]


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _tokenize_words(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"[a-z']+", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ? ! separators."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# LLMDetectionResult
# ---------------------------------------------------------------------------

@dataclass
class LLMDetectionResult:
    """Full output of LLMDetector.detect()."""
    llm_probability: float          # 0 (human) → 1 (LLM)
    confidence: float               # 0–1
    is_llm_generated: bool
    vocab_diversity: float          # type-token ratio
    sentence_length_uniformity: float  # low = variable (human), high = uniform (LLM)
    burstiness: float               # coefficient of variation of sentence lengths
    llm_phrase_count: int           # how many LLM-tell phrases matched
    repetition_score: float         # n-gram repetition rate
    ai_hedging_density: float       # hedging phrases per sentence
    evidence: list[str] = field(default_factory=list)  # matched phrases / explanations


# ---------------------------------------------------------------------------
# LLMDetector
# ---------------------------------------------------------------------------

class LLMDetector:
    """Detect AI-generated job posting text using statistical and pattern signals.

    Signals:
    1. Vocabulary diversity (type-token ratio)
    2. Sentence length uniformity (coefficient of variation)
    3. Burstiness: human writing has high CoV; LLM is uniform
    4. Token repetition: characteristic n-gram distributions
    5. LLM-tell phrase regex patterns
    6. AI hedging density
    """

    # Threshold above which we classify as LLM-generated
    DEFAULT_THRESHOLD: float = 0.55

    # Minimum text length (chars) for a reliable analysis
    MIN_TEXT_LENGTH: int = 100

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> LLMDetectionResult:
        """Analyse *text* and return an LLMDetectionResult."""
        evidence: list[str] = []

        if len(text) < self.MIN_TEXT_LENGTH:
            return LLMDetectionResult(
                llm_probability=0.5,
                confidence=0.0,
                is_llm_generated=False,
                vocab_diversity=0.0,
                sentence_length_uniformity=0.0,
                burstiness=0.0,
                llm_phrase_count=0,
                repetition_score=0.0,
                ai_hedging_density=0.0,
                evidence=["text too short for reliable analysis"],
            )

        words = _tokenize_words(text)
        sentences = _split_sentences(text)

        # ---- Feature extraction ----------------------------------------

        ttr = self._type_token_ratio(words)
        uniformity, burstiness = self._sentence_length_stats(sentences)
        phrase_count, matched_phrases = self._count_llm_phrases(text)
        repetition = self._repetition_score(words)
        hedging_density = self._hedging_density(phrase_count, sentences)

        # ---- Evidence accumulation -------------------------------------

        if ttr < 0.35:
            evidence.append(f"low vocabulary diversity (TTR={ttr:.2f})")
        if uniformity > 0.7:
            evidence.append(f"highly uniform sentence lengths (uniformity={uniformity:.2f})")
        if burstiness < 0.25:
            evidence.append(f"low burstiness — LLM-like uniformity (CoV={burstiness:.2f})")
        if phrase_count > 0:
            evidence.extend([f"LLM phrase: '{p}'" for p in matched_phrases[:5]])
        if repetition > 0.15:
            evidence.append(f"high n-gram repetition (rate={repetition:.2f})")
        if hedging_density > 0.3:
            evidence.append(f"high AI hedging density ({hedging_density:.2f} per sentence)")

        # ---- Scoring ---------------------------------------------------

        llm_prob = self._combine_signals(
            ttr=ttr,
            uniformity=uniformity,
            burstiness=burstiness,
            phrase_count=phrase_count,
            repetition=repetition,
            hedging_density=hedging_density,
            n_sentences=len(sentences),
        )

        # Confidence scales with text length and signal count
        n_signals_fired = sum([
            ttr < 0.35,
            uniformity > 0.7,
            burstiness < 0.25,
            phrase_count > 0,
            repetition > 0.15,
            hedging_density > 0.3,
        ])
        text_length_factor = 1.0 - math.exp(-len(text) / 500.0)
        confidence = round(text_length_factor * (0.3 + 0.12 * n_signals_fired), 4)
        confidence = min(1.0, confidence)

        return LLMDetectionResult(
            llm_probability=round(llm_prob, 4),
            confidence=confidence,
            is_llm_generated=llm_prob >= self.threshold,
            vocab_diversity=round(ttr, 4),
            sentence_length_uniformity=round(uniformity, 4),
            burstiness=round(burstiness, 4),
            llm_phrase_count=phrase_count,
            repetition_score=round(repetition, 4),
            ai_hedging_density=round(hedging_density, 4),
            evidence=evidence,
        )

    def score(self, text: str) -> float:
        """Convenience method: return just the llm_probability float."""
        return self.detect(text).llm_probability

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def _type_token_ratio(self, words: list[str]) -> float:
        """Type-token ratio: unique_words / total_words.

        LLM text has lower TTR (more repetitive vocabulary).
        Clipped to a 200-word window for comparability across lengths.
        """
        if not words:
            return 0.0
        window = words[:200]
        return len(set(window)) / len(window)

    def _sentence_length_stats(
        self, sentences: list[str]
    ) -> tuple[float, float]:
        """Return (uniformity, burstiness) of sentence word-lengths.

        uniformity = 1 - CoV  (high CoV → variable → human-like)
        burstiness = CoV itself (high = variable = human, low = uniform = LLM)
        """
        if len(sentences) < 2:
            return 0.5, 0.5
        lengths = [len(s.split()) for s in sentences if s.split()]
        if len(lengths) < 2:
            return 0.5, 0.5
        mean_len = statistics.mean(lengths)
        if mean_len == 0:
            return 0.5, 0.5
        stdev = statistics.stdev(lengths)
        cov = stdev / mean_len     # coefficient of variation
        burstiness = round(cov, 4)
        uniformity = round(max(0.0, 1.0 - cov), 4)
        return uniformity, burstiness

    def _count_llm_phrases(self, text: str) -> tuple[int, list[str]]:
        """Count and collect matched LLM-tell phrases."""
        matched: list[str] = []
        for pattern in _LLM_PHRASES:
            m = pattern.search(text)
            if m:
                matched.append(m.group(0))
        return len(matched), matched

    def _repetition_score(self, words: list[str]) -> float:
        """Measure n-gram repetition (trigrams).

        Returns fraction of trigrams that are repeated (appear > once).
        LLM text tends to reuse certain phrase structures more often.
        """
        if len(words) < 3:
            return 0.0
        trigrams = _ngrams(words, 3)
        if not trigrams:
            return 0.0
        counts = Counter(trigrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        return round(repeated / len(counts), 4)

    def _hedging_density(self, phrase_count: int, sentences: list[str]) -> float:
        """Hedging phrases per sentence."""
        if not sentences:
            return 0.0
        return round(phrase_count / len(sentences), 4)

    def _combine_signals(
        self,
        ttr: float,
        uniformity: float,
        burstiness: float,
        phrase_count: int,
        repetition: float,
        hedging_density: float,
        n_sentences: int,
    ) -> float:
        """Combine signals into a single LLM probability via log-odds.

        Each signal contributes a log-odds vote toward "LLM" or "human".
        """
        log_odds = 0.0  # start at 50/50 prior

        # ---- Vocabulary diversity: low TTR → LLM ----
        # Human median ~0.5, LLM median ~0.3 for job postings
        ttr_signal = 0.5 - ttr   # positive = LLM evidence
        log_odds += ttr_signal * 3.0

        # ---- Sentence uniformity: high uniformity → LLM ----
        log_odds += (uniformity - 0.5) * 2.5

        # ---- Burstiness: low CoV → LLM ----
        log_odds += (0.4 - burstiness) * 2.0

        # ---- LLM phrases: each match adds evidence ----
        # Each phrase adds ~0.3 log-odds unit, capped at 3 phrases
        log_odds += min(phrase_count, 3) * 0.5

        # ---- Repetition ----
        log_odds += (repetition - 0.05) * 4.0

        # ---- Hedging density ----
        log_odds += (hedging_density - 0.1) * 2.0

        prob = 1.0 / (1.0 + math.exp(-log_odds))
        return round(max(0.0, min(1.0, prob)), 4)


# ---------------------------------------------------------------------------
# StyleFingerprinter
# ---------------------------------------------------------------------------

@dataclass
class StyleFeatures:
    """Stylometric features extracted from a text document."""
    avg_sentence_length: float         # mean words per sentence
    sentence_length_std: float         # std-dev of words per sentence
    ttr: float                         # type-token ratio (200-word window)
    vocab_richness: float              # unique words / sqrt(total words) (Yule K approximation)
    function_word_ratio: float         # function words / total words
    avg_word_length: float             # mean chars per word
    comma_density: float               # commas per sentence
    exclamation_density: float         # exclamation marks per sentence
    passive_voice_ratio: float         # proxy: "is/are/was/were + past participle"
    hedge_density: float               # hedging words per sentence


@dataclass
class FingerprintResult:
    """Result of comparing a document's style to a reference corpus."""
    features: StyleFeatures
    mahalanobis_distance: float        # distance from "human writing" centroid
    llm_probability: float             # derived from Mahalanobis distance
    is_llm_style: bool
    nearest_class: str                 # "human" or "llm"
    feature_vector: list[float] = field(default_factory=list)


# Common English function words
_FUNCTION_WORDS: frozenset[str] = frozenset([
    "the", "a", "an", "in", "on", "at", "of", "to", "for", "with",
    "by", "from", "up", "about", "into", "through", "during", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "this", "that", "these", "those", "it",
    "he", "she", "they", "we", "you", "i", "and", "or", "but", "not",
    "as", "if", "so", "yet", "both", "either", "neither", "while",
])

# Hedging words
_HEDGE_WORDS: frozenset[str] = frozenset([
    "approximately", "generally", "typically", "usually", "often",
    "sometimes", "possibly", "potentially", "ideally", "preferably",
    "ideally", "perhaps", "mainly", "largely", "broadly", "relatively",
    "somewhat", "fairly", "quite", "rather", "may", "might", "could",
    "should", "would", "tend", "tends", "likely", "unlikely",
])

# Passive-voice proxy: "is/are/was/were" followed by an -ed/-en word
_PASSIVE_RE = re.compile(
    r"\b(is|are|was|were)\s+\w*(?:ed|en)\b", re.IGNORECASE
)


class StyleFingerprinter:
    """Extract stylometric features and compare to reference corpora.

    The fingerprinter maintains two reference centroids:
    - ``human_centroid``: mean feature vector for known human-written postings
    - ``llm_centroid``: mean feature vector for known LLM-written postings

    Mahalanobis distance from the human centroid is used as the LLM
    probability proxy.  When no labelled corpus has been loaded, the
    fingerprinter uses built-in default centroids estimated from research.
    """

    # Default feature centroids (10-dimensional feature vector):
    # [avg_sent_len, sent_len_std, ttr, vocab_richness, fn_word_ratio,
    #  avg_word_len, comma_density, exclamation_density, passive_ratio, hedge_density]

    _DEFAULT_HUMAN_CENTROID: list[float] = [
        14.0,   # avg_sentence_length (words)
        6.5,    # sentence_length_std (high variance = human)
        0.52,   # ttr
        6.0,    # vocab_richness
        0.48,   # function_word_ratio
        5.1,    # avg_word_length (chars)
        1.8,    # comma_density
        0.05,   # exclamation_density
        0.08,   # passive_voice_ratio
        0.12,   # hedge_density
    ]

    _DEFAULT_LLM_CENTROID: list[float] = [
        18.0,   # longer, more verbose sentences
        3.0,    # low std-dev (uniform)
        0.36,   # lower TTR (repetitive vocab)
        4.5,    # lower richness
        0.44,   # slightly lower function word ratio
        5.6,    # slightly longer words (formal register)
        2.4,    # more commas (complex syntax)
        0.03,   # fewer exclamations
        0.12,   # more passive voice
        0.28,   # high hedging density
    ]

    # Approximate diagonal of covariance matrix (for simplified Mahalanobis)
    _DEFAULT_VARIANCE: list[float] = [
        25.0, 10.0, 0.02, 4.0, 0.01, 0.5, 1.0, 0.01, 0.005, 0.02
    ]

    # Mahalanobis threshold (above = LLM-like)
    DEFAULT_THRESHOLD: float = 3.0

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._human_centroid: list[float] = list(self._DEFAULT_HUMAN_CENTROID)
        self._llm_centroid: list[float] = list(self._DEFAULT_LLM_CENTROID)
        self._variance: list[float] = list(self._DEFAULT_VARIANCE)
        self._corpus_human: list[list[float]] = []
        self._corpus_llm: list[list[float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, text: str) -> StyleFeatures:
        """Extract stylometric features from *text*."""
        words = _tokenize_words(text)
        sentences = _split_sentences(text)

        # Sentence length stats
        sent_lengths = [len(s.split()) for s in sentences if s.split()]
        avg_sent_len = statistics.mean(sent_lengths) if sent_lengths else 0.0
        sent_len_std = statistics.stdev(sent_lengths) if len(sent_lengths) >= 2 else 0.0

        # TTR
        window = words[:200]
        ttr = len(set(window)) / len(window) if window else 0.0

        # Vocabulary richness: unique / sqrt(total) (Yule K proxy)
        vocab_richness = (
            len(set(words)) / math.sqrt(len(words)) if words else 0.0
        )

        # Function word ratio
        fn_count = sum(1 for w in words if w in _FUNCTION_WORDS)
        fn_ratio = fn_count / len(words) if words else 0.0

        # Average word length
        char_lengths = [len(w) for w in words]
        avg_word_len = statistics.mean(char_lengths) if char_lengths else 0.0

        # Punctuation densities
        n_sents = max(1, len(sentences))
        comma_density = text.count(",") / n_sents
        excl_density = text.count("!") / n_sents

        # Passive voice proxy
        passive_matches = len(_PASSIVE_RE.findall(text))
        passive_ratio = passive_matches / n_sents

        # Hedge density
        hedge_count = sum(1 for w in words if w in _HEDGE_WORDS)
        hedge_density = hedge_count / n_sents

        return StyleFeatures(
            avg_sentence_length=round(avg_sent_len, 3),
            sentence_length_std=round(sent_len_std, 3),
            ttr=round(ttr, 4),
            vocab_richness=round(vocab_richness, 3),
            function_word_ratio=round(fn_ratio, 4),
            avg_word_length=round(avg_word_len, 3),
            comma_density=round(comma_density, 3),
            exclamation_density=round(excl_density, 4),
            passive_voice_ratio=round(passive_ratio, 4),
            hedge_density=round(hedge_density, 4),
        )

    def fingerprint(self, text: str) -> FingerprintResult:
        """Compare *text* style to reference corpora.

        Returns FingerprintResult with Mahalanobis distance and LLM probability.
        """
        features = self.extract_features(text)
        vec = self._to_vector(features)

        dist_human = self._mahalanobis(vec, self._human_centroid, self._variance)
        dist_llm = self._mahalanobis(vec, self._llm_centroid, self._variance)

        # LLM probability: proportion of distance attributable to LLM centroid
        total_dist = dist_human + dist_llm
        if total_dist == 0.0:
            llm_prob = 0.5
        else:
            # Closer to LLM centroid → higher probability
            llm_prob = 1.0 - (dist_llm / total_dist)

        llm_prob = round(max(0.0, min(1.0, llm_prob)), 4)
        is_llm = dist_human > self.threshold or llm_prob > 0.55
        nearest = "llm" if dist_llm < dist_human else "human"

        return FingerprintResult(
            features=features,
            mahalanobis_distance=round(dist_human, 4),
            llm_probability=llm_prob,
            is_llm_style=is_llm,
            nearest_class=nearest,
            feature_vector=vec,
        )

    def add_to_corpus(self, text: str, label: str) -> None:
        """Add a labelled example to the corpus for centroid updating.

        Args:
            text:  raw text of the job posting.
            label: "human" or "llm".
        """
        features = self.extract_features(text)
        vec = self._to_vector(features)
        if label == "human":
            self._corpus_human.append(vec)
            self._human_centroid = self._mean_vectors(self._corpus_human)
        elif label == "llm":
            self._corpus_llm.append(vec)
            self._llm_centroid = self._mean_vectors(self._corpus_llm)
        # Recompute variance from union of both corpora
        all_vecs = self._corpus_human + self._corpus_llm
        if len(all_vecs) >= 2:
            self._variance = self._compute_variance(all_vecs)

    def corpus_size(self) -> dict[str, int]:
        return {"human": len(self._corpus_human), "llm": len(self._corpus_llm)}

    def get_centroid(self, label: str) -> list[float]:
        """Return the current centroid for *label* ('human' or 'llm')."""
        if label == "human":
            return list(self._human_centroid)
        if label == "llm":
            return list(self._llm_centroid)
        raise ValueError(f"Unknown label '{label}'. Use 'human' or 'llm'.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _to_vector(self, f: StyleFeatures) -> list[float]:
        return [
            f.avg_sentence_length,
            f.sentence_length_std,
            f.ttr,
            f.vocab_richness,
            f.function_word_ratio,
            f.avg_word_length,
            f.comma_density,
            f.exclamation_density,
            f.passive_voice_ratio,
            f.hedge_density,
        ]

    def _mahalanobis(
        self, vec: list[float], centroid: list[float], variance: list[float]
    ) -> float:
        """Simplified Mahalanobis distance using diagonal covariance."""
        dist_sq = sum(
            ((v - c) ** 2) / max(var, 1e-9)
            for v, c, var in zip(vec, centroid, variance, strict=False)
        )
        return math.sqrt(dist_sq)

    def _mean_vectors(self, vecs: list[list[float]]) -> list[float]:
        if not vecs:
            return list(self._DEFAULT_HUMAN_CENTROID)
        n = len(vecs)
        return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]

    def _compute_variance(self, vecs: list[list[float]]) -> list[float]:
        if len(vecs) < 2:
            return list(self._DEFAULT_VARIANCE)
        n_dims = len(vecs[0])
        result = []
        for i in range(n_dims):
            col = [v[i] for v in vecs]
            var = statistics.variance(col) if len(col) >= 2 else 1.0
            result.append(max(var, 1e-9))
        return result
