"""Stylometric Operator Fingerprinting for job scam detection.

Links multiple scam postings to the same operator by analyzing writing style,
even across different company names and accounts. Scammers reuse writing
patterns even when they change identities.

Design constraints:
- stdlib only: re, math, statistics, collections
- @dataclass for all data classes
- No numpy / scipy / pandas
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Common English function words
_FUNCTION_WORDS: list[str] = [
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "of", "to", "from", "up", "and", "but",
    "or", "nor", "so", "yet", "this", "that", "these", "those", "it",
    "he", "she", "they", "we", "you", "i", "not", "no", "also", "just",
]

# Punctuation marks tracked
_TRACKED_PUNCT: list[str] = [".", ",", ";", ":", "!", "?", "-", "(", ")"]

# English contractions
_CONTRACTION_RE = re.compile(
    r"\b(i'm|you're|he's|she's|it's|we're|they're|"
    r"i've|you've|we've|they've|"
    r"i'd|you'd|he'd|she'd|we'd|they'd|"
    r"i'll|you'll|he'll|she'll|we'll|they'll|"
    r"isn't|aren't|wasn't|weren't|"
    r"haven't|hasn't|hadn't|"
    r"won't|wouldn't|can't|couldn't|shouldn't|"
    r"don't|doesn't|didn't|"
    r"there's|that's|what's|here's|who's|"
    r"let's|that'll|there'll)\b",
    re.IGNORECASE,
)

# Simple passive voice heuristic: was/were/been/is/are/being + past-participle-like word
# We detect "was/were/been + word ending in -ed/-en/-t"
_PASSIVE_RE = re.compile(
    r"\b(was|were|been|is|are|being)\s+\w+(?:ed|en|t)\b",
    re.IGNORECASE,
)

# Sentence boundary: ends with .!? followed by space+capital or end of string
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$")

# Word tokenizer — only alphabetic tokens
_WORD_RE = re.compile(r"[a-zA-Z]+")

# Variable slot markers used inside templates
_SLOT_TOKEN = "{{SLOT}}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize_words(text: str) -> list[str]:
    """Return lowercase alphabetic word tokens."""
    return _WORD_RE.findall(text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    # Normalize newlines to spaces first
    flat = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", flat)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines."""
    paras = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in paras if p.strip()]


def _count_syllables(word: str) -> int:
    """Rough syllable counter for English words (Flesch-Kincaid)."""
    word = word.lower().rstrip("e")
    count = len(re.findall(r"[aeiou]+", word))
    return max(1, count)


def _flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level estimate (0-18)."""
    words = _tokenize_words(text)
    sentences = _split_sentences(text)
    if not words or not sentences:
        return 0.0
    n_words = len(words)
    n_sentences = len(sentences)
    n_syllables = sum(_count_syllables(w) for w in words)
    asl = n_words / n_sentences          # average sentence length
    asw = n_syllables / n_words          # average syllables per word
    grade = 0.39 * asl + 11.8 * asw - 15.59
    return max(0.0, round(grade, 2))


# ---------------------------------------------------------------------------
# StyleFingerprint dataclass
# ---------------------------------------------------------------------------

@dataclass
class StyleFingerprint:
    """Writing-style fingerprint extracted from a text document.

    All float values are in [0, ∞) unless noted. Ratios are in [0, 1].
    """
    # Sentence-level
    avg_sentence_length: float = 0.0       # words per sentence
    sentence_length_std: float = 0.0       # std-dev of sentence lengths

    # Word-level
    avg_word_length: float = 0.0           # characters per word

    # Lexical diversity
    vocabulary_richness: float = 0.0       # type-token ratio over first 200 words

    # Function words
    function_word_ratios: dict[str, float] = field(default_factory=dict)

    # Punctuation
    punctuation_ratios: dict[str, float] = field(default_factory=dict)

    # Paragraph structure
    paragraph_count: int = 0
    avg_paragraph_length: float = 0.0      # words per paragraph

    # Character-level style
    capitalization_rate: float = 0.0       # fraction of letters that are uppercase

    # Contraction and syntactic features
    contraction_usage_rate: float = 0.0    # contractions per 100 words
    passive_voice_ratio: float = 0.0       # passive clauses / total sentences

    # Readability
    readability_score: float = 0.0         # Flesch-Kincaid grade level

    # Metadata
    word_count: int = 0
    is_short: bool = False                 # True if < 50 words (low confidence)

    def to_vector(self) -> list[float]:
        """Flatten to a numeric vector for distance computation.

        Order: [avg_sentence_length, sentence_length_std, avg_word_length,
                vocabulary_richness, *function_word_ratios.values(),
                *punctuation_ratios.values(), capitalization_rate,
                contraction_usage_rate, passive_voice_ratio, readability_score]
        """
        fw_keys = sorted(self.function_word_ratios.keys())
        pu_keys = sorted(self.punctuation_ratios.keys())
        vec: list[float] = [
            self.avg_sentence_length,
            self.sentence_length_std,
            self.avg_word_length,
            self.vocabulary_richness,
        ]
        vec += [self.function_word_ratios.get(k, 0.0) for k in fw_keys]
        vec += [self.punctuation_ratios.get(k, 0.0) for k in pu_keys]
        vec += [
            self.capitalization_rate,
            self.contraction_usage_rate,
            self.passive_voice_ratio,
            self.readability_score,
        ]
        return vec


# ---------------------------------------------------------------------------
# StyleExtractor
# ---------------------------------------------------------------------------

class StyleExtractor:
    """Extract a StyleFingerprint from arbitrary text.

    Usage::

        extractor = StyleExtractor()
        fp = extractor.extract("We are hiring... apply now!")
    """

    _SHORT_THRESHOLD = 50   # words

    def extract(self, text: str) -> StyleFingerprint:
        """Return a StyleFingerprint for *text*. Handles empty/short text."""
        if not text or not text.strip():
            return StyleFingerprint(is_short=True)

        words = _tokenize_words(text)
        word_count = len(words)
        is_short = word_count < self._SHORT_THRESHOLD

        sentences = _split_sentences(text)
        paragraphs = _split_paragraphs(text)

        # Sentence lengths (in words)
        sent_lengths = self._sentence_word_counts(sentences)
        avg_sl = statistics.mean(sent_lengths) if sent_lengths else 0.0
        std_sl = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0.0

        # Average word length
        avg_wl = (
            sum(len(w) for w in words) / word_count if word_count else 0.0
        )

        # Vocabulary richness: TTR on first 200 words
        sample = words[:200]
        vocab_richness = len(set(sample)) / len(sample) if sample else 0.0

        # Function word ratios
        fw_ratios = self._function_word_ratios(words)

        # Punctuation ratios (per 100 characters)
        pu_ratios = self._punctuation_ratios(text)

        # Paragraph stats
        para_count = len(paragraphs)
        para_word_counts = [len(_tokenize_words(p)) for p in paragraphs]
        avg_pl = statistics.mean(para_word_counts) if para_word_counts else 0.0

        # Capitalization rate: uppercase letters / total letters
        letters = [c for c in text if c.isalpha()]
        cap_rate = (
            sum(1 for c in letters if c.isupper()) / len(letters)
            if letters else 0.0
        )

        # Contraction usage rate (per 100 words)
        contractions = _CONTRACTION_RE.findall(text)
        contraction_rate = (len(contractions) / word_count * 100) if word_count else 0.0

        # Passive voice ratio
        passive_matches = _PASSIVE_RE.findall(text)
        passive_ratio = (
            len(passive_matches) / len(sentences) if sentences else 0.0
        )

        # Readability
        readability = _flesch_kincaid_grade(text)

        return StyleFingerprint(
            avg_sentence_length=round(avg_sl, 3),
            sentence_length_std=round(std_sl, 3),
            avg_word_length=round(avg_wl, 3),
            vocabulary_richness=round(vocab_richness, 3),
            function_word_ratios=fw_ratios,
            punctuation_ratios=pu_ratios,
            paragraph_count=para_count,
            avg_paragraph_length=round(avg_pl, 3),
            capitalization_rate=round(cap_rate, 3),
            contraction_usage_rate=round(contraction_rate, 3),
            passive_voice_ratio=round(passive_ratio, 3),
            readability_score=readability,
            word_count=word_count,
            is_short=is_short,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sentence_word_counts(sentences: list[str]) -> list[int]:
        counts = []
        for s in sentences:
            wc = len(_tokenize_words(s))
            if wc > 0:
                counts.append(wc)
        return counts or [0]

    @staticmethod
    def _function_word_ratios(words: list[str]) -> dict[str, float]:
        n = len(words)
        if n == 0:
            return {w: 0.0 for w in _FUNCTION_WORDS}
        counts = Counter(words)
        return {fw: round(counts[fw] / n, 6) for fw in _FUNCTION_WORDS}

    @staticmethod
    def _punctuation_ratios(text: str) -> dict[str, float]:
        n_chars = len(text)
        if n_chars == 0:
            return {p: 0.0 for p in _TRACKED_PUNCT}
        counts = Counter(text)
        return {p: round(counts[p] / n_chars, 6) for p in _TRACKED_PUNCT}


# ---------------------------------------------------------------------------
# OperatorLinker
# ---------------------------------------------------------------------------

@dataclass
class OperatorProfile:
    """A known scam operator's style profile."""
    operator_id: str
    fingerprints: list[StyleFingerprint] = field(default_factory=list)
    posting_ids: list[str] = field(default_factory=list)
    # Optional: centroid fingerprint (recomputed on update)
    centroid: StyleFingerprint | None = None

    def update_centroid(self) -> None:
        """Recompute centroid as mean over all collected fingerprints."""
        if not self.fingerprints:
            self.centroid = None
            return
        self.centroid = _mean_fingerprint(self.fingerprints)


@dataclass
class LinkResult:
    """Result of linking a posting to a known operator."""
    posting_id: str
    operator_id: str | None   # None = no match found
    distance: float           # lower = more similar
    confidence: float         # 0–1
    is_match: bool


def _feature_weights() -> list[float]:
    """Feature importance weights for distance computation.

    Ordered to match StyleFingerprint.to_vector():
    [avg_sentence_length, sentence_length_std, avg_word_length,
     vocabulary_richness, *function_words (sorted), *punct (sorted),
     capitalization_rate, contraction_usage_rate, passive_voice_ratio,
     readability_score]
    """
    n_fw = len(_FUNCTION_WORDS)
    n_pu = len(_TRACKED_PUNCT)
    weights: list[float] = [
        2.0,   # avg_sentence_length — strong stylometric signal
        1.5,   # sentence_length_std
        1.5,   # avg_word_length
        2.0,   # vocabulary_richness
    ]
    # function words — moderate weight each
    weights += [1.0] * n_fw
    # punctuation — lower weight each
    weights += [0.5] * n_pu
    weights += [
        1.5,   # capitalization_rate
        1.5,   # contraction_usage_rate
        1.0,   # passive_voice_ratio
        1.0,   # readability_score
    ]
    return weights


def _weighted_euclidean(v1: list[float], v2: list[float], weights: list[float]) -> float:
    """Weighted Euclidean distance between two feature vectors."""
    if len(v1) != len(v2) or len(v1) != len(weights):
        # Vectors have different lengths; use pairwise minimum length
        min_len = min(len(v1), len(v2), len(weights))
        v1, v2, weights = v1[:min_len], v2[:min_len], weights[:min_len]
    total = sum(w * (a - b) ** 2 for w, a, b in zip(weights, v1, v2, strict=False))
    return math.sqrt(total)


def _mean_fingerprint(fps: list[StyleFingerprint]) -> StyleFingerprint:
    """Compute element-wise mean of a list of fingerprints."""
    n = len(fps)
    if n == 0:
        return StyleFingerprint()
    if n == 1:
        return fps[0]

    # Build per-field means
    avg_sl = sum(f.avg_sentence_length for f in fps) / n
    std_sl = sum(f.sentence_length_std for f in fps) / n
    avg_wl = sum(f.avg_word_length for f in fps) / n
    vr = sum(f.vocabulary_richness for f in fps) / n
    cap = sum(f.capitalization_rate for f in fps) / n
    cont = sum(f.contraction_usage_rate for f in fps) / n
    pv = sum(f.passive_voice_ratio for f in fps) / n
    rs = sum(f.readability_score for f in fps) / n
    wc = sum(f.word_count for f in fps) // n
    para_c = sum(f.paragraph_count for f in fps) // n
    avg_pl = sum(f.avg_paragraph_length for f in fps) / n

    # function words
    fw: dict[str, float] = {}
    for w in _FUNCTION_WORDS:
        fw[w] = sum(f.function_word_ratios.get(w, 0.0) for f in fps) / n

    # punctuation
    pu: dict[str, float] = {}
    for p in _TRACKED_PUNCT:
        pu[p] = sum(f.punctuation_ratios.get(p, 0.0) for f in fps) / n

    return StyleFingerprint(
        avg_sentence_length=round(avg_sl, 3),
        sentence_length_std=round(std_sl, 3),
        avg_word_length=round(avg_wl, 3),
        vocabulary_richness=round(vr, 3),
        function_word_ratios=fw,
        punctuation_ratios=pu,
        paragraph_count=para_c,
        avg_paragraph_length=round(avg_pl, 3),
        capitalization_rate=round(cap, 3),
        contraction_usage_rate=round(cont, 3),
        passive_voice_ratio=round(pv, 3),
        readability_score=round(rs, 2),
        word_count=wc,
        is_short=wc < StyleExtractor._SHORT_THRESHOLD,
    )


class OperatorLinker:
    """Maintain a library of known operator fingerprints.

    Usage::

        linker = OperatorLinker()
        linker.add_fingerprint("op_001", "posting_A", fingerprint_A)
        result = linker.link("posting_B", fingerprint_B)
    """

    # Distance below which two fingerprints are considered the same operator
    MATCH_THRESHOLD: float = 3.0
    # Distance above which we are confident there is NO match
    NO_MATCH_THRESHOLD: float = 8.0

    def __init__(self, match_threshold: float | None = None) -> None:
        self._operators: dict[str, OperatorProfile] = {}
        self._weights = _feature_weights()
        if match_threshold is not None:
            self.MATCH_THRESHOLD = match_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fingerprint(
        self,
        operator_id: str,
        posting_id: str,
        fp: StyleFingerprint,
    ) -> None:
        """Add a fingerprint to the operator's profile."""
        if operator_id not in self._operators:
            self._operators[operator_id] = OperatorProfile(
                operator_id=operator_id
            )
        op = self._operators[operator_id]
        op.fingerprints.append(fp)
        op.posting_ids.append(posting_id)
        op.update_centroid()

    def link(
        self,
        posting_id: str,
        fp: StyleFingerprint,
    ) -> LinkResult:
        """Compare *fp* against all known operators; return best match."""
        if not self._operators:
            return LinkResult(
                posting_id=posting_id,
                operator_id=None,
                distance=float("inf"),
                confidence=0.0,
                is_match=False,
            )

        best_op: str | None = None
        best_dist = float("inf")
        v = fp.to_vector()

        for op_id, profile in self._operators.items():
            if profile.centroid is None:
                continue
            d = _weighted_euclidean(v, profile.centroid.to_vector(), self._weights)
            if d < best_dist:
                best_dist = d
                best_op = op_id

        is_match = best_dist < self.MATCH_THRESHOLD
        confidence = self._confidence(best_dist)

        return LinkResult(
            posting_id=posting_id,
            operator_id=best_op if is_match else None,
            distance=round(best_dist, 4),
            confidence=round(confidence, 4),
            is_match=is_match,
        )

    def cluster(
        self,
        postings: dict[str, StyleFingerprint],
    ) -> dict[str, list[str]]:
        """Agglomerative clustering of postings into operator groups.

        Returns a dict mapping cluster_id -> list of posting_ids.
        Uses complete-linkage agglomeration with MATCH_THRESHOLD.
        """
        posting_ids = list(postings.keys())
        n = len(posting_ids)
        if n == 0:
            return {}

        # Each posting starts in its own cluster
        clusters: list[list[str]] = [[pid] for pid in posting_ids]

        def cluster_distance(c1: list[str], c2: list[str]) -> float:
            """Complete-linkage distance between two clusters."""
            max_d = 0.0
            for a in c1:
                for b in c2:
                    d = _weighted_euclidean(
                        postings[a].to_vector(),
                        postings[b].to_vector(),
                        self._weights,
                    )
                    if d > max_d:
                        max_d = d
            return max_d

        changed = True
        while changed and len(clusters) > 1:
            changed = False
            best_i, best_j = -1, -1
            best_d = float("inf")
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = cluster_distance(clusters[i], clusters[j])
                    if d < best_d:
                        best_d = d
                        best_i, best_j = i, j
            if best_d < self.MATCH_THRESHOLD:
                # Merge
                clusters[best_i] = clusters[best_i] + clusters[best_j]
                clusters.pop(best_j)
                changed = True

        return {f"cluster_{i}": c for i, c in enumerate(clusters)}

    def known_operators(self) -> list[str]:
        return list(self._operators.keys())

    def get_profile(self, operator_id: str) -> OperatorProfile | None:
        return self._operators.get(operator_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _confidence(self, distance: float) -> float:
        """Convert distance to a [0, 1] confidence score."""
        if distance >= self.NO_MATCH_THRESHOLD:
            return 0.0
        # Sigmoid-like decay
        ratio = distance / self.NO_MATCH_THRESHOLD
        return round(1.0 - ratio, 4)


# ---------------------------------------------------------------------------
# TemplateDetector
# ---------------------------------------------------------------------------

@dataclass
class TemplateMatch:
    """Two postings that appear to share a template."""
    posting_id_a: str
    posting_id_b: str
    similarity: float          # 0-1; 1 = identical
    variable_slots: list[str]  # tokens that differ between the two
    is_template: bool


@dataclass
class TemplateFamilyRecord:
    """A family of postings sharing a common template."""
    family_id: str
    template_text: str          # canonical template with {{SLOT}} markers
    member_ids: list[str] = field(default_factory=list)
    slot_values: dict[str, list[str]] = field(default_factory=dict)
    spread_count: int = 0       # how many distinct postings


class TemplateDetector:
    """Detect when multiple postings use the same fill-in-the-blank template.

    Usage::

        detector = TemplateDetector()
        detector.add_posting("posting_1", text_1)
        detector.add_posting("posting_2", text_2)
        match = detector.compare("posting_1", "posting_2")
    """

    TEMPLATE_SIMILARITY_THRESHOLD: float = 0.75

    def __init__(self, threshold: float | None = None) -> None:
        self._postings: dict[str, str] = {}
        self._families: dict[str, TemplateFamilyRecord] = {}
        if threshold is not None:
            self.TEMPLATE_SIMILARITY_THRESHOLD = threshold

    def add_posting(self, posting_id: str, text: str) -> None:
        """Register a posting text."""
        self._postings[posting_id] = text

    def compare(self, id_a: str, id_b: str) -> TemplateMatch:
        """Compare two registered postings for template similarity."""
        text_a = self._postings.get(id_a, "")
        text_b = self._postings.get(id_b, "")
        return self.compare_texts(id_a, id_b, text_a, text_b)

    def compare_texts(
        self,
        id_a: str,
        id_b: str,
        text_a: str,
        text_b: str,
    ) -> TemplateMatch:
        """Compare two raw texts directly."""
        tokens_a = _tokenize_words(text_a)
        tokens_b = _tokenize_words(text_b)
        similarity = self._token_similarity(tokens_a, tokens_b)
        slots = self._find_slots(text_a, text_b)
        is_tmpl = similarity >= self.TEMPLATE_SIMILARITY_THRESHOLD
        return TemplateMatch(
            posting_id_a=id_a,
            posting_id_b=id_b,
            similarity=round(similarity, 4),
            variable_slots=slots,
            is_template=is_tmpl,
        )

    def detect_template_families(self) -> list[TemplateFamilyRecord]:
        """Group all registered postings into template families."""
        ids = list(self._postings.keys())
        # Build similarity graph
        edges: list[tuple[str, str]] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                m = self.compare(ids[i], ids[j])
                if m.is_template:
                    edges.append((ids[i], ids[j]))

        # Union-Find
        parent = {pid: pid for pid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for a, b in edges:
            union(a, b)

        groups: dict[str, list[str]] = defaultdict(list)
        for pid in ids:
            groups[find(pid)].append(pid)

        families = []
        for root, members in groups.items():
            if len(members) < 2:
                continue
            canonical = self._build_template(members)
            family = TemplateFamilyRecord(
                family_id=f"family_{root}",
                template_text=canonical,
                member_ids=members,
                spread_count=len(members),
            )
            families.append(family)
        return families

    def extract_slot_values(self, template: str, text: str) -> dict[str, str]:
        """Given a template with {{SLOT}} markers, extract the slot values
        from a concrete posting text. Returns slot_index -> value mapping."""
        # Simple: split on {{SLOT}} and find matching portions
        parts = template.split(_SLOT_TOKEN)
        values: dict[str, str] = {}
        remaining = text
        for i, part in enumerate(parts[:-1]):
            idx = remaining.find(part)
            if idx == -1:
                break
            remaining = remaining[idx + len(part):]
            # The next slot value ends where the next template part starts
            next_part = parts[i + 1]
            end = remaining.find(next_part) if next_part else len(remaining)
            if end == -1:
                end = len(remaining)
            values[f"slot_{i}"] = remaining[:end].strip()
            remaining = remaining[end:]
        return values

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_similarity(a: list[str], b: list[str]) -> float:
        """Jaccard similarity on bigrams of token sequences."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        # Use bigrams for more structural comparison
        bigrams_a = set(zip(a, a[1:], strict=False)) if len(a) > 1 else set(zip(a, a, strict=False))
        bigrams_b = set(zip(b, b[1:], strict=False)) if len(b) > 1 else set(zip(b, b, strict=False))
        intersection = len(bigrams_a & bigrams_b)
        union = len(bigrams_a | bigrams_b)
        return intersection / union if union else 0.0

    @staticmethod
    def _find_slots(text_a: str, text_b: str) -> list[str]:
        """Find words present in one text but not the other (likely filled slots)."""
        words_a = set(_tokenize_words(text_a))
        words_b = set(_tokenize_words(text_b))
        # Symmetric difference = changed tokens
        diff = (words_a - words_b) | (words_b - words_a)
        # Filter out very common words
        common = {w for w in _FUNCTION_WORDS}
        return sorted(diff - common)[:20]  # cap at 20

    def _build_template(self, member_ids: list[str]) -> str:
        """Build a canonical template by replacing differing tokens with {{SLOT}}."""
        if not member_ids:
            return ""
        texts = [self._postings[pid] for pid in member_ids if pid in self._postings]
        if not texts:
            return ""
        # Use first text as base; replace words that differ across all texts
        base_words = _tokenize_words(texts[0])
        result_words = []
        for i, word in enumerate(base_words):
            same_everywhere = all(
                i < len(_tokenize_words(t)) and _tokenize_words(t)[i] == word
                for t in texts[1:]
            )
            result_words.append(word if same_everywhere else _SLOT_TOKEN)
        return " ".join(result_words)


# ---------------------------------------------------------------------------
# WritingEvolutionTracker
# ---------------------------------------------------------------------------

@dataclass
class StyleSnapshot:
    """A fingerprint at a specific point in time."""
    timestamp: str              # ISO date or datetime string
    fingerprint: StyleFingerprint
    posting_id: str = ""


@dataclass
class StyleShift:
    """A detected shift in writing style between two snapshots."""
    from_timestamp: str
    to_timestamp: str
    magnitude: float            # distance between fingerprints
    is_significant: bool        # True if > threshold
    possible_ai_assisted: bool  # True if sudden vocabulary richness spike
    description: str = ""


@dataclass
class IntraDocInconsistency:
    """Style inconsistency within a single document (copy-paste detection)."""
    posting_id: str
    segment_a_index: int
    segment_b_index: int
    distance: float
    is_inconsistent: bool
    description: str = ""


class WritingEvolutionTracker:
    """Track how a single operator's writing changes over time.

    Usage::

        tracker = WritingEvolutionTracker("op_001")
        tracker.record(snapshot_t1)
        tracker.record(snapshot_t2)
        shifts = tracker.detect_shifts()
    """

    # A shift is significant if the distance exceeds this value
    SHIFT_THRESHOLD: float = 2.0

    # Vocabulary richness increase that suggests AI assistance
    AI_VOCAB_SPIKE: float = 0.15

    # Intra-document inconsistency threshold
    INTRA_DOC_THRESHOLD: float = 2.5

    def __init__(self, operator_id: str) -> None:
        self.operator_id = operator_id
        self._snapshots: list[StyleSnapshot] = []
        self._weights = _feature_weights()

    def record(self, snapshot: StyleSnapshot) -> None:
        """Add a new style snapshot (must be in chronological order)."""
        self._snapshots.append(snapshot)

    def detect_shifts(self) -> list[StyleShift]:
        """Detect significant style shifts between consecutive snapshots."""
        shifts: list[StyleShift] = []
        snaps = self._snapshots
        for i in range(1, len(snaps)):
            prev = snaps[i - 1]
            curr = snaps[i]
            d = _weighted_euclidean(
                prev.fingerprint.to_vector(),
                curr.fingerprint.to_vector(),
                self._weights,
            )
            is_sig = d >= self.SHIFT_THRESHOLD
            ai_flag = self._check_ai_assistance(prev.fingerprint, curr.fingerprint)
            desc = self._describe_shift(prev.fingerprint, curr.fingerprint, d, ai_flag)
            shifts.append(StyleShift(
                from_timestamp=prev.timestamp,
                to_timestamp=curr.timestamp,
                magnitude=round(d, 4),
                is_significant=is_sig,
                possible_ai_assisted=ai_flag,
                description=desc,
            ))
        return shifts

    def detect_ai_assistance(self) -> list[StyleSnapshot]:
        """Return snapshots where AI assistance is suspected."""
        flagged: list[StyleSnapshot] = []
        snaps = self._snapshots
        for i in range(1, len(snaps)):
            prev = snaps[i - 1]
            curr = snaps[i]
            if self._check_ai_assistance(prev.fingerprint, curr.fingerprint):
                flagged.append(curr)
        return flagged

    def detect_intra_doc_inconsistencies(
        self,
        posting_id: str,
        text: str,
        extractor: StyleExtractor | None = None,
    ) -> list[IntraDocInconsistency]:
        """Find style inconsistencies within a single document.

        Splits the document into paragraphs and compares adjacent pairs.
        """
        if extractor is None:
            extractor = StyleExtractor()
        paras = _split_paragraphs(text)
        inconsistencies: list[IntraDocInconsistency] = []

        for i in range(len(paras) - 1):
            fp_a = extractor.extract(paras[i])
            fp_b = extractor.extract(paras[i + 1])
            if fp_a.is_short or fp_b.is_short:
                continue
            d = _weighted_euclidean(
                fp_a.to_vector(), fp_b.to_vector(), self._weights
            )
            is_incons = d >= self.INTRA_DOC_THRESHOLD
            desc = ""
            if is_incons:
                desc = (
                    f"Paragraphs {i} and {i + 1} have divergent style "
                    f"(distance={d:.2f}), suggesting copy-paste from different sources."
                )
            inconsistencies.append(IntraDocInconsistency(
                posting_id=posting_id,
                segment_a_index=i,
                segment_b_index=i + 1,
                distance=round(d, 4),
                is_inconsistent=is_incons,
                description=desc,
            ))
        return inconsistencies

    def style_trajectory(self) -> list[dict]:
        """Return chronological style metrics for trend analysis."""
        return [
            {
                "timestamp": s.timestamp,
                "posting_id": s.posting_id,
                "vocab_richness": s.fingerprint.vocabulary_richness,
                "avg_sentence_length": s.fingerprint.avg_sentence_length,
                "readability_score": s.fingerprint.readability_score,
                "capitalization_rate": s.fingerprint.capitalization_rate,
            }
            for s in self._snapshots
        ]

    def snapshot_count(self) -> int:
        return len(self._snapshots)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_ai_assistance(
        self,
        prev: StyleFingerprint,
        curr: StyleFingerprint,
    ) -> bool:
        """Heuristic: sudden jump in vocab richness or readability = possible AI."""
        vocab_jump = curr.vocabulary_richness - prev.vocabulary_richness
        readability_jump = curr.readability_score - prev.readability_score
        # AI tends to increase vocabulary richness and moderate readability
        return vocab_jump > self.AI_VOCAB_SPIKE or readability_jump > 3.0

    @staticmethod
    def _describe_shift(
        prev: StyleFingerprint,
        curr: StyleFingerprint,
        distance: float,
        ai_flag: bool,
    ) -> str:
        parts: list[str] = []
        if curr.avg_sentence_length > prev.avg_sentence_length + 5:
            parts.append("sentences became longer")
        elif prev.avg_sentence_length > curr.avg_sentence_length + 5:
            parts.append("sentences became shorter")
        if curr.vocabulary_richness > prev.vocabulary_richness + 0.1:
            parts.append("vocabulary richness increased")
        if ai_flag:
            parts.append("possible AI writing assistance")
        if not parts:
            parts.append("general style drift")
        return f"Distance {distance:.2f}: {', '.join(parts)}."
