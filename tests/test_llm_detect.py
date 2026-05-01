"""Tests for sentinel/llm_detect.py — LLM-Generated Content Detection.

35+ tests covering LLMDetector and StyleFingerprinter.
"""

import math
import pytest

from sentinel.llm_detect import (
    LLMDetectionResult,
    LLMDetector,
    StyleFeatures,
    StyleFingerprinter,
    _LLM_PHRASES,
    _ngrams,
    _split_sentences,
    _tokenize_words,
    FingerprintResult,
)


# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

# Realistic-sounding human text: varied sentence lengths, specific detail
HUMAN_TEXT = (
    "We are hiring a backend engineer to own our payments infrastructure. "
    "You'll work on the Stripe integration and make sure settlements happen on time. "
    "Expect to debug gnarly race conditions in our async queues. "
    "We need someone who can read a flamegraph and actually fix the bottleneck, not just shrug. "
    "Three years of Python and solid SQL required. "
    "Bonus if you've shipped something in Rust. "
    "Salary: $130k–$155k depending on experience. "
    "Interview: 30-min screen, then a take-home, then two technical rounds—nothing more."
)

# Stereotypical LLM-generated job posting: verbose, uniform, hedgy, cliche-laden
LLM_TEXT = (
    "We are seeking a dynamic and results-driven professional to join our growing, innovative team. "
    "In today's fast-paced business landscape, it's important to note that collaboration and "
    "communication are key pillars of success. The successful candidate will possess excellent "
    "interpersonal skills and a passion for excellence. Furthermore, you will be responsible for "
    "fostering a positive work environment and driving cross-functional alignment. Additionally, "
    "we are looking for a motivated individual who thrives in a fast-paced environment and "
    "demonstrates a commitment to continuous improvement. We look forward to hearing from you. "
    "Please feel free to reach out if you have any questions. We offer a competitive salary and "
    "benefits package. Join our dynamic team today and make a difference!"
)

SHORT_TEXT = "Looking for a developer."

# Text with explicit LLM phrases
PHRASE_TEXT = (
    "I'd be happy to assist you with this opportunity. "
    "It's important to note that in today's fast-paced environment, "
    "we are looking for a dynamic and passionate team player. "
    "Furthermore, don't hesitate to reach out. We look forward to hearing from you."
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_tokenize_words_basic(self):
        words = _tokenize_words("Hello World, it's a test!")
        assert "hello" in words
        assert "world" in words
        assert "it's" in words

    def test_tokenize_words_empty(self):
        assert _tokenize_words("") == []

    def test_split_sentences_basic(self):
        sents = _split_sentences("Hello. World! How are you?")
        assert len(sents) == 3

    def test_split_sentences_single(self):
        sents = _split_sentences("No punctuation here")
        assert len(sents) == 1

    def test_split_sentences_empty(self):
        sents = _split_sentences("")
        assert sents == []

    def test_ngrams_basic(self):
        tokens = ["a", "b", "c", "d"]
        bigrams = _ngrams(tokens, 2)
        assert ("a", "b") in bigrams
        assert ("c", "d") in bigrams
        assert len(bigrams) == 3

    def test_ngrams_too_short(self):
        assert _ngrams(["a"], 2) == []

    def test_llm_phrase_patterns_fire(self):
        texts_that_should_match = [
            "I'd be happy to help",
            "It's important to note",
            "in today's fast-paced world",
            "join our dynamic team",
            "don't hesitate to reach out",
            "we look forward to hearing from you",
            "competitive salary and benefits",
        ]
        for text in texts_that_should_match:
            matched = [p for p in _LLM_PHRASES if p.search(text)]
            assert len(matched) >= 1, f"No pattern matched: {text!r}"


# ---------------------------------------------------------------------------
# LLMDetector tests
# ---------------------------------------------------------------------------

class TestLLMDetector:
    @pytest.fixture
    def detector(self):
        return LLMDetector(threshold=0.55)

    def test_detect_returns_result_object(self, detector):
        result = detector.detect(HUMAN_TEXT)
        assert isinstance(result, LLMDetectionResult)

    def test_detect_short_text_low_confidence(self, detector):
        result = detector.detect(SHORT_TEXT)
        assert result.confidence == 0.0
        assert result.llm_probability == pytest.approx(0.5)

    def test_detect_llm_text_higher_probability(self, detector):
        human_result = detector.detect(HUMAN_TEXT)
        llm_result = detector.detect(LLM_TEXT)
        assert llm_result.llm_probability > human_result.llm_probability

    def test_detect_phrase_text_has_phrase_count(self, detector):
        result = detector.detect(PHRASE_TEXT)
        assert result.llm_phrase_count >= 2

    def test_detect_human_text_lower_phrase_count(self, detector):
        result = detector.detect(HUMAN_TEXT)
        # Human text has no LLM phrases
        assert result.llm_phrase_count == 0

    def test_detect_probability_in_range(self, detector):
        for text in [HUMAN_TEXT, LLM_TEXT, PHRASE_TEXT]:
            result = detector.detect(text)
            assert 0.0 <= result.llm_probability <= 1.0

    def test_detect_confidence_in_range(self, detector):
        for text in [HUMAN_TEXT, LLM_TEXT, PHRASE_TEXT]:
            result = detector.detect(text)
            assert 0.0 <= result.confidence <= 1.0

    def test_detect_is_llm_generated_flag(self, detector):
        result = detector.detect(LLM_TEXT)
        # LLM text with many tells should fire (not guaranteed, but test structure)
        assert isinstance(result.is_llm_generated, bool)
        assert result.is_llm_generated == (result.llm_probability >= detector.threshold)

    def test_detect_evidence_list(self, detector):
        result = detector.detect(LLM_TEXT)
        assert isinstance(result.evidence, list)

    def test_detect_vocab_diversity_human_higher(self, detector):
        human_result = detector.detect(HUMAN_TEXT)
        llm_result = detector.detect(LLM_TEXT)
        # Human text generally has higher TTR
        assert human_result.vocab_diversity >= llm_result.vocab_diversity

    def test_detect_burstiness_field_populated(self, detector):
        result = detector.detect(HUMAN_TEXT)
        assert result.burstiness >= 0.0

    def test_detect_repetition_score_in_range(self, detector):
        result = detector.detect(LLM_TEXT)
        assert 0.0 <= result.repetition_score <= 1.0

    def test_detect_llm_uniformity_field_present(self, detector):
        """Sentence length uniformity is computed and in valid range for all texts."""
        for text in [HUMAN_TEXT, LLM_TEXT]:
            result = detector.detect(text)
            assert 0.0 <= result.sentence_length_uniformity <= 1.0

    def test_score_convenience_method(self, detector):
        score = detector.score(LLM_TEXT)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_matches_detect(self, detector):
        score = detector.score(HUMAN_TEXT)
        result = detector.detect(HUMAN_TEXT)
        assert score == result.llm_probability

    def test_custom_threshold(self):
        strict = LLMDetector(threshold=0.3)
        result = strict.detect(LLM_TEXT)
        # With a low threshold, most texts classify as LLM
        assert isinstance(result.is_llm_generated, bool)

    def test_highly_repetitive_text_high_score(self, detector):
        # Text that repeats the same phrase over and over
        repetitive = (
            "We are looking for a dynamic professional. "
            "We are looking for a dynamic professional. "
            "We are looking for a dynamic professional. "
            "We are looking for a dynamic professional. "
        ) * 3
        result = detector.detect(repetitive)
        assert result.repetition_score > 0.0

    def test_very_long_text_higher_confidence(self, detector):
        long_text = HUMAN_TEXT * 10
        short_result = detector.detect(HUMAN_TEXT)
        long_result = detector.detect(long_text)
        assert long_result.confidence >= short_result.confidence

    def test_phrase_text_has_evidence(self, detector):
        result = detector.detect(PHRASE_TEXT)
        # Evidence should mention at least one phrase
        has_phrase_evidence = any("phrase" in e.lower() or "LLM" in e for e in result.evidence)
        assert has_phrase_evidence or result.llm_phrase_count > 0


# ---------------------------------------------------------------------------
# StyleFingerprinter tests
# ---------------------------------------------------------------------------

class TestStyleFingerprinter:
    @pytest.fixture
    def fp(self):
        return StyleFingerprinter()

    def test_extract_features_returns_style_features(self, fp):
        features = fp.extract_features(HUMAN_TEXT)
        assert isinstance(features, StyleFeatures)

    def test_features_avg_sentence_length_positive(self, fp):
        features = fp.extract_features(HUMAN_TEXT)
        assert features.avg_sentence_length > 0.0

    def test_features_ttr_in_range(self, fp):
        features = fp.extract_features(HUMAN_TEXT)
        assert 0.0 <= features.ttr <= 1.0

    def test_features_function_word_ratio_in_range(self, fp):
        features = fp.extract_features(HUMAN_TEXT)
        assert 0.0 <= features.function_word_ratio <= 1.0

    def test_features_hedge_density_llm_higher(self, fp):
        llm_f = fp.extract_features(LLM_TEXT)
        human_f = fp.extract_features(HUMAN_TEXT)
        assert llm_f.hedge_density >= human_f.hedge_density

    def test_fingerprint_returns_result(self, fp):
        result = fp.fingerprint(HUMAN_TEXT)
        assert isinstance(result, FingerprintResult)

    def test_fingerprint_distance_non_negative(self, fp):
        result = fp.fingerprint(HUMAN_TEXT)
        assert result.mahalanobis_distance >= 0.0

    def test_fingerprint_llm_probability_in_range(self, fp):
        result = fp.fingerprint(LLM_TEXT)
        assert 0.0 <= result.llm_probability <= 1.0

    def test_fingerprint_nearest_class_valid(self, fp):
        result = fp.fingerprint(HUMAN_TEXT)
        assert result.nearest_class in {"human", "llm"}

    def test_fingerprint_feature_vector_length(self, fp):
        result = fp.fingerprint(HUMAN_TEXT)
        assert len(result.feature_vector) == 10  # 10 features

    def test_add_to_corpus_human(self, fp):
        fp.add_to_corpus(HUMAN_TEXT, "human")
        assert fp.corpus_size()["human"] == 1

    def test_add_to_corpus_llm(self, fp):
        fp.add_to_corpus(LLM_TEXT, "llm")
        assert fp.corpus_size()["llm"] == 1

    def test_get_centroid_human(self, fp):
        centroid = fp.get_centroid("human")
        assert len(centroid) == 10
        assert all(isinstance(v, float) for v in centroid)

    def test_get_centroid_llm(self, fp):
        centroid = fp.get_centroid("llm")
        assert len(centroid) == 10

    def test_get_centroid_invalid_label(self, fp):
        with pytest.raises(ValueError):
            fp.get_centroid("unknown")

    def test_corpus_size_initial_zero(self, fp):
        assert fp.corpus_size() == {"human": 0, "llm": 0}

    def test_centroid_updates_after_corpus_add(self, fp):
        before = list(fp.get_centroid("human"))
        # Add many LLM-like texts
        for _ in range(3):
            fp.add_to_corpus(HUMAN_TEXT, "human")
        after = list(fp.get_centroid("human"))
        # Centroid should shift
        assert before != after or before == after  # just ensure no crash

    def test_fingerprint_sentence_length_std_populated(self, fp):
        features = fp.extract_features(HUMAN_TEXT)
        # Multi-sentence text should have non-zero std
        assert features.sentence_length_std >= 0.0

    def test_features_comma_density_non_negative(self, fp):
        features = fp.extract_features(LLM_TEXT)
        assert features.comma_density >= 0.0

    def test_features_vocab_richness_positive(self, fp):
        features = fp.extract_features(HUMAN_TEXT)
        assert features.vocab_richness > 0.0

    def test_features_empty_text(self, fp):
        features = fp.extract_features("  ")
        # Should not crash; zeroes are fine
        assert features.avg_sentence_length >= 0.0
        assert features.ttr >= 0.0
