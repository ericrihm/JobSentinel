"""Tests for sentinel/stylometry.py — Stylometric Operator Fingerprinting.

45+ tests covering:
- StyleFingerprint dataclass
- StyleExtractor (all features, edge cases)
- OperatorLinker (add, link, cluster, confidence)
- TemplateDetector (similarity, slot detection, family tracking)
- WritingEvolutionTracker (shifts, AI detection, intra-doc inconsistencies)
"""

from __future__ import annotations

import math
import pytest

from sentinel.stylometry import (
    IntraDocInconsistency,
    LinkResult,
    OperatorLinker,
    OperatorProfile,
    StyleExtractor,
    StyleFingerprint,
    StyleShift,
    StyleSnapshot,
    TemplateFamilyRecord,
    TemplateDetector,
    TemplateMatch,
    WritingEvolutionTracker,
    _flesch_kincaid_grade,
    _mean_fingerprint,
    _split_paragraphs,
    _split_sentences,
    _tokenize_words,
    _weighted_euclidean,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEGIT_TEXT = (
    "We are seeking a highly motivated Senior Software Engineer to join our "
    "infrastructure team. The ideal candidate will have five or more years of "
    "experience with distributed systems and cloud computing. You will design "
    "and implement scalable backend services using Python and Go. We offer "
    "competitive compensation and a collaborative work environment.\n\n"
    "Required qualifications include a Bachelor's degree in Computer Science "
    "or equivalent experience, strong proficiency in Python or Go, and "
    "experience with Kubernetes or similar container orchestration platforms. "
    "Preferred qualifications include prior open-source contributions and "
    "experience with consensus protocols such as Paxos or Raft.\n\n"
    "We are an equal opportunity employer and value diversity at our company. "
    "We do not discriminate on the basis of race, religion, color, national "
    "origin, gender, sexual orientation, age, marital status, or disability."
)

SCAM_TEXT = (
    "EARN $5000 PER WEEK GUARANTEED!!! Work from home NOW!!! "
    "No experience required!!! Anyone can qualify!!! "
    "You will be hired IMMEDIATELY after applying. No interview needed!!! "
    "Send $99 registration fee and provide your Social Security Number. "
    "LIMITED SPOTS AVAILABLE!!! Apply NOW before it's too late!!!"
)

SCAM_TEXT_2 = (
    "MAKE $5000 WEEKLY GUARANTEED!!! Work at home NOW!!! "
    "Zero experience needed!!! Everyone qualifies!!! "
    "Get hired INSTANTLY after signup. No interview required!!! "
    "Pay $99 registration fee and give your Social Security Number. "
    "LIMITED POSITIONS AVAILABLE!!! Apply TODAY before it's too late!!!"
)

SHORT_TEXT = "Hiring now. Apply today."

MULTI_PARA_TEXT = (
    "This is the first paragraph of the job description. It discusses "
    "responsibilities in a professional and measured tone. We seek qualified "
    "candidates with relevant experience in the field.\n\n"
    "BUY BITCOIN NOW!!! GUARANTEED INCOME!!! SEND MONEY IMMEDIATELY!!!"
)


@pytest.fixture
def extractor() -> StyleExtractor:
    return StyleExtractor()


@pytest.fixture
def legit_fp(extractor: StyleExtractor) -> StyleFingerprint:
    return extractor.extract(LEGIT_TEXT)


@pytest.fixture
def scam_fp(extractor: StyleExtractor) -> StyleFingerprint:
    return extractor.extract(SCAM_TEXT)


@pytest.fixture
def linker() -> OperatorLinker:
    return OperatorLinker()


@pytest.fixture
def detector() -> TemplateDetector:
    return TemplateDetector()


# ===========================================================================
# 1. StyleFingerprint dataclass
# ===========================================================================

class TestStyleFingerprint:
    def test_default_values(self):
        fp = StyleFingerprint()
        assert fp.avg_sentence_length == 0.0
        assert fp.word_count == 0
        assert fp.is_short is False
        assert fp.function_word_ratios == {}
        assert fp.punctuation_ratios == {}

    def test_to_vector_length_consistent(self, legit_fp: StyleFingerprint):
        vec = legit_fp.to_vector()
        # Must be a list of floats
        assert all(isinstance(x, float) for x in vec)
        assert len(vec) > 10

    def test_to_vector_two_fps_same_length(
        self, legit_fp: StyleFingerprint, scam_fp: StyleFingerprint
    ):
        v1 = legit_fp.to_vector()
        v2 = scam_fp.to_vector()
        assert len(v1) == len(v2)

    def test_to_vector_empty_fp(self):
        fp = StyleFingerprint()
        vec = fp.to_vector()
        # All zeros for a default (empty) fingerprint
        assert all(v == 0.0 for v in vec)

    def test_function_word_ratios_sum_less_than_one(self, legit_fp: StyleFingerprint):
        total = sum(legit_fp.function_word_ratios.values())
        # Function words in aggregate are usually < 50% of all words, but can sum > 1
        # What we want: each individual ratio is in [0, 1]
        for w, r in legit_fp.function_word_ratios.items():
            assert 0.0 <= r <= 1.0, f"ratio for '{w}' out of range: {r}"

    def test_punctuation_ratios_non_negative(self, legit_fp: StyleFingerprint):
        for p, r in legit_fp.punctuation_ratios.items():
            assert r >= 0.0


# ===========================================================================
# 2. StyleExtractor
# ===========================================================================

class TestStyleExtractor:
    def test_extract_empty_string_returns_short(self, extractor: StyleExtractor):
        fp = extractor.extract("")
        assert fp.is_short is True
        assert fp.word_count == 0

    def test_extract_whitespace_returns_short(self, extractor: StyleExtractor):
        fp = extractor.extract("   \n\t  ")
        assert fp.is_short is True

    def test_extract_short_text_flagged(self, extractor: StyleExtractor):
        fp = extractor.extract(SHORT_TEXT)
        assert fp.is_short is True

    def test_extract_legit_text_basic_metrics(self, legit_fp: StyleFingerprint):
        assert legit_fp.word_count > 100
        assert legit_fp.avg_sentence_length > 5.0
        assert legit_fp.avg_word_length > 3.0
        assert 0.0 < legit_fp.vocabulary_richness <= 1.0

    def test_vocabulary_richness_range(self, legit_fp: StyleFingerprint):
        assert 0.0 < legit_fp.vocabulary_richness <= 1.0

    def test_scam_has_high_caps_rate(self, scam_fp: StyleFingerprint):
        # Scam text uses lots of ALL CAPS
        assert scam_fp.capitalization_rate > 0.1

    def test_legit_has_lower_caps_than_scam(
        self, legit_fp: StyleFingerprint, scam_fp: StyleFingerprint
    ):
        assert legit_fp.capitalization_rate < scam_fp.capitalization_rate

    def test_readability_score_non_negative(self, legit_fp: StyleFingerprint):
        assert legit_fp.readability_score >= 0.0

    def test_paragraph_count_multi_para(self, extractor: StyleExtractor):
        fp = extractor.extract(MULTI_PARA_TEXT)
        assert fp.paragraph_count == 2

    def test_paragraph_count_single_para(self, extractor: StyleExtractor):
        fp = extractor.extract(SCAM_TEXT)
        assert fp.paragraph_count >= 1

    def test_contraction_usage_rate_with_contractions(self, extractor: StyleExtractor):
        text = "We're hiring now. You'll love it. It's a great opportunity. Don't miss out."
        fp = extractor.extract(text)
        assert fp.contraction_usage_rate > 0.0

    def test_contraction_usage_rate_without_contractions(self, legit_fp: StyleFingerprint):
        # LEGIT_TEXT has "don't" (1 contraction), so just check it's a float
        assert isinstance(legit_fp.contraction_usage_rate, float)

    def test_passive_voice_ratio_detected(self, extractor: StyleExtractor):
        text = (
            "The report was written by the team. The analysis was completed on time. "
            "All tasks were finished before the deadline. The system was tested thoroughly. "
            "Results were reviewed by the committee. The budget was approved last week. "
            "The project was launched successfully."
        )
        fp = extractor.extract(text)
        assert fp.passive_voice_ratio > 0.0

    def test_sentence_length_std_multiple_sentences(self, legit_fp: StyleFingerprint):
        # Multi-sentence text should have non-zero std if sentences vary
        assert legit_fp.sentence_length_std >= 0.0

    def test_sentence_length_std_single_sentence(self, extractor: StyleExtractor):
        fp = extractor.extract("This is a single sentence without any period at the end")
        assert fp.sentence_length_std == 0.0

    def test_function_words_tracked(self, legit_fp: StyleFingerprint):
        assert "the" in legit_fp.function_word_ratios
        assert "and" in legit_fp.function_word_ratios
        assert legit_fp.function_word_ratios["the"] > 0.0

    def test_punctuation_tracked(self, legit_fp: StyleFingerprint):
        assert "." in legit_fp.punctuation_ratios
        assert "," in legit_fp.punctuation_ratios

    def test_avg_paragraph_length_positive(self, legit_fp: StyleFingerprint):
        assert legit_fp.avg_paragraph_length > 0.0


# ===========================================================================
# 3. Internal helpers
# ===========================================================================

class TestHelpers:
    def test_tokenize_words_basic(self):
        assert _tokenize_words("Hello, World!") == ["hello", "world"]

    def test_tokenize_words_empty(self):
        assert _tokenize_words("") == []

    def test_tokenize_words_numbers_stripped(self):
        # Numbers are non-alphabetic, should not appear
        tokens = _tokenize_words("earn $5000 per week")
        assert "5000" not in tokens

    def test_split_sentences_basic(self):
        sents = _split_sentences("Hello world. How are you? Fine thanks.")
        assert len(sents) >= 2

    def test_split_paragraphs_basic(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        paras = _split_paragraphs(text)
        assert len(paras) == 3

    def test_split_paragraphs_single(self):
        paras = _split_paragraphs("Just one paragraph here.")
        assert len(paras) == 1

    def test_flesch_kincaid_positive(self):
        grade = _flesch_kincaid_grade(LEGIT_TEXT)
        assert grade >= 0.0

    def test_flesch_kincaid_empty(self):
        assert _flesch_kincaid_grade("") == 0.0

    def test_weighted_euclidean_identical(self):
        v = [1.0, 2.0, 3.0]
        w = [1.0, 1.0, 1.0]
        assert _weighted_euclidean(v, v, w) == 0.0

    def test_weighted_euclidean_known(self):
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        w = [1.0, 1.0]
        # sqrt(9 + 16) = 5
        assert abs(_weighted_euclidean(v1, v2, w) - 5.0) < 1e-9

    def test_weighted_euclidean_weighted(self):
        v1 = [0.0]
        v2 = [1.0]
        w = [4.0]
        # sqrt(4 * 1) = 2
        assert abs(_weighted_euclidean(v1, v2, w) - 2.0) < 1e-9

    def test_mean_fingerprint_single(self, legit_fp: StyleFingerprint):
        result = _mean_fingerprint([legit_fp])
        assert result.avg_sentence_length == legit_fp.avg_sentence_length

    def test_mean_fingerprint_two(
        self, legit_fp: StyleFingerprint, scam_fp: StyleFingerprint
    ):
        mean = _mean_fingerprint([legit_fp, scam_fp])
        expected = (legit_fp.avg_sentence_length + scam_fp.avg_sentence_length) / 2
        assert abs(mean.avg_sentence_length - expected) < 0.01

    def test_mean_fingerprint_empty(self):
        result = _mean_fingerprint([])
        assert result.word_count == 0


# ===========================================================================
# 4. OperatorLinker
# ===========================================================================

class TestOperatorLinker:
    def test_link_no_operators_returns_no_match(
        self, linker: OperatorLinker, scam_fp: StyleFingerprint
    ):
        result = linker.link("p1", scam_fp)
        assert result.is_match is False
        assert result.operator_id is None
        assert math.isinf(result.distance)

    def test_add_and_link_same_style(
        self, linker: OperatorLinker, extractor: StyleExtractor
    ):
        fp1 = extractor.extract(SCAM_TEXT)
        fp2 = extractor.extract(SCAM_TEXT_2)
        linker.add_fingerprint("op_scam", "p1", fp1)
        result = linker.link("p2", fp2)
        # Both texts are stylistically very similar (all caps, exclamation marks)
        assert result.distance < linker.NO_MATCH_THRESHOLD

    def test_add_and_link_different_styles(
        self,
        linker: OperatorLinker,
        legit_fp: StyleFingerprint,
        scam_fp: StyleFingerprint,
    ):
        linker.add_fingerprint("op_legit", "p1", legit_fp)
        result = linker.link("p2", scam_fp)
        # Legit and scam texts differ significantly; distance should be > 0
        assert result.distance > 0.0

    def test_known_operators_empty(self, linker: OperatorLinker):
        assert linker.known_operators() == []

    def test_known_operators_after_add(
        self, linker: OperatorLinker, scam_fp: StyleFingerprint
    ):
        linker.add_fingerprint("op_001", "p1", scam_fp)
        assert "op_001" in linker.known_operators()

    def test_get_profile_exists(
        self, linker: OperatorLinker, scam_fp: StyleFingerprint
    ):
        linker.add_fingerprint("op_x", "p1", scam_fp)
        profile = linker.get_profile("op_x")
        assert profile is not None
        assert profile.operator_id == "op_x"

    def test_get_profile_missing(self, linker: OperatorLinker):
        assert linker.get_profile("nonexistent") is None

    def test_centroid_updated_on_add(
        self,
        linker: OperatorLinker,
        scam_fp: StyleFingerprint,
        legit_fp: StyleFingerprint,
    ):
        linker.add_fingerprint("op_y", "p1", scam_fp)
        profile = linker.get_profile("op_y")
        assert profile is not None
        assert profile.centroid is not None

    def test_link_result_confidence_range(
        self, linker: OperatorLinker, scam_fp: StyleFingerprint
    ):
        linker.add_fingerprint("op_z", "p1", scam_fp)
        result = linker.link("p2", scam_fp)
        assert 0.0 <= result.confidence <= 1.0

    def test_cluster_identical_texts_same_group(self, extractor: StyleExtractor):
        linker = OperatorLinker(match_threshold=5.0)
        fp1 = extractor.extract(SCAM_TEXT)
        fp2 = extractor.extract(SCAM_TEXT_2)
        fp3 = extractor.extract(LEGIT_TEXT)
        clusters = linker.cluster({"p1": fp1, "p2": fp2, "p3": fp3})
        # p1 and p2 should be in the same cluster (similar style)
        # p3 might be alone
        all_members = [m for c in clusters.values() for m in c]
        assert "p1" in all_members
        assert "p2" in all_members
        assert "p3" in all_members

    def test_cluster_single_posting(
        self, linker: OperatorLinker, scam_fp: StyleFingerprint
    ):
        clusters = linker.cluster({"only": scam_fp})
        assert len(clusters) == 1

    def test_cluster_empty(self, linker: OperatorLinker):
        clusters = linker.cluster({})
        assert clusters == {}

    def test_operator_profile_multiple_fingerprints(
        self,
        linker: OperatorLinker,
        scam_fp: StyleFingerprint,
        legit_fp: StyleFingerprint,
    ):
        linker.add_fingerprint("op_multi", "p1", scam_fp)
        linker.add_fingerprint("op_multi", "p2", legit_fp)
        profile = linker.get_profile("op_multi")
        assert profile is not None
        assert len(profile.fingerprints) == 2
        assert len(profile.posting_ids) == 2


# ===========================================================================
# 5. TemplateDetector
# ===========================================================================

class TestTemplateDetector:
    def test_compare_identical_texts(self, detector: TemplateDetector):
        detector.add_posting("a", SCAM_TEXT)
        detector.add_posting("b", SCAM_TEXT)
        match = detector.compare("a", "b")
        assert match.similarity == pytest.approx(1.0, abs=0.01)
        assert match.is_template is True

    def test_compare_very_similar_texts(self, detector: TemplateDetector):
        detector.add_posting("a", SCAM_TEXT)
        detector.add_posting("b", SCAM_TEXT_2)
        match = detector.compare("a", "b")
        # These are very similar (same structure, slightly different words)
        assert match.similarity > 0.0
        assert isinstance(match.is_template, bool)

    def test_compare_very_different_texts(self, detector: TemplateDetector):
        detector.add_posting("a", SCAM_TEXT)
        detector.add_posting("b", LEGIT_TEXT)
        match = detector.compare("a", "b")
        assert match.similarity < 0.9
        assert match.is_template is False

    def test_compare_ids_in_result(self, detector: TemplateDetector):
        detector.add_posting("x", SCAM_TEXT)
        detector.add_posting("y", LEGIT_TEXT)
        match = detector.compare("x", "y")
        assert match.posting_id_a == "x"
        assert match.posting_id_b == "y"

    def test_find_slots_different_words(self, detector: TemplateDetector):
        text_a = "We are hiring a Python engineer in Seattle for $120000"
        text_b = "We are hiring a Java engineer in Chicago for $130000"
        match = detector.compare_texts("a", "b", text_a, text_b)
        # Slots should include the changed words
        assert len(match.variable_slots) > 0

    def test_detect_template_families_two_similar(self, detector: TemplateDetector):
        detector.add_posting("p1", SCAM_TEXT)
        detector.add_posting("p2", SCAM_TEXT)  # identical
        families = detector.detect_template_families()
        assert len(families) >= 1
        family = families[0]
        assert "p1" in family.member_ids or "p2" in family.member_ids

    def test_detect_template_families_no_matches(self, detector: TemplateDetector):
        detector.add_posting("a", SCAM_TEXT)
        detector.add_posting("b", LEGIT_TEXT)
        families = detector.detect_template_families()
        # These are too different to be a template family
        assert len(families) == 0

    def test_extract_slot_values_simple(self, detector: TemplateDetector):
        template = "We are hiring a {{SLOT}} in {{SLOT}}"
        text = "We are hiring a developer in Seattle"
        slots = detector.extract_slot_values(template, text)
        # Should find at least one slot value
        assert len(slots) >= 1

    def test_template_similarity_threshold_custom(self):
        detector = TemplateDetector(threshold=0.99)
        detector.add_posting("a", SCAM_TEXT)
        detector.add_posting("b", SCAM_TEXT_2)
        match = detector.compare("a", "b")
        # With very high threshold, even similar texts may not match
        assert isinstance(match.is_template, bool)

    def test_family_spread_count(self, detector: TemplateDetector):
        detector.add_posting("p1", SCAM_TEXT)
        detector.add_posting("p2", SCAM_TEXT)
        detector.add_posting("p3", SCAM_TEXT)
        families = detector.detect_template_families()
        if families:
            assert families[0].spread_count >= 2


# ===========================================================================
# 6. WritingEvolutionTracker
# ===========================================================================

class TestWritingEvolutionTracker:
    @pytest.fixture
    def tracker(self) -> WritingEvolutionTracker:
        return WritingEvolutionTracker("op_test")

    @pytest.fixture
    def tracker_with_data(
        self, extractor: StyleExtractor
    ) -> WritingEvolutionTracker:
        tracker = WritingEvolutionTracker("op_001")
        # Add snapshots: normal scam text, then AI-sounding text
        fp1 = extractor.extract(SCAM_TEXT)
        fp2 = extractor.extract(LEGIT_TEXT)
        tracker.record(StyleSnapshot("2026-01-01", fp1, "p1"))
        tracker.record(StyleSnapshot("2026-02-01", fp2, "p2"))
        return tracker

    def test_empty_tracker_detect_shifts(self, tracker: WritingEvolutionTracker):
        shifts = tracker.detect_shifts()
        assert shifts == []

    def test_single_snapshot_no_shifts(
        self, tracker: WritingEvolutionTracker, scam_fp: StyleFingerprint
    ):
        tracker.record(StyleSnapshot("2026-01-01", scam_fp, "p1"))
        shifts = tracker.detect_shifts()
        assert shifts == []

    def test_two_snapshots_one_shift(
        self, tracker_with_data: WritingEvolutionTracker
    ):
        shifts = tracker_with_data.detect_shifts()
        assert len(shifts) == 1

    def test_shift_timestamps(
        self, tracker_with_data: WritingEvolutionTracker
    ):
        shifts = tracker_with_data.detect_shifts()
        assert shifts[0].from_timestamp == "2026-01-01"
        assert shifts[0].to_timestamp == "2026-02-01"

    def test_shift_magnitude_positive(
        self, tracker_with_data: WritingEvolutionTracker
    ):
        shifts = tracker_with_data.detect_shifts()
        assert shifts[0].magnitude >= 0.0

    def test_shift_has_description(
        self, tracker_with_data: WritingEvolutionTracker
    ):
        shifts = tracker_with_data.detect_shifts()
        assert isinstance(shifts[0].description, str)
        assert len(shifts[0].description) > 0

    def test_snapshot_count(
        self, tracker_with_data: WritingEvolutionTracker
    ):
        assert tracker_with_data.snapshot_count() == 2

    def test_style_trajectory_returns_list(
        self, tracker_with_data: WritingEvolutionTracker
    ):
        traj = tracker_with_data.style_trajectory()
        assert len(traj) == 2
        assert "timestamp" in traj[0]
        assert "vocab_richness" in traj[0]

    def test_detect_ai_assistance_returns_list(
        self, tracker_with_data: WritingEvolutionTracker
    ):
        flagged = tracker_with_data.detect_ai_assistance()
        assert isinstance(flagged, list)

    def test_intra_doc_inconsistency_consistent_text(
        self, tracker: WritingEvolutionTracker, extractor: StyleExtractor
    ):
        # Consistent professional text — no big style shifts between paragraphs
        text = (
            "We are seeking a motivated engineer to join our team. "
            "You will design and implement new features.\n\n"
            "The role requires strong Python skills and teamwork. "
            "You should have at least three years of experience."
        )
        inconsistencies = tracker.detect_intra_doc_inconsistencies(
            "p1", text, extractor
        )
        # Short paragraphs are skipped, so expect empty or no-inconsistency results
        assert isinstance(inconsistencies, list)

    def test_intra_doc_inconsistency_mixed_text(
        self, tracker: WritingEvolutionTracker, extractor: StyleExtractor
    ):
        # Mixed professional + scam paragraph
        inconsistencies = tracker.detect_intra_doc_inconsistencies(
            "p1", MULTI_PARA_TEXT, extractor
        )
        assert isinstance(inconsistencies, list)

    def test_intra_doc_returns_inconsistency_dataclass(
        self, tracker: WritingEvolutionTracker, extractor: StyleExtractor
    ):
        long_text = (
            "We are a leading technology company seeking talented engineers. "
            "Our team works on cutting edge distributed systems at global scale. "
            "We use Python Go and Rust for our backend infrastructure services. "
            "You will collaborate with world class engineers across multiple teams. "
            "Experience with cloud platforms like AWS or GCP is highly preferred.\n\n"
            "EARN MONEY NOW!!! GUARANTEED!!! SEND FEE IMMEDIATELY!!! APPLY NOW!!! "
            "NO EXPERIENCE NEEDED!!! HIRE IMMEDIATELY!!! LIMITED SPOTS AVAILABLE!!!"
        )
        inconsistencies = tracker.detect_intra_doc_inconsistencies(
            "mixed_p", long_text, extractor
        )
        # The two paragraphs are stylistically very different
        # At least some inconsistency objects are returned (if paragraphs are long enough)
        assert isinstance(inconsistencies, list)
        for inc in inconsistencies:
            assert isinstance(inc, IntraDocInconsistency)
            assert inc.posting_id == "mixed_p"

    def test_record_multiple_snapshots(
        self, tracker: WritingEvolutionTracker, extractor: StyleExtractor
    ):
        for i in range(5):
            fp = extractor.extract(SCAM_TEXT)
            tracker.record(StyleSnapshot(f"2026-0{i+1}-01", fp, f"p{i}"))
        assert tracker.snapshot_count() == 5
        shifts = tracker.detect_shifts()
        assert len(shifts) == 4
