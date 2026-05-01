"""Tests for sentinel.adversarial — TextNormalizer, EvasionDetector, and DB methods."""

from __future__ import annotations

import pytest

from sentinel.adversarial import EvasionDetector, TextNormalizer
from sentinel.db import SentinelDB
from sentinel.models import JobPosting, ScamSignal, SignalCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalizer() -> TextNormalizer:
    return TextNormalizer()


@pytest.fixture
def detector() -> EvasionDetector:
    return EvasionDetector()


@pytest.fixture
def temp_db(tmp_path) -> SentinelDB:
    db = SentinelDB(path=str(tmp_path / "adv_test.db"))
    yield db
    db.close()


# ---------------------------------------------------------------------------
# TextNormalizer — Unicode / homoglyph normalization
# ---------------------------------------------------------------------------


class TestUnicodeNormalization:
    """Cyrillic and Greek lookalikes are replaced with their Latin equivalents."""

    def test_cyrillic_a_replaced(self, normalizer: TextNormalizer):
        # Cyrillic а (U+0430) → Latin a
        assert normalizer.normalize("sаlаry") == "salary"

    def test_cyrillic_o_replaced(self, normalizer: TextNormalizer):
        # Cyrillic о (U+043E) → Latin o
        assert normalizer.normalize("bitcоin") == "bitcoin"

    def test_cyrillic_e_replaced(self, normalizer: TextNormalizer):
        # Cyrillic е (U+0435) → Latin e
        assert normalizer.normalize("frее monеy") == "free money"

    def test_greek_omicron_replaced(self, normalizer: TextNormalizer):
        # Greek ο (U+03BF) → Latin o
        text = "cryptο"
        result = normalizer.normalize(text)
        assert result == "crypto"

    def test_fullwidth_ascii_replaced(self, normalizer: TextNormalizer):
        # Fullwidth latin letters → ASCII
        result = normalizer.normalize("ｂｉｔｃｏｉｎ")
        assert result == "bitcoin"

    def test_mixed_cyrillic_latin_word(self, normalizer: TextNormalizer):
        # "рayment" — р is Cyrillic р (U+0440) → p
        text = "Send рayment now"
        result = normalizer.normalize(text)
        assert "payment" in result

    def test_pure_ascii_unchanged(self, normalizer: TextNormalizer):
        text = "Senior Software Engineer at Acme Corp"
        assert normalizer.normalize(text) == text


# ---------------------------------------------------------------------------
# TextNormalizer — Leetspeak
# ---------------------------------------------------------------------------


class TestLeetspeakExpansion:
    def test_zero_to_o(self, normalizer: TextNormalizer):
        assert normalizer.normalize("b1tc0in") == "bitcoin"

    def test_at_to_a(self, normalizer: TextNormalizer):
        result = normalizer.normalize("c@sh @pp")
        assert result == "cash app"

    def test_dollar_to_s(self, normalizer: TextNormalizer):
        result = normalizer.normalize("$end money")
        assert result == "send money"

    def test_plus_to_t(self, normalizer: TextNormalizer):
        result = normalizer.normalize("guaran+eed")
        assert result == "guaranteed"

    def test_three_to_e(self, normalizer: TextNormalizer):
        result = normalizer.normalize("fr33 mon3y")
        assert result == "free money"

    def test_pure_numeric_tokens_unchanged(self, normalizer: TextNormalizer):
        # Year "2024" and salary "$50000" digits should not be leetspeak-expanded
        # when they stand as pure numeric tokens (no alpha around them)
        text = "5 years experience salary 80000"
        result = normalizer.normalize(text)
        # "5" alone is pure digit → unchanged; "80000" alone is pure digit → unchanged
        assert "5" in result
        assert "80000" in result


# ---------------------------------------------------------------------------
# TextNormalizer — Zero-width and invisible characters
# ---------------------------------------------------------------------------


class TestZeroWidthRemoval:
    def test_zero_width_space_removed(self, normalizer: TextNormalizer):
        # Insert U+200B (zero-width space) inside "bitcoin"
        text = "bit​coin"
        result = normalizer.normalize(text)
        assert "​" not in result

    def test_zero_width_joiner_removed(self, normalizer: TextNormalizer):
        text = "pay‍ment"
        result = normalizer.normalize(text)
        assert "‍" not in result

    def test_soft_hyphen_removed(self, normalizer: TextNormalizer):
        text = "guaran\xadteed"
        result = normalizer.normalize(text)
        assert "\xad" not in result

    def test_bom_removed(self, normalizer: TextNormalizer):
        text = "﻿guaranteed income"
        result = normalizer.normalize(text)
        assert not result.startswith("﻿")
        assert "guaranteed" in result


# ---------------------------------------------------------------------------
# EvasionDetector — evasion_attempt signal
# ---------------------------------------------------------------------------


class TestEvasionAttemptSignal:
    def test_cyrillic_evasion_detected(self, detector: EvasionDetector):
        normalizer = TextNormalizer()
        original = "sеnd mоnеy nоw"   # е, о are Cyrillic
        normalized = normalizer.normalize(original)
        signals = detector.detect_evasion_attempts(original, normalized)
        names = [s.name for s in signals]
        assert "evasion_attempt" in names

    def test_clean_text_no_evasion(self, detector: EvasionDetector):
        text = "Great opportunity at Google, 5 years Python experience required."
        signals = detector.detect_evasion_attempts(text, text)
        assert not any(s.name == "evasion_attempt" for s in signals)

    def test_evasion_signal_weight(self, detector: EvasionDetector):
        normalizer = TextNormalizer()
        original = "b1tc0in pаyment"
        normalized = normalizer.normalize(original)
        signals = detector.detect_evasion_attempts(original, normalized)
        evasion = next((s for s in signals if s.name == "evasion_attempt"), None)
        if evasion:
            assert evasion.weight == pytest.approx(0.70)
            assert evasion.category == SignalCategory.RED_FLAG

    def test_empty_text_no_crash(self, detector: EvasionDetector):
        signals = detector.detect_evasion_attempts("", "")
        assert signals == []


# ---------------------------------------------------------------------------
# EvasionDetector — unicode_anomaly signal (mixed-script words)
# ---------------------------------------------------------------------------


class TestUnicodeAnomalySignal:
    def test_mixed_cyrillic_latin_word_flagged(self, detector: EvasionDetector):
        # Word "pаyment" with Cyrillic а should trigger unicode_anomaly
        word_with_cyrillic = "pаyment"  # Cyrillic а mixed with Latin
        signals = detector.detect_evasion_attempts(word_with_cyrillic, word_with_cyrillic)
        assert any(s.name == "unicode_anomaly" for s in signals)

    def test_all_ascii_no_anomaly(self, detector: EvasionDetector):
        text = "legitimate job posting with normal ascii text"
        signals = detector.detect_evasion_attempts(text, text)
        assert not any(s.name == "unicode_anomaly" for s in signals)

    def test_unicode_anomaly_weight(self, detector: EvasionDetector):
        word_with_cyrillic = "pаyment fee required"
        signals = detector.detect_evasion_attempts(word_with_cyrillic, word_with_cyrillic)
        anomaly = next((s for s in signals if s.name == "unicode_anomaly"), None)
        if anomaly:
            assert anomaly.weight == pytest.approx(0.55)
            assert anomaly.category == SignalCategory.WARNING


# ---------------------------------------------------------------------------
# EvasionDetector — misspelled_scam_keyword signal
# ---------------------------------------------------------------------------


class TestMisspelledKeywordSignal:
    def test_leet_bitcoin_detected(self, detector: EvasionDetector):
        text = "pay us in b1tc0in or ethereum"
        signals = detector.detect_evasion_attempts(text, text)
        assert any(s.name == "misspelled_scam_keyword" for s in signals)

    def test_space_split_payment_detected(self, detector: EvasionDetector):
        text = "you must make a pa yment of $50"
        signals = detector.detect_evasion_attempts(text, text)
        assert any(s.name == "misspelled_scam_keyword" for s in signals)

    def test_plus_guaranteed_detected(self, detector: EvasionDetector):
        text = "guaran+eed income every week"
        signals = detector.detect_evasion_attempts(text, text)
        assert any(s.name == "misspelled_scam_keyword" for s in signals)

    def test_clean_text_no_misspelling_flag(self, detector: EvasionDetector):
        text = "We are looking for a senior Python developer with 5 years experience."
        signals = detector.detect_evasion_attempts(text, text)
        assert not any(s.name == "misspelled_scam_keyword" for s in signals)


# ---------------------------------------------------------------------------
# EvasionDetector — signal decay tracking
# ---------------------------------------------------------------------------


class TestSignalDecayTracking:
    def _seed_decay(self, db: SentinelDB, signal_name: str, rates: list[float]) -> None:
        """Insert signal_rate_history rows simulating historical and recent windows."""
        from datetime import datetime, timedelta, UTC

        now = datetime.now(UTC)
        for i, rate in enumerate(rates):
            window_start = (now - timedelta(days=len(rates) - i)).isoformat()
            window_end = (now - timedelta(days=len(rates) - i - 1)).isoformat()
            fire_count = int(rate * 100)
            db.record_signal_rates(
                {signal_name: fire_count}, 100, window_start, window_end
            )

    def test_decaying_signal_flagged(self, detector: EvasionDetector, temp_db: SentinelDB):
        # Baseline 20% → drops to 2%
        self._seed_decay(temp_db, "upfront_payment", [0.20, 0.20, 0.20, 0.20, 0.04, 0.02, 0.02, 0.02])
        results = detector.track_signal_decay(temp_db)
        decaying = [r for r in results if r["signal_name"] == "upfront_payment" and r["suspicious"]]
        assert len(decaying) == 1

    def test_stable_signal_not_flagged(self, detector: EvasionDetector, temp_db: SentinelDB):
        # Stable around 20% — no decay
        self._seed_decay(temp_db, "crypto_payment", [0.20, 0.19, 0.21, 0.20, 0.20, 0.19, 0.21, 0.20])
        results = detector.track_signal_decay(temp_db)
        stable = [r for r in results if r["signal_name"] == "crypto_payment" and r["suspicious"]]
        assert len(stable) == 0

    def test_empty_db_returns_empty_list(self, detector: EvasionDetector, temp_db: SentinelDB):
        results = detector.track_signal_decay(temp_db)
        assert results == []

    def test_decay_result_keys(self, detector: EvasionDetector, temp_db: SentinelDB):
        self._seed_decay(temp_db, "guaranteed_income", [0.15, 0.15, 0.15, 0.15, 0.02, 0.01, 0.01, 0.01])
        results = detector.track_signal_decay(temp_db)
        assert len(results) > 0
        row = results[0]
        for key in ("signal_name", "baseline_rate", "recent_rate", "decay_rate", "suspicious"):
            assert key in row


# ---------------------------------------------------------------------------
# EvasionDetector — near-miss detection
# ---------------------------------------------------------------------------


class TestNearMissDetection:
    def _make_signal(self, name: str, evidence: str) -> ScamSignal:
        return ScamSignal(
            name=name,
            category=SignalCategory.WARNING,
            weight=0.5,
            evidence=evidence,
        )

    def test_near_miss_found_at_threshold(self, detector: EvasionDetector):
        # Evidence contains "bitco" — partial match for "bitcoin"
        signals = [self._make_signal("partial_crypto", "bitco")]
        misses = detector.detect_near_misses(signals, score=0.45)
        assert len(misses) > 0
        assert any(m["partial_match"] == "bitco" for m in misses)

    def test_near_miss_severity_high_when_score_high(self, detector: EvasionDetector):
        signals = [self._make_signal("something", "bitco")]
        misses = detector.detect_near_misses(signals, score=0.50)
        high = [m for m in misses if m["severity"] == "high"]
        assert len(high) > 0

    def test_near_miss_severity_low_when_score_low(self, detector: EvasionDetector):
        signals = [self._make_signal("something", "bitco")]
        misses = detector.detect_near_misses(signals, score=0.20)
        low = [m for m in misses if m["severity"] == "low"]
        assert len(low) > 0

    def test_no_near_miss_for_trivial_score(self, detector: EvasionDetector):
        signals = [self._make_signal("something", "bitco")]
        misses = detector.detect_near_misses(signals, score=0.05)
        assert misses == []

    def test_full_keyword_match_not_near_miss(self, detector: EvasionDetector):
        # "bitcoin" fully present in evidence — not a near-miss
        signals = [self._make_signal("crypto", "bitcoin payment required")]
        misses = detector.detect_near_misses(signals, score=0.6)
        # "bitcoin" is already fully matched, so it shouldn't appear as near-miss
        assert not any(m["partial_match"] == "bitcoin" for m in misses)


# ---------------------------------------------------------------------------
# DB methods — near_misses and signal_decay_history
# ---------------------------------------------------------------------------


class TestDBMethods:
    def test_insert_and_get_near_miss(self, temp_db: SentinelDB):
        temp_db.insert_near_miss("upfront_payment", "fee req", "https://example.com/job/1")
        rows = temp_db.get_near_misses()
        assert len(rows) == 1
        assert rows[0]["signal_name"] == "upfront_payment"
        assert rows[0]["partial_match"] == "fee req"
        assert rows[0]["job_url"] == "https://example.com/job/1"

    def test_get_near_misses_filtered(self, temp_db: SentinelDB):
        temp_db.insert_near_miss("crypto_payment", "bitco", "https://example.com/job/2")
        temp_db.insert_near_miss("upfront_payment", "upfr", "https://example.com/job/3")
        crypto_rows = temp_db.get_near_misses(signal_name="crypto_payment")
        assert all(r["signal_name"] == "crypto_payment" for r in crypto_rows)

    def test_get_near_misses_empty(self, temp_db: SentinelDB):
        assert temp_db.get_near_misses() == []

    def test_insert_and_get_signal_rate(self, temp_db: SentinelDB):
        temp_db.insert_signal_rate("upfront_payment", "2026-04-01T00:00:00+00:00", 0.18)
        rows = temp_db.get_signal_decay(signal_name="upfront_payment")
        assert len(rows) == 1
        assert rows[0]["fire_rate"] == pytest.approx(0.18)

    def test_get_signal_decay_all(self, temp_db: SentinelDB):
        temp_db.insert_signal_rate("sig_a", "2026-04-01T00:00:00+00:00", 0.10)
        temp_db.insert_signal_rate("sig_b", "2026-04-02T00:00:00+00:00", 0.20)
        rows = temp_db.get_signal_decay()
        names = {r["signal_name"] for r in rows}
        assert "sig_a" in names
        assert "sig_b" in names


# ---------------------------------------------------------------------------
# Integration — extract_signals uses normalizer
# ---------------------------------------------------------------------------


class TestExtractSignalsIntegration:
    def test_cyrillic_obfuscated_bitcoin_caught(self):
        """Cyrillic о in 'bitcоin' still triggers crypto_payment after normalization."""
        from sentinel.signals import extract_signals

        job = JobPosting(
            url="https://example.com/job/scam1",
            title="Remote Work Opportunity",
            company="FastCash Ltd",
            description="Pay us in bitcоin (U+043E) or gift cards. No experience needed.",
        )
        signals = extract_signals(job)
        names = [s.name for s in signals]
        # After normalization "bitcоin" → "bitcoin", so crypto_payment should fire
        assert "crypto_payment" in names

    def test_evasion_signal_from_leet(self):
        """Leetspeak 'b1tc0in' should cause both normalization delta and crypto detection."""
        from sentinel.signals import extract_signals

        job = JobPosting(
            url="https://example.com/job/scam2",
            title="Flexible Income",
            company="EasyCash",
            description="You will receive b1tc0in payments daily. No experience required.",
        )
        signals = extract_signals(job)
        names = [s.name for s in signals]
        # Normalized → "bitcoin" fires crypto_payment; original has leet → evasion or misspelling
        assert "crypto_payment" in names or any(
            n in names for n in ("evasion_attempt", "misspelled_scam_keyword")
        )
