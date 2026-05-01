"""Tests for flywheel-scorer integration: learned weights flow from DB into scoring."""

import pytest

from sentinel.db import SentinelDB
from sentinel.knowledge import KnowledgeBase
from sentinel.flywheel import DetectionFlywheel
from sentinel.models import (
    JobPosting,
    ScamSignal,
    SignalCategory,
    UserReport,
    ValidationResult,
)
from sentinel.scorer import (
    _load_learned_weights,
    _reset_learned_weights_cache,
    score_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(name: str, weight: float, category: SignalCategory = SignalCategory.RED_FLAG) -> ScamSignal:
    return ScamSignal(
        name=name,
        category=category,
        weight=weight,
        confidence=0.80,
        detail=f"test signal {name}",
        evidence="test",
    )


def _insert_pattern(db: SentinelDB, pattern_id: str, *, alpha: float, beta: float, observations: int, status: str = "active") -> None:
    """Insert a pattern directly into the DB for testing."""
    db.save_pattern({
        "pattern_id": pattern_id,
        "name": pattern_id,
        "description": f"Test pattern {pattern_id}",
        "category": "red_flag",
        "regex": "",
        "keywords_json": "[]",
        "alpha": alpha,
        "beta": beta,
        "observations": observations,
        "true_positives": int(alpha - 1),
        "false_positives": int(beta - 1),
        "status": status,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLearnedWeightsOverride:
    """After inserting a pattern with high alpha, score_signals should use the learned weight."""

    def test_high_alpha_pattern_overrides_static_weight(self, tmp_path):
        """A pattern with alpha=100, beta=1 should yield a learned weight near 0.99,
        significantly different from the static weight of 0.50."""
        _reset_learned_weights_cache()

        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)

        # Insert a pattern with overwhelming positive evidence
        _insert_pattern(db, "upfront_payment", alpha=100.0, beta=1.0, observations=99)
        db.close()

        # Score with a signal whose static weight is 0.50
        signal = _make_signal("upfront_payment", weight=0.50)

        # Score WITH learned weights
        score_learned, _ = score_signals([signal], db_path=db_path, use_learned_weights=True)

        _reset_learned_weights_cache()

        # Score WITHOUT learned weights (static only)
        score_static, _ = score_signals([signal], db_path=db_path, use_learned_weights=False)

        # The learned score should be much higher than static (0.99 vs 0.50)
        assert score_learned > score_static, (
            f"Learned score ({score_learned}) should exceed static score ({score_static})"
        )
        # With alpha=100, beta=1 the learned weight is ~0.99 -> log-odds pushes score high
        assert score_learned > 0.95, f"Expected score > 0.95 with alpha=100, got {score_learned}"
        # Static weight of 0.50 -> log-odds 0 -> score 0.50
        assert abs(score_static - 0.50) < 0.01, f"Static score should be ~0.50, got {score_static}"

        _reset_learned_weights_cache()


class TestStaticFallback:
    """With no DB data, scoring should fall back to static weights."""

    def test_no_db_falls_back_to_static(self, tmp_path):
        """When the DB has no patterns at all, score_signals uses static weights."""
        _reset_learned_weights_cache()

        db_path = str(tmp_path / "empty.db")
        db = SentinelDB(path=db_path)
        db.close()

        signal = _make_signal("upfront_payment", weight=0.80)

        score_with_db, _ = score_signals([signal], db_path=db_path, use_learned_weights=True)
        _reset_learned_weights_cache()

        score_static, _ = score_signals([signal], db_path=db_path, use_learned_weights=False)
        _reset_learned_weights_cache()

        # Both should be identical because the DB has no qualifying patterns
        assert abs(score_with_db - score_static) < 1e-6, (
            f"Expected same score, got {score_with_db} vs {score_static}"
        )

    def test_nonexistent_db_falls_back_gracefully(self, tmp_path):
        """If the DB path does not exist, scoring should still work with static weights."""
        _reset_learned_weights_cache()

        bad_path = str(tmp_path / "nonexistent" / "dir" / "missing.db")
        signal = _make_signal("upfront_payment", weight=0.80)

        # Should not raise; falls back to static
        score, confidence = score_signals([signal], db_path=bad_path, use_learned_weights=True)
        _reset_learned_weights_cache()

        # Static weight 0.80 -> score should be 0.80
        assert abs(score - 0.80) < 0.01, f"Expected ~0.80, got {score}"


class TestInsufficientObservations:
    """Signals with < 10 observations should keep static weights."""

    def test_low_observation_pattern_ignored(self, tmp_path):
        """A pattern with only 5 observations should NOT override the static weight."""
        _reset_learned_weights_cache()

        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)

        # alpha=100 but only 5 observations — should be ignored
        _insert_pattern(db, "upfront_payment", alpha=100.0, beta=1.0, observations=5)
        db.close()

        signal = _make_signal("upfront_payment", weight=0.50)

        score_with_db, _ = score_signals([signal], db_path=db_path, use_learned_weights=True)
        _reset_learned_weights_cache()

        score_static, _ = score_signals([signal], db_path=db_path, use_learned_weights=False)
        _reset_learned_weights_cache()

        # Both should be the same because observations < 10
        assert abs(score_with_db - score_static) < 1e-6, (
            f"Low-obs pattern should be ignored: {score_with_db} vs {score_static}"
        )

    def test_boundary_observation_count(self, tmp_path):
        """A pattern with exactly 10 observations should use the learned weight."""
        _reset_learned_weights_cache()

        db_path = str(tmp_path / "test.db")
        db = SentinelDB(path=db_path)

        _insert_pattern(db, "upfront_payment", alpha=100.0, beta=1.0, observations=10)
        db.close()

        signal = _make_signal("upfront_payment", weight=0.50)

        score_learned, _ = score_signals([signal], db_path=db_path, use_learned_weights=True)
        _reset_learned_weights_cache()

        score_static, _ = score_signals([signal], db_path=db_path, use_learned_weights=False)
        _reset_learned_weights_cache()

        # With exactly 10 observations, learned weight should kick in
        assert score_learned > score_static, (
            f"At boundary (10 obs), learned should apply: {score_learned} vs {score_static}"
        )


class TestFullFlywheelLoop:
    """End-to-end: seed patterns, report via flywheel, verify weight changes in scorer."""

    def test_report_updates_weight_in_scorer(self, tmp_path):
        """Seed a pattern, submit enough true-positive reports via the flywheel,
        then verify score_signals picks up the learned weight."""
        _reset_learned_weights_cache()

        db_path = str(tmp_path / "flywheel.db")
        db = SentinelDB(path=db_path)

        # Seed a pattern with neutral prior
        _insert_pattern(db, "upfront_payment", alpha=1.0, beta=1.0, observations=0)

        # Score before any reports — static weight 0.50 -> score ~0.50
        signal = _make_signal("upfront_payment", weight=0.50)
        score_before, _ = score_signals([signal], db_path=db_path, use_learned_weights=True)
        _reset_learned_weights_cache()

        # Simulate 20 true-positive reports via the flywheel
        flywheel = DetectionFlywheel(db=db)
        job = JobPosting(url="https://example.com/job/1", title="Test", company="Test Co")
        result = ValidationResult(job=job, signals=[signal])
        report = UserReport(url=job.url, is_scam=True, reason="test")

        for _ in range(20):
            flywheel.learn_from_report(report, result)

        # Now the pattern should have alpha ~21, beta ~1, observations ~20
        patterns = db.get_patterns(status="active")
        pattern = [p for p in patterns if p["pattern_id"] == "upfront_payment"][0]
        assert pattern["observations"] >= 20, f"Expected >=20 observations, got {pattern['observations']}"
        assert pattern["alpha"] > 15.0, f"Expected alpha > 15, got {pattern['alpha']}"

        # Score after reports — should now use learned weight
        score_after, _ = score_signals([signal], db_path=db_path, use_learned_weights=True)
        _reset_learned_weights_cache()

        # The learned weight (alpha/(alpha+beta) ~ 21/22 ~ 0.95) should push score up
        assert score_after > score_before, (
            f"After reports, score should increase: {score_after} vs {score_before}"
        )
        assert score_after > 0.90, (
            f"After 20 true-positive reports, score should be > 0.90, got {score_after}"
        )

        db.close()

    def test_knowledge_base_report_propagates(self, tmp_path):
        """Reports submitted through KnowledgeBase.report_scam should also feed
        the DB state that scorer reads — confirming the full integration chain."""
        _reset_learned_weights_cache()

        db_path = str(tmp_path / "kb.db")
        db = SentinelDB(path=db_path)

        # Seed pattern
        _insert_pattern(db, "crypto_payment", alpha=1.0, beta=1.0, observations=0)

        kb = KnowledgeBase(db=db)
        flywheel = DetectionFlywheel(db=db)

        signal = _make_signal("crypto_payment", weight=0.50)
        job = JobPosting(url="https://example.com/job/crypto", title="Crypto Job")
        result = ValidationResult(job=job, signals=[signal])
        report = UserReport(url=job.url, is_scam=True, reason="crypto scam")

        # Submit via KnowledgeBase (records the report)
        kb.report_scam(url=job.url, is_scam=True, reason="crypto scam", our_prediction=0.5)

        # Then learn via flywheel (updates pattern stats) — 15 times to exceed threshold
        for _ in range(15):
            flywheel.learn_from_report(report, result)

        # Verify the pattern was updated
        patterns = db.get_patterns(status="active")
        pattern = [p for p in patterns if p["pattern_id"] == "crypto_payment"][0]
        assert pattern["observations"] >= 15

        # Score should now use learned weight
        score, _ = score_signals([signal], db_path=db_path, use_learned_weights=True)
        _reset_learned_weights_cache()

        assert score > 0.85, f"After 15 TP reports, expected score > 0.85, got {score}"

        db.close()
