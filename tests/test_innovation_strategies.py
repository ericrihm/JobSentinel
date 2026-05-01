"""Tests for the three implemented innovation strategies:
_correlate_signals, _expand_keywords, and _mine_patterns.
"""

import json

import pytest

from sentinel.db import SentinelDB
from sentinel.innovation import InnovationEngine
from sentinel.knowledge import KnowledgeBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_job_with_signals(db: SentinelDB, url: str, signals: list[dict], scam_score: float = 0.8) -> None:
    """Insert a job record with the given signals."""
    db.save_job({
        "url": url,
        "title": "Test Job",
        "company": "Test Co",
        "location": "Remote",
        "description": "A test job posting.",
        "salary_min": 0.0,
        "salary_max": 0.0,
        "scam_score": scam_score,
        "risk_level": "high" if scam_score >= 0.6 else "low",
        "signal_count": len(signals),
        "signals_json": json.dumps(signals),
        "user_reported": 1,
        "user_verdict": "scam",
    })


def _seed_scam_report(db: SentinelDB, url: str, reason: str = "", our_prediction: float = 0.8) -> None:
    """Insert a scam report for the given URL."""
    db.save_report({
        "url": url,
        "is_scam": True,
        "reason": reason,
        "our_prediction": our_prediction,
        "was_correct": True,
    })


def _make_engine(db: SentinelDB) -> InnovationEngine:
    """Create an InnovationEngine backed by the given DB (avoids filesystem state)."""
    engine = InnovationEngine.__new__(InnovationEngine)
    engine.db = db
    from sentinel.flywheel import DetectionFlywheel
    engine.flywheel = DetectionFlywheel(db)
    # Initialize STRATEGIES as fresh instances (no shared state)
    from sentinel.innovation import ImprovementArm
    engine.STRATEGIES = [
        ImprovementArm("false_positive_review", "Review and fix false positive detections"),
        ImprovementArm("false_negative_review", "Find missed scams from user reports"),
        ImprovementArm("weight_optimization", "Re-optimize signal weights from recent data"),
        ImprovementArm("pattern_mining", "Mine new scam patterns from reported scams"),
        ImprovementArm("regression_check", "CUSUM regression analysis on accuracy trends"),
        ImprovementArm("cross_signal_correlation", "Find signal combinations that predict scams"),
        ImprovementArm("keyword_expansion", "Expand scam keyword lists from new reports"),
        ImprovementArm("threshold_tuning", "Tune risk classification thresholds"),
    ]
    return engine


# ===========================================================================
# _correlate_signals
# ===========================================================================


class TestCorrelateSignals:
    def test_correlate_signals_with_data(self, temp_db):
        """Seed DB with overlapping signals across scam jobs; verify pairs found."""
        engine = _make_engine(temp_db)

        # Create 5 scam jobs, each with signals A and B (co-occur 5 times)
        # plus a third signal C on some
        for i in range(5):
            url = f"https://example.com/job/{i}"
            signals = [
                {"name": "upfront_payment", "category": "red_flag"},
                {"name": "guaranteed_income", "category": "red_flag"},
            ]
            if i < 3:
                signals.append({"name": "no_interview", "category": "red_flag"})
            _seed_job_with_signals(temp_db, url, signals)
            _seed_scam_report(temp_db, url, reason=f"Scam #{i}")

        result = engine._correlate_signals(0.0)

        assert result.success is True
        assert result.strategy == "cross_signal_correlation"
        # The detail should mention the top pair(s)
        assert "upfront_payment" in result.detail or "guaranteed_income" in result.detail

    def test_correlate_signals_empty(self, temp_db):
        """No reports at all -- returns gracefully with success=False."""
        engine = _make_engine(temp_db)

        result = engine._correlate_signals(0.0)

        assert result.success is False
        assert "No reports" in result.detail or "No scam" in result.detail or "No data" in result.detail

    def test_correlate_signals_insufficient_cooccurrences(self, temp_db):
        """Only 2 co-occurrences (below the >3 threshold) -- no pairs found."""
        engine = _make_engine(temp_db)

        for i in range(2):
            url = f"https://example.com/job/{i}"
            signals = [
                {"name": "sig_a", "category": "red_flag"},
                {"name": "sig_b", "category": "red_flag"},
            ]
            _seed_job_with_signals(temp_db, url, signals)
            _seed_scam_report(temp_db, url, reason=f"Scam #{i}")

        result = engine._correlate_signals(0.0)

        assert result.success is False


# ===========================================================================
# _expand_keywords
# ===========================================================================


class TestExpandKeywords:
    def test_expand_keywords_finds_new(self, temp_db):
        """Seed scam reports with repeated bigrams; verify candidate keywords created."""
        engine = _make_engine(temp_db)

        # Create 5 scam reports all mentioning "fake check deposit"
        for i in range(5):
            url = f"https://example.com/kw/{i}"
            _seed_scam_report(
                temp_db, url,
                reason="They asked me to deposit fake check and wire money back immediately"
            )

        result = engine._expand_keywords(0.0)

        assert result.success is True
        assert result.strategy == "keyword_expansion"
        # Verify candidate pattern was saved
        candidates = temp_db.get_patterns(status="candidate")
        assert len(candidates) >= 1
        # Check that at least one candidate has keywords
        found_keywords = False
        for pat in candidates:
            kw = pat.get("keywords") or pat.get("keywords_json", "[]")
            if isinstance(kw, str):
                kw = json.loads(kw)
            if kw:
                found_keywords = True
                break
        assert found_keywords

    def test_expand_keywords_no_duplicates(self, temp_db):
        """Existing pattern keywords should not be re-proposed."""
        engine = _make_engine(temp_db)

        # Seed a pattern that already has "wire back" as a keyword
        temp_db.save_pattern({
            "pattern_id": "existing_pat",
            "name": "Existing Pattern",
            "description": "Test",
            "category": "red_flag",
            "regex": "",
            "keywords_json": json.dumps(["wire back", "deposit check"]),
            "status": "active",
        })

        # Create reports whose bigrams overlap heavily with existing keywords
        for i in range(5):
            url = f"https://example.com/nodup/{i}"
            _seed_scam_report(
                temp_db, url,
                reason="They told me to wire back the deposit check amount"
            )

        result = engine._expand_keywords(0.0)

        # If any new candidates were found, verify "wire back" is not among them
        if result.success:
            candidates = temp_db.get_patterns(status="candidate")
            for pat in candidates:
                kw = pat.get("keywords") or pat.get("keywords_json", "[]")
                if isinstance(kw, str):
                    kw = json.loads(kw)
                for keyword in kw:
                    assert keyword.lower() != "wire back", "Existing keyword 'wire back' should not be re-proposed"
                    assert keyword.lower() != "deposit check", "Existing keyword 'deposit check' should not be re-proposed"

    def test_expand_keywords_no_reasons(self, temp_db):
        """No scam report reasons at all -- returns gracefully."""
        engine = _make_engine(temp_db)

        result = engine._expand_keywords(0.0)

        assert result.success is False
        assert "No" in result.detail


# ===========================================================================
# _mine_patterns
# ===========================================================================


class TestMinePatterns:
    def test_mine_patterns_creates_candidates(self, temp_db):
        """Seed multiple similar reasons; verify candidate patterns created."""
        engine = _make_engine(temp_db)

        # 5 reports with similar language about payment and training fees
        similar_reasons = [
            "Scam asked for upfront payment training fee before starting work",
            "Required upfront payment for training materials fee to begin",
            "Had to pay upfront payment training fee registration before hire",
            "They demanded upfront payment for training fee certification",
            "Asked me for upfront payment and training fee before interview",
        ]
        for i, reason in enumerate(similar_reasons):
            url = f"https://example.com/mine/{i}"
            _seed_scam_report(temp_db, url, reason=reason)

        result = engine._mine_patterns(0.0)

        assert result.success is True
        assert result.strategy == "pattern_mining"
        assert result.new_patterns >= 1

        # Verify candidate patterns exist in DB
        candidates = temp_db.get_patterns(status="candidate")
        assert len(candidates) >= 1

        # Verify pattern has keywords and regex
        pat = candidates[0]
        kw = pat.get("keywords") or pat.get("keywords_json", "[]")
        if isinstance(kw, str):
            kw = json.loads(kw)
        assert len(kw) > 0
        assert pat.get("regex", "") != ""

    def test_mine_patterns_insufficient_data(self, temp_db):
        """Fewer than 3 reports with reasons -- no patterns mined."""
        engine = _make_engine(temp_db)

        # Only 2 reports
        for i in range(2):
            url = f"https://example.com/few/{i}"
            _seed_scam_report(temp_db, url, reason="Some scam reason")

        result = engine._mine_patterns(0.0)

        assert result.success is False
        assert result.new_patterns == 0
        candidates = temp_db.get_patterns(status="candidate")
        assert len(candidates) == 0

    def test_mine_patterns_no_cluster_large_enough(self, temp_db):
        """3 reports but all with very different text -- no overlapping cluster."""
        engine = _make_engine(temp_db)

        distinct_reasons = [
            "Cryptocurrency bitcoin wallet transfer required immediately",
            "Company asked for passport social security number",
            "Mystery shopping envelope stuffing work from home",
        ]
        for i, reason in enumerate(distinct_reasons):
            url = f"https://example.com/distinct/{i}"
            _seed_scam_report(temp_db, url, reason=reason)

        result = engine._mine_patterns(0.0)

        # Should fail because no cluster has 3+ members with word overlap >= 3
        assert result.success is False
