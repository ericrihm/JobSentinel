"""Tests for sentinel.active_learning — QBC + margin review selection."""

import pytest

from sentinel.active_learning import ActiveLearner, ReviewCandidate, select_review_batch


def _job(url, scam_score, signals=None):
    return {
        "url": url,
        "title": f"Job {url}",
        "company": "Acme",
        "scam_score": scam_score,
        "confidence": 0.5,
        "signals": signals or [],
    }


def _signal(weight, category="red_flag"):
    return {"name": "sig", "weight": weight, "category": category}


class TestActiveLearner:
    def test_boundary_job_ranked_highest(self):
        """A job at score=0.5 is maximally uncertain and must rank #1."""
        jobs = [
            _job("boundary", 0.5),
            _job("clear_scam", 0.95),
            _job("clear_legit", 0.02),
        ]
        learner = ActiveLearner()
        ranked = learner.rank_for_review(jobs)

        assert ranked[0].url == "boundary"

    def test_clear_extremes_ranked_lower_than_uncertain(self):
        """Jobs with scores near 0 or 1 should have lower informativeness than near-0.5 jobs."""
        jobs = [
            _job("scam", 0.98),
            _job("legit", 0.01),
            _job("uncertain_high", 0.55),
            _job("uncertain_low", 0.45),
        ]
        learner = ActiveLearner()
        ranked = learner.rank_for_review(jobs)

        informative_urls = {c.url for c in ranked[:2]}
        assert "uncertain_high" in informative_urls
        assert "uncertain_low" in informative_urls

    def test_high_signal_disagreement_boosts_rank(self):
        """A job with signals that cause method disagreement should outscore a same-margin job with no signals."""
        disagreeing_signals = [
            _signal(0.9, "red_flag"),
            _signal(0.9, "positive"),
            _signal(0.9, "positive"),
        ]
        job_with_signals = _job("disagreement", 0.5, signals=disagreeing_signals)
        job_no_signals = _job("no_signals", 0.5)

        learner = ActiveLearner()
        ranked = learner.rank_for_review([job_with_signals, job_no_signals])

        assert ranked[0].url == "disagreement"
        assert ranked[0].disagreement >= ranked[1].disagreement

    def test_top_n_limits_output(self):
        jobs = [_job(f"job{i}", i / 20.0) for i in range(20)]
        learner = ActiveLearner()
        ranked = learner.rank_for_review(jobs, top_n=5)

        assert len(ranked) == 5

    def test_returns_review_candidate_dataclass(self):
        jobs = [_job("test", 0.6)]
        learner = ActiveLearner()
        result = learner.rank_for_review(jobs)

        assert len(result) == 1
        c = result[0]
        assert isinstance(c, ReviewCandidate)
        assert c.url == "test"
        assert 0.0 <= c.informativeness <= 1.0
        assert 0.0 <= c.disagreement <= 1.0
        assert 0.0 <= c.margin <= 1.0
        assert c.scam_score == pytest.approx(0.6)

    def test_empty_input_returns_empty(self):
        learner = ActiveLearner()
        assert learner.rank_for_review([]) == []

    def test_informativeness_formula(self):
        """Informativeness must equal 0.6 * disagreement + 0.4 * margin."""
        learner = ActiveLearner()
        jobs = [_job("x", 0.5)]
        c = learner.rank_for_review(jobs)[0]
        expected = round(0.6 * c.disagreement + 0.4 * c.margin, 6)
        assert c.informativeness == pytest.approx(expected, abs=1e-5)

    def test_select_review_batch_with_db(self, temp_db):
        """select_review_batch returns candidates from the DB in informativeness order."""
        import json
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        for i, score in enumerate([0.51, 0.99, 0.01, 0.48]):
            temp_db.conn.execute(
                """
                INSERT INTO jobs (url, title, company, scam_score, confidence,
                                  signals_json, analyzed_at, user_reported)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (f"http://job{i}", f"Job {i}", "Co", score, 0.5, "[]", now),
            )
        temp_db.conn.commit()

        candidates = select_review_batch(temp_db, batch_size=4)

        assert len(candidates) <= 4
        assert all(isinstance(c, ReviewCandidate) for c in candidates)

        scores = [c.informativeness for c in candidates]
        assert scores == sorted(scores, reverse=True)

        top_urls = {c.url for c in candidates[:2]}
        assert "http://job0" in top_urls or "http://job3" in top_urls
