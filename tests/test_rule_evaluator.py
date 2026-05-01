"""Tests for sentinel.rule_evaluator — offline pattern quality scoring."""

import pytest

from sentinel.rule_evaluator import (
    RuleEvaluator,
    _coverage_score,
    _novelty_score,
    _parse_keywords,
    _precision_score,
    _specificity_score,
)


# ---------------------------------------------------------------------------
# _parse_keywords
# ---------------------------------------------------------------------------


def test_parse_keywords_list():
    assert _parse_keywords(["Pay", "Bitcoin"]) == ["pay", "bitcoin"]


def test_parse_keywords_json_string():
    assert _parse_keywords('["wire", "transfer"]') == ["wire", "transfer"]


def test_parse_keywords_csv_string():
    assert _parse_keywords("fee, deposit, upfront") == ["fee", "deposit", "upfront"]


def test_parse_keywords_none():
    assert _parse_keywords(None) == []


def test_parse_keywords_empty_list():
    assert _parse_keywords([]) == []


# ---------------------------------------------------------------------------
# Specificity
# ---------------------------------------------------------------------------


def test_specificity_long_regex():
    pat = {"regex": r"\b(send|wire|transfer)\s+\$?\d+\s+(fee|deposit)\b", "keywords": []}
    score = _specificity_score(pat)
    assert score > 0.4


def test_specificity_no_regex_no_keywords():
    assert _specificity_score({}) == 0.0


def test_specificity_keywords_only():
    pat = {"keywords": ["bitcoin", "crypto", "payment", "wire", "fee"]}
    score = _specificity_score(pat)
    assert 0.3 < score < 0.8


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_coverage_high():
    pat = {"observations": 50, "true_positives": 45}
    assert _coverage_score(pat) == pytest.approx(0.9)


def test_coverage_zero_obs():
    assert _coverage_score({"observations": 0, "true_positives": 0}) == 0.0


def test_coverage_low_obs_shrinkage():
    pat = {"observations": 2, "true_positives": 2}
    score = _coverage_score(pat)
    assert score < 1.0, "Should shrink toward 0.5 with few observations"


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


def test_precision_high():
    pat = {"true_positives": 40, "false_positives": 2, "observations": 42}
    score = _precision_score(pat)
    assert score > 0.8


def test_precision_low_obs_bayesian_fallback():
    pat = {"observations": 2, "true_positives": 1, "false_positives": 0, "alpha": 3.0, "beta": 1.0}
    score = _precision_score(pat)
    assert score == pytest.approx(0.75)


def test_precision_zero_outcomes():
    pat = {"observations": 0, "true_positives": 0, "false_positives": 0, "alpha": 1.0, "beta": 1.0}
    assert _precision_score(pat) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------


def test_novelty_no_active_patterns():
    pat = {"keywords": ["bitcoin"], "regex": ""}
    assert _novelty_score(pat, []) == 1.0


def test_novelty_exact_duplicate():
    pat = {"pattern_id": "new", "keywords": ["bitcoin", "payment"], "regex": ""}
    active = [{"pattern_id": "old", "keywords": ["bitcoin", "payment"], "regex": ""}]
    score = _novelty_score(pat, active)
    assert score == pytest.approx(0.0)


def test_novelty_different_keywords():
    pat = {"pattern_id": "new", "keywords": ["telegram", "whatsapp"], "regex": ""}
    active = [{"pattern_id": "old", "keywords": ["bitcoin", "payment"], "regex": ""}]
    score = _novelty_score(pat, active)
    assert score == 1.0


def test_novelty_partial_overlap():
    pat = {"pattern_id": "new", "keywords": ["bitcoin", "telegram"], "regex": ""}
    active = [{"pattern_id": "old", "keywords": ["bitcoin", "payment"], "regex": ""}]
    score = _novelty_score(pat, active)
    assert 0.3 < score < 0.9


# ---------------------------------------------------------------------------
# RuleEvaluator
# ---------------------------------------------------------------------------


def test_evaluate_strong_candidate():
    evaluator = RuleEvaluator()
    pat = {
        "pattern_id": "test_1",
        "regex": r"\b(upfront|advance)\s+(fee|payment|deposit)\b",
        "keywords": ["fee", "deposit", "wire", "payment", "advance"],
        "observations": 100,
        "true_positives": 90,
        "false_positives": 3,
        "alpha": 10.0,
        "beta": 2.0,
    }
    result = evaluator.evaluate_candidate(pat, [])
    assert result["recommendation"] == "promote"
    assert result["composite_score"] > 0.55


def test_evaluate_weak_candidate_rejected():
    evaluator = RuleEvaluator()
    pat = {
        "pattern_id": "test_2",
        "regex": "",
        "keywords": [],
        "observations": 3,
        "true_positives": 1,
        "false_positives": 1,
        "alpha": 1.0,
        "beta": 1.0,
    }
    result = evaluator.evaluate_candidate(pat, [])
    assert result["recommendation"] == "reject"
    assert result["composite_score"] < 0.55


def test_evaluate_duplicate_rejected():
    evaluator = RuleEvaluator()
    active = [{
        "pattern_id": "existing",
        "regex": r"\bbitcoin\b",
        "keywords": ["bitcoin", "crypto", "payment"],
    }]
    pat = {
        "pattern_id": "candidate",
        "regex": r"\bbitcoin\b",
        "keywords": ["bitcoin", "crypto", "payment"],
        "observations": 50,
        "true_positives": 40,
        "false_positives": 5,
        "alpha": 5.0,
        "beta": 1.0,
    }
    result = evaluator.evaluate_candidate(pat, active)
    assert result["novelty_score"] < 0.2, "Duplicate should have near-zero novelty"


def test_rank_candidates_sorted():
    evaluator = RuleEvaluator()
    strong = {
        "pattern_id": "strong",
        "regex": r"\bsend\s+\$\d+\b",
        "keywords": ["wire", "fee", "deposit", "payment"],
        "observations": 80,
        "true_positives": 75,
        "false_positives": 2,
        "alpha": 8.0,
        "beta": 1.0,
    }
    weak = {
        "pattern_id": "weak",
        "regex": "",
        "keywords": [],
        "observations": 1,
        "true_positives": 0,
        "false_positives": 1,
        "alpha": 1.0,
        "beta": 1.0,
    }
    ranked = evaluator.rank_candidates([weak, strong], [])
    assert ranked[0]["pattern_id"] == "strong"
    assert ranked[1]["pattern_id"] == "weak"
    assert ranked[0]["composite_score"] >= ranked[1]["composite_score"]


def test_rank_candidates_empty():
    evaluator = RuleEvaluator()
    assert evaluator.rank_candidates([], []) == []
