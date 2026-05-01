"""Tests for the engram push integration — publishing sentinel findings to engram."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sentinel.engram_publisher import EngramPublisher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ingest_path(tmp_path) -> Path:
    return tmp_path / "engram_ingest.jsonl"


@pytest.fixture
def publisher(ingest_path) -> EngramPublisher:
    return EngramPublisher(engram_cli="engram", ingest_path=ingest_path)


@pytest.fixture
def sample_finding() -> dict:
    return {
        "strategy": "false_positive_review",
        "success": True,
        "detail": "Signal 'vague_description' caused 5 false positives — weight reduced",
        "precision_delta": 0.02,
        "new_patterns": 0,
        "deprecated_patterns": 0,
    }


@pytest.fixture
def strategy_state_file(tmp_path) -> Path:
    state_path = tmp_path / "innovation_state.json"
    state_path.write_text(json.dumps({
        "false_positive_review": {"alpha": 8.0, "beta": 3.0, "attempts": 11},
        "pattern_mining": {"alpha": 5.0, "beta": 5.0, "attempts": 10},
        "weight_optimization": {"alpha": 3.0, "beta": 7.0, "attempts": 10},
    }))
    return state_path


# ---------------------------------------------------------------------------
# _convert_finding
# ---------------------------------------------------------------------------


class TestConvertFinding:
    def test_converts_successful_finding(self, publisher, sample_finding):
        pattern = publisher._convert_finding(sample_finding)
        assert pattern["source"] == "sentinel"
        assert pattern["type"] == "finding"
        assert pattern["name"] == "sentinel_false_positive_review"
        assert pattern["category"] == "success"
        assert pattern["confidence"] == 0.8
        assert pattern["domain"] == "security"
        assert "sentinel" in pattern["tags"]
        assert "innovation" in pattern["tags"]

    def test_converts_failed_finding(self, publisher):
        finding = {"strategy": "regression_check", "success": False, "detail": "CUSUM alarm"}
        pattern = publisher._convert_finding(finding)
        assert pattern["confidence"] == 0.4
        assert pattern["category"] == "pattern"

    def test_converts_deprecation_finding(self, publisher):
        finding = {
            "strategy": "weight_optimization",
            "success": False,
            "detail": "Deprecated 3 patterns",
            "deprecated_patterns": 3,
        }
        pattern = publisher._convert_finding(finding)
        assert pattern["category"] == "regression"
        assert "deprecation" in pattern["tags"]

    def test_handles_new_patterns_tag(self, publisher):
        finding = {
            "strategy": "pattern_mining",
            "success": True,
            "detail": "Mined 2 patterns",
            "new_patterns": 2,
        }
        pattern = publisher._convert_finding(finding)
        assert "new_pattern" in pattern["tags"]

    def test_handles_missing_keys(self, publisher):
        finding = {}
        pattern = publisher._convert_finding(finding)
        assert pattern["name"] == "sentinel_unknown"
        assert pattern["description"] == ""

    def test_uses_name_fallback(self, publisher):
        finding = {"name": "custom_strategy", "detail": "test"}
        pattern = publisher._convert_finding(finding)
        assert pattern["name"] == "sentinel_custom_strategy"

    def test_metadata_populated(self, publisher, sample_finding):
        pattern = publisher._convert_finding(sample_finding)
        assert "metadata" in pattern
        assert pattern["metadata"]["success"] is True
        assert pattern["metadata"]["precision_delta"] == 0.02

    def test_timestamp_present(self, publisher, sample_finding):
        pattern = publisher._convert_finding(sample_finding)
        assert "timestamp" in pattern
        assert isinstance(pattern["timestamp"], float)


# ---------------------------------------------------------------------------
# publish_pattern
# ---------------------------------------------------------------------------


class TestPublishPattern:
    def test_publish_writes_to_file_when_cli_unavailable(self, publisher, ingest_path, sample_finding):
        # CLI will fail (engram not installed), should fall back to file
        result = publisher.publish_pattern(sample_finding)
        assert result is True
        assert ingest_path.exists()

        lines = ingest_path.read_text().strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["source"] == "sentinel"
        assert data["name"] == "sentinel_false_positive_review"

    def test_publish_via_cli_success(self, publisher, sample_finding):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = publisher.publish_pattern(sample_finding)
            assert result is True
            mock_run.assert_called_once()

    def test_publish_cli_failure_falls_back_to_file(self, publisher, ingest_path, sample_finding):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            result = publisher.publish_pattern(sample_finding)
            assert result is True  # file fallback succeeds
            assert ingest_path.exists()

    def test_publish_multiple_patterns(self, publisher, ingest_path):
        findings = [
            {"strategy": f"s{i}", "success": True, "detail": f"detail {i}"}
            for i in range(5)
        ]
        for f in findings:
            publisher.publish_pattern(f)

        lines = ingest_path.read_text().strip().split("\n")
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# publish_evolved_strategies
# ---------------------------------------------------------------------------


class TestPublishEvolvedStrategies:
    def test_publishes_strategies_from_state_file(self, tmp_path, strategy_state_file):
        ingest_path = tmp_path / "ingest.jsonl"
        pub = EngramPublisher(ingest_path=ingest_path)

        with patch.object(pub, "_read_strategies") as mock_read:
            mock_read.return_value = [
                {"name": "false_positive_review", "alpha": 8.0, "beta": 3.0,
                 "mean": 0.727, "attempts": 11},
                {"name": "pattern_mining", "alpha": 5.0, "beta": 5.0,
                 "mean": 0.5, "attempts": 10},
            ]
            results = pub.publish_evolved_strategies()

        assert results["total"] == 2
        assert results["pushed"] == 2
        assert results["failed"] == 0

        lines = ingest_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert data["source"] == "sentinel"
            assert data["type"] == "strategy"
            assert data["domain"] == "security"
            assert "thompson_sampling" in data["tags"]

    def test_returns_zero_when_no_strategies(self, tmp_path):
        ingest_path = tmp_path / "ingest.jsonl"
        pub = EngramPublisher(ingest_path=ingest_path)
        results = pub.publish_evolved_strategies()
        assert results["total"] == 0
        assert results["pushed"] == 0

    def test_strategy_pattern_format(self, tmp_path):
        ingest_path = tmp_path / "ingest.jsonl"
        pub = EngramPublisher(ingest_path=ingest_path)

        with patch.object(pub, "_read_strategies") as mock_read:
            mock_read.return_value = [
                {"name": "test_strat", "alpha": 10.0, "beta": 2.0,
                 "mean": 0.833, "attempts": 12, "description": "A test strategy"},
            ]
            pub.publish_evolved_strategies()

        data = json.loads(ingest_path.read_text().strip())
        assert data["name"] == "sentinel_strategy_test_strat"
        assert data["confidence"] == 0.833
        assert data["metadata"]["alpha"] == 10.0
        assert data["metadata"]["beta"] == 2.0
        assert data["metadata"]["attempts"] == 12


# ---------------------------------------------------------------------------
# _read_strategies (state file fallback)
# ---------------------------------------------------------------------------


class TestReadStrategies:
    def test_reads_from_state_file(self, tmp_path, strategy_state_file):
        pub = EngramPublisher()

        with patch(
            "sentinel.engram_publisher.Path.home",
            return_value=tmp_path,
        ):
            # Manually set up the path structure
            sentinel_dir = tmp_path / ".sentinel"
            sentinel_dir.mkdir(exist_ok=True)
            state_path = sentinel_dir / "innovation_state.json"
            state_path.write_text(json.dumps({
                "fp_review": {"alpha": 8.0, "beta": 3.0, "attempts": 11},
                "mining": {"alpha": 5.0, "beta": 5.0, "attempts": 10},
            }))

        # Direct test with the actual file
        strategies = pub._read_strategies_from_file(strategy_state_file)
        assert len(strategies) == 3
        assert all("name" in s for s in strategies)
        assert all("mean" in s for s in strategies)

    def test_returns_empty_when_no_state(self, publisher):
        strategies = publisher._read_strategies(db=None)
        assert strategies == []


class TestReadStrategiesFromFile:
    """Test the file-based strategy reader directly."""

    def test_parses_state_file(self, strategy_state_file):
        pub = EngramPublisher()
        strategies = pub._read_strategies_from_file(strategy_state_file)
        assert len(strategies) == 3

        names = {s["name"] for s in strategies}
        assert "false_positive_review" in names
        assert "pattern_mining" in names
        assert "weight_optimization" in names

        # Check mean calculation
        fp = next(s for s in strategies if s["name"] == "false_positive_review")
        assert fp["mean"] == round(8.0 / 11.0, 3)

    def test_returns_empty_for_missing_file(self, tmp_path):
        pub = EngramPublisher()
        assert pub._read_strategies_from_file(tmp_path / "nonexistent.json") == []

    def test_returns_empty_for_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json")
        pub = EngramPublisher()
        assert pub._read_strategies_from_file(bad_file) == []


# ---------------------------------------------------------------------------
# _push_via_file
# ---------------------------------------------------------------------------


class TestPushViaFile:
    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "ingest.jsonl"
        pub = EngramPublisher(ingest_path=deep_path)
        result = pub._push_via_file({"test": True})
        assert result is True
        assert deep_path.exists()

    def test_appends_jsonl(self, ingest_path, publisher):
        publisher._push_via_file({"first": True})
        publisher._push_via_file({"second": True})
        lines = ingest_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["first"] is True
        assert json.loads(lines[1])["second"] is True
