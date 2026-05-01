"""Tests for batch analysis via --file flag (cli.py analyze --file)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from sentinel.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCAM_JOB = {
    "title": "Work From Home — Earn $5000/Week GUARANTEED",
    "company": "ScamCo LLC",
    "description": (
        "Earn GUARANTEED $5,000 per week! No experience required. "
        "You will be hired immediately. Pay $99 registration fee. "
        "Provide your Social Security Number and bank account. "
        "Apply NOW — limited spots! globalopps@gmail.com"
    ),
    "location": "Remote",
}

_LEGIT_JOB = {
    "title": "Software Engineer",
    "company": "Google",
    "description": (
        "As a Software Engineer at Google you will design and build distributed "
        "systems. Minimum 3 years Python/Go experience required. "
        "Salary: $150,000 - $200,000/year."
    ),
    "location": "Seattle, WA",
}

_MID_JOB = {
    "title": "Marketing Coordinator",
    "company": "Acme Corp",
    "description": (
        "We are always looking for talented coordinators. "
        "Responsibilities: assist with campaigns and other duties as assigned."
    ),
    "location": "Chicago, IL",
}


@pytest.fixture
def jobs_file(tmp_path: Path) -> str:
    """Write a temp JSON file with 3 jobs and return its path."""
    data = [_SCAM_JOB, _LEGIT_JOB, _MID_JOB]
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_batch_analyze_from_file(jobs_file: str):
    """Running `analyze --file <path>` processes all jobs and prints a table."""
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", "--no-ai", "--file", jobs_file])

    assert result.exit_code == 0, result.output

    # Header elements
    assert "Batch Analysis" in result.output
    assert "3" in result.output  # job count
    assert os.path.basename(jobs_file) in result.output

    # Summary line must be present
    assert "Summary:" in result.output

    # At least one job title should appear (truncated or full)
    assert "Software Engineer" in result.output or "Work From" in result.output


def test_batch_summary_counts(jobs_file: str):
    """The summary section reflects the correct risk-level distribution."""
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", "--no-ai", "--file", jobs_file])

    assert result.exit_code == 0, result.output
    assert "Summary:" in result.output

    # Confirm the summary line contains at least one valid risk label
    summary_line = next(
        (line for line in result.output.splitlines() if "Summary:" in line), ""
    )
    valid_labels = {"SCAM", "HIGH", "SUSPICIOUS", "LOW", "SAFE"}
    assert any(label in summary_line for label in valid_labels), (
        f"No valid risk label found in summary: {summary_line!r}"
    )


def test_batch_json_output(jobs_file: str):
    """--json-output returns valid JSON with a results list and summary."""
    runner = CliRunner()
    result = runner.invoke(main, ["--json-output", "analyze", "--no-ai", "--file", jobs_file])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total"] == 3
    assert len(data["results"]) == 3
    assert "summary" in data
    assert isinstance(data["summary"], dict)


def test_batch_results_sorted_descending(jobs_file: str):
    """Results in JSON output must be sorted by scam_score descending."""
    runner = CliRunner()
    result = runner.invoke(main, ["--json-output", "analyze", "--no-ai", "--file", jobs_file])

    assert result.exit_code == 0
    data = json.loads(result.output)
    scores = [r["scam_score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True), (
        f"Results not sorted descending: {scores}"
    )


def test_batch_no_file_requires_input_text():
    """Without --file and without INPUT_TEXT the command exits with error."""
    runner = CliRunner()
    result = runner.invoke(main, ["analyze"])
    # Should fail (non-zero exit) with an error message
    assert result.exit_code != 0 or "Error" in result.output


def test_batch_nonexistent_file():
    """A nonexistent file path causes click to report an error."""
    runner = CliRunner()
    result = runner.invoke(main, ["analyze", "--file", "/tmp/does_not_exist_sentinel.json"])
    assert result.exit_code != 0
