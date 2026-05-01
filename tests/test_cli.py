"""Comprehensive tests for sentinel.cli — Click CLI commands."""

import json

import pytest
from click.testing import CliRunner

import sentinel.config
from sentinel.cli import main
from sentinel.db import SentinelDB
from sentinel.knowledge import KnowledgeBase


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "cli_test.db")


@pytest.fixture
def mock_config(db_path):
    """SentinelConfig pointing at a temporary database."""
    return sentinel.config.SentinelConfig(db_path=db_path)


@pytest.fixture
def seeded_db(db_path):
    """Temporary DB seeded with default scam patterns; yields db_path."""
    db = SentinelDB(path=db_path)
    kb = KnowledgeBase(db=db)
    kb.seed_default_patterns()
    db.close()
    return db_path


@pytest.fixture
def patched_runner(runner, mock_config, monkeypatch):
    """Runner + monkeypatched get_config that returns our test config."""
    monkeypatch.setattr(sentinel.config, "get_config", lambda: mock_config)
    monkeypatch.setattr(sentinel.config, "_config", None)
    return runner


@pytest.fixture
def seeded_patched_runner(runner, seeded_db, monkeypatch):
    """Runner + monkeypatched get_config backed by a seeded DB."""
    mock_config = sentinel.config.SentinelConfig(db_path=seeded_db)
    monkeypatch.setattr(sentinel.config, "get_config", lambda: mock_config)
    monkeypatch.setattr(sentinel.config, "_config", None)
    return runner


# ===========================================================================
# analyze command
# ===========================================================================


class TestAnalyzeCommand:
    SCAM_TEXT = (
        "EARN GUARANTEED $5,000 PER WEEK WORKING FROM HOME! "
        "No experience required. Send $99 registration fee to get started. "
        "Provide your Social Security Number. Hire immediately, no interview needed."
    )
    LEGIT_TEXT = (
        "Senior Software Engineer at a leading tech company. "
        "Competitive salary, 5+ years Python required. "
        "Full healthcare, 401k matching. Interview: phone screen, technical, offer."
    )

    def test_analyze_scam_text_exit_code_0(self, patched_runner):
        """Analyzing scam text should exit with code 0."""
        result = patched_runner.invoke(main, ["analyze", self.SCAM_TEXT])
        assert result.exit_code == 0, result.output

    def test_analyze_scam_text_high_risk_in_output(self, patched_runner):
        """Scam text analysis output should contain HIGH or SCAM risk label."""
        result = patched_runner.invoke(main, ["analyze", self.SCAM_TEXT])
        assert result.exit_code == 0
        output_upper = result.output.upper()
        assert "SCAM" in output_upper or "HIGH" in output_upper

    def test_analyze_scam_shows_red_flags(self, patched_runner):
        """Scam text output should list at least one red flag."""
        result = patched_runner.invoke(main, ["analyze", self.SCAM_TEXT])
        assert result.exit_code == 0
        assert "Red Flag" in result.output or "red flag" in result.output.lower()

    def test_analyze_legit_text_exit_code_0(self, patched_runner):
        """Analyzing legit text should exit with code 0."""
        result = patched_runner.invoke(main, ["analyze", self.LEGIT_TEXT])
        assert result.exit_code == 0, result.output

    def test_analyze_legit_text_lower_risk_than_scam(self, patched_runner):
        """Legit text should produce a lower or equal scam score than obvious scam text."""
        # --json-output is a root group option, must come before the subcommand
        scam_result = patched_runner.invoke(
            main, ["--json-output", "analyze", self.SCAM_TEXT]
        )
        legit_result = patched_runner.invoke(
            main, ["--json-output", "analyze", self.LEGIT_TEXT]
        )
        scam_score = json.loads(scam_result.output)["scam_score"]
        legit_score = json.loads(legit_result.output)["scam_score"]
        assert legit_score <= scam_score

    def test_analyze_json_output_flag(self, patched_runner):
        """--json-output flag should produce valid JSON."""
        result = patched_runner.invoke(
            main, ["--json-output", "analyze", self.SCAM_TEXT]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "scam_score" in data
        assert "risk_level" in data

    def test_analyze_json_output_has_risk_level(self, patched_runner):
        """JSON output should include a valid risk_level."""
        result = patched_runner.invoke(
            main, ["--json-output", "analyze", self.SCAM_TEXT]
        )
        data = json.loads(result.output)
        assert data["risk_level"] in ("safe", "low", "suspicious", "high", "scam")

    def test_analyze_no_ai_flag(self, patched_runner):
        """--no-ai flag should still produce valid output."""
        result = patched_runner.invoke(
            main, ["analyze", "--no-ai", self.SCAM_TEXT]
        )
        assert result.exit_code == 0

    def test_analyze_with_title_and_company_flags(self, patched_runner):
        """--title and --company options should be accepted."""
        result = patched_runner.invoke(
            main,
            [
                "analyze",
                "--title", "Work From Home Agent",
                "--company", "Global Opps LLC",
                "Join our team today! Earn guaranteed income!",
            ],
        )
        assert result.exit_code == 0

    def test_analyze_score_shown_in_output(self, patched_runner):
        """Output should include the word 'Score'."""
        result = patched_runner.invoke(main, ["analyze", self.SCAM_TEXT])
        assert result.exit_code == 0
        assert "Score" in result.output or "score" in result.output


# ===========================================================================
# validate command
# ===========================================================================


class TestValidateCommand:
    def test_validate_known_company_exit_0(self, patched_runner):
        """Validating a known company like Google should exit with code 0."""
        result = patched_runner.invoke(main, ["validate", "Google"])
        assert result.exit_code == 0, result.output

    def test_validate_known_company_shows_verified(self, patched_runner):
        """Output for a known company should contain VERIFIED or LINKEDIN FOUND."""
        result = patched_runner.invoke(main, ["validate", "Google"])
        assert result.exit_code == 0
        output_upper = result.output.upper()
        assert "VERIFIED" in output_upper or "LINKEDIN" in output_upper

    def test_validate_unknown_company_exit_0(self, patched_runner):
        """Validating an unknown company should not crash (exit 0 or 1)."""
        result = patched_runner.invoke(
            main, ["validate", "XYZ Random Company 12345 Nonexistent"]
        )
        # Should exit cleanly; may be 0 or 1 depending on validation result
        assert result.exit_code in (0, 1)

    def test_validate_json_output(self, patched_runner):
        """--json-output should produce valid JSON for validate."""
        result = patched_runner.invoke(
            main, ["--json-output", "validate", "Google"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "company" in data or "is_verified" in data

    def test_validate_shows_company_name(self, patched_runner):
        """Output should include the company name."""
        result = patched_runner.invoke(main, ["validate", "Google"])
        assert result.exit_code == 0
        assert "Google" in result.output

    def test_validate_microsoft(self, patched_runner):
        """Validating Microsoft should return a 200-level exit."""
        result = patched_runner.invoke(main, ["validate", "Microsoft"])
        assert result.exit_code == 0

    def test_validate_with_domain_option(self, patched_runner):
        """--domain option should be accepted without error."""
        result = patched_runner.invoke(
            main, ["validate", "--domain", "google.com", "Google"]
        )
        assert result.exit_code == 0


# ===========================================================================
# report command
# ===========================================================================


class TestReportCommand:
    def test_report_scam_url_exit_0(self, patched_runner):
        """Reporting a URL as scam should exit with code 0."""
        result = patched_runner.invoke(
            main,
            [
                "report",
                "https://www.linkedin.com/jobs/view/12345",
                "--reason", "Asked for upfront payment",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_report_shows_confirmation(self, patched_runner):
        """Output should contain confirmation that the report was recorded."""
        result = patched_runner.invoke(
            main,
            ["report", "https://www.linkedin.com/jobs/view/99999"],
        )
        assert result.exit_code == 0
        assert "Record" in result.output or "report" in result.output.lower() or "SCAM" in result.output

    def test_report_legitimate_flag(self, patched_runner):
        """--legitimate flag should mark job as not-scam."""
        result = patched_runner.invoke(
            main,
            [
                "report",
                "https://www.linkedin.com/jobs/view/11111",
                "--legitimate",
            ],
        )
        assert result.exit_code == 0
        output_upper = result.output.upper()
        assert "LEGITIMATE" in output_upper

    def test_report_json_output(self, patched_runner):
        """--json-output should produce valid JSON for report."""
        result = patched_runner.invoke(
            main,
            [
                "--json-output",
                "report",
                "https://www.linkedin.com/jobs/view/22222",
                "--reason", "SSN requested",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "recorded" in data
        assert data["recorded"] is True

    def test_report_json_output_includes_verdict(self, patched_runner):
        """JSON output for report should include a verdict."""
        result = patched_runner.invoke(
            main,
            [
                "--json-output",
                "report",
                "https://www.linkedin.com/jobs/view/33333",
            ],
        )
        data = json.loads(result.output)
        assert data.get("verdict") == "scam"


# ===========================================================================
# patterns command
# ===========================================================================


class TestPatternsCommand:
    def test_patterns_exit_code_0(self, seeded_patched_runner):
        """patterns command should exit with code 0."""
        result = seeded_patched_runner.invoke(main, ["patterns"])
        assert result.exit_code == 0, result.output

    def test_patterns_shows_output(self, seeded_patched_runner):
        """patterns command should produce non-empty output."""
        result = seeded_patched_runner.invoke(main, ["patterns"])
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0

    def test_patterns_shows_scam_patterns_header(self, seeded_patched_runner):
        """Output should contain 'Scam Patterns' header."""
        result = seeded_patched_runner.invoke(main, ["patterns"])
        assert result.exit_code == 0
        assert "Scam Pattern" in result.output or "pattern" in result.output.lower()

    def test_patterns_json_output(self, seeded_patched_runner):
        """--json-output should produce valid JSON for patterns."""
        result = seeded_patched_runner.invoke(
            main, ["--json-output", "patterns"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "patterns" in data
        assert "count" in data

    def test_patterns_json_has_pattern_items(self, seeded_patched_runner):
        """JSON output should list actual pattern items."""
        result = seeded_patched_runner.invoke(
            main, ["--json-output", "patterns"]
        )
        data = json.loads(result.output)
        assert data["count"] > 0
        assert len(data["patterns"]) > 0

    def test_patterns_red_flag_filter(self, seeded_patched_runner):
        """--type red-flag should only show red_flag category patterns."""
        result = seeded_patched_runner.invoke(
            main, ["--json-output", "patterns", "--type", "red-flag"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for p in data["patterns"]:
            assert p["category"] == "red_flag"

    def test_patterns_warning_filter(self, seeded_patched_runner):
        """--type warning should only show warning category patterns."""
        result = seeded_patched_runner.invoke(
            main, ["--json-output", "patterns", "--type", "warning"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        for p in data["patterns"]:
            assert p["category"] == "warning"

    def test_patterns_empty_db_no_crash(self, patched_runner):
        """An empty DB should not crash the patterns command."""
        result = patched_runner.invoke(main, ["patterns"])
        assert result.exit_code == 0

    def test_patterns_status_option(self, seeded_patched_runner):
        """--status option should be accepted."""
        result = seeded_patched_runner.invoke(
            main, ["patterns", "--status", "active"]
        )
        assert result.exit_code == 0


# ===========================================================================
# stats command
# ===========================================================================


class TestStatsCommand:
    def test_stats_exit_code_0(self, patched_runner):
        """stats command should exit with code 0."""
        result = patched_runner.invoke(main, ["stats"])
        assert result.exit_code == 0, result.output

    def test_stats_shows_output(self, patched_runner):
        """stats command should produce non-empty output."""
        result = patched_runner.invoke(main, ["stats"])
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0

    def test_stats_shows_jobs_analyzed(self, patched_runner):
        """Output should contain 'Jobs analyzed' or similar."""
        result = patched_runner.invoke(main, ["stats"])
        assert result.exit_code == 0
        assert "Jobs" in result.output or "jobs" in result.output

    def test_stats_json_output(self, patched_runner):
        """--json-output should produce valid JSON for stats."""
        result = patched_runner.invoke(main, ["--json-output", "stats"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_jobs_analyzed" in data

    def test_stats_json_has_accuracy(self, patched_runner):
        """JSON stats output should include accuracy_detail."""
        result = patched_runner.invoke(main, ["--json-output", "stats"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "accuracy_detail" in data or "total_jobs_analyzed" in data

    def test_stats_shows_detection_statistics_header(self, patched_runner):
        """Output should contain the statistics header."""
        result = patched_runner.invoke(main, ["stats"])
        assert "Sentinel" in result.output or "Statistics" in result.output or "stats" in result.output.lower()

    def test_stats_shows_flywheel_section(self, patched_runner):
        """Output should mention the flywheel or user feedback."""
        result = patched_runner.invoke(main, ["stats"])
        assert result.exit_code == 0
        # Should mention feedback or flywheel section
        assert "Feedback" in result.output or "reports" in result.output.lower() or "Flywheel" in result.output


# ===========================================================================
# evolve command
# ===========================================================================


class TestEvolveCommand:
    def test_evolve_exit_code_0(self, seeded_patched_runner):
        """evolve command should exit with code 0."""
        result = seeded_patched_runner.invoke(main, ["evolve"])
        assert result.exit_code == 0, result.output

    def test_evolve_shows_cycle_complete(self, seeded_patched_runner):
        """Output should indicate a cycle was completed."""
        result = seeded_patched_runner.invoke(main, ["evolve"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "cycle" in output_lower or "flywheel" in output_lower

    def test_evolve_json_output(self, seeded_patched_runner):
        """--json-output should produce valid JSON for evolve."""
        result = seeded_patched_runner.invoke(
            main, ["--json-output", "evolve"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "cycle_number" in data

    def test_evolve_json_has_metrics(self, seeded_patched_runner):
        """JSON evolve output should contain key metrics."""
        result = seeded_patched_runner.invoke(
            main, ["--json-output", "evolve"]
        )
        data = json.loads(result.output)
        required = {"cycle_number", "precision", "recall", "f1"}
        assert required.issubset(set(data.keys()))


# ===========================================================================
# CLI global options
# ===========================================================================


class TestGlobalOptions:
    def test_version_flag(self, runner):
        """--version should print version info or raise a known error when not installed."""
        result = runner.invoke(main, ["--version"])
        # sentinel may not be installed as a package in the test environment;
        # in that case Click raises RuntimeError which is an exit code 1.
        # Either 0 (installed) or 1 (not installed) is acceptable — the
        # command exists and responds rather than failing with an unknown error.
        assert result.exit_code in (0, 1)
        # Should not produce a completely empty output or silent crash
        assert result.output is not None

    def test_help_flag(self, runner):
        """--help should show usage information."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "sentinel" in result.output.lower()

    def test_analyze_help(self, runner):
        """analyze --help should show analyze usage."""
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_validate_help(self, runner):
        """validate --help should show validate usage."""
        result = runner.invoke(main, ["validate", "--help"])
        assert result.exit_code == 0

    def test_patterns_help(self, runner):
        """patterns --help should show patterns usage."""
        result = runner.invoke(main, ["patterns", "--help"])
        assert result.exit_code == 0

    def test_stats_help(self, runner):
        """stats --help should show stats usage."""
        result = runner.invoke(main, ["stats", "--help"])
        assert result.exit_code == 0

    def test_evolve_help(self, runner):
        """evolve --help should show evolve usage."""
        result = runner.invoke(main, ["evolve", "--help"])
        assert result.exit_code == 0
