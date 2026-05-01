"""Tests for sentinel.cloud — cloud-ready execution module.

Covers: CloudConfig, TursoAdapter, WebhookReporter, CloudRunner, PhaseResult,
_TursoDBWrapper, and CLI command registration.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

from sentinel.cloud import (
    CloudConfig,
    CloudRunner,
    PhaseResult,
    TursoAdapter,
    WebhookReporter,
    _TursoDBWrapper,
    _now_iso,
    register_cloud_commands,
)
from sentinel.db import SentinelDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path):
    """SentinelDB backed by a temporary file."""
    db_path = str(tmp_path / "test_cloud.db")
    db = SentinelDB(path=db_path)
    yield db
    db.close()


@pytest.fixture
def temp_db_path(tmp_path):
    """Return a path string for a temp DB (not opened yet)."""
    return str(tmp_path / "test_cloud.db")


@pytest.fixture
def cloud_config():
    """A basic CloudConfig with defaults."""
    return CloudConfig(
        db_url="",
        api_key="test-key",
        scan_queries=["software engineer", "data analyst"],
        scan_interval_hours=2.0,
        report_webhook="",
    )


@pytest.fixture
def runner(temp_db_path):
    """CloudRunner backed by a temporary SQLite database."""
    config = CloudConfig(
        db_url=temp_db_path,
        scan_queries=["test query"],
    )
    r = CloudRunner(db_url=temp_db_path, config=config)
    yield r
    r.close()


# ---------------------------------------------------------------------------
# CloudConfig tests
# ---------------------------------------------------------------------------

class TestCloudConfig:
    def test_from_env_defaults(self):
        """from_env returns sensible defaults when env vars are unset."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = CloudConfig.from_env()
        assert cfg.db_url == ""
        assert cfg.api_key == ""
        assert cfg.scan_queries == []
        assert cfg.scan_interval_hours == 1.0
        assert cfg.report_webhook == ""

    def test_from_env_with_values(self):
        """from_env picks up all SENTINEL_ env vars."""
        env = {
            "SENTINEL_DB_URL": "libsql://my-db.turso.io",
            "SENTINEL_API_KEY": "sk-test-123",
            "SENTINEL_SCAN_QUERIES": "python developer,remote jobs,data science",
            "SENTINEL_SCAN_INTERVAL": "4",
            "SENTINEL_REPORT_WEBHOOK": "https://hooks.slack.com/services/T/B/X",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = CloudConfig.from_env()
        assert cfg.db_url == "libsql://my-db.turso.io"
        assert cfg.api_key == "sk-test-123"
        assert cfg.scan_queries == ["python developer", "remote jobs", "data science"]
        assert cfg.scan_interval_hours == 4.0
        assert cfg.report_webhook == "https://hooks.slack.com/services/T/B/X"

    def test_from_env_invalid_interval(self):
        """Non-numeric interval falls back to 1.0."""
        with patch.dict(os.environ, {"SENTINEL_SCAN_INTERVAL": "not-a-number"}, clear=True):
            cfg = CloudConfig.from_env()
        assert cfg.scan_interval_hours == 1.0

    def test_validate_no_db_url(self):
        """validate reports missing DB URL."""
        cfg = CloudConfig(db_url="")
        issues = cfg.validate()
        assert any("SENTINEL_DB_URL" in i for i in issues)

    def test_validate_no_queries(self):
        """validate reports missing queries."""
        cfg = CloudConfig(db_url="something", scan_queries=[])
        issues = cfg.validate()
        assert any("SENTINEL_SCAN_QUERIES" in i for i in issues)

    def test_validate_fully_configured(self):
        """validate returns fewer issues when fully configured."""
        cfg = CloudConfig(
            db_url="test.db",
            scan_queries=["test"],
        )
        issues = cfg.validate()
        # db_url and queries are set, so those warnings should not appear
        assert not any("SENTINEL_DB_URL" in i for i in issues)
        assert not any("SENTINEL_SCAN_QUERIES" in i for i in issues)

    def test_to_dict_masks_api_key(self):
        """to_dict masks the API key."""
        cfg = CloudConfig(api_key="secret-key-123")
        d = cfg.to_dict()
        assert d["api_key"] == "***"

    def test_to_dict_no_api_key(self):
        """to_dict shows '(not set)' when no API key."""
        cfg = CloudConfig(api_key="")
        d = cfg.to_dict()
        assert d["api_key"] == "(not set)"

    def test_from_env_queries_with_whitespace(self):
        """Queries with surrounding whitespace are trimmed."""
        with patch.dict(os.environ, {"SENTINEL_SCAN_QUERIES": " q1 , q2 , q3 "}, clear=True):
            cfg = CloudConfig.from_env()
        assert cfg.scan_queries == ["q1", "q2", "q3"]

    def test_from_env_empty_queries_string(self):
        """Empty query string yields empty list."""
        with patch.dict(os.environ, {"SENTINEL_SCAN_QUERIES": ""}, clear=True):
            cfg = CloudConfig.from_env()
        assert cfg.scan_queries == []


# ---------------------------------------------------------------------------
# TursoAdapter tests
# ---------------------------------------------------------------------------

class TestTursoAdapter:
    def test_fallback_to_local_sqlite(self, temp_db_path):
        """When no Turso URL given, falls back to local SentinelDB."""
        adapter = TursoAdapter(db_url=temp_db_path)
        assert not adapter.is_turso
        assert adapter.db is not None
        adapter.close()

    def test_fallback_on_empty_url(self):
        """Empty URL uses default SentinelDB."""
        adapter = TursoAdapter(db_url="")
        assert not adapter.is_turso
        adapter.close()

    def test_check_connectivity_local(self, temp_db_path):
        """check_connectivity returns connected=True for local DB."""
        adapter = TursoAdapter(db_url=temp_db_path)
        result = adapter.check_connectivity()
        assert result["connected"] is True
        assert result["backend"] == "sqlite"
        adapter.close()

    def test_turso_url_fallback(self):
        """A libsql:// URL without the library installed falls back to local."""
        adapter = TursoAdapter(db_url="libsql://test.turso.io")
        # Should have fallen back (libsql_experimental likely not installed)
        assert not adapter.is_turso
        adapter.close()

    def test_close_is_safe(self, temp_db_path):
        """close() can be called multiple times safely."""
        adapter = TursoAdapter(db_url=temp_db_path)
        adapter.close()
        adapter.close()  # should not raise


# ---------------------------------------------------------------------------
# _TursoDBWrapper tests
# ---------------------------------------------------------------------------

class TestTursoDBWrapper:
    @pytest.fixture
    def wrapper(self, tmp_path):
        """Create a _TursoDBWrapper with a real SQLite connection."""
        from sentinel.db import SCHEMA
        db_path = str(tmp_path / "wrapper_test.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        conn.commit()
        wrapper = _TursoDBWrapper(conn)
        yield wrapper
        conn.close()

    def test_save_and_get_stats(self, wrapper):
        """save_job + get_stats round-trip."""
        wrapper.save_job({
            "url": "https://example.com/job/1",
            "title": "Test Engineer",
            "company": "TestCo",
            "scam_score": 0.8,
            "risk_level": "high",
        })
        stats = wrapper.get_stats()
        assert stats["total_jobs_analyzed"] == 1
        assert stats["scam_jobs_detected"] == 1  # score >= 0.6

    def test_save_job_with_signals_list(self, wrapper):
        """signals_json as a list gets JSON-serialized."""
        wrapper.save_job({
            "url": "https://example.com/job/2",
            "title": "Dev",
            "signals_json": [{"name": "test_signal"}],
        })
        stats = wrapper.get_stats()
        assert stats["total_jobs_analyzed"] == 1

    def test_get_reports_empty(self, wrapper):
        """get_reports returns empty list when no reports exist."""
        assert wrapper.get_reports() == []

    def test_get_patterns_empty(self, wrapper):
        """get_patterns returns empty list when no patterns exist."""
        assert wrapper.get_patterns(status="active") == []

    def test_save_and_get_patterns(self, wrapper):
        """save_pattern + get_patterns round-trip."""
        wrapper.save_pattern({
            "pattern_id": "test_pattern_1",
            "name": "test_pattern",
            "description": "A test pattern",
            "category": "red_flag",
            "status": "active",
        })
        patterns = wrapper.get_patterns(status="active")
        assert len(patterns) == 1
        assert patterns[0]["name"] == "test_pattern"

    def test_get_source_stats_empty(self, wrapper):
        """get_source_stats returns empty list when no stats exist."""
        assert wrapper.get_source_stats() == []

    def test_get_best_sources_empty(self, wrapper):
        """get_best_sources returns empty list when no stats exist."""
        assert wrapper.get_best_sources() == []

    def test_close_is_noop(self, wrapper):
        """close() is a no-op on the wrapper."""
        wrapper.close()  # should not raise

    def test_path_attribute(self, wrapper):
        assert wrapper.path == "(turso)"


# ---------------------------------------------------------------------------
# WebhookReporter tests
# ---------------------------------------------------------------------------

class TestWebhookReporter:
    def test_not_configured(self):
        """Reporter with no URL is not configured."""
        reporter = WebhookReporter()
        assert not reporter.configured

    def test_configured(self):
        """Reporter with a URL is configured."""
        reporter = WebhookReporter(webhook_url="https://example.com/hook")
        assert reporter.configured

    def test_detect_type_slack(self):
        reporter = WebhookReporter(webhook_url="https://hooks.slack.com/services/T/B/X")
        assert reporter.detect_type() == "slack"

    def test_detect_type_discord(self):
        reporter = WebhookReporter(webhook_url="https://discord.com/api/webhooks/123/abc")
        assert reporter.detect_type() == "discord"

    def test_detect_type_discord_old_domain(self):
        reporter = WebhookReporter(webhook_url="https://discordapp.com/api/webhooks/123/abc")
        assert reporter.detect_type() == "discord"

    def test_detect_type_generic(self):
        reporter = WebhookReporter(webhook_url="https://example.com/webhook")
        assert reporter.detect_type() == "generic"

    def test_detect_type_none(self):
        reporter = WebhookReporter()
        assert reporter.detect_type() == "none"

    def test_format_slack(self):
        reporter = WebhookReporter(webhook_url="https://hooks.slack.com/services/T/B/X")
        results = {
            "summary": {
                "jobs_new": 10,
                "high_risk_count": 2,
                "errors": ["test error"],
                "duration_seconds": 5.5,
            },
        }
        payload = reporter.format_slack(results)
        assert "blocks" in payload
        assert len(payload["blocks"]) >= 2  # header + section + errors

    def test_format_slack_no_errors(self):
        reporter = WebhookReporter(webhook_url="https://hooks.slack.com/services/T/B/X")
        results = {"summary": {"jobs_new": 5, "high_risk_count": 0, "errors": [], "duration_seconds": 1.0}}
        payload = reporter.format_slack(results)
        assert "blocks" in payload

    def test_format_slack_with_phases(self):
        reporter = WebhookReporter(webhook_url="https://hooks.slack.com/services/T/B/X")
        results = {
            "summary": {"jobs_new": 0, "high_risk_count": 0, "errors": [], "duration_seconds": 1.0},
            "phases": {"ingest": {"status": "success"}, "score": {"status": "error"}},
        }
        payload = reporter.format_slack(results)
        # Phases block should be appended
        assert len(payload["blocks"]) >= 3

    def test_format_discord(self):
        reporter = WebhookReporter(webhook_url="https://discord.com/api/webhooks/123/abc")
        results = {
            "summary": {
                "jobs_new": 10,
                "high_risk_count": 3,
                "errors": [],
                "duration_seconds": 2.0,
            },
        }
        payload = reporter.format_discord(results)
        assert "embeds" in payload
        assert payload["embeds"][0]["color"] == 0xFF0000  # red for high risk

    def test_format_discord_no_risk(self):
        reporter = WebhookReporter(webhook_url="https://discord.com/api/webhooks/123/abc")
        results = {"summary": {"jobs_new": 5, "high_risk_count": 0, "errors": [], "duration_seconds": 1.0}}
        payload = reporter.format_discord(results)
        assert payload["embeds"][0]["color"] == 0x00FF00  # green

    def test_format_discord_with_errors(self):
        reporter = WebhookReporter(webhook_url="https://discord.com/api/webhooks/123/abc")
        results = {
            "summary": {"jobs_new": 0, "high_risk_count": 0, "errors": ["e1", "e2"], "duration_seconds": 1.0},
        }
        payload = reporter.format_discord(results)
        fields = payload["embeds"][0]["fields"]
        error_field = [f for f in fields if f["name"] == "Errors"]
        assert len(error_field) == 1

    def test_format_generic(self):
        reporter = WebhookReporter(webhook_url="https://example.com/hook")
        results = {"summary": {"jobs_new": 1}}
        payload = reporter.format_generic(results)
        assert payload["event"] == "sentinel_scan_complete"
        assert "timestamp" in payload
        assert payload["results"] == results

    def test_format_results_auto_detects_slack(self):
        reporter = WebhookReporter(webhook_url="https://hooks.slack.com/services/T/B/X")
        results = {"summary": {"jobs_new": 0, "high_risk_count": 0, "errors": [], "duration_seconds": 0}}
        payload = reporter.format_results(results)
        assert "blocks" in payload  # Slack format

    def test_format_results_auto_detects_discord(self):
        reporter = WebhookReporter(webhook_url="https://discord.com/api/webhooks/1/a")
        results = {"summary": {"jobs_new": 0, "high_risk_count": 0, "errors": [], "duration_seconds": 0}}
        payload = reporter.format_results(results)
        assert "embeds" in payload

    def test_format_results_auto_detects_generic(self):
        reporter = WebhookReporter(webhook_url="https://example.com/hook")
        results = {"summary": {}}
        payload = reporter.format_results(results)
        assert payload["event"] == "sentinel_scan_complete"

    def test_send_no_url(self):
        reporter = WebhookReporter()
        result = reporter.send({"summary": {}})
        assert result["sent"] is False
        assert "no webhook URL" in result["reason"]

    def test_send_to_mock_server(self):
        """Actually POST to a local HTTP server."""
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received["body"] = json.loads(body)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass  # suppress output

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()

        reporter = WebhookReporter(webhook_url=f"http://127.0.0.1:{port}/hook")
        result = reporter.send({"summary": {"jobs_new": 42, "high_risk_count": 0, "errors": [], "duration_seconds": 1.0}})

        thread.join(timeout=5)
        server.server_close()

        assert result["sent"] is True
        assert result["status_code"] == 200
        assert received["body"]["event"] == "sentinel_scan_complete"

    def test_send_http_error(self):
        """Send to a server that returns 500."""
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"error")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = Thread(target=server.handle_request, daemon=True)
        thread.start()

        reporter = WebhookReporter(webhook_url=f"http://127.0.0.1:{port}/hook")
        result = reporter.send({"summary": {"jobs_new": 0, "high_risk_count": 0, "errors": [], "duration_seconds": 0}})

        thread.join(timeout=5)
        server.server_close()

        assert result["sent"] is False
        assert result["status_code"] == 500

    def test_send_connection_refused(self):
        """Send to unreachable host."""
        reporter = WebhookReporter(webhook_url="http://127.0.0.1:1/hook", timeout=1.0)
        result = reporter.send({"summary": {}})
        assert result["sent"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# PhaseResult tests
# ---------------------------------------------------------------------------

class TestPhaseResult:
    def test_to_dict(self):
        pr = PhaseResult(
            phase="ingest",
            status="success",
            started_at="2026-01-01T00:00:00",
            completed_at="2026-01-01T00:01:00",
            duration_seconds=60.0,
            data={"jobs_new": 5},
        )
        d = pr.to_dict()
        assert d["phase"] == "ingest"
        assert d["status"] == "success"
        assert d["data"]["jobs_new"] == 5
        assert d["error"] == ""

    def test_to_dict_error(self):
        pr = PhaseResult(phase="score", status="error", error="connection timeout")
        d = pr.to_dict()
        assert d["status"] == "error"
        assert d["error"] == "connection timeout"

    def test_default_values(self):
        pr = PhaseResult(phase="test", status="success")
        assert pr.data == {}
        assert pr.duration_seconds == 0.0
        assert pr.started_at == ""

    def test_to_dict_is_json_serializable(self):
        pr = PhaseResult(
            phase="innovate",
            status="success",
            data={"strategies": [{"name": "a"}]},
        )
        serialized = json.dumps(pr.to_dict())
        assert "innovate" in serialized


# ---------------------------------------------------------------------------
# CloudRunner tests
# ---------------------------------------------------------------------------

class TestCloudRunner:
    def test_init_with_config(self, temp_db_path):
        config = CloudConfig(db_url=temp_db_path, scan_queries=["test"])
        runner = CloudRunner(config=config)
        assert runner._config.scan_queries == ["test"]
        runner.close()

    def test_init_with_db_url(self, temp_db_path):
        runner = CloudRunner(db_url=temp_db_path)
        assert runner._db_url == temp_db_path
        runner.close()

    def test_queries_from_config(self, temp_db_path):
        config = CloudConfig(db_url=temp_db_path, scan_queries=["q1", "q2"])
        runner = CloudRunner(config=config)
        assert runner._queries() == ["q1", "q2"]
        runner.close()

    def test_queries_defaults(self, temp_db_path):
        config = CloudConfig(db_url=temp_db_path, scan_queries=[])
        runner = CloudRunner(config=config)
        defaults = runner._queries()
        assert len(defaults) == 3
        assert "software engineer" in defaults
        runner.close()

    def test_get_status(self, runner):
        status = runner.get_status()
        assert "config" in status
        assert "database" in status
        assert "webhook" in status
        assert "checked_at" in status
        assert status["database"]["connected"] is True

    def test_get_status_db_backend(self, runner):
        status = runner.get_status()
        assert status["database"]["backend"] == "sqlite"

    def test_run_score(self, runner):
        """run_score completes successfully on an empty database."""
        result = runner.run_score()
        assert result.phase == "score"
        assert result.status == "success"
        assert "precision" in result.data
        assert "recall" in result.data

    def test_run_evolve(self, runner):
        """run_evolve completes successfully on an empty database."""
        result = runner.run_evolve()
        assert result.phase == "evolve"
        assert result.status == "success"
        assert "promoted" in result.data
        assert "deprecated" in result.data

    def test_run_innovate(self, runner):
        """run_innovate completes successfully on an empty database."""
        result = runner.run_innovate(max_strategies=1)
        assert result.phase == "innovate"
        assert result.status == "success"
        assert "strategies_run" in result.data

    def test_run_innovate_max_strategies(self, runner):
        result = runner.run_innovate(max_strategies=2)
        assert result.data.get("strategies_run", 0) <= 2

    @patch("sentinel.cloud.CloudRunner.run_research")
    def test_run_full_cycle_skip_research(self, mock_research, runner):
        """Full cycle with skip_research=True skips the research phase."""
        result = runner.run_full_cycle(skip_research=True)
        mock_research.assert_not_called()
        assert "research" not in result.get("phases", {})
        assert "summary" in result

    def test_run_full_cycle_summary_structure(self, runner):
        """Full cycle returns a summary with expected keys."""
        with patch.object(runner, "run_research", return_value=PhaseResult(
            phase="research", status="success", data={"topics_researched": 0, "patterns_found": 0}
        )):
            result = runner.run_full_cycle(skip_research=False)
        summary = result["summary"]
        assert "started_at" in summary
        assert "completed_at" in summary
        assert "duration_seconds" in summary
        assert "jobs_fetched" in summary
        assert "errors" in summary

    def test_run_full_cycle_with_webhook(self, runner):
        """Full cycle sends webhook notification when configured."""
        # Mock webhook
        mock_webhook = MagicMock()
        mock_webhook.configured = True
        mock_webhook.send.return_value = {"sent": True, "status_code": 200}
        runner._webhook = mock_webhook

        with patch.object(runner, "run_research", return_value=PhaseResult(
            phase="research", status="success", data={}
        )):
            result = runner.run_full_cycle(skip_research=False)

        mock_webhook.send.assert_called_once()
        assert result.get("webhook", {}).get("sent") is True

    def test_run_full_cycle_webhook_failure(self, runner):
        """Full cycle handles webhook failure gracefully."""
        mock_webhook = MagicMock()
        mock_webhook.configured = True
        mock_webhook.send.side_effect = Exception("webhook down")
        runner._webhook = mock_webhook

        with patch.object(runner, "run_research", return_value=PhaseResult(
            phase="research", status="success", data={}
        )):
            result = runner.run_full_cycle(skip_research=False)

        assert result.get("webhook", {}).get("sent") is False

    def test_run_ingest_error_handling(self, temp_db_path):
        """run_ingest returns error result on failure."""
        config = CloudConfig(db_url=temp_db_path)
        runner = CloudRunner(config=config)

        with patch("sentinel.ingest.IngestionPipeline.auto_ingest", side_effect=RuntimeError("test error")):
            result = runner.run_ingest(queries=["test"])

        assert result.status == "error"
        assert "test error" in result.error
        runner.close()

    def test_run_score_error_handling(self, temp_db_path):
        """run_score returns error result on failure."""
        config = CloudConfig(db_url=temp_db_path)
        runner = CloudRunner(config=config)

        with patch("sentinel.flywheel.DetectionFlywheel.run_cycle", side_effect=RuntimeError("flywheel exploded")):
            result = runner.run_score()

        assert result.status == "error"
        assert "flywheel exploded" in result.error
        runner.close()

    def test_run_evolve_error_handling(self, temp_db_path):
        """run_evolve returns error result on failure."""
        config = CloudConfig(db_url=temp_db_path)
        runner = CloudRunner(config=config)

        with patch("sentinel.flywheel.DetectionFlywheel.evolve_patterns", side_effect=RuntimeError("evolve boom")):
            result = runner.run_evolve()

        assert result.status == "error"
        assert "evolve boom" in result.error
        runner.close()

    def test_run_innovate_error_handling(self, temp_db_path):
        """run_innovate returns error result on failure."""
        config = CloudConfig(db_url=temp_db_path)
        runner = CloudRunner(config=config)

        with patch("sentinel.innovation.InnovationEngine.run_cycle", side_effect=RuntimeError("innovate fail")):
            result = runner.run_innovate()

        assert result.status == "error"
        assert "innovate fail" in result.error
        runner.close()

    def test_run_research_error_handling(self, temp_db_path):
        """run_research returns error result on failure."""
        config = CloudConfig(db_url=temp_db_path)
        runner = CloudRunner(config=config)

        with patch("sentinel.research.ResearchEngine.run_cycle", side_effect=RuntimeError("research fail")):
            result = runner.run_research()

        assert result.status == "error"
        assert "research fail" in result.error
        runner.close()

    def test_close_is_idempotent(self, temp_db_path):
        """close() can be called multiple times."""
        runner = CloudRunner(db_url=temp_db_path)
        runner.get_status()  # forces adapter init
        runner.close()
        runner.close()  # should not raise

    def test_run_ingest_result_structure(self, runner):
        """run_ingest result has correct phase name and timing fields."""
        result = runner.run_ingest(queries=["nonexistent query xyz"])
        assert result.phase == "ingest"
        assert result.started_at != ""
        assert result.completed_at != ""
        assert result.duration_seconds >= 0

    def test_run_score_timing(self, runner):
        """run_score records timing information."""
        result = runner.run_score()
        assert result.started_at != ""
        assert result.completed_at != ""
        assert result.duration_seconds >= 0

    def test_run_full_cycle_phases_present(self, runner):
        """Full cycle includes all expected phases."""
        with patch.object(runner, "run_research", return_value=PhaseResult(
            phase="research", status="success", data={}
        )):
            result = runner.run_full_cycle()

        phases = result["phases"]
        assert "ingest" in phases
        assert "score" in phases
        assert "evolve" in phases
        assert "innovate" in phases
        assert "research" in phases

    def test_run_full_cycle_error_accumulation(self, runner):
        """Full cycle accumulates errors from all phases."""
        with patch.object(runner, "run_ingest", return_value=PhaseResult(
            phase="ingest", status="error", error="ingest failed", data={}
        )):
            with patch.object(runner, "run_research", return_value=PhaseResult(
                phase="research", status="error", error="research failed", data={}
            )):
                result = runner.run_full_cycle()

        errors = result["summary"]["errors"]
        assert any("ingest" in e for e in errors)
        assert any("research" in e for e in errors)


# ---------------------------------------------------------------------------
# CLI registration tests
# ---------------------------------------------------------------------------

class TestCLIRegistration:
    def test_register_cloud_commands(self):
        """register_cloud_commands adds 3 commands to a click group."""
        import click

        @click.group()
        @click.pass_context
        def test_cli(ctx):
            ctx.ensure_object(dict)

        register_cloud_commands(test_cli)

        command_names = list(test_cli.commands.keys())
        assert "cloud-scan" in command_names
        assert "cloud-evolve" in command_names
        assert "cloud-status" in command_names

    def test_cloud_status_command(self, tmp_path):
        """cloud-status runs without error."""
        import click
        from click.testing import CliRunner

        @click.group()
        @click.pass_context
        def test_cli(ctx):
            ctx.ensure_object(dict)
            ctx.obj["json"] = True

        register_cloud_commands(test_cli)

        runner = CliRunner()
        db_path = str(tmp_path / "cli_test.db")
        result = runner.invoke(test_cli, ["cloud-status"], env={"SENTINEL_DB_URL": db_path})
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "config" in data
        assert "database" in data

    def test_cloud_evolve_command(self, tmp_path):
        """cloud-evolve runs on empty DB without error."""
        import click
        from click.testing import CliRunner

        @click.group()
        @click.pass_context
        def test_cli(ctx):
            ctx.ensure_object(dict)
            ctx.obj["json"] = True

        register_cloud_commands(test_cli)

        runner = CliRunner()
        db_path = str(tmp_path / "cli_test.db")
        result = runner.invoke(test_cli, ["cloud-evolve", "--db-url", db_path])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "evolve" in data
        assert "innovate" in data


# ---------------------------------------------------------------------------
# _now_iso helper test
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_now_iso_format(self):
        """_now_iso returns a valid ISO 8601 timestamp."""
        ts = _now_iso()
        # Should be parseable
        dt = datetime.fromisoformat(ts)
        assert dt.year >= 2026
