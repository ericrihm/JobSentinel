"""Tests for analyzer.py AI escalation tier (_escalate_to_ai and analyze_job)."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from sentinel.models import JobPosting, ScamSignal, SignalCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job(description: str = "Remote data entry job, apply now!") -> JobPosting:
    return JobPosting(
        title="Data Entry Specialist",
        company="Acme Corp",
        location="Remote",
        description=description,
    )


def _make_signals() -> list[ScamSignal]:
    return [
        ScamSignal(
            name="urgency_language",
            category=SignalCategory.WARNING,
            weight=0.5,
            detail="apply now",
        )
    ]


def _make_haiku_response(text: str) -> MagicMock:
    """Build a mock Anthropic response with a single text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


# ---------------------------------------------------------------------------
# _escalate_to_ai — AI disabled via config
# ---------------------------------------------------------------------------

class TestEscalateToAiDisabled:
    def test_returns_disabled_when_ai_off(self):
        """When ai_enabled=False in config, escalation returns ('', 'disabled')."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = False
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        with patch("sentinel.analyzer.get_config", return_value=mock_config):
            result = analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        assert result == ("", "disabled")


# ---------------------------------------------------------------------------
# _escalate_to_ai — anthropic module not installed
# ---------------------------------------------------------------------------

class TestEscalateToAiNoModule:
    def test_returns_none_when_anthropic_missing(self):
        """When anthropic is not importable, escalation returns ('', 'none')."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", False),
        ):
            result = analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        assert result == ("", "none")


# ---------------------------------------------------------------------------
# _escalate_to_ai — Haiku succeeds
# ---------------------------------------------------------------------------

class TestEscalateToAiHaikuSuccess:
    def test_haiku_response_returned(self):
        """When Haiku returns a text response, it is used as the analysis."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        analysis_text = "This posting appears legitimate based on the details provided."
        haiku_response = _make_haiku_response(analysis_text)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = haiku_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", True),
            patch("sentinel.analyzer._anthropic", mock_anthropic_module),
        ):
            text, tier = analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        assert text == analysis_text
        assert tier == "claude-haiku-4-5"
        mock_client.messages.create.assert_called_once()

    def test_haiku_uses_correct_model_and_params(self):
        """Verify Haiku is called with max_tokens=512 and the correct model."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        haiku_response = _make_haiku_response("Looks suspicious.")

        mock_client = MagicMock()
        mock_client.messages.create.return_value = haiku_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", True),
            patch("sentinel.analyzer._anthropic", mock_anthropic_module),
        ):
            analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5"
        assert call_kwargs["max_tokens"] == 512


# ---------------------------------------------------------------------------
# _escalate_to_ai — Haiku fails, Sonnet takes over
# ---------------------------------------------------------------------------

class TestEscalateToAiHaikuFailsSonnetSucceeds:
    def test_sonnet_used_when_haiku_raises(self):
        """When Haiku raises an exception, Sonnet is tried and its response returned."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        sonnet_text = "Deep analysis: multiple red flags detected."
        sonnet_response = _make_haiku_response(sonnet_text)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            RuntimeError("Haiku quota exceeded"),
            sonnet_response,
        ]

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", True),
            patch("sentinel.analyzer._anthropic", mock_anthropic_module),
        ):
            text, tier = analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        assert text == sonnet_text
        assert tier == "claude-sonnet-4-6"
        assert mock_client.messages.create.call_count == 2

    def test_sonnet_called_with_max_tokens_1024(self):
        """Sonnet fallback uses max_tokens=1024 for deeper analysis."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        sonnet_response = _make_haiku_response("Deep analysis result.")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            Exception("network error"),
            sonnet_response,
        ]

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", True),
            patch("sentinel.analyzer._anthropic", mock_anthropic_module),
        ):
            analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        # Second call is Sonnet
        sonnet_kwargs = mock_client.messages.create.call_args_list[1][1]
        assert sonnet_kwargs["model"] == "claude-sonnet-4-6"
        assert sonnet_kwargs["max_tokens"] == 1024


# ---------------------------------------------------------------------------
# _escalate_to_ai — both tiers fail
# ---------------------------------------------------------------------------

class TestEscalateToAiBothFail:
    def test_returns_failed_when_both_tiers_raise(self):
        """When both Haiku and Sonnet raise, returns ('', 'failed')."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            ConnectionError("network timeout"),
            ConnectionError("network timeout"),
        ]

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", True),
            patch("sentinel.analyzer._anthropic", mock_anthropic_module),
        ):
            text, tier = analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        assert text == ""
        assert tier == "failed"

    def test_returns_failed_on_single_api_exception(self):
        """Single-tier API exception on Haiku plus Sonnet both fail gracefully."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = OSError("Connection refused")

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", True),
            patch("sentinel.analyzer._anthropic", mock_anthropic_module),
        ):
            text, tier = analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        assert text == ""
        assert tier == "failed"


# ---------------------------------------------------------------------------
# analyze_job — use_ai=True with mid-range score (hits AI path)
# ---------------------------------------------------------------------------

class TestAnalyzeJobWithAI:
    def test_analyze_job_use_ai_true_mid_range_score(self):
        """analyze_job with use_ai=True and mid-range score populates ai_analysis."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        ai_text = "This posting has several red flags including vague description."
        haiku_response = _make_haiku_response(ai_text)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = haiku_response

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        # A job description that will produce a mid-range (ambiguous) score
        job = JobPosting(
            title="Flexible Opportunity",
            company="Opportunity Corp",
            description="Work from home. Flexible hours. Apply now! No experience required.",
        )

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", True),
            patch("sentinel.analyzer._anthropic", mock_anthropic_module),
        ):
            # Force the AI path by patching build_result to return a mid-range score
            mock_result = MagicMock()
            mock_result.scam_score = 0.5  # mid-range: triggers AI escalation
            mock_result.ai_analysis = None
            mock_result.ai_tier_used = None
            mock_result.analysis_time_ms = 0.0
            mock_result.risk_level.value = "suspicious"

            with patch("sentinel.analyzer.build_result", return_value=mock_result):
                result = analyzer.analyze_job(job, use_ai=True)

        # AI analysis should have been set
        assert result.ai_analysis == ai_text
        assert result.ai_tier_used == "claude-haiku-4-5"

    def test_analyze_job_ai_called_for_ambiguous_score(self):
        """analyze_job calls _escalate_to_ai when score is in [0.3, 0.7]."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        job = _make_job()

        with patch("sentinel.analyzer.get_config", return_value=mock_config):
            mock_result = MagicMock()
            mock_result.scam_score = 0.45
            mock_result.ai_analysis = None
            mock_result.ai_tier_used = None
            mock_result.analysis_time_ms = 0.0
            mock_result.risk_level.value = "suspicious"

            with (
                patch("sentinel.analyzer.build_result", return_value=mock_result),
                patch("sentinel.analyzer._escalate_to_ai", return_value=("AI says suspicious", "claude-haiku-4-5")) as mock_escalate,
            ):
                analyzer.analyze_job(job, use_ai=True)

        mock_escalate.assert_called_once()


# ---------------------------------------------------------------------------
# analyze_job — use_ai=False skips AI entirely
# ---------------------------------------------------------------------------

class TestAnalyzeJobWithoutAI:
    def test_analyze_job_use_ai_false_skips_ai(self):
        """analyze_job with use_ai=False never calls _escalate_to_ai."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        job = _make_job()

        with patch("sentinel.analyzer.get_config", return_value=mock_config):
            mock_result = MagicMock()
            mock_result.scam_score = 0.5  # would trigger AI if use_ai=True
            mock_result.ai_analysis = None
            mock_result.ai_tier_used = None
            mock_result.analysis_time_ms = 0.0
            mock_result.risk_level.value = "suspicious"

            with (
                patch("sentinel.analyzer.build_result", return_value=mock_result),
                patch("sentinel.analyzer._escalate_to_ai") as mock_escalate,
            ):
                result = analyzer.analyze_job(job, use_ai=False)

        mock_escalate.assert_not_called()

    def test_analyze_job_use_ai_false_no_ai_fields_set(self):
        """With use_ai=False, ai_analysis and ai_tier_used remain unset."""
        from sentinel import analyzer

        job = _make_job("Normal job description with standard requirements.")

        result = analyzer.analyze_job(job, use_ai=False)

        # AI fields should not be populated when use_ai=False
        assert not result.ai_analysis  # empty string or None
        assert not result.ai_tier_used  # empty string or None

    def test_analyze_job_high_score_no_ai_escalation(self):
        """A high scam score (>0.7) does NOT trigger AI escalation regardless."""
        from sentinel import analyzer

        job = _make_job()

        with patch("sentinel.analyzer._escalate_to_ai") as mock_escalate:
            mock_result = MagicMock()
            mock_result.scam_score = 0.9  # above ambiguous range
            mock_result.ai_analysis = None
            mock_result.ai_tier_used = None
            mock_result.analysis_time_ms = 0.0
            mock_result.risk_level.value = "scam"

            with patch("sentinel.analyzer.build_result", return_value=mock_result):
                analyzer.analyze_job(job, use_ai=True)

        mock_escalate.assert_not_called()

    def test_analyze_job_low_score_no_ai_escalation(self):
        """A low scam score (<0.3) does NOT trigger AI escalation."""
        from sentinel import analyzer

        job = _make_job()

        with patch("sentinel.analyzer._escalate_to_ai") as mock_escalate:
            mock_result = MagicMock()
            mock_result.scam_score = 0.1  # below ambiguous range
            mock_result.ai_analysis = None
            mock_result.ai_tier_used = None
            mock_result.analysis_time_ms = 0.0
            mock_result.risk_level.value = "safe"

            with patch("sentinel.analyzer.build_result", return_value=mock_result):
                analyzer.analyze_job(job, use_ai=True)

        mock_escalate.assert_not_called()


# ---------------------------------------------------------------------------
# _escalate_to_ai — ImportError (anthropic not installed simulation)
# ---------------------------------------------------------------------------

class TestEscalateToAiImportError:
    def test_import_error_scenario(self):
        """Simulate the module-level ImportError path: _ANTHROPIC_AVAILABLE=False."""
        # This tests the already-handled import path (module loaded without anthropic).
        # The module-level try/except sets _ANTHROPIC_AVAILABLE=False when anthropic
        # is missing; _escalate_to_ai checks this flag and returns early.
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", False),
        ):
            text, tier = analyzer._escalate_to_ai(_make_job(), [], 0.5)

        assert text == ""
        assert tier == "none"


# ---------------------------------------------------------------------------
# _escalate_to_ai — empty response content
# ---------------------------------------------------------------------------

class TestEscalateToAiEmptyResponse:
    def test_empty_content_falls_through_to_sonnet(self):
        """Haiku returning empty text content falls through to Sonnet."""
        from sentinel import analyzer

        mock_config = MagicMock()
        mock_config.ai_enabled = True
        mock_config.ai_model = "claude-haiku-4-5"
        mock_config.ai_model_deep = "claude-sonnet-4-6"

        # Haiku returns block with empty text
        empty_block = MagicMock()
        empty_block.type = "text"
        empty_block.text = ""
        empty_response = MagicMock()
        empty_response.content = [empty_block]

        sonnet_text = "Sonnet deep analysis: suspicious patterns detected."
        sonnet_response = _make_haiku_response(sonnet_text)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [empty_response, sonnet_response]

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with (
            patch("sentinel.analyzer.get_config", return_value=mock_config),
            patch("sentinel.analyzer._ANTHROPIC_AVAILABLE", True),
            patch("sentinel.analyzer._anthropic", mock_anthropic_module),
        ):
            text, tier = analyzer._escalate_to_ai(_make_job(), _make_signals(), 0.5)

        assert text == sonnet_text
        assert tier == "claude-sonnet-4-6"
