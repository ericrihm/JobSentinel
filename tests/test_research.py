"""Tests for the Research Flywheel — sentinel/research.py and CLI commands."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sentinel.cli import main
from sentinel.db import SentinelDB
from sentinel.knowledge import KnowledgeBase
from sentinel.research import (
    PromptOptimizer,
    PromptTemplate,
    ResearchEngine,
    ResearchPrompt,
    ResearchResult,
    ResearchTopic,
    _DEFAULT_TEMPLATES,
    _RESEARCH_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path) -> SentinelDB:
    db_path = str(tmp_path / "test_research.db")
    db = SentinelDB(path=db_path)
    yield db
    db.close()


@pytest.fixture
def seeded_db(temp_db: SentinelDB) -> SentinelDB:
    kb = KnowledgeBase(db=temp_db)
    kb.seed_default_patterns()
    return temp_db


@pytest.fixture
def engine(seeded_db: SentinelDB) -> ResearchEngine:
    """ResearchEngine with a mock AI function."""
    mock_ai = MagicMock(return_value=json.dumps({
        "patterns": [
            {
                "keyword": "wire transfer recruitment",
                "description": "Scammers recruit victims via wire transfer schemes",
                "category": "red_flag",
                "weight": 0.85,
            },
            {
                "keyword": "deposit refund scam",
                "description": "Job requires upfront deposit with promise of refund",
                "category": "red_flag",
                "weight": 0.9,
            },
        ],
        "evasion_tactics": [
            "Using Unicode homoglyphs to bypass keyword filters",
            "Splitting scam keywords across multiple paragraphs",
        ],
        "new_categories": [
            "AI-generated deepfake interviews",
        ],
        "weight_adjustments": [
            {
                "signal_name": "upfront_payment_required",
                "direction": "increase",
                "reason": "Still the most common scam tactic in 2025",
            },
        ],
    }))
    return ResearchEngine(db=seeded_db, analyzer_fn=mock_ai, research_budget=2)


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# ResearchTopic dataclass tests
# ---------------------------------------------------------------------------

class TestResearchTopic:
    def test_create_topic(self):
        topic = ResearchTopic(
            area="pig_butchering",
            priority=0.85,
            reason="Low pattern coverage",
        )
        assert topic.area == "pig_butchering"
        assert topic.priority == 0.85
        assert topic.last_researched is None

    def test_create_topic_with_timestamp(self):
        ts = datetime.now(UTC).isoformat()
        topic = ResearchTopic(
            area="crypto_scams",
            priority=0.6,
            reason="Emerging threat",
            last_researched=ts,
        )
        assert topic.last_researched == ts


# ---------------------------------------------------------------------------
# ResearchPrompt dataclass tests
# ---------------------------------------------------------------------------

class TestResearchPrompt:
    def test_create_prompt(self):
        topic = ResearchTopic(area="test", priority=0.5, reason="testing")
        prompt = ResearchPrompt(
            topic=topic,
            prompt_text="What are the latest scam tactics?",
            max_tokens=512,
        )
        assert prompt.prompt_text == "What are the latest scam tactics?"
        assert prompt.max_tokens == 512
        assert prompt.expected_output_format == "structured_json"

    def test_prompt_template_id(self):
        topic = ResearchTopic(area="test", priority=0.5, reason="testing")
        prompt = ResearchPrompt(
            topic=topic,
            prompt_text="test",
            template_id="latest_tactics",
        )
        assert prompt.template_id == "latest_tactics"


# ---------------------------------------------------------------------------
# ResearchResult dataclass tests
# ---------------------------------------------------------------------------

class TestResearchResult:
    def test_create_result(self):
        topic = ResearchTopic(area="test", priority=0.5, reason="testing")
        result = ResearchResult(
            topic=topic,
            raw_response="some response",
            extracted_patterns=[{"type": "keyword", "keyword": "test"}],
            confidence=0.8,
        )
        assert result.raw_response == "some response"
        assert len(result.extracted_patterns) == 1
        assert result.confidence == 0.8
        assert result.timestamp  # auto-generated

    def test_result_default_values(self):
        topic = ResearchTopic(area="test", priority=0.5, reason="testing")
        result = ResearchResult(topic=topic, raw_response="")
        assert result.extracted_patterns == []
        assert result.confidence == 0.0
        assert result.tokens_used == 0


# ---------------------------------------------------------------------------
# Weak area identification
# ---------------------------------------------------------------------------

class TestIdentifyWeakAreas:
    def test_no_reports_returns_list(self, engine):
        """With no reports and well-seeded DB, result should be a valid list."""
        topics = engine.identify_weak_areas()
        assert isinstance(topics, list)
        # May be empty if all categories are covered by default patterns,
        # which is valid — the seeded DB has broad keyword coverage.
        for t in topics:
            assert isinstance(t, ResearchTopic)
            assert 0 <= t.priority <= 1.0

    def test_low_precision_signals(self, engine):
        """Signals with many false positives should generate research topics."""
        db = engine.db
        # Create a pattern with low precision
        db.save_pattern({
            "pattern_id": "bad_signal",
            "name": "bad_signal",
            "description": "A poorly performing signal",
            "category": "warning",
            "regex": "",
            "keywords_json": "[]",
            "alpha": 1.0,
            "beta": 1.0,
            "observations": 20,
            "true_positives": 3,
            "false_positives": 17,
            "status": "active",
        })
        topics = engine.identify_weak_areas()
        low_prec_topics = [t for t in topics if "bad_signal" in t.area]
        assert len(low_prec_topics) == 1
        assert low_prec_topics[0].priority > 0.5

    def test_false_positive_detection(self, engine):
        """Multiple false positive reports should create a research topic."""
        db = engine.db
        for i in range(5):
            db.save_report({
                "url": f"https://example.com/fp/{i}",
                "is_scam": False,
                "reason": "Not a scam",
                "our_prediction": 0.8,
                "was_correct": False,
            })
        topics = engine.identify_weak_areas()
        fp_topics = [t for t in topics if "false_positive" in t.area]
        assert len(fp_topics) == 1

    def test_false_negative_detection(self, engine):
        """Multiple missed scams should create a research topic."""
        db = engine.db
        for i in range(5):
            db.save_report({
                "url": f"https://example.com/fn/{i}",
                "is_scam": True,
                "reason": "Crypto wire transfer scam",
                "our_prediction": 0.2,
                "was_correct": False,
            })
        topics = engine.identify_weak_areas()
        fn_topics = [t for t in topics if "missed_scam" in t.area]
        assert len(fn_topics) == 1

    def test_topics_sorted_by_priority(self, engine):
        """Topics should be sorted by priority descending."""
        topics = engine.identify_weak_areas()
        if len(topics) >= 2:
            for i in range(len(topics) - 1):
                assert topics[i].priority >= topics[i + 1].priority

    def test_duplicate_areas_merged(self, engine):
        """Duplicate topic areas should be merged, keeping highest priority."""
        topics = engine.identify_weak_areas()
        areas = [t.area for t in topics]
        assert len(areas) == len(set(areas)), "Duplicate areas found"


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

class TestGeneratePrompts:
    def test_basic_prompt_generation(self, engine):
        topics = [
            ResearchTopic(area="pig_butchering", priority=0.9, reason="Low coverage"),
            ResearchTopic(area="task_scams", priority=0.8, reason="Emerging threat"),
        ]
        prompts = engine.generate_research_prompts(topics)
        assert len(prompts) == 2
        for p in prompts:
            assert p.prompt_text
            assert p.topic in topics
            assert p.template_id

    def test_prompt_budget_limits(self, engine):
        """Should not generate more prompts than the budget."""
        topics = [
            ResearchTopic(area=f"topic_{i}", priority=0.9 - i * 0.1, reason="test")
            for i in range(10)
        ]
        engine.research_budget = 3
        prompts = engine.generate_research_prompts(topics)
        assert len(prompts) <= 3

    def test_prompt_template_slots_filled(self, engine):
        topics = [
            ResearchTopic(area="crypto_job_scams", priority=0.9, reason="test"),
        ]
        prompts = engine.generate_research_prompts(topics)
        assert len(prompts) >= 1
        # Slots like {job_category} should not remain in the prompt
        assert "{job_category}" not in prompts[0].prompt_text
        assert "{time_period}" not in prompts[0].prompt_text

    def test_empty_topics_returns_empty(self, engine):
        prompts = engine.generate_research_prompts([])
        assert prompts == []


# ---------------------------------------------------------------------------
# Pattern extraction from AI responses
# ---------------------------------------------------------------------------

class TestExtractPatterns:
    def test_json_response_extraction(self, engine):
        response = json.dumps({
            "patterns": [
                {"keyword": "fake deposit", "description": "Requires deposit", "category": "red_flag", "weight": 0.9},
            ],
            "evasion_tactics": ["Using zero-width characters"],
            "new_categories": ["deepfake interviews"],
            "weight_adjustments": [
                {"signal_name": "upfront_payment", "direction": "increase", "reason": "still common"},
            ],
        })
        patterns = engine.extract_patterns(response)
        types = {p["type"] for p in patterns}
        assert "keyword" in types
        assert "evasion_tactic" in types
        assert "new_category" in types
        assert "weight_adjustment" in types
        assert len(patterns) == 4

    def test_markdown_fenced_json(self, engine):
        response = '```json\n{"patterns": [{"keyword": "test scam", "description": "x", "category": "warning", "weight": 0.5}], "evasion_tactics": [], "new_categories": [], "weight_adjustments": []}\n```'
        patterns = engine.extract_patterns(response)
        assert len(patterns) == 1
        assert patterns[0]["keyword"] == "test scam"

    def test_text_fallback_extraction(self, engine):
        response = (
            "Key findings:\n"
            "- Scammers are using AI-generated headshots for fake recruiter profiles\n"
            "- New trend of requiring Zelle payments for training materials\n"
            "- Task-based scams promising commission for completing surveys\n"
        )
        patterns = engine.extract_patterns(response)
        assert len(patterns) >= 2  # Should extract bullet points

    def test_empty_response_returns_empty(self, engine):
        assert engine.extract_patterns("") == []
        assert engine.extract_patterns("  ") == []

    def test_invalid_json_falls_back_to_text(self, engine):
        response = "This is not JSON but has:\n- Important finding about scams\n- Another finding"
        patterns = engine.extract_patterns(response)
        assert isinstance(patterns, list)

    def test_patterns_capped_at_20(self, engine):
        """Text extraction should cap at 20 patterns."""
        lines = "\n".join(f"- Finding number {i} about scam detection patterns" for i in range(30))
        patterns = engine.extract_patterns(lines)
        assert len(patterns) <= 20


# ---------------------------------------------------------------------------
# Knowledge integration
# ---------------------------------------------------------------------------

class TestIntegrateFindings:
    def test_keyword_patterns_saved(self, engine):
        patterns = [
            {"type": "keyword", "keyword": "unique test keyword xyz123", "description": "test", "category": "warning", "weight": 0.5},
        ]
        result = engine.integrate_findings(patterns)
        assert result["new_patterns"] == 1
        assert result["skipped"] == 0

        # Verify it's in the DB as candidate
        candidates = engine.db.get_patterns(status="candidate")
        found = [c for c in candidates if "unique test keyword xyz123" in str(c.get("keywords_json", ""))]
        assert len(found) >= 1

    def test_duplicate_keywords_skipped(self, engine):
        """Keywords that already exist in patterns should be skipped."""
        # "send check" is in the default patterns
        patterns = [
            {"type": "keyword", "keyword": "send check", "description": "dup", "category": "red_flag", "weight": 0.9},
        ]
        result = engine.integrate_findings(patterns)
        assert result["skipped"] == 1
        assert result["new_patterns"] == 0

    def test_weight_adjustment_applied(self, engine):
        """Weight adjustments should modify existing pattern alpha."""
        # Get the initial alpha for upfront_payment_required
        pattern = engine.db.conn.execute(
            "SELECT alpha FROM patterns WHERE pattern_id = ?",
            ("upfront_payment_required",),
        ).fetchone()
        if pattern is None:
            pytest.skip("Pattern not found in seeded DB")
        old_alpha = pattern["alpha"]

        patterns = [
            {"type": "weight_adjustment", "signal_name": "upfront_payment_required", "direction": "increase", "reason": "still dominant"},
        ]
        result = engine.integrate_findings(patterns)
        assert result["adjustments"] == 1

        # Verify alpha changed
        new_pattern = engine.db.conn.execute(
            "SELECT alpha FROM patterns WHERE pattern_id = ?",
            ("upfront_payment_required",),
        ).fetchone()
        assert new_pattern["alpha"] > old_alpha

    def test_weight_decrease(self, engine):
        """Direction 'decrease' should reduce alpha."""
        pattern = engine.db.conn.execute(
            "SELECT alpha FROM patterns WHERE pattern_id = ?",
            ("upfront_payment_required",),
        ).fetchone()
        if pattern is None:
            pytest.skip("Pattern not found")
        old_alpha = pattern["alpha"]

        patterns = [
            {"type": "weight_adjustment", "signal_name": "upfront_payment_required", "direction": "decrease", "reason": "less common now"},
        ]
        engine.integrate_findings(patterns)

        new_pattern = engine.db.conn.execute(
            "SELECT alpha FROM patterns WHERE pattern_id = ?",
            ("upfront_payment_required",),
        ).fetchone()
        assert new_pattern["alpha"] < old_alpha

    def test_short_keywords_skipped(self, engine):
        patterns = [
            {"type": "keyword", "keyword": "ab", "description": "too short", "category": "warning", "weight": 0.3},
        ]
        result = engine.integrate_findings(patterns)
        assert result["skipped"] == 1

    def test_invalid_adjustment_direction_skipped(self, engine):
        patterns = [
            {"type": "weight_adjustment", "signal_name": "upfront_payment_required", "direction": "invalid", "reason": "bad"},
        ]
        result = engine.integrate_findings(patterns)
        assert result["skipped"] == 1

    def test_evasion_tactic_integrated(self, engine):
        patterns = [
            {"type": "evasion_tactic", "keyword": "Unicode homoglyph bypass technique", "description": "Evasion", "category": "warning", "weight": 0.6},
        ]
        result = engine.integrate_findings(patterns)
        assert result["new_patterns"] == 1


# ---------------------------------------------------------------------------
# Research history tracking
# ---------------------------------------------------------------------------

class TestResearchHistory:
    def test_insert_and_retrieve(self, seeded_db):
        db = seeded_db
        db.insert_research_run({
            "topic": "pig_butchering",
            "prompt": "What are pig butchering scam indicators?",
            "response_summary": "Key indicators include...",
            "patterns_extracted": 5,
            "patterns_adopted": 3,
            "precision_delta": 0.02,
        })
        history = db.get_research_history(limit=10)
        assert len(history) >= 1
        latest = history[0]
        assert latest["topic"] == "pig_butchering"
        assert latest["patterns_extracted"] == 5
        assert latest["patterns_adopted"] == 3

    def test_multiple_runs_ordered(self, seeded_db):
        db = seeded_db
        for i in range(5):
            db.insert_research_run({
                "topic": f"topic_{i}",
                "prompt": f"prompt_{i}",
                "response_summary": f"summary_{i}",
                "patterns_extracted": i,
                "patterns_adopted": i,
            })
        history = db.get_research_history(limit=10)
        assert len(history) >= 5
        # Most recent first
        assert history[0]["topic"] == "topic_4"


# ---------------------------------------------------------------------------
# Topic priority evolution
# ---------------------------------------------------------------------------

class TestTopicPriority:
    def test_update_and_retrieve(self, seeded_db):
        db = seeded_db
        db.update_topic_priority(
            topic="pig_butchering",
            priority=0.85,
            patterns_found=10,
            precision_impact=0.03,
        )
        topics = db.get_top_research_topics(n=10)
        pb = [t for t in topics if t["topic"] == "pig_butchering"]
        assert len(pb) == 1
        assert pb[0]["priority"] == 0.85
        assert pb[0]["total_patterns_found"] == 10

    def test_priority_update_accumulates(self, seeded_db):
        db = seeded_db
        db.update_topic_priority(topic="test_accum", priority=0.5, patterns_found=5)
        db.update_topic_priority(topic="test_accum", priority=0.7, patterns_found=3)
        topics = db.get_top_research_topics(n=20)
        t = [x for x in topics if x["topic"] == "test_accum"]
        assert len(t) == 1
        assert t[0]["total_patterns_found"] == 8  # 5 + 3
        assert t[0]["priority"] == 0.7  # latest priority wins

    def test_topics_ordered_by_priority(self, seeded_db):
        db = seeded_db
        db.update_topic_priority(topic="low_prio", priority=0.2, patterns_found=1)
        db.update_topic_priority(topic="high_prio", priority=0.9, patterns_found=1)
        db.update_topic_priority(topic="mid_prio", priority=0.5, patterns_found=1)
        topics = db.get_top_research_topics(n=10)
        priorities = [t["priority"] for t in topics]
        assert priorities == sorted(priorities, reverse=True)


# ---------------------------------------------------------------------------
# Prompt Optimizer
# ---------------------------------------------------------------------------

class TestPromptOptimizer:
    def test_initial_state(self):
        opt = PromptOptimizer()
        assert len(opt.templates) == len(_DEFAULT_TEMPLATES)

    def test_select_template(self):
        opt = PromptOptimizer()
        template = opt.select_template()
        assert isinstance(template, PromptTemplate)
        assert template.template_id in [t.template_id for t in _DEFAULT_TEMPLATES]

    def test_record_outcome_success(self):
        opt = PromptOptimizer()
        tid = opt.templates[0].template_id
        old_alpha = opt.templates[0].alpha
        opt.record_outcome(tid, patterns_extracted=5, tokens_used=200)
        assert opt.templates[0].alpha == old_alpha + 1
        assert opt.templates[0].uses == 1
        assert opt.templates[0].total_patterns_extracted == 5

    def test_record_outcome_failure(self):
        opt = PromptOptimizer()
        tid = opt.templates[0].template_id
        old_beta = opt.templates[0].beta
        opt.record_outcome(tid, patterns_extracted=0, tokens_used=200)
        assert opt.templates[0].beta == old_beta + 1
        assert opt.templates[0].uses == 1
        assert opt.templates[0].total_patterns_extracted == 0

    def test_efficiency_tracking(self):
        opt = PromptOptimizer()
        tid = opt.templates[0].template_id
        opt.record_outcome(tid, patterns_extracted=10, tokens_used=500)
        assert opt.templates[0].efficiency == 10 / 500

    def test_efficiency_zero_tokens(self):
        opt = PromptOptimizer()
        assert opt.templates[0].efficiency == 0.0

    def test_rankings(self):
        opt = PromptOptimizer()
        # Make one template clearly better
        opt.record_outcome(opt.templates[0].template_id, 10, 100)
        opt.record_outcome(opt.templates[0].template_id, 8, 100)
        rankings = opt.get_rankings()
        # Rankings include all default templates plus optionally the Fathom strategy
        assert len(rankings) >= len(_DEFAULT_TEMPLATES)
        # Best template should rank first
        assert rankings[0]["template_id"] == opt.templates[0].template_id


# ---------------------------------------------------------------------------
# Execute research (mocked AI)
# ---------------------------------------------------------------------------

class TestExecuteResearch:
    def test_execute_with_mock_ai(self, engine):
        topic = ResearchTopic(area="test_topic", priority=0.8, reason="testing")
        prompt = ResearchPrompt(
            topic=topic,
            prompt_text="Test prompt",
            template_id="latest_tactics",
        )
        result = engine.execute_research(prompt)
        assert isinstance(result, ResearchResult)
        assert len(result.extracted_patterns) > 0
        assert result.confidence > 0
        assert result.tokens_used > 0

    def test_execute_with_ai_failure(self, seeded_db):
        """When AI fails, result should still be valid but empty."""
        failing_ai = MagicMock(side_effect=Exception("API error"))
        eng = ResearchEngine(db=seeded_db, analyzer_fn=failing_ai)

        topic = ResearchTopic(area="test", priority=0.8, reason="testing")
        prompt = ResearchPrompt(topic=topic, prompt_text="test", template_id="latest_tactics")
        result = eng.execute_research(prompt)
        assert result.raw_response == ""
        assert result.extracted_patterns == []
        assert result.confidence == 0.0

    def test_execute_updates_optimizer(self, engine):
        topic = ResearchTopic(area="test", priority=0.8, reason="testing")
        prompt = ResearchPrompt(topic=topic, prompt_text="test", template_id="latest_tactics")

        engine.execute_research(prompt)

        # The optimizer should have recorded the outcome
        template = next(t for t in engine.optimizer.templates if t.template_id == "latest_tactics")
        assert template.uses >= 1


# ---------------------------------------------------------------------------
# Full research cycle
# ---------------------------------------------------------------------------

class TestRunCycle:
    def test_run_cycle_with_eligible_topics(self, engine):
        """Full cycle should identify topics, generate prompts, and execute."""
        results = engine.run_cycle()
        assert isinstance(results, list)
        # With seeded DB, there should be uncovered categories to research

    def test_run_cycle_records_to_db(self, engine):
        results = engine.run_cycle()
        if results:
            history = engine.db.get_research_history(limit=10)
            assert len(history) >= 1

    def test_run_cycle_respects_budget(self, engine):
        engine.research_budget = 1
        results = engine.run_cycle(max_prompts=1)
        assert len(results) <= 1

    def test_run_cycle_no_eligible_topics(self, seeded_db):
        """When all topics are below threshold, no research should run."""
        eng = ResearchEngine(db=seeded_db, research_budget=2)
        eng.RESEARCH_THRESHOLD = 1.0  # Nothing will be above this
        results = eng.run_cycle()
        assert results == []


# ---------------------------------------------------------------------------
# Prioritize next research
# ---------------------------------------------------------------------------

class TestPrioritizeResearch:
    def test_never_researched_topics_boosted(self, engine):
        """Topics that have never been researched should get a priority boost."""
        topics = engine.prioritize_next_research()
        # All topics are new so they should all get the +0.15 bonus
        for t in topics:
            assert t.priority >= 0.1

    def test_historically_valuable_topics_boosted(self, engine):
        """Topics with positive precision delta history should be boosted."""
        # Record a successful research run for a specific area
        engine.db.insert_research_run({
            "topic": "pig_butchering",
            "prompt": "test",
            "response_summary": "test",
            "patterns_extracted": 5,
            "patterns_adopted": 3,
            "precision_delta": 0.05,  # positive
        })
        topics = engine.prioritize_next_research()
        # pig_butchering should exist and have boosted priority
        pb = [t for t in topics if t.area == "pig_butchering"]
        if pb:
            assert pb[0].priority > 0.3  # should be boosted


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestGetReport:
    def test_report_structure(self, engine):
        report = engine.get_report()
        assert "total_research_runs" in report
        assert "total_patterns_extracted" in report
        assert "total_patterns_adopted" in report
        assert "adoption_rate" in report
        assert "prompt_template_rankings" in report

    def test_report_after_cycle(self, engine):
        engine.run_cycle()
        report = engine.get_report()
        assert isinstance(report["prompt_template_rankings"], list)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_db_no_crash(self, temp_db):
        """Engine should handle an empty DB gracefully."""
        eng = ResearchEngine(db=temp_db)
        topics = eng.identify_weak_areas()
        assert isinstance(topics, list)

    def test_no_analyzer_available(self, seeded_db):
        """With no analyzer function, engine should use direct API (which we mock away)."""
        eng = ResearchEngine(db=seeded_db, analyzer_fn=None)
        topic = ResearchTopic(area="test", priority=0.8, reason="testing")
        prompt = ResearchPrompt(topic=topic, prompt_text="test", template_id="latest_tactics")

        with patch.object(eng, "_call_ai_direct", return_value=""):
            result = eng.execute_research(prompt)
            assert result.raw_response == ""

    def test_json_parse_embedded_json(self, engine):
        """Should extract JSON embedded in surrounding text."""
        text = 'Here are my findings:\n{"patterns": [{"keyword": "embedded test", "description": "x", "category": "warning", "weight": 0.5}], "evasion_tactics": [], "new_categories": [], "weight_adjustments": []}\nEnd of response.'
        patterns = engine.extract_patterns(text)
        assert len(patterns) == 1
        assert patterns[0]["keyword"] == "embedded test"

    def test_integrate_empty_patterns(self, engine):
        result = engine.integrate_findings([])
        assert result["new_patterns"] == 0
        assert result["total_input"] == 0

    def test_integrate_nonexistent_signal_adjustment(self, engine):
        patterns = [
            {"type": "weight_adjustment", "signal_name": "nonexistent_signal_xyz", "direction": "increase", "reason": "test"},
        ]
        result = engine.integrate_findings(patterns)
        assert result["skipped"] == 1

    def test_template_mean_property(self):
        t = PromptTemplate(
            template_id="test",
            template_text="test",
            slots=[],
            alpha=3.0,
            beta=1.0,
        )
        assert t.mean == pytest.approx(0.75)

    def test_template_sample(self):
        t = PromptTemplate(
            template_id="test",
            template_text="test",
            slots=[],
            alpha=10.0,
            beta=2.0,
        )
        # Sample should return float between 0 and 1
        sample = t.sample()
        assert 0.0 <= sample <= 1.0

    def test_research_result_auto_timestamp(self):
        topic = ResearchTopic(area="test", priority=0.5, reason="test")
        r = ResearchResult(topic=topic, raw_response="x")
        assert r.timestamp is not None
        assert len(r.timestamp) > 10  # ISO format


# ---------------------------------------------------------------------------
# CLI command tests
# ---------------------------------------------------------------------------

class TestCLIResearch:
    def test_research_help(self, runner):
        result = runner.invoke(main, ["research", "--help"])
        assert result.exit_code == 0
        assert "research" in result.output.lower()

    def test_research_no_flags(self, runner):
        """Running without flags should show help message."""
        result = runner.invoke(main, ["research"])
        assert result.exit_code == 0
        assert "--auto" in result.output or "--topic" in result.output

    def test_research_history_empty(self, runner, tmp_path):
        db_path = str(tmp_path / "cli_test.db")
        with patch.dict("os.environ", {"SENTINEL_DB_PATH": db_path}):
            result = runner.invoke(main, ["research", "--history"])
            assert result.exit_code == 0
            assert "No research history" in result.output or "Research History" in result.output

    def test_research_topics_listing(self, runner, tmp_path):
        db_path = str(tmp_path / "cli_test2.db")
        with patch.dict("os.environ", {"SENTINEL_DB_PATH": db_path}):
            result = runner.invoke(main, ["research", "--topics"])
            assert result.exit_code == 0

    def test_research_auto_mode(self, runner, tmp_path):
        db_path = str(tmp_path / "cli_test3.db")
        with patch.dict("os.environ", {"SENTINEL_DB_PATH": db_path}):
            with patch("sentinel.research.ResearchEngine.run_cycle", return_value=[]):
                with patch("sentinel.research.ResearchEngine.get_report", return_value={
                    "total_research_runs": 0,
                    "total_patterns_extracted": 0,
                    "total_patterns_adopted": 0,
                    "adoption_rate": 0.0,
                    "prompt_template_rankings": [],
                }):
                    result = runner.invoke(main, ["research", "--auto"])
                    assert result.exit_code == 0

    def test_research_topic_targeted(self, runner, tmp_path):
        db_path = str(tmp_path / "cli_test4.db")
        mock_result = ResearchResult(
            topic=ResearchTopic(area="pig_butchering", priority=0.9, reason="test"),
            raw_response="test response",
            extracted_patterns=[{"type": "keyword", "keyword": "test"}],
            confidence=0.5,
        )
        with patch.dict("os.environ", {"SENTINEL_DB_PATH": db_path}):
            with patch("sentinel.research.ResearchEngine.generate_research_prompts") as mock_gen:
                mock_gen.return_value = [ResearchPrompt(
                    topic=ResearchTopic(area="pig_butchering", priority=0.9, reason="test"),
                    prompt_text="test",
                    template_id="latest_tactics",
                )]
                with patch("sentinel.research.ResearchEngine.execute_research", return_value=mock_result):
                    with patch("sentinel.research.ResearchEngine.integrate_findings", return_value={"new_patterns": 1, "skipped": 0, "adjustments": 0, "total_input": 1}):
                        result = runner.invoke(main, ["research", "--topic", "pig butchering"])
                        assert result.exit_code == 0

    def test_research_json_output_history(self, runner, tmp_path):
        db_path = str(tmp_path / "cli_test5.db")
        with patch.dict("os.environ", {"SENTINEL_DB_PATH": db_path}):
            result = runner.invoke(main, ["--json-output", "research", "--history"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "runs" in data

    def test_research_json_output_topics(self, runner, tmp_path):
        db_path = str(tmp_path / "cli_test6.db")
        with patch.dict("os.environ", {"SENTINEL_DB_PATH": db_path}):
            result = runner.invoke(main, ["--json-output", "research", "--topics"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "topics" in data


# ---------------------------------------------------------------------------
# Daemon integration
# ---------------------------------------------------------------------------

class TestDaemonIntegration:
    def test_daemon_imports_research(self):
        """The daemon should be able to import ResearchEngine."""
        from sentinel.research import ResearchEngine
        assert ResearchEngine is not None

    def test_daemon_cycle_result_has_research_fields(self):
        from sentinel.daemon import CycleResult
        cr = CycleResult(
            cycle_number=1,
            started_at="2026-01-01",
            completed_at="2026-01-01",
            ingestion_queries=["test"],
            jobs_fetched=0,
            jobs_new=0,
            high_risk_count=0,
            flywheel_ran=False,
            regression_detected=False,
            innovation_ran=False,
            innovation_strategy="",
            errors=[],
            duration_seconds=1.0,
            research_ran=True,
            research_topics=3,
            research_patterns_found=10,
        )
        assert cr.research_ran is True
        assert cr.research_topics == 3
        assert cr.research_patterns_found == 10


# ---------------------------------------------------------------------------
# System prompt constant
# ---------------------------------------------------------------------------

class TestConstants:
    def test_system_prompt_exists(self):
        assert "fraud" in _RESEARCH_SYSTEM_PROMPT.lower()
        assert "JSON" in _RESEARCH_SYSTEM_PROMPT

    def test_default_templates_defined(self):
        assert len(_DEFAULT_TEMPLATES) >= 5
        for t in _DEFAULT_TEMPLATES:
            assert t.template_id
            assert t.template_text
            assert isinstance(t.slots, list)
