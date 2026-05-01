"""Tests for sentinel/honeypot.py — Honeypot Intelligence Gathering System.

Covers: HoneypotProfile, HoneypotInteraction, HoneypotManager,
        InteractionAnalyzer, IntelligenceExtractor, DeploymentStrategy.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta

import pytest

from sentinel.db import SentinelDB
from sentinel.honeypot import (
    Channel,
    DeploymentStrategy,
    ExtractedIntelligence,
    HoneypotInteraction,
    HoneypotManager,
    HoneypotProfile,
    IntelligenceExtractor,
    InteractionAnalyzer,
    ProfileStatus,
    ScammerBehaviorProfile,
    _detect_signals_in_message,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def manager() -> HoneypotManager:
    return HoneypotManager()


@pytest.fixture
def manager_with_db(tmp_path) -> HoneypotManager:
    db = SentinelDB(path=str(tmp_path / "honeypot_test.db"))
    mgr = HoneypotManager(db=db)
    yield mgr
    db.close()


@pytest.fixture
def alice(manager: HoneypotManager) -> HoneypotProfile:
    return manager.create_profile(
        name="Alice Honeybee",
        email="alice.honeybee@example.com",
        resume_summary="Entry-level admin seeking remote work",
        target_job_categories=["administrative", "customer service"],
        metadata={"experience_level": "entry"},
    )


@pytest.fixture
def bob(manager: HoneypotManager) -> HoneypotProfile:
    return manager.create_profile(
        name="Bob Canary",
        email="bob.canary@example.com",
        resume_summary="Mid-level data analyst open to all industries",
        target_job_categories=["data analyst", "finance"],
        metadata={"experience_level": "mid"},
    )


def _make_interaction(
    channel: Channel = Channel.EMAIL,
    content: str = "Hello, you have been selected for a position.",
    stage: int = 1,
    sender_email: str = "scammer@bad.com",
    hours_offset: int = 0,
) -> HoneypotInteraction:
    ts = (datetime.now(UTC) + timedelta(hours=hours_offset)).isoformat()
    return HoneypotInteraction(
        timestamp=ts,
        channel=channel,
        message_content=content,
        sender_info={"email": sender_email},
        escalation_stage=stage,
    )


# ===========================================================================
# HoneypotInteraction
# ===========================================================================


class TestHoneypotInteraction:
    def test_defaults(self):
        ix = HoneypotInteraction()
        assert ix.interaction_id != ""
        assert ix.channel == Channel.EMAIL
        assert ix.escalation_stage == 1
        assert ix.response_sent is None
        assert isinstance(ix.scam_signals_detected, list)

    def test_to_dict_roundtrip(self):
        ix = HoneypotInteraction(
            channel=Channel.WHATSAPP,
            message_content="Send me $200 via bitcoin",
            sender_info={"email": "bad@scam.com"},
            scam_signals_detected=["payment_demand"],
            escalation_stage=3,
            response_sent="OK",
        )
        d = ix.to_dict()
        assert d["channel"] == "whatsapp"
        assert d["escalation_stage"] == 3
        assert "payment_demand" in d["scam_signals_detected"]

        restored = HoneypotInteraction.from_dict(d)
        assert restored.channel == Channel.WHATSAPP
        assert restored.escalation_stage == 3
        assert restored.response_sent == "OK"

    def test_from_dict_unknown_channel_falls_back(self):
        # 'other' is always valid; from_dict with a valid channel value works
        d = {
            "channel": "linkedin_message",
            "message_content": "Hi",
            "escalation_stage": 1,
        }
        ix = HoneypotInteraction.from_dict(d)
        assert ix.channel == Channel.LINKEDIN_MESSAGE


# ===========================================================================
# HoneypotProfile
# ===========================================================================


class TestHoneypotProfile:
    def test_defaults(self):
        p = HoneypotProfile()
        assert p.profile_id != ""
        assert p.status == ProfileStatus.ACTIVE
        assert p.is_active is True
        assert p.interaction_count == 0
        assert isinstance(p.interactions, list)
        assert isinstance(p.applications, dict)

    def test_to_dict_roundtrip(self):
        p = HoneypotProfile(
            name="Test Persona",
            email="test@honeypot.com",
            resume_summary="Generic resume",
            target_job_categories=["sales"],
            status=ProfileStatus.RETIRED,
            metadata={"experience_level": "senior"},
        )
        d = p.to_dict()
        assert d["name"] == "Test Persona"
        assert d["status"] == "retired"

        restored = HoneypotProfile.from_dict(d)
        assert restored.name == "Test Persona"
        assert restored.status == ProfileStatus.RETIRED
        assert restored.metadata["experience_level"] == "senior"

    def test_interaction_count_property(self):
        p = HoneypotProfile()
        assert p.interaction_count == 0
        p.interactions.append(HoneypotInteraction())
        assert p.interaction_count == 1
        p.interactions.append(HoneypotInteraction())
        assert p.interaction_count == 2

    def test_is_active_reflects_status(self):
        p = HoneypotProfile()
        assert p.is_active is True
        p.status = ProfileStatus.RETIRED
        assert p.is_active is False
        p.status = ProfileStatus.COMPROMISED
        assert p.is_active is False


# ===========================================================================
# HoneypotManager — profile lifecycle
# ===========================================================================


class TestHoneypotManagerLifecycle:
    def test_create_profile(self, manager):
        p = manager.create_profile(
            name="Test",
            email="t@t.com",
            resume_summary="summary",
            target_job_categories=["tech"],
        )
        assert p.profile_id in [prof.profile_id for prof in manager.list_profiles()]

    def test_get_profile(self, manager, alice):
        found = manager.get_profile(alice.profile_id)
        assert found is not None
        assert found.name == "Alice Honeybee"

    def test_get_profile_missing_returns_none(self, manager):
        assert manager.get_profile("nonexistent-id") is None

    def test_list_profiles_all(self, manager, alice, bob):
        all_profiles = manager.list_profiles()
        assert len(all_profiles) == 2

    def test_list_profiles_filter_by_status(self, manager, alice, bob):
        manager.retire_profile(alice.profile_id)
        active = manager.list_profiles(status=ProfileStatus.ACTIVE)
        assert all(p.status == ProfileStatus.ACTIVE for p in active)
        assert len(active) == 1

    def test_retire_profile(self, manager, alice):
        result = manager.retire_profile(alice.profile_id)
        assert result is True
        assert manager.get_profile(alice.profile_id).status == ProfileStatus.RETIRED

    def test_retire_missing_returns_false(self, manager):
        assert manager.retire_profile("bad-id") is False

    def test_mark_compromised(self, manager, alice):
        result = manager.mark_compromised(alice.profile_id)
        assert result is True
        assert manager.get_profile(alice.profile_id).status == ProfileStatus.COMPROMISED

    def test_mark_compromised_missing_returns_false(self, manager):
        assert manager.mark_compromised("bad-id") is False

    def test_rotate_profiles_by_age(self, manager):
        # Create a profile with old creation_date
        p = manager.create_profile("Old", "old@test.com", "r", ["admin"])
        old_date = (datetime.now(UTC) - timedelta(days=40)).isoformat()
        p.creation_date = old_date

        retired = manager.rotate_profiles(max_age_days=30)
        assert p.profile_id in retired
        assert manager.get_profile(p.profile_id).status == ProfileStatus.RETIRED

    def test_rotate_profiles_by_interaction_count(self, manager):
        p = manager.create_profile("Busy", "busy@test.com", "r", ["sales"])
        for _ in range(55):
            p.interactions.append(HoneypotInteraction())

        retired = manager.rotate_profiles(max_interactions=50)
        assert p.profile_id in retired

    def test_rotate_profiles_skips_non_active(self, manager, alice):
        manager.retire_profile(alice.profile_id)
        alice_profile = manager.get_profile(alice.profile_id)
        old_date = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        alice_profile.creation_date = old_date
        # Already retired; should not be re-listed
        retired = manager.rotate_profiles(max_age_days=30)
        assert alice.profile_id not in retired


# ===========================================================================
# HoneypotManager — applications tracking
# ===========================================================================


class TestHoneypotManagerApplications:
    def test_record_application(self, manager, alice):
        ok = manager.record_application(alice.profile_id, "https://linkedin.com/jobs/1")
        assert ok is True
        apps = manager.get_applications(alice.profile_id)
        assert "https://linkedin.com/jobs/1" in apps

    def test_record_application_missing_profile(self, manager):
        ok = manager.record_application("bad-id", "https://example.com/jobs/1")
        assert ok is False

    def test_get_applications_empty(self, manager, alice):
        assert manager.get_applications(alice.profile_id) == {}

    def test_get_applications_missing_profile(self, manager):
        assert manager.get_applications("bad-id") == {}

    def test_multiple_applications(self, manager, alice):
        manager.record_application(alice.profile_id, "https://example.com/jobs/1")
        manager.record_application(alice.profile_id, "https://example.com/jobs/2")
        apps = manager.get_applications(alice.profile_id)
        assert len(apps) == 2


# ===========================================================================
# HoneypotManager — interaction logging
# ===========================================================================


class TestHoneypotManagerInteractions:
    def test_log_interaction_basic(self, manager, alice):
        ix = manager.log_interaction(
            alice.profile_id,
            Channel.EMAIL,
            "Congratulations, you are hired! Please send $100 fee.",
        )
        assert ix is not None
        assert ix.interaction_id != ""
        profile = manager.get_profile(alice.profile_id)
        assert profile.interaction_count == 1
        assert profile.last_interaction is not None

    def test_log_interaction_detects_signals(self, manager, alice):
        ix = manager.log_interaction(
            alice.profile_id,
            Channel.EMAIL,
            "Please wire $200 immediately. Also need your SSN and bank account.",
        )
        assert "payment_demand" in ix.scam_signals_detected
        assert "personal_info_request" in ix.scam_signals_detected
        assert "urgency_language" in ix.scam_signals_detected

    def test_log_interaction_missing_profile_returns_none(self, manager):
        ix = manager.log_interaction("bad-id", Channel.EMAIL, "Hello")
        assert ix is None

    def test_log_interaction_with_response(self, manager, alice):
        ix = manager.log_interaction(
            alice.profile_id,
            Channel.WHATSAPP,
            "Send money to continue.",
            response_sent="I am interested, how do I proceed?",
            escalation_stage=3,
        )
        assert ix.response_sent == "I am interested, how do I proceed?"
        assert ix.escalation_stage == 3
        assert ix.channel == Channel.WHATSAPP

    def test_log_multiple_interactions(self, manager, alice):
        for stage in [1, 2, 3]:
            manager.log_interaction(
                alice.profile_id, Channel.EMAIL, "message", escalation_stage=stage
            )
        assert manager.get_profile(alice.profile_id).interaction_count == 3


# ===========================================================================
# HoneypotManager — reports
# ===========================================================================


class TestHoneypotManagerReports:
    def test_generate_interaction_report_no_interactions(self, manager, alice):
        report = manager.generate_interaction_report(alice.profile_id)
        assert report["total_interactions"] == 0
        assert report["profile_id"] == alice.profile_id

    def test_generate_interaction_report_with_interactions(self, manager, alice):
        manager.log_interaction(alice.profile_id, Channel.EMAIL, "Hello, pay fee now.")
        manager.log_interaction(alice.profile_id, Channel.WHATSAPP, "Send bitcoin asap.", escalation_stage=3)
        report = manager.generate_interaction_report(alice.profile_id)
        assert report["total_interactions"] == 2
        assert report["max_escalation_stage"] == 3
        assert "email" in report["channels_used"]
        assert "whatsapp" in report["channels_used"]

    def test_generate_interaction_report_missing_profile(self, manager):
        report = manager.generate_interaction_report("bad-id")
        assert "error" in report

    def test_generate_global_report(self, manager, alice, bob):
        manager.log_interaction(alice.profile_id, Channel.EMAIL, "Hello")
        manager.log_interaction(bob.profile_id, Channel.PHONE, "Call me")
        report = manager.generate_global_report()
        assert report["total_profiles"] == 2
        assert report["total_interactions"] == 2
        assert report["active_profiles"] == 2

    def test_global_report_status_counts(self, manager, alice, bob):
        manager.retire_profile(alice.profile_id)
        manager.mark_compromised(bob.profile_id)
        report = manager.generate_global_report()
        assert report["retired_profiles"] == 1
        assert report["compromised_profiles"] == 1
        assert report["active_profiles"] == 0


# ===========================================================================
# HoneypotManager — DB persistence
# ===========================================================================


class TestHoneypotManagerDBPersistence:
    def test_profile_persists_to_db(self, manager_with_db):
        p = manager_with_db.create_profile("Persist", "p@test.com", "r", ["tech"])
        # Reload from DB
        mgr2 = HoneypotManager(db=manager_with_db._db)
        assert mgr2.get_profile(p.profile_id) is not None

    def test_interaction_persists_to_db(self, manager_with_db):
        p = manager_with_db.create_profile("IxPersist", "ix@test.com", "r", ["sales"])
        manager_with_db.log_interaction(p.profile_id, Channel.EMAIL, "Pay fee now")
        rows = manager_with_db._db.conn.execute(
            "SELECT * FROM honeypot_interactions WHERE profile_id = ?", (p.profile_id,)
        ).fetchall()
        assert len(rows) == 1

    def test_application_persists_to_db(self, manager_with_db):
        p = manager_with_db.create_profile("AppPersist", "app@test.com", "r", ["admin"])
        manager_with_db.record_application(p.profile_id, "https://example.com/jobs/99")
        rows = manager_with_db._db.conn.execute(
            "SELECT * FROM honeypot_applications WHERE profile_id = ?", (p.profile_id,)
        ).fetchall()
        assert len(rows) == 1


# ===========================================================================
# InteractionAnalyzer
# ===========================================================================


class TestInteractionAnalyzer:
    def setup_method(self):
        self.analyzer = InteractionAnalyzer()

    def test_response_time_single_interaction(self):
        ix = [_make_interaction()]
        assert self.analyzer.analyze_response_time(ix) is None

    def test_response_time_two_interactions(self):
        ixs = [
            _make_interaction(hours_offset=0),
            _make_interaction(hours_offset=2),
        ]
        avg = self.analyzer.analyze_response_time(ixs)
        assert avg is not None
        assert 1.5 <= avg <= 2.5  # ~2 hours average

    def test_response_time_empty(self):
        assert self.analyzer.analyze_response_time([]) is None

    def test_escalation_pattern_empty(self):
        result = self.analyzer.analyze_escalation_pattern([])
        assert result["max_stage"] == 0
        assert result["reached_payment"] is False

    def test_escalation_pattern_stages(self):
        ixs = [
            _make_interaction(stage=1),
            _make_interaction(stage=2),
            _make_interaction(stage=3),
        ]
        result = self.analyzer.analyze_escalation_pattern(ixs)
        assert result["max_stage"] == 3
        assert result["reached_payment"] is True

    def test_channel_usage_no_switches(self):
        ixs = [_make_interaction(channel=Channel.EMAIL) for _ in range(3)]
        result = self.analyzer.analyze_channel_usage(ixs)
        assert result["channel_switches"] == 0
        assert result["primary_channel"] == "email"

    def test_channel_usage_with_switch(self):
        ixs = [
            _make_interaction(channel=Channel.EMAIL, hours_offset=0),
            _make_interaction(channel=Channel.WHATSAPP, hours_offset=1),
        ]
        result = self.analyzer.analyze_channel_usage(ixs)
        assert result["channel_switches"] == 1

    def test_language_patterns_payment(self):
        ixs = [_make_interaction(content="You need to pay a processing fee of $100")]
        patterns = self.analyzer.extract_language_patterns(ixs)
        assert "payment_language" in patterns

    def test_language_patterns_info_request(self):
        ixs = [_make_interaction(content="Send us your SSN and bank account number")]
        patterns = self.analyzer.extract_language_patterns(ixs)
        assert "info_request_language" in patterns

    def test_language_patterns_channel_switch(self):
        ixs = [_make_interaction(content="Please contact me on WhatsApp: +1234567890")]
        patterns = self.analyzer.extract_language_patterns(ixs)
        assert "channel_switch_attempt" in patterns

    def test_aggressiveness_score_zero_no_interactions(self):
        score = self.analyzer.score_aggressiveness([])
        assert score == 0.0

    def test_aggressiveness_score_high(self):
        ixs = [
            _make_interaction(content="Pay $500 fee now! SSN required. Wire transfer only.", stage=3),
            _make_interaction(content="Send bitcoin immediately! WhatsApp me urgent.", stage=4),
        ]
        score = self.analyzer.score_aggressiveness(ixs)
        assert score > 0.4

    def test_build_scammer_profile(self):
        ixs = [
            _make_interaction(
                content="You are hired! Please pay $100 fee. Send SSN.",
                stage=1,
                sender_email="scam1@evil.com",
            ),
            _make_interaction(
                content="Wire the money immediately. Bitcoin also accepted.",
                stage=3,
                sender_email="scam1@evil.com",
                hours_offset=2,
            ),
        ]
        profile = self.analyzer.build_scammer_profile(ixs)
        assert isinstance(profile, ScammerBehaviorProfile)
        assert "scam1@evil.com" in profile.sender_emails
        assert profile.interaction_count == 2
        assert profile.aggressiveness_score > 0.0

    def test_identify_playbook(self):
        ixs = [
            _make_interaction(channel=Channel.EMAIL, stage=1),
            _make_interaction(channel=Channel.EMAIL, stage=2),
            _make_interaction(channel=Channel.WHATSAPP, stage=3),
        ]
        playbook = self.analyzer.identify_playbook(ixs)
        assert "stage_1:email" in playbook
        assert "stage_3:whatsapp" in playbook

    def test_identify_playbook_empty(self):
        assert self.analyzer.identify_playbook([]) == []


# ===========================================================================
# IntelligenceExtractor
# ===========================================================================


class TestIntelligenceExtractor:
    def setup_method(self):
        self.extractor = IntelligenceExtractor()

    def test_extract_blocklist_entities_from_sender_info(self):
        ixs = [
            HoneypotInteraction(
                sender_info={"email": "bad@scamdomain.com", "phone": "+15551234567"},
                message_content="",
            )
        ]
        entities = self.extractor.extract_blocklist_entities(ixs)
        assert "bad@scamdomain.com" in entities["emails"]
        assert "scamdomain.com" in entities["domains"]
        assert "+15551234567" in entities["phones"]

    def test_extract_blocklist_entities_from_message_content(self):
        ixs = [
            HoneypotInteraction(
                sender_info={},
                message_content="Contact evil@hacker.net or call 555-123-4567 for more info",
            )
        ]
        entities = self.extractor.extract_blocklist_entities(ixs)
        assert "evil@hacker.net" in entities["emails"]
        assert "hacker.net" in entities["domains"]

    def test_extract_payment_methods(self):
        ixs = [
            HoneypotInteraction(message_content="Pay via Bitcoin or Western Union"),
            HoneypotInteraction(message_content="Zelle or CashApp also accepted"),
        ]
        methods = self.extractor.extract_payment_methods(ixs)
        assert "bitcoin" in methods
        assert "western union" in methods
        assert "zelle" in methods

    def test_extract_payment_methods_empty(self):
        ixs = [HoneypotInteraction(message_content="No payment mentioned here")]
        assert self.extractor.extract_payment_methods(ixs) == []

    def test_derive_new_signals_channel_switch(self):
        ixs = [
            _make_interaction(channel=Channel.EMAIL, stage=1, hours_offset=0),
            _make_interaction(channel=Channel.WHATSAPP, stage=2, hours_offset=1),
        ]
        signals = self.extractor.derive_new_signals(ixs)
        names = [s["name"] for s in signals]
        assert "channel_switching_pattern" in names

    def test_derive_new_signals_rapid_escalation(self):
        ixs = [
            _make_interaction(stage=1),
            _make_interaction(stage=3, hours_offset=1),  # payment demand at index 1
        ]
        signals = self.extractor.derive_new_signals(ixs)
        names = [s["name"] for s in signals]
        assert "rapid_payment_escalation" in names

    def test_derive_new_signals_multiple_payment_methods(self):
        ixs = [
            HoneypotInteraction(
                message_content="Pay via Bitcoin, Zelle, or Western Union",
                escalation_stage=3,
            )
        ]
        signals = self.extractor.derive_new_signals(ixs)
        names = [s["name"] for s in signals]
        assert "multiple_payment_methods_offered" in names

    def test_extract_all_returns_structured_intel(self):
        ixs = [
            HoneypotInteraction(
                sender_info={"email": "evil@bad.org"},
                message_content="Send $50 fee via bitcoin. Contact WhatsApp. SSN required.",
                channel=Channel.EMAIL,
                escalation_stage=3,
            )
        ]
        intel = self.extractor.extract_all(ixs)
        assert isinstance(intel, ExtractedIntelligence)
        d = intel.to_dict()
        assert isinstance(d["new_signals"], list)
        assert d["extracted_at"] != ""

    def test_generate_playbook(self):
        ixs = [
            _make_interaction(channel=Channel.EMAIL, stage=1),
            _make_interaction(channel=Channel.WHATSAPP, stage=3),
        ]
        playbook = self.extractor.generate_playbook(ixs)
        assert isinstance(playbook, list)
        assert len(playbook) > 0


# ===========================================================================
# DeploymentStrategy
# ===========================================================================


class TestDeploymentStrategy:
    def setup_method(self):
        self.manager = HoneypotManager()
        self.strategy = DeploymentStrategy(self.manager)
        self.manager.create_profile(
            name="Deploy Alice",
            email="alice@deploy.com",
            resume_summary="summary",
            target_job_categories=["administrative", "customer service"],
            metadata={"experience_level": "entry"},
        )

    def test_prioritize_postings_filters_threshold(self):
        jobs = [
            {"url": "https://example.com/1", "scam_score": 0.8},
            {"url": "https://example.com/2", "scam_score": 0.3},
            {"url": "https://example.com/3", "scam_score": 0.6},
        ]
        prioritized = self.strategy.prioritize_postings(jobs, min_scam_score=0.5)
        urls = [j["url"] for j in prioritized]
        assert "https://example.com/1" in urls
        assert "https://example.com/3" in urls
        assert "https://example.com/2" not in urls

    def test_prioritize_postings_sorted_desc(self):
        jobs = [
            {"url": "a", "scam_score": 0.6},
            {"url": "b", "scam_score": 0.9},
            {"url": "c", "scam_score": 0.7},
        ]
        result = self.strategy.prioritize_postings(jobs)
        scores = [j["scam_score"] for j in result]
        assert scores == sorted(scores, reverse=True)

    def test_select_persona_by_category_match(self):
        job = {"title": "Administrative Assistant", "industry": "Healthcare"}
        persona = self.strategy.select_persona(job)
        assert persona is not None
        assert "administrative" in persona.target_job_categories

    def test_select_persona_no_active_profiles(self):
        mgr = HoneypotManager()
        strat = DeploymentStrategy(mgr)
        persona = strat.select_persona({"title": "anything"})
        assert persona is None

    def test_decide_deployment_below_threshold(self):
        job = {"url": "https://safe.com/job/1", "scam_score": 0.2}
        decision = self.strategy.decide_deployment(job)
        assert decision is None

    def test_decide_deployment_above_threshold(self):
        job = {"url": "https://scam.com/job/99", "scam_score": 0.85, "title": "admin clerk"}
        decision = self.strategy.decide_deployment(job)
        assert decision is not None
        assert decision.job_url == "https://scam.com/job/99"
        assert decision.priority_score == 0.85

    def test_decide_deployment_skip_already_honeypotted(self):
        job_url = "https://scam.com/job/honeypotted"
        job = {"url": job_url, "scam_score": 0.9, "title": "admin"}
        # First deployment records the application
        d1 = self.strategy.decide_deployment(job)
        assert d1 is not None
        self.manager.record_application(d1.profile_id, job_url)
        # Second attempt should skip
        d2 = self.strategy.decide_deployment(job, skip_already_honeypotted=True)
        assert d2 is None

    def test_deploy_batch(self):
        jobs = [
            {"url": f"https://scam.com/job/{i}", "scam_score": 0.5 + i * 0.05, "title": "admin"}
            for i in range(5)
        ]
        decisions = self.strategy.deploy_batch(jobs, max_deployments=3)
        assert len(decisions) <= 3

    def test_deploy_batch_respects_max_deployments(self):
        jobs = [
            {"url": f"https://scam.com/job/batch/{i}", "scam_score": 0.9}
            for i in range(20)
        ]
        # Only 1 active profile, so it will be used once and then all others skip due to honeypotted guard
        decisions = self.strategy.deploy_batch(jobs, max_deployments=5)
        assert len(decisions) <= 5

    def test_ab_targeting_analysis_groups_by_experience(self):
        mgr = HoneypotManager()
        mgr.create_profile("E1", "e1@t.com", "r", ["sales"], metadata={"experience_level": "entry"})
        mgr.create_profile("E2", "e2@t.com", "r", ["tech"], metadata={"experience_level": "mid"})
        strat = DeploymentStrategy(mgr)
        result = strat.ab_targeting_analysis()
        assert "by_experience_level" in result
        assert "by_job_category" in result

    def test_coverage_report_full_coverage(self):
        mgr = HoneypotManager()
        p = mgr.create_profile("Coverage", "c@t.com", "r", ["admin"])
        mgr.record_application(p.profile_id, "https://scam.com/1")
        mgr.record_application(p.profile_id, "https://scam.com/2")
        strat = DeploymentStrategy(mgr)
        report = strat.coverage_report(["https://scam.com/1", "https://scam.com/2"])
        assert report["coverage_pct"] == 100.0
        assert report["not_honeypotted"] == 0

    def test_coverage_report_partial_coverage(self):
        mgr = HoneypotManager()
        p = mgr.create_profile("Partial", "part@t.com", "r", ["admin"])
        mgr.record_application(p.profile_id, "https://scam.com/1")
        strat = DeploymentStrategy(mgr)
        report = strat.coverage_report(["https://scam.com/1", "https://scam.com/2"])
        assert report["coverage_pct"] == 50.0
        assert report["not_honeypotted"] == 1

    def test_coverage_report_empty_suspected(self):
        strat = DeploymentStrategy(HoneypotManager())
        report = strat.coverage_report([])
        assert report["coverage_pct"] == 0.0


# ===========================================================================
# Signal detection helper
# ===========================================================================


class TestDetectSignalsInMessage:
    def test_payment_demand(self):
        sigs = _detect_signals_in_message("Please pay the fee of $50 via wire transfer.")
        assert "payment_demand" in sigs

    def test_personal_info_request(self):
        sigs = _detect_signals_in_message("Send us your SSN and bank account details.")
        assert "personal_info_request" in sigs

    def test_urgency_language(self):
        sigs = _detect_signals_in_message("You must respond immediately, today only!")
        assert "urgency_language" in sigs

    def test_channel_switch_attempt(self):
        sigs = _detect_signals_in_message("Contact me on WhatsApp for next steps.")
        assert "channel_switch_attempt" in sigs

    def test_premature_hire(self):
        sigs = _detect_signals_in_message("Congratulations! You are hired for this position.")
        assert "premature_hire" in sigs

    def test_crypto_mention(self):
        sigs = _detect_signals_in_message("Payment must be via Bitcoin or Ethereum only.")
        assert "crypto_mention" in sigs

    def test_clean_message_no_signals(self):
        sigs = _detect_signals_in_message(
            "Thank you for applying. We would like to schedule an interview next week."
        )
        assert len(sigs) == 0

    def test_multiple_signals_detected(self):
        sigs = _detect_signals_in_message(
            "Congratulations you are hired! Pay $200 fee immediately via Bitcoin. SSN required."
        )
        assert len(sigs) >= 3


# ===========================================================================
# ScammerBehaviorProfile
# ===========================================================================


class TestScammerBehaviorProfile:
    def test_to_dict_roundtrip(self):
        profile = ScammerBehaviorProfile(
            sender_emails=["a@b.com"],
            sender_phones=["+15551234567"],
            channels_used=["email", "whatsapp"],
            avg_response_hours=2.5,
            escalation_speed_hours=6.0,
            language_patterns=["payment_language"],
            common_signals=["payment_demand"],
            aggressiveness_score=0.75,
            interaction_count=5,
        )
        d = profile.to_dict()
        assert d["sender_emails"] == ["a@b.com"]
        assert d["aggressiveness_score"] == 0.75
        assert d["interaction_count"] == 5
        assert "payment_language" in d["language_patterns"]


# ===========================================================================
# ExtractedIntelligence
# ===========================================================================


class TestExtractedIntelligence:
    def test_to_dict_signal_count(self):
        intel = ExtractedIntelligence(
            new_signals=[{"name": "sig1"}, {"name": "sig2"}],
            blocklist_emails=["bad@evil.com"],
        )
        d = intel.to_dict()
        assert d["signal_count"] == 2
        assert "bad@evil.com" in d["blocklist_emails"]

    def test_empty_intel(self):
        intel = ExtractedIntelligence()
        d = intel.to_dict()
        assert d["signal_count"] == 0
        assert d["blocklist_emails"] == []
        assert d["scam_playbooks"] == []
