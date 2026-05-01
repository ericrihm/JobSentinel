"""Honeypot Intelligence Gathering System.

Manages canary/decoy job applications and profiles to attract scammers,
log all interactions, analyze scammer behavior patterns, extract new
detection signals, and deploy honeypots strategically against suspected
scam postings.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProfileStatus(Enum):
    ACTIVE = "active"
    RETIRED = "retired"
    COMPROMISED = "compromised"


class Channel(Enum):
    EMAIL = "email"
    LINKEDIN_MESSAGE = "linkedin_message"
    PHONE = "phone"
    WHATSAPP = "whatsapp"
    SMS = "sms"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class HoneypotInteraction:
    """A single inbound contact from a scammer to a honeypot persona."""

    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    channel: Channel = Channel.EMAIL
    message_content: str = ""
    sender_info: dict = field(default_factory=dict)
    scam_signals_detected: list[str] = field(default_factory=list)
    response_sent: Optional[str] = None
    escalation_stage: int = 1  # 1=initial contact, 2=info request, 3=payment demand, 4=follow-up

    def to_dict(self) -> dict:
        return {
            "interaction_id": self.interaction_id,
            "timestamp": self.timestamp,
            "channel": self.channel.value,
            "message_content": self.message_content,
            "sender_info": self.sender_info,
            "scam_signals_detected": self.scam_signals_detected,
            "response_sent": self.response_sent,
            "escalation_stage": self.escalation_stage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HoneypotInteraction":
        obj = cls()
        obj.interaction_id = data.get("interaction_id", obj.interaction_id)
        obj.timestamp = data.get("timestamp", obj.timestamp)
        obj.channel = Channel(data.get("channel", "email"))
        obj.message_content = data.get("message_content", "")
        obj.sender_info = data.get("sender_info", {})
        obj.scam_signals_detected = data.get("scam_signals_detected", [])
        obj.response_sent = data.get("response_sent")
        obj.escalation_stage = data.get("escalation_stage", 1)
        return obj


@dataclass
class HoneypotProfile:
    """A decoy job-seeker persona deployed to attract scammers."""

    profile_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    email: str = ""
    resume_summary: str = ""
    target_job_categories: list[str] = field(default_factory=list)
    status: ProfileStatus = ProfileStatus.ACTIVE
    creation_date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_interaction: Optional[str] = None
    interactions: list[HoneypotInteraction] = field(default_factory=list)
    # Postings this persona has applied to: {job_url: applied_at}
    applications: dict[str, str] = field(default_factory=dict)
    # Demographic / experience metadata for A/B targeting analysis
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "email": self.email,
            "resume_summary": self.resume_summary,
            "target_job_categories": self.target_job_categories,
            "status": self.status.value,
            "creation_date": self.creation_date,
            "last_interaction": self.last_interaction,
            "interactions": [i.to_dict() for i in self.interactions],
            "applications": self.applications,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HoneypotProfile":
        obj = cls()
        obj.profile_id = data.get("profile_id", obj.profile_id)
        obj.name = data.get("name", "")
        obj.email = data.get("email", "")
        obj.resume_summary = data.get("resume_summary", "")
        obj.target_job_categories = data.get("target_job_categories", [])
        obj.status = ProfileStatus(data.get("status", "active"))
        obj.creation_date = data.get("creation_date", obj.creation_date)
        obj.last_interaction = data.get("last_interaction")
        obj.interactions = [
            HoneypotInteraction.from_dict(i) for i in data.get("interactions", [])
        ]
        obj.applications = data.get("applications", {})
        obj.metadata = data.get("metadata", {})
        return obj

    @property
    def interaction_count(self) -> int:
        return len(self.interactions)

    @property
    def is_active(self) -> bool:
        return self.status == ProfileStatus.ACTIVE


# ---------------------------------------------------------------------------
# HoneypotManager
# ---------------------------------------------------------------------------


class HoneypotManager:
    """Create, track, and retire honeypot personas.

    Uses an in-memory store by default; pass a SentinelDB instance to persist
    interactions in the DB honeypot tables (created lazily if absent).
    """

    def __init__(self, db=None) -> None:
        self._profiles: dict[str, HoneypotProfile] = {}
        self._db = db
        if db is not None:
            self._ensure_schema()
            self._load_from_db()

    # ------------------------------------------------------------------
    # Schema / persistence
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create honeypot tables if they don't exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS honeypot_profiles (
            profile_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS honeypot_interactions (
            interaction_id TEXT PRIMARY KEY,
            profile_id TEXT NOT NULL,
            interaction_json TEXT NOT NULL,
            channel TEXT NOT NULL,
            escalation_stage INTEGER NOT NULL DEFAULT 1,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS honeypot_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id TEXT NOT NULL,
            job_url TEXT NOT NULL,
            applied_at TEXT NOT NULL,
            UNIQUE(profile_id, job_url)
        );
        """
        self._db.conn.executescript(ddl)
        self._db.conn.commit()

    def _load_from_db(self) -> None:
        rows = self._db.conn.execute(
            "SELECT profile_json FROM honeypot_profiles"
        ).fetchall()
        for row in rows:
            try:
                data = json.loads(row[0])
                profile = HoneypotProfile.from_dict(data)
                self._profiles[profile.profile_id] = profile
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    def _save_profile_to_db(self, profile: HoneypotProfile) -> None:
        if self._db is None:
            return
        now = datetime.now(UTC).isoformat()
        self._db.conn.execute(
            """
            INSERT INTO honeypot_profiles (profile_id, profile_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                status       = excluded.status,
                updated_at   = excluded.updated_at
            """,
            (profile.profile_id, json.dumps(profile.to_dict()), profile.status.value, now, now),
        )
        self._db.conn.commit()

    # ------------------------------------------------------------------
    # Profile lifecycle
    # ------------------------------------------------------------------

    def create_profile(
        self,
        name: str,
        email: str,
        resume_summary: str,
        target_job_categories: list[str],
        metadata: Optional[dict] = None,
    ) -> HoneypotProfile:
        """Create a new honeypot persona and register it."""
        profile = HoneypotProfile(
            name=name,
            email=email,
            resume_summary=resume_summary,
            target_job_categories=target_job_categories,
            metadata=metadata or {},
        )
        self._profiles[profile.profile_id] = profile
        self._save_profile_to_db(profile)
        logger.info("Created honeypot profile %s (%s)", profile.profile_id, name)
        return profile

    def get_profile(self, profile_id: str) -> Optional[HoneypotProfile]:
        return self._profiles.get(profile_id)

    def list_profiles(self, status: Optional[ProfileStatus] = None) -> list[HoneypotProfile]:
        profiles = list(self._profiles.values())
        if status is not None:
            profiles = [p for p in profiles if p.status == status]
        return profiles

    def retire_profile(self, profile_id: str) -> bool:
        """Retire a profile (soft delete — keeps interaction history)."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return False
        profile.status = ProfileStatus.RETIRED
        self._save_profile_to_db(profile)
        logger.info("Retired honeypot profile %s", profile_id)
        return True

    def mark_compromised(self, profile_id: str) -> bool:
        """Mark a profile as compromised (scammer identified it as fake)."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return False
        profile.status = ProfileStatus.COMPROMISED
        self._save_profile_to_db(profile)
        logger.warning("Honeypot profile %s marked as COMPROMISED", profile_id)
        return True

    def rotate_profiles(self, max_age_days: int = 30, max_interactions: int = 50) -> list[str]:
        """Retire profiles that are too old or over-exposed.

        Returns list of retired profile_ids.
        """
        now = datetime.now(UTC)
        retired: list[str] = []
        for profile in list(self._profiles.values()):
            if profile.status != ProfileStatus.ACTIVE:
                continue
            try:
                created = datetime.fromisoformat(profile.creation_date)
                # Ensure tz-aware comparison
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                age = (now - created).days
            except (ValueError, TypeError):
                age = 0
            if age >= max_age_days or profile.interaction_count >= max_interactions:
                self.retire_profile(profile.profile_id)
                retired.append(profile.profile_id)
        return retired

    # ------------------------------------------------------------------
    # Applications tracking
    # ------------------------------------------------------------------

    def record_application(self, profile_id: str, job_url: str) -> bool:
        """Record that this persona applied to a job posting."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return False
        now = datetime.now(UTC).isoformat()
        profile.applications[job_url] = now
        self._save_profile_to_db(profile)
        if self._db is not None:
            self._db.conn.execute(
                """
                INSERT OR IGNORE INTO honeypot_applications (profile_id, job_url, applied_at)
                VALUES (?, ?, ?)
                """,
                (profile_id, job_url, now),
            )
            self._db.conn.commit()
        return True

    def get_applications(self, profile_id: str) -> dict[str, str]:
        """Return {job_url: applied_at} for a profile."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return {}
        return dict(profile.applications)

    # ------------------------------------------------------------------
    # Interaction logging
    # ------------------------------------------------------------------

    def log_interaction(
        self,
        profile_id: str,
        channel: Channel,
        message_content: str,
        sender_info: Optional[dict] = None,
        response_sent: Optional[str] = None,
        escalation_stage: int = 1,
    ) -> Optional[HoneypotInteraction]:
        """Log a new interaction against a honeypot profile."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            logger.warning("log_interaction: profile %s not found", profile_id)
            return None

        signals = _detect_signals_in_message(message_content)
        interaction = HoneypotInteraction(
            channel=channel,
            message_content=message_content,
            sender_info=sender_info or {},
            scam_signals_detected=signals,
            response_sent=response_sent,
            escalation_stage=escalation_stage,
        )
        profile.interactions.append(interaction)
        profile.last_interaction = interaction.timestamp
        self._save_profile_to_db(profile)

        if self._db is not None:
            self._db.conn.execute(
                """
                INSERT INTO honeypot_interactions
                    (interaction_id, profile_id, interaction_json, channel, escalation_stage, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction.interaction_id,
                    profile_id,
                    json.dumps(interaction.to_dict()),
                    channel.value,
                    escalation_stage,
                    interaction.timestamp,
                ),
            )
            self._db.conn.commit()

        logger.info(
            "Logged interaction %s for profile %s via %s (stage %d, %d signals)",
            interaction.interaction_id,
            profile_id,
            channel.value,
            escalation_stage,
            len(signals),
        )
        return interaction

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_interaction_report(self, profile_id: str) -> dict:
        """Generate a summary report for all interactions on a profile."""
        profile = self._profiles.get(profile_id)
        if profile is None:
            return {"error": f"Profile {profile_id} not found"}

        interactions = profile.interactions
        if not interactions:
            return {
                "profile_id": profile_id,
                "name": profile.name,
                "status": profile.status.value,
                "total_interactions": 0,
                "channels_used": [],
                "escalation_distribution": {},
                "top_signals": [],
                "applications_count": len(profile.applications),
            }

        channel_counts: Counter = Counter(i.channel.value for i in interactions)
        escalation_dist: Counter = Counter(i.escalation_stage for i in interactions)
        all_signals: list[str] = []
        for i in interactions:
            all_signals.extend(i.scam_signals_detected)
        top_signals = Counter(all_signals).most_common(10)

        max_escalation = max(i.escalation_stage for i in interactions)

        return {
            "profile_id": profile_id,
            "name": profile.name,
            "email": profile.email,
            "status": profile.status.value,
            "creation_date": profile.creation_date,
            "last_interaction": profile.last_interaction,
            "total_interactions": len(interactions),
            "channels_used": dict(channel_counts),
            "escalation_distribution": {str(k): v for k, v in escalation_dist.items()},
            "max_escalation_stage": max_escalation,
            "top_signals": [{"signal": s, "count": c} for s, c in top_signals],
            "applications_count": len(profile.applications),
            "target_categories": profile.target_job_categories,
        }

    def generate_global_report(self) -> dict:
        """Aggregate report across all profiles."""
        all_interactions: list[HoneypotInteraction] = []
        active = 0
        retired = 0
        compromised = 0
        for profile in self._profiles.values():
            all_interactions.extend(profile.interactions)
            if profile.status == ProfileStatus.ACTIVE:
                active += 1
            elif profile.status == ProfileStatus.RETIRED:
                retired += 1
            elif profile.status == ProfileStatus.COMPROMISED:
                compromised += 1

        channel_counts: Counter = Counter(i.channel.value for i in all_interactions)
        all_signals: list[str] = []
        for i in all_interactions:
            all_signals.extend(i.scam_signals_detected)

        return {
            "total_profiles": len(self._profiles),
            "active_profiles": active,
            "retired_profiles": retired,
            "compromised_profiles": compromised,
            "total_interactions": len(all_interactions),
            "channels_used": dict(channel_counts),
            "top_signals": Counter(all_signals).most_common(10),
        }


# ---------------------------------------------------------------------------
# InteractionAnalyzer
# ---------------------------------------------------------------------------


@dataclass
class ScammerBehaviorProfile:
    """Behavioral fingerprint assembled from multiple interactions."""

    scammer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_emails: list[str] = field(default_factory=list)
    sender_phones: list[str] = field(default_factory=list)
    channels_used: list[str] = field(default_factory=list)
    avg_response_hours: float = 0.0
    escalation_speed_hours: float = 0.0  # time from first contact to payment ask
    language_patterns: list[str] = field(default_factory=list)
    common_signals: list[str] = field(default_factory=list)
    aggressiveness_score: float = 0.0  # 0.0–1.0
    interaction_count: int = 0

    def to_dict(self) -> dict:
        return {
            "scammer_id": self.scammer_id,
            "sender_emails": self.sender_emails,
            "sender_phones": self.sender_phones,
            "channels_used": self.channels_used,
            "avg_response_hours": round(self.avg_response_hours, 2),
            "escalation_speed_hours": round(self.escalation_speed_hours, 2),
            "language_patterns": self.language_patterns,
            "common_signals": self.common_signals,
            "aggressiveness_score": round(self.aggressiveness_score, 3),
            "interaction_count": self.interaction_count,
        }


class InteractionAnalyzer:
    """Analyze patterns in scammer responses to honeypot personas."""

    # Keywords that indicate escalation toward payment demand
    _PAYMENT_KEYWORDS = re.compile(
        r"\b(fee|payment|deposit|wire|transfer|bitcoin|gift card|zelle|"
        r"cashapp|western union|moneygram|processing fee|registration fee|"
        r"background check fee|equipment fee|startup cost)\b",
        re.IGNORECASE,
    )
    _INFO_REQUEST_KEYWORDS = re.compile(
        r"\b(ssn|social security|bank account|routing number|date of birth|"
        r"passport|driver.?s? licen[sc]e|credit card|debit card|address)\b",
        re.IGNORECASE,
    )
    _CHANNEL_SWITCH_KEYWORDS = re.compile(
        r"\b(whatsapp|telegram|signal|text me|call me|my number is|"
        r"reach me at|contact me on)\b",
        re.IGNORECASE,
    )

    def analyze_response_time(
        self, interactions: list[HoneypotInteraction]
    ) -> Optional[float]:
        """Return average hours between consecutive interactions, or None if < 2."""
        if len(interactions) < 2:
            return None
        deltas: list[float] = []
        sorted_ix = sorted(interactions, key=lambda i: i.timestamp)
        for a, b in zip(sorted_ix, sorted_ix[1:]):
            try:
                ta = datetime.fromisoformat(a.timestamp)
                tb = datetime.fromisoformat(b.timestamp)
                if ta.tzinfo is None:
                    ta = ta.replace(tzinfo=UTC)
                if tb.tzinfo is None:
                    tb = tb.replace(tzinfo=UTC)
                delta_h = abs((tb - ta).total_seconds()) / 3600
                deltas.append(delta_h)
            except (ValueError, TypeError):
                pass
        return sum(deltas) / len(deltas) if deltas else None

    def analyze_escalation_pattern(
        self, interactions: list[HoneypotInteraction]
    ) -> dict:
        """Summarize how escalation stages progress across interactions."""
        if not interactions:
            return {"stages": [], "max_stage": 0, "reached_payment": False}
        sorted_ix = sorted(interactions, key=lambda i: i.timestamp)
        stages = [i.escalation_stage for i in sorted_ix]
        return {
            "stages": stages,
            "max_stage": max(stages),
            "reached_payment": max(stages) >= 3,
            "stage_sequence": stages,
        }

    def analyze_channel_usage(
        self, interactions: list[HoneypotInteraction]
    ) -> dict:
        """Count and rank the channels a scammer uses."""
        counts: Counter = Counter(i.channel.value for i in interactions)
        channel_switches = 0
        sorted_ix = sorted(interactions, key=lambda i: i.timestamp)
        prev_channel = None
        for ix in sorted_ix:
            if prev_channel is not None and ix.channel.value != prev_channel:
                channel_switches += 1
            prev_channel = ix.channel.value
        return {
            "channel_counts": dict(counts),
            "channel_switches": channel_switches,
            "primary_channel": counts.most_common(1)[0][0] if counts else None,
        }

    def extract_language_patterns(self, interactions: list[HoneypotInteraction]) -> list[str]:
        """Extract notable language patterns from message content."""
        patterns: list[str] = []
        for ix in interactions:
            content = ix.message_content.lower()
            if self._PAYMENT_KEYWORDS.search(content):
                patterns.append("payment_language")
            if self._INFO_REQUEST_KEYWORDS.search(content):
                patterns.append("info_request_language")
            if self._CHANNEL_SWITCH_KEYWORDS.search(content):
                patterns.append("channel_switch_attempt")
            if re.search(r"\b(congratulations?|you.ve been selected|you are hired)\b", content, re.IGNORECASE):
                patterns.append("premature_hire_language")
            if re.search(r"\b(urgent|immediately|asap|right away|today only)\b", content, re.IGNORECASE):
                patterns.append("urgency_language")
        return list(set(patterns))

    def score_aggressiveness(self, interactions: list[HoneypotInteraction]) -> float:
        """Score 0.0–1.0 how aggressive a scammer is based on escalation and signals."""
        if not interactions:
            return 0.0
        max_stage = max(i.escalation_stage for i in interactions)
        total_signals = sum(len(i.scam_signals_detected) for i in interactions)
        lang_patterns = self.extract_language_patterns(interactions)
        channel_info = self.analyze_channel_usage(interactions)
        switches = channel_info.get("channel_switches", 0)

        # Weighted components
        stage_score = min(max_stage / 4.0, 1.0)
        signal_score = min(total_signals / 20.0, 1.0)
        lang_score = min(len(lang_patterns) / 5.0, 1.0)
        switch_score = min(switches / 3.0, 1.0)

        return round(
            0.4 * stage_score + 0.3 * signal_score + 0.2 * lang_score + 0.1 * switch_score,
            3,
        )

    def build_scammer_profile(
        self,
        interactions: list[HoneypotInteraction],
        scammer_id: Optional[str] = None,
    ) -> ScammerBehaviorProfile:
        """Build a behavioral fingerprint from a list of interactions."""
        profile = ScammerBehaviorProfile(
            scammer_id=scammer_id or str(uuid.uuid4())
        )
        profile.interaction_count = len(interactions)

        # Collect sender info
        emails: set[str] = set()
        phones: set[str] = set()
        for ix in interactions:
            si = ix.sender_info
            if si.get("email"):
                emails.add(si["email"])
            if si.get("phone"):
                phones.add(si["phone"])
        profile.sender_emails = list(emails)
        profile.sender_phones = list(phones)

        # Channels
        channel_info = self.analyze_channel_usage(interactions)
        profile.channels_used = list(channel_info.get("channel_counts", {}).keys())

        # Response time
        avg_response = self.analyze_response_time(interactions)
        profile.avg_response_hours = avg_response or 0.0

        # Escalation speed (time from first contact to first payment stage)
        sorted_ix = sorted(interactions, key=lambda i: i.timestamp)
        first_ts = sorted_ix[0].timestamp if sorted_ix else None
        payment_ts = None
        for ix in sorted_ix:
            if ix.escalation_stage >= 3:
                payment_ts = ix.timestamp
                break
        if first_ts and payment_ts:
            try:
                t1 = datetime.fromisoformat(first_ts)
                t2 = datetime.fromisoformat(payment_ts)
                if t1.tzinfo is None:
                    t1 = t1.replace(tzinfo=UTC)
                if t2.tzinfo is None:
                    t2 = t2.replace(tzinfo=UTC)
                profile.escalation_speed_hours = abs((t2 - t1).total_seconds()) / 3600
            except (ValueError, TypeError):
                pass

        profile.language_patterns = self.extract_language_patterns(interactions)

        # Top signals
        all_signals: list[str] = []
        for ix in interactions:
            all_signals.extend(ix.scam_signals_detected)
        profile.common_signals = [s for s, _ in Counter(all_signals).most_common(10)]

        profile.aggressiveness_score = self.score_aggressiveness(interactions)
        return profile

    def identify_playbook(self, interactions: list[HoneypotInteraction]) -> list[str]:
        """Identify the scam playbook steps from the interaction sequence."""
        if not interactions:
            return []
        sorted_ix = sorted(interactions, key=lambda i: i.timestamp)
        playbook: list[str] = []
        for ix in sorted_ix:
            step = f"stage_{ix.escalation_stage}:{ix.channel.value}"
            if step not in playbook:
                playbook.append(step)
        return playbook


# ---------------------------------------------------------------------------
# IntelligenceExtractor
# ---------------------------------------------------------------------------


@dataclass
class ExtractedIntelligence:
    """Structured intelligence extracted from honeypot interactions."""

    new_signals: list[dict] = field(default_factory=list)
    blocklist_emails: list[str] = field(default_factory=list)
    blocklist_domains: list[str] = field(default_factory=list)
    blocklist_phones: list[str] = field(default_factory=list)
    payment_methods_observed: list[str] = field(default_factory=list)
    scam_playbooks: list[list[str]] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "new_signals": self.new_signals,
            "blocklist_emails": self.blocklist_emails,
            "blocklist_domains": self.blocklist_domains,
            "blocklist_phones": self.blocklist_phones,
            "payment_methods_observed": self.payment_methods_observed,
            "scam_playbooks": self.scam_playbooks,
            "extracted_at": self.extracted_at,
            "signal_count": len(self.new_signals),
        }


_PAYMENT_METHOD_RE = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|gift card|western union|moneygram|"
    r"wire transfer|zelle|cashapp|cash app|venmo|paypal|crypto)\b",
    re.IGNORECASE,
)
_DOMAIN_RE = re.compile(r"@([\w.-]+\.[a-z]{2,})", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(\+?1?\s*[-.]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE)


class IntelligenceExtractor:
    """Convert honeypot interactions into new detection signals and blocklists."""

    def __init__(self) -> None:
        self._analyzer = InteractionAnalyzer()

    def extract_blocklist_entities(
        self, interactions: list[HoneypotInteraction]
    ) -> dict:
        """Pull emails, domains, and phone numbers from sender_info and message content."""
        emails: set[str] = set()
        domains: set[str] = set()
        phones: set[str] = set()

        for ix in interactions:
            si = ix.sender_info
            if si.get("email"):
                e = si["email"].lower().strip()
                emails.add(e)
                m = _DOMAIN_RE.search(e)
                if m:
                    domains.add(m.group(1).lower())
            if si.get("phone"):
                phones.add(si["phone"].strip())

            # Also scrape content for embedded contact info
            for email_match in _EMAIL_RE.finditer(ix.message_content):
                e = email_match.group().lower().strip()
                emails.add(e)
                m = _DOMAIN_RE.search(e)
                if m:
                    domains.add(m.group(1).lower())
            for phone_match in _PHONE_RE.finditer(ix.message_content):
                phones.add(phone_match.group().strip())

        return {
            "emails": sorted(emails),
            "domains": sorted(domains),
            "phones": sorted(phones),
        }

    def extract_payment_methods(self, interactions: list[HoneypotInteraction]) -> list[str]:
        """Find payment methods mentioned across interactions."""
        found: set[str] = set()
        for ix in interactions:
            for m in _PAYMENT_METHOD_RE.finditer(ix.message_content):
                found.add(m.group().lower())
        return sorted(found)

    def derive_new_signals(
        self, interactions: list[HoneypotInteraction]
    ) -> list[dict]:
        """Convert observed scammer behaviors into candidate detection signals."""
        signals: list[dict] = []
        channel_info = self._analyzer.analyze_channel_usage(interactions)

        # Channel-switching pattern
        if channel_info.get("channel_switches", 0) >= 1:
            signals.append(
                {
                    "name": "channel_switching_pattern",
                    "category": "red_flag",
                    "weight": 0.6,
                    "detail": (
                        f"Scammer switched channels {channel_info['channel_switches']} time(s) "
                        f"across {len(interactions)} interactions"
                    ),
                    "source": "honeypot",
                }
            )

        # Fast escalation to payment demand
        sorted_ix = sorted(interactions, key=lambda i: i.timestamp)
        first_payment_idx = next(
            (idx for idx, ix in enumerate(sorted_ix) if ix.escalation_stage >= 3),
            None,
        )
        if first_payment_idx is not None and first_payment_idx <= 1:
            signals.append(
                {
                    "name": "rapid_payment_escalation",
                    "category": "red_flag",
                    "weight": 0.75,
                    "detail": "Payment demand reached within 1-2 interactions",
                    "source": "honeypot",
                }
            )

        # Payment method diversity
        methods = self.extract_payment_methods(interactions)
        if len(methods) >= 2:
            signals.append(
                {
                    "name": "multiple_payment_methods_offered",
                    "category": "red_flag",
                    "weight": 0.7,
                    "detail": f"Scammer offered {len(methods)} payment methods: {', '.join(methods)}",
                    "source": "honeypot",
                }
            )

        # Personal info harvesting
        info_requests = [
            ix for ix in interactions
            if any(
                s in ix.scam_signals_detected
                for s in ("personal_info_request", "ssn_request", "bank_info_request")
            )
        ]
        if info_requests:
            signals.append(
                {
                    "name": "personal_info_harvesting",
                    "category": "red_flag",
                    "weight": 0.8,
                    "detail": f"Scammer requested personal info in {len(info_requests)} interaction(s)",
                    "source": "honeypot",
                }
            )

        return signals

    def generate_playbook(self, interactions: list[HoneypotInteraction]) -> list[str]:
        """Generate a human-readable scam playbook from interaction sequences."""
        return self._analyzer.identify_playbook(interactions)

    def extract_all(
        self, interactions: list[HoneypotInteraction]
    ) -> ExtractedIntelligence:
        """Run all extraction pipelines and return a structured result."""
        entities = self.extract_blocklist_entities(interactions)
        intel = ExtractedIntelligence(
            new_signals=self.derive_new_signals(interactions),
            blocklist_emails=entities["emails"],
            blocklist_domains=entities["domains"],
            blocklist_phones=entities["phones"],
            payment_methods_observed=self.extract_payment_methods(interactions),
            scam_playbooks=[self.generate_playbook(interactions)],
        )
        return intel

    def feed_signals_to_db(
        self,
        db,
        interactions: list[HoneypotInteraction],
    ) -> int:
        """Save derived candidate signals to the patterns table. Returns count saved."""
        signals = self.derive_new_signals(interactions)
        from datetime import UTC, datetime
        import uuid as _uuid

        now = datetime.now(UTC).isoformat()
        saved = 0
        for sig in signals:
            pattern_id = f"honeypot_{sig['name']}_{_uuid.uuid4().hex[:8]}"
            db.save_pattern(
                {
                    "pattern_id": pattern_id,
                    "name": sig["name"],
                    "description": sig.get("detail", ""),
                    "category": sig.get("category", "red_flag"),
                    "regex": "",
                    "keywords": [],
                    "alpha": 1.0,
                    "beta": 1.0,
                    "observations": 1,
                    "true_positives": 1,
                    "false_positives": 0,
                    "status": "candidate",
                    "created_at": now,
                    "updated_at": now,
                }
            )
            saved += 1
        return saved


# ---------------------------------------------------------------------------
# DeploymentStrategy
# ---------------------------------------------------------------------------


@dataclass
class DeploymentDecision:
    """Result of deciding which honeypot to deploy against a posting."""

    job_url: str
    profile_id: str
    priority_score: float
    reason: str


class DeploymentStrategy:
    """Decide which job postings to honeypot and which persona to use."""

    def __init__(self, manager: HoneypotManager) -> None:
        self._manager = manager

    # ------------------------------------------------------------------
    # Posting prioritization
    # ------------------------------------------------------------------

    def prioritize_postings(
        self, job_scores: list[dict], min_scam_score: float = 0.4
    ) -> list[dict]:
        """Return job postings sorted by scam likelihood, above threshold.

        *job_scores* is a list of dicts with at least {"url": str, "scam_score": float}.
        """
        suspicious = [j for j in job_scores if j.get("scam_score", 0.0) >= min_scam_score]
        return sorted(suspicious, key=lambda j: j["scam_score"], reverse=True)

    def select_persona(
        self, job: dict, profiles: Optional[list[HoneypotProfile]] = None
    ) -> Optional[HoneypotProfile]:
        """Select the best active persona for a given job posting.

        Prefers personas whose target_job_categories overlap with the job title/industry.
        Falls back to any active persona.
        """
        active = profiles or self._manager.list_profiles(status=ProfileStatus.ACTIVE)
        if not active:
            return None

        job_title = (job.get("title") or "").lower()
        job_industry = (job.get("industry") or "").lower()

        # Score each persona by category overlap
        def category_score(p: HoneypotProfile) -> float:
            cats = [c.lower() for c in p.target_job_categories]
            hits = sum(1 for c in cats if c in job_title or c in job_industry)
            return hits

        ranked = sorted(active, key=category_score, reverse=True)
        return ranked[0]

    def decide_deployment(
        self,
        job: dict,
        min_scam_score: float = 0.4,
        skip_already_honeypotted: bool = True,
    ) -> Optional[DeploymentDecision]:
        """Decide whether and which persona to deploy against a single job.

        Returns None if the posting is below threshold or already honeypotted.
        """
        scam_score = job.get("scam_score", 0.0)
        job_url = job.get("url", "")

        if scam_score < min_scam_score:
            return None

        # Check if already applied
        if skip_already_honeypotted:
            for profile in self._manager.list_profiles():
                if job_url in profile.applications:
                    return None

        persona = self.select_persona(job)
        if persona is None:
            return None

        priority_score = scam_score
        return DeploymentDecision(
            job_url=job_url,
            profile_id=persona.profile_id,
            priority_score=round(priority_score, 3),
            reason=f"scam_score={scam_score:.2f} >= threshold={min_scam_score:.2f}",
        )

    def deploy_batch(
        self,
        job_scores: list[dict],
        min_scam_score: float = 0.4,
        max_deployments: int = 10,
    ) -> list[DeploymentDecision]:
        """Deploy honeypots across a batch of job postings.

        Returns a list of DeploymentDecisions (up to max_deployments).
        """
        prioritized = self.prioritize_postings(job_scores, min_scam_score)
        decisions: list[DeploymentDecision] = []
        for job in prioritized:
            if len(decisions) >= max_deployments:
                break
            decision = self.decide_deployment(job, min_scam_score)
            if decision is not None:
                self._manager.record_application(decision.profile_id, decision.job_url)
                decisions.append(decision)
        return decisions

    # ------------------------------------------------------------------
    # A/B targeting analysis
    # ------------------------------------------------------------------

    def ab_targeting_analysis(
        self, profiles: Optional[list[HoneypotProfile]] = None
    ) -> dict:
        """Analyze which persona characteristics attract more scammer contact.

        Compares interaction rates across personas grouped by metadata attributes.
        """
        target_profiles = profiles or self._manager.list_profiles()
        if not target_profiles:
            return {}

        by_experience: defaultdict[str, list[int]] = defaultdict(list)
        by_category: defaultdict[str, list[int]] = defaultdict(list)

        for p in target_profiles:
            exp = p.metadata.get("experience_level", "unknown")
            by_experience[exp].append(p.interaction_count)
            for cat in p.target_job_categories:
                by_category[cat].append(p.interaction_count)

        def avg(vals: list[int]) -> float:
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        return {
            "by_experience_level": {k: {"avg_interactions": avg(v), "profiles": len(v)}
                                     for k, v in by_experience.items()},
            "by_job_category": {k: {"avg_interactions": avg(v), "profiles": len(v)}
                                  for k, v in by_category.items()},
        }

    def coverage_report(self, all_suspected_scam_urls: list[str]) -> dict:
        """Report what % of suspected scam postings have been honeypotted."""
        honeypotted: set[str] = set()
        for profile in self._manager.list_profiles():
            honeypotted.update(profile.applications.keys())

        suspected = set(all_suspected_scam_urls)
        covered = suspected & honeypotted
        coverage_pct = round(len(covered) / len(suspected) * 100, 1) if suspected else 0.0

        return {
            "total_suspected": len(suspected),
            "honeypotted": len(covered),
            "not_honeypotted": len(suspected - honeypotted),
            "coverage_pct": coverage_pct,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SIGNAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("payment_demand", re.compile(
        r"\b(fee|payment|deposit|buy equipment|wire|transfer|gift card)\b", re.IGNORECASE
    )),
    ("personal_info_request", re.compile(
        r"\b(ssn|social security|bank account|routing number|passport|driver.?s? licen[sc]e|"
        r"credit card|debit card)\b",
        re.IGNORECASE,
    )),
    ("guaranteed_income", re.compile(
        r"\b(guaranteed (salary|income|pay|earnings?))\b", re.IGNORECASE
    )),
    ("urgency_language", re.compile(
        r"\b(urgent|immediately|asap|right away|today only|limited time)\b", re.IGNORECASE
    )),
    ("channel_switch_attempt", re.compile(
        r"\b(whatsapp|telegram|signal|text me|call me|reach me at)\b", re.IGNORECASE
    )),
    ("premature_hire", re.compile(
        r"\b(you.?re hired|you.?ve been selected|congratulations.{0,30}position|"
        r"welcome to the team)\b",
        re.IGNORECASE,
    )),
    ("crypto_mention", re.compile(
        r"\b(bitcoin|btc|ethereum|eth|crypto)\b", re.IGNORECASE
    )),
]


def _detect_signals_in_message(content: str) -> list[str]:
    """Lightweight signal detection on inbound scammer message content."""
    detected: list[str] = []
    for name, pattern in _SIGNAL_PATTERNS:
        if pattern.search(content):
            detected.append(name)
    return detected
