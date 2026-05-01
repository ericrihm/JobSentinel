"""Ecosystem integration — connects sentinel to ctools intelligence mesh.

Publishes observations to engram, events to interop mesh, crash signals
to autopsy, and flywheel state to session-bridge.
"""
import contextlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OBSERVATIONS_PATH = Path.home() / ".config" / "ctools" / "observations.jsonl"
EVENTS_PATH = Path.home() / ".config" / "ctools" / "events.jsonl"

def publish_observation(category: str, evidence: str, context: str = "") -> None:
    """Append an observation to the ctools JSONL for engram ingestion.

    Categories: success, failure, partial, pattern, regression, evolution
    """
    entry = {
        "subject": "sentinel",
        "tool": "sentinel",
        "category": category,
        "evidence": evidence,
        "context": context,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
    }
    try:
        OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OBSERVATIONS_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass

def publish_event(event_type: str, data: dict) -> None:
    """Publish an event to the interop mesh."""
    entry = {
        "event": event_type,
        "source": "sentinel",
        "data": data,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass

def publish_to_engram(outcome: str, patterns: list[str] = None, notes: str = "") -> None:
    """Publish outcome to engram via FoundryBridge if available."""
    try:
        from engram.integrations.foundry_bridge import FoundryBridge
        bridge = FoundryBridge()
        bridge.record_project_outcome(
            project_name="sentinel",
            outcome=outcome,
            patterns_used=patterns or [],
            domain="security",
            notes=notes,
        )
    except (ImportError, Exception):
        publish_observation(outcome, notes)

def publish_flywheel_state(metrics: dict) -> None:
    """Publish flywheel cycle results to session-bridge and engram."""
    publish_event("sentinel_flywheel_cycle", metrics)
    grade = metrics.get("grade", "?")
    precision = metrics.get("precision", 0)
    publish_observation(
        "success" if grade in ("A", "B") else "partial",
        f"flywheel cycle: grade={grade} precision={precision:.0%}",
    )

def publish_detection_result(scam_score: float, signal_count: int, risk_level: str) -> None:
    """Publish a detection result for cross-tool learning."""
    publish_observation(
        "pattern",
        f"scam_score={scam_score:.2f} signals={signal_count} risk={risk_level}",
    )

def read_ecosystem_context() -> dict:
    """Read relevant context from ecosystem tools."""
    context = {}
    # Read session-bridge briefing for project context
    briefing_path = Path.home() / ".config" / "ctools" / "session-bridge" / "briefings" / "sentinel.json"
    if briefing_path.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            context["session_briefing"] = json.loads(briefing_path.read_text())
    # Read engram patterns relevant to security/detection
    try:
        from engram.store.db import EngineDB
        db = EngineDB()
        patterns = db.search_patterns("scam detection security", limit=5)
        context["engram_patterns"] = [p.to_dict() for p in patterns] if patterns else []
    except (ImportError, Exception):
        pass
    return context
