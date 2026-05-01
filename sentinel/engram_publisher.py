"""Engram push integration — publishes sentinel findings to engram.

When sentinel's innovation flywheel discovers new patterns or evolves its
Thompson Sampling strategies, this module converts findings into
engram-compatible format and pushes them via ``engram ingest`` CLI or
direct JSONL file write.

This is a one-way push: sentinel → engram.  Sentinel's ecosystem.py already
handles the bidirectional ctools mesh plumbing; this module focuses
specifically on structured engram pattern publishing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENGRAM_INGEST_PATH = Path.home() / ".config" / "ctools" / "engram_ingest.jsonl"


class EngramPublisher:
    """Publishes sentinel findings and evolved strategies to engram.

    Usage::

        pub = EngramPublisher()
        pub.publish_pattern(finding)
        pub.publish_evolved_strategies()

    The publisher tries the ``engram ingest`` CLI first; if the tool is not
    installed it falls back to appending to the shared JSONL ingest file.
    """

    def __init__(
        self,
        engram_cli: str = "engram",
        ingest_path: Path | None = None,
    ) -> None:
        self._engram_cli = engram_cli
        self._ingest_path = ingest_path or ENGRAM_INGEST_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish_pattern(self, finding: dict[str, Any]) -> bool:
        """Convert a sentinel finding to engram-compatible format and push it.

        Parameters
        ----------
        finding:
            Dict from sentinel's innovation or flywheel engine.  Expected keys
            include ``strategy`` (or ``name``), ``detail`` (or ``description``),
            ``success``, and optionally ``new_patterns``, ``precision_delta``,
            ``deprecated_patterns``.

        Returns
        -------
        True if the pattern was successfully pushed (CLI or file).
        """
        pattern = self._convert_finding(finding)
        return self._push_pattern(pattern)

    def publish_evolved_strategies(
        self,
        db: Any | None = None,
    ) -> dict[str, int]:
        """Push sentinel's Thompson Sampling evolved strategies as patterns.

        Reads strategies from the innovation engine state or from the
        provided SentinelDB, converts each strategy arm into an engram
        pattern, and pushes them.

        Returns
        -------
        Dict with ``pushed``, ``failed``, ``total`` counts.
        """
        strategies = self._read_strategies(db)
        results = {"pushed": 0, "failed": 0, "total": len(strategies)}

        for strategy in strategies:
            pattern = {
                "source": "sentinel",
                "type": "strategy",
                "name": f"sentinel_strategy_{strategy['name']}",
                "description": strategy.get("description", ""),
                "confidence": strategy.get("mean", 0.5),
                "domain": "security",
                "tags": ["sentinel", "thompson_sampling", "strategy", strategy["name"]],
                "metadata": {
                    "alpha": strategy.get("alpha", 1.0),
                    "beta": strategy.get("beta", 1.0),
                    "attempts": strategy.get("attempts", 0),
                },
                "timestamp": time.time(),
            }
            if self._push_pattern(pattern):
                results["pushed"] += 1
            else:
                results["failed"] += 1

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_finding(self, finding: dict[str, Any]) -> dict[str, Any]:
        """Convert a sentinel finding dict into engram pattern format."""
        name = finding.get("strategy") or finding.get("name") or "unknown"
        detail = finding.get("detail") or finding.get("description") or ""
        success = finding.get("success", False)

        # Determine category from the finding
        category = "pattern"
        if success:
            category = "success"
        elif finding.get("deprecated_patterns", 0) > 0:
            category = "regression"

        tags = ["sentinel", "innovation", name]
        if finding.get("new_patterns", 0) > 0:
            tags.append("new_pattern")
        if finding.get("deprecated_patterns", 0) > 0:
            tags.append("deprecation")

        return {
            "source": "sentinel",
            "type": "finding",
            "name": f"sentinel_{name}",
            "description": detail,
            "confidence": 0.8 if success else 0.4,
            "domain": "security",
            "category": category,
            "tags": tags,
            "metadata": {
                "success": success,
                "precision_delta": finding.get("precision_delta", 0.0),
                "new_patterns": finding.get("new_patterns", 0),
                "deprecated_patterns": finding.get("deprecated_patterns", 0),
            },
            "timestamp": time.time(),
        }

    def _push_pattern(self, pattern: dict[str, Any]) -> bool:
        """Try CLI first, fall back to JSONL file write."""
        if self._push_via_cli(pattern):
            return True
        return self._push_via_file(pattern)

    def _push_via_cli(self, pattern: dict[str, Any]) -> bool:
        """Push a single pattern via ``engram ingest`` CLI."""
        try:
            result = subprocess.run(
                [self._engram_cli, "ingest", "--source", "sentinel", "--json"],
                input=json.dumps(pattern),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True
            logger.debug(
                "engram ingest failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return False
        except FileNotFoundError:
            logger.debug("engram CLI not installed")
            return False
        except subprocess.TimeoutExpired:
            logger.debug("engram ingest timed out")
            return False
        except Exception as exc:
            logger.debug("engram ingest error: %s", exc)
            return False

    def _push_via_file(self, pattern: dict[str, Any]) -> bool:
        """Append a pattern to the shared JSONL ingest file."""
        try:
            self._ingest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._ingest_path, "a") as f:
                f.write(json.dumps(pattern) + "\n")
            return True
        except OSError as exc:
            logger.warning("Failed to write to ingest file %s: %s", self._ingest_path, exc)
            return False

    def _read_strategies(self, db: Any | None = None) -> list[dict[str, Any]]:
        """Read current Thompson Sampling strategy state.

        Tries the DB first, then the innovation state file on disk.
        """
        # Try DB if provided (SentinelDB instance)
        if db is not None:
            try:
                from sentinel.innovation import InnovationEngine
                engine = InnovationEngine(db=db)
                return engine.get_strategy_rankings()
            except Exception:
                logger.debug("Failed to read strategies from DB", exc_info=True)

        # Fall back to state file
        state_path = Path.home() / ".sentinel" / "innovation_state.json"
        return self._read_strategies_from_file(state_path)

    def _read_strategies_from_file(self, state_path: Path) -> list[dict[str, Any]]:
        """Parse strategy state from a JSON file on disk."""
        if not state_path.exists():
            return []
        try:
            data = json.loads(state_path.read_text())
            strategies = []
            for name, state in data.items():
                alpha = state.get("alpha", 1.0)
                beta = state.get("beta", 1.0)
                strategies.append({
                    "name": name,
                    "alpha": alpha,
                    "beta": beta,
                    "mean": round(alpha / (alpha + beta), 3),
                    "attempts": state.get("attempts", 0),
                })
            return strategies
        except (json.JSONDecodeError, OSError):
            logger.debug("Failed to read innovation state file %s", state_path, exc_info=True)
            return []
