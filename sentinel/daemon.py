"""Autonomous continuous operation daemon for Sentinel.

Runs the full self-improving loop on a configurable schedule:
  INGEST -> SCORE -> LEARN -> EVOLVE -> INNOVATE -> (sleep) -> repeat
"""

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CycleResult:
    cycle_number: int
    started_at: str
    completed_at: str
    ingestion_queries: list[str]
    jobs_fetched: int
    jobs_new: int
    high_risk_count: int
    flywheel_ran: bool
    regression_detected: bool
    innovation_ran: bool
    innovation_strategy: str
    errors: list[str]
    duration_seconds: float


class SentinelDaemon:
    """Autonomous self-improving daemon.

    Orchestrates the full cycle:
    1. INGEST: Fetch jobs from all configured sources
    2. SCORE: Analyze and score each new job
    3. LEARN: Update signal weights from any new reports
    4. EVOLVE: Promote/deprecate patterns, check for regression
    5. INNOVATE: Run one Thompson Sampling strategy to explore improvements
    6. SLEEP: Wait for next cycle
    """

    def __init__(
        self,
        queries: list[str] | None = None,
        location: str = "",
        interval_seconds: int = 3600,  # 1 hour default
        use_ai: bool = False,
        max_cycles: int = 0,  # 0 = unlimited
        db_path: str | None = None,
    ):
        self.queries = queries or [
            "software engineer",
            "data analyst",
            "remote work from home",
            "customer service representative",
            "administrative assistant",
        ]
        self.location = location
        self.interval_seconds = interval_seconds
        self.use_ai = use_ai
        self.max_cycles = max_cycles
        self.db_path = db_path
        self._running = False
        self._cycle_count = 0
        self._history: list[CycleResult] = []

    def run(self) -> list[CycleResult]:
        """Run the daemon loop. Blocks until stopped or max_cycles reached."""
        self._running = True
        # Register signal handlers for graceful shutdown
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info(
            "Sentinel daemon starting — interval=%ds, queries=%s, max_cycles=%s",
            self.interval_seconds,
            self.queries,
            self.max_cycles or "unlimited",
        )

        try:
            while self._running:
                self._cycle_count += 1
                if self.max_cycles and self._cycle_count > self.max_cycles:
                    break

                result = self._run_one_cycle()
                self._history.append(result)

                logger.info(
                    "Cycle %d complete: %d new jobs, %d high-risk, regression=%s",
                    result.cycle_number,
                    result.jobs_new,
                    result.high_risk_count,
                    result.regression_detected,
                )

                if self.max_cycles and self._cycle_count >= self.max_cycles:
                    break

                if self._running:
                    logger.info("Sleeping %ds until next cycle...", self.interval_seconds)
                    # Sleep in small increments so we can respond to shutdown signals
                    self._interruptible_sleep(self.interval_seconds)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            logger.info("Sentinel daemon stopped after %d cycles", self._cycle_count)

        return self._history

    def _run_one_cycle(self) -> CycleResult:
        """Execute one full INGEST->SCORE->LEARN->EVOLVE->INNOVATE cycle."""
        started = datetime.now()
        errors: list[str] = []
        jobs_fetched = 0
        jobs_new = 0
        high_risk = 0
        flywheel_ran = False
        regression = False
        innovation_ran = False
        innovation_strategy = ""

        # --- INGEST + SCORE ---
        try:
            from sentinel.ingest import IngestionPipeline

            pipeline = IngestionPipeline(db_path=self.db_path)
            runs = pipeline.auto_ingest(
                queries=self.queries,
                location=self.location,
                run_flywheel=False,  # we'll run flywheel separately for more control
            )
            for r in runs:
                jobs_fetched += r.jobs_fetched
                jobs_new += r.jobs_new
                high_risk += r.high_risk_count
                errors.extend(r.errors)
        except Exception as e:
            logger.error("Ingestion failed: %s", e)
            errors.append(f"ingestion: {e}")

        # --- LEARN + EVOLVE ---
        try:
            from sentinel.db import SentinelDB
            from sentinel.flywheel import DetectionFlywheel

            db = SentinelDB(path=self.db_path) if self.db_path else SentinelDB()
            fw = DetectionFlywheel(db=db)
            metrics = fw.run_cycle()
            flywheel_ran = True
            regression = metrics.get("regression_alarm", False)
        except Exception as e:
            logger.error("Flywheel cycle failed: %s", e)
            errors.append(f"flywheel: {e}")

        # --- INNOVATE ---
        try:
            from sentinel.db import SentinelDB
            from sentinel.innovation import InnovationEngine

            db = SentinelDB(path=self.db_path) if self.db_path else SentinelDB()
            engine = InnovationEngine(db=db)
            result = engine.run_cycle()
            innovation_ran = True
            # result is a list of ImprovementResult
            if result:
                innovation_strategy = result[0].strategy
        except Exception as e:
            logger.error("Innovation cycle failed: %s", e)
            errors.append(f"innovation: {e}")

        completed = datetime.now()
        return CycleResult(
            cycle_number=self._cycle_count,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            ingestion_queries=list(self.queries),
            jobs_fetched=jobs_fetched,
            jobs_new=jobs_new,
            high_risk_count=high_risk,
            flywheel_ran=flywheel_ran,
            regression_detected=regression,
            innovation_ran=innovation_ran,
            innovation_strategy=innovation_strategy,
            errors=errors,
            duration_seconds=(completed - started).total_seconds(),
        )

    def _interruptible_sleep(self, seconds: int):
        """Sleep in 1-second increments so shutdown signals are responsive."""
        for _ in range(seconds):
            if not self._running:
                break
            time.sleep(1)

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received (signal %d), finishing current cycle...", signum)
        self._running = False

    def stop(self):
        """Stop the daemon after the current cycle completes."""
        self._running = False

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def history(self) -> list[CycleResult]:
        return list(self._history)
