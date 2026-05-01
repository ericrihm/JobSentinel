"""Autonomous continuous operation daemon for Sentinel.

Runs the full self-improving loop on a configurable schedule:
  INGEST -> SCORE -> LEARN -> EVOLVE -> INNOVATE -> RESEARCH -> (sleep) -> repeat
"""

import logging
import signal
import time
from dataclasses import dataclass
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
    research_ran: bool = False
    research_topics: int = 0
    research_patterns_found: int = 0
    feedback_ran: bool = False
    feedback_jobs_rescanned: int = 0
    feedback_jobs_drifted: int = 0
    feedback_synthetic_generated: int = 0
    deep_research_ran: bool = False
    deep_research_patterns_found: int = 0
    cortex_ran: bool = False
    cortex_health_grade: str = ""
    cortex_strategic_mode: str = ""
    cortex_signals_routed: int = 0
    cortex_actions_dispatched: int = 0
    meta_evolution_ran: bool = False
    meta_evolution_fitness: float = 0.0
    meta_evolution_surgeries: int = 0
    meta_evolution_trend: str = ""


class SentinelDaemon:
    """Autonomous self-improving daemon.

    Orchestrates the full cycle:
    1. INGEST: Fetch jobs from all configured sources
    2. SCORE: Analyze and score each new job
    3. LEARN: Update signal weights from any new reports
    4. EVOLVE: Promote/deprecate patterns, check for regression
    5. INNOVATE: Run one Thompson Sampling strategy to explore improvements
    6. RESEARCH: Discover and integrate new fraud detection knowledge
    7. CORTEX: Meta-cognitive orchestration
    8. META-EVOLVE: Self-improve the flywheel system itself
    9. SLEEP: Wait for next cycle
    """

    def __init__(
        self,
        queries: list[str] | None = None,
        location: str = "",
        interval_seconds: int = 3600,  # 1 hour default
        use_ai: bool = False,
        max_cycles: int = 0,  # 0 = unlimited
        db_path: str | None = None,
        research_budget: int = 2,  # research prompts per cycle
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
        self._research_budget = research_budget
        self._running = False
        self._cycle_count = 0
        self._history: list[CycleResult] = []
        self._flywheel = None  # Persistent across cycles (warm-start)
        self._cortex = None    # Persistent across cycles (warm-start)
        self._meta_evolution = None  # Persistent across cycles (warm-start)

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
        """Execute one full INGEST->SCORE->LEARN->EVOLVE->INNOVATE->RESEARCH cycle."""
        started = datetime.now()
        errors: list[str] = []
        jobs_fetched = 0
        jobs_new = 0
        high_risk = 0
        flywheel_ran = False
        regression = False
        innovation_ran = False
        innovation_strategy = ""
        research_ran = False
        research_topics = 0
        research_patterns_found = 0
        metrics: dict = {}  # populated by flywheel, consumed by cortex

        # --- INGEST + SCORE ---
        try:
            from sentinel.db import SentinelDB as _SentinelDB
            from sentinel.ingest import IngestionPipeline

            pipeline = IngestionPipeline(db_path=self.db_path)
            _db_for_prio = _SentinelDB(path=self.db_path) if self.db_path else _SentinelDB()
            ordered_queries = self._prioritized_queries(_db_for_prio)
            _db_for_prio.close()
            runs = pipeline.auto_ingest(
                queries=ordered_queries,
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

            if self._flywheel is None:
                db = SentinelDB(path=self.db_path) if self.db_path else SentinelDB()
                self._flywheel = DetectionFlywheel(db=db)
            metrics = self._flywheel.run_cycle()
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

        # --- RESEARCH ---
        try:
            from sentinel.db import SentinelDB
            from sentinel.research import ResearchEngine

            db = SentinelDB(path=self.db_path) if self.db_path else SentinelDB()
            research = ResearchEngine(db=db, research_budget=self._research_budget)
            results = research.run_cycle()
            research_ran = True
            research_topics = len(results)
            research_patterns_found = sum(
                len(r.extracted_patterns) for r in results
            )
        except Exception as e:
            logger.error("Research cycle failed: %s", e)
            errors.append(f"research: {e}")

        # --- DEEP RESEARCH PLUGIN (optional) ---
        try:
            from sentinel._research_plugin import plugin_available, run_deep_research_phase

            if plugin_available():
                from sentinel.db import SentinelDB
                _dr_db = SentinelDB(path=self.db_path) if self.db_path else SentinelDB()
                dr_result = run_deep_research_phase(
                    db=_dr_db,
                    research_budget=max(1, self._research_budget // 2),
                )
                dr_patterns = dr_result.get("patterns_found", 0)
                research_patterns_found += dr_patterns
                if dr_patterns > 0:
                    logger.info(
                        "Deep research: %d topics, %d patterns",
                        dr_result.get("topics_researched", 0),
                        dr_patterns,
                    )
        except (ImportError, Exception) as e:
            logger.debug("Deep research phase skipped: %s", e)

        # --- FEEDBACK ---
        feedback_ran = False
        feedback_jobs_rescanned = 0
        feedback_jobs_drifted = 0
        feedback_synthetic_generated = 0
        try:
            from sentinel.db import SentinelDB
            from sentinel.feedback import FeedbackPipeline

            db = SentinelDB(path=self.db_path) if self.db_path else SentinelDB()
            pipeline = FeedbackPipeline(db=db)

            rescan = pipeline.rescan_and_compare(days=7, sample_size=50)
            feedback_jobs_rescanned = rescan.jobs_rescanned
            feedback_jobs_drifted = rescan.jobs_drifted

            synthetic = pipeline.generate_synthetic_feedback(n=20)
            feedback_synthetic_generated = len(synthetic)

            stats = pipeline.get_feedback_stats()
            feedback_ran = True
            logger.info(
                "Feedback phase: rescanned=%d drifted=%d synthetic=%d coverage=%.1f%%",
                feedback_jobs_rescanned,
                feedback_jobs_drifted,
                feedback_synthetic_generated,
                stats.get("feedback_coverage", 0.0) * 100,
            )
        except Exception as e:
            logger.error("Feedback phase failed: %s", e)
            errors.append(f"feedback: {e}")

        # --- CORTEX (meta-cognitive layer) ---
        cortex_ran = False
        cortex_health_grade = ""
        cortex_strategic_mode = ""
        cortex_signals_routed = 0
        cortex_actions_dispatched = 0
        try:
            from sentinel.cortex import Cortex
            from sentinel.db import SentinelDB

            if self._cortex is None:
                _cortex_db = SentinelDB(path=self.db_path) if self.db_path else SentinelDB()
                self._cortex = Cortex(db=_cortex_db)

            # Gather metrics from all subsystems for the cortex
            cortex_metrics = {
                "precision": metrics.get("precision", 0.0) if flywheel_ran else 0.0,
                "recall": metrics.get("recall", 0.0) if flywheel_ran else 0.0,
                "f1": metrics.get("f1", 0.0) if flywheel_ran else 0.0,
                "accuracy": metrics.get("accuracy", 0.0) if flywheel_ran else 0.0,
                "regression_alarm": regression,
                "cusum_statistic": metrics.get("cusum_statistic", 0.0) if flywheel_ran else 0.0,
                "calibration_ece": metrics.get("calibration_ece", 0.0) if flywheel_ran else 0.0,
                "innovation_ran": innovation_ran,
                "innovation_strategy": innovation_strategy,
                "research_ran": research_ran,
                "research_patterns_found": research_patterns_found,
                "errors": errors,
            }

            cortex_state = self._cortex.observe_cycle(cortex_metrics)
            cortex_actions = self._cortex.route_signals(cortex_metrics)
            cortex_ran = True
            cortex_health_grade = self._cortex.generate_report().get("health_grade", "")
            cortex_strategic_mode = (
                cortex_state.strategic_priorities[0]
                if cortex_state.strategic_priorities
                else "OBSERVE"
            )
            cortex_signals_routed = len(cortex_state.cross_system_signals)
            cortex_actions_dispatched = len(cortex_actions)

            logger.info(
                "Cortex: grade=%s mode=%s signals=%d actions=%d",
                cortex_health_grade,
                cortex_strategic_mode,
                cortex_signals_routed,
                cortex_actions_dispatched,
            )
        except Exception as e:
            logger.error("Cortex phase failed: %s", e)
            errors.append(f"cortex: {e}")

        # --- META-EVOLVE (self-improving the flywheel system itself) ---
        meta_evolution_ran = False
        meta_evolution_fitness = 0.0
        meta_evolution_surgeries = 0
        meta_evolution_trend = ""
        try:
            from sentinel.db import SentinelDB
            from sentinel.meta_evolution import MetaEvolutionEngine

            if self._meta_evolution is None:
                _me_db = SentinelDB(path=self.db_path) if self.db_path else SentinelDB()
                self._meta_evolution = MetaEvolutionEngine(db=_me_db)

            me_metrics = {
                "precision": metrics.get("precision", 0.0) if flywheel_ran else 0.0,
                "recall": metrics.get("recall", 0.0) if flywheel_ran else 0.0,
                "f1": metrics.get("f1", 0.0) if flywheel_ran else 0.0,
                "cusum_statistic": metrics.get("cusum_statistic", 0.0) if flywheel_ran else 0.0,
                "calibration_ece": metrics.get("calibration_ece", 0.0) if flywheel_ran else 0.0,
            }
            me_result = self._meta_evolution.run_cycle(me_metrics)
            meta_evolution_ran = True
            meta_evolution_fitness = me_result.get("fitness", 0.0)
            meta_evolution_surgeries = me_result.get("surgeries_performed", 0)
            meta_evolution_trend = me_result.get("velocity_trend", "")

            logger.info(
                "MetaEvolution: fitness=%.4f trend=%s surgeries=%d",
                meta_evolution_fitness,
                meta_evolution_trend,
                meta_evolution_surgeries,
            )
        except Exception as e:
            logger.error("MetaEvolution phase failed: %s", e)
            errors.append(f"meta_evolution: {e}")

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
            research_ran=research_ran,
            research_topics=research_topics,
            research_patterns_found=research_patterns_found,
            feedback_ran=feedback_ran,
            feedback_jobs_rescanned=feedback_jobs_rescanned,
            feedback_jobs_drifted=feedback_jobs_drifted,
            feedback_synthetic_generated=feedback_synthetic_generated,
            cortex_ran=cortex_ran,
            cortex_health_grade=cortex_health_grade,
            cortex_strategic_mode=cortex_strategic_mode,
            cortex_signals_routed=cortex_signals_routed,
            cortex_actions_dispatched=cortex_actions_dispatched,
            meta_evolution_ran=meta_evolution_ran,
            meta_evolution_fitness=meta_evolution_fitness,
            meta_evolution_surgeries=meta_evolution_surgeries,
            meta_evolution_trend=meta_evolution_trend,
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

    def _prioritized_queries(self, db) -> list[str]:
        """Sort self.queries so higher scam-yield sources run first.

        Uses source_stats to rank queries that match known source names.
        Queries without stats are appended after ranked ones in original order.

        Args:
            db: An open SentinelDB instance to read source_stats from.
        """
        try:
            best = db.get_best_sources(n=len(self.queries) + 10)
        except Exception:
            return list(self.queries)

        if not best:
            return list(self.queries)

        # Build a rank map: source_name -> position (lower = better)
        rank_map = {src: i for i, src in enumerate(best)}

        # Sort queries: those matching a ranked source come first; rest appended
        ranked = [q for q in self.queries if q in rank_map]
        unranked = [q for q in self.queries if q not in rank_map]
        ranked.sort(key=lambda q: rank_map[q])
        return ranked + unranked

    def _active_source_names(self) -> set[str]:
        """Return the set of source names whose circuit breaker is currently active.

        Inspects the module-level SmartThrottler via sentinel.sources.get_throttler().
        Returns source names (not domains) for circuit-broken domains.
        """
        broken: set[str] = set()
        try:
            from sentinel.sources import get_throttler

            throttler = get_throttler()
            stats = throttler.get_stats()
            # Map domain prefixes back to source names
            domain_to_source = {
                "remoteok.com": "remoteok",
                "api.adzuna.com": "adzuna",
                "www.themuse.com": "themuse",
                "data.usajobs.gov": "usajobs",
                "remotive.com": "remotive",
            }
            for domain, info in stats.items():
                if info.get("circuit_broken"):
                    src = domain_to_source.get(domain)
                    if src:
                        broken.add(src)
        except Exception as exc:
            logger.debug("Could not check throttler state: %s", exc)
        return broken

    def stop(self):
        """Stop the daemon after the current cycle completes."""
        self._running = False

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def history(self) -> list[CycleResult]:
        return list(self._history)
