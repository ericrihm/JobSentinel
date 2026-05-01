"""Research Flywheel — autonomous knowledge discovery for JobSentinel.

Adds a RESEARCH phase to the daemon cycle that:
1. Identifies weakest detection areas (lowest precision, most FP/FN)
2. Generates optimized research prompts targeting those gaps
3. Executes research using the AI analyzer tier system
4. Extracts actionable patterns, keywords, and signals from results
5. Integrates findings into the knowledge base and pattern DB
6. Measures improvement and adjusts research priorities

Self-optimizing: learns which topics and prompt formats yield the most
detection improvements and allocates more research cycles to high-value areas.
"""

import json
import logging
import random
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sentinel.db import SentinelDB

try:
    from sentinel.ecosystem import publish_observation
except ImportError:
    def publish_observation(category: str, evidence: str, context: str = "") -> None: pass

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResearchTopic:
    """A topic area identified as needing research."""
    area: str
    priority: float  # 0-1, based on detection weakness
    reason: str
    last_researched: str | None = None  # ISO timestamp


@dataclass
class ResearchPrompt:
    """An optimized prompt ready for AI execution."""
    topic: ResearchTopic
    prompt_text: str
    expected_output_format: str = "structured_json"
    max_tokens: int = 1024
    template_id: str = ""


@dataclass
class ResearchResult:
    """The output of executing a research prompt."""
    topic: ResearchTopic
    raw_response: str
    extracted_patterns: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: str = field(default_factory=_now_iso)
    prompt_template_id: str = ""
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_RESEARCH_SYSTEM_PROMPT = (
    "You are a fraud detection researcher specializing in employment scams. "
    "Analyze the topic and provide structured insights that can be converted "
    "into detection rules. Be specific about linguistic patterns, red flags, "
    "and distinguishing features. Output your findings as a JSON object with "
    "keys: 'patterns' (list of {keyword, description, category, weight}), "
    "'evasion_tactics' (list of strings), 'new_categories' (list of strings), "
    "'weight_adjustments' (list of {signal_name, direction, reason})."
)


@dataclass
class PromptTemplate:
    """A reusable prompt template with performance tracking."""
    template_id: str
    template_text: str
    slots: list[str]  # e.g. ["{scam_type}", "{industry}"]
    alpha: float = 1.0  # Thompson Sampling success
    beta: float = 1.0   # Thompson Sampling failure
    total_patterns_extracted: int = 0
    total_tokens_used: int = 0
    uses: int = 0

    def sample(self) -> float:
        """Thompson sample from Beta(alpha, beta)."""
        return random.betavariate(self.alpha, self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def efficiency(self) -> float:
        """Patterns per token — higher is better."""
        if self.total_tokens_used == 0:
            return 0.0
        return self.total_patterns_extracted / self.total_tokens_used


_DEFAULT_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        template_id="latest_tactics",
        template_text=(
            "What are the latest employment fraud tactics targeting {job_category} "
            "roles in {time_period}? Focus on new scam types that may evade "
            "keyword-based detection. Provide specific phrases and patterns."
        ),
        slots=["{job_category}", "{time_period}"],
    ),
    PromptTemplate(
        template_id="linguistic_patterns",
        template_text=(
            "What linguistic patterns distinguish legitimate {industry} job postings "
            "from scams? Include specific word choices, sentence structures, and "
            "formatting conventions that differ between real and fake postings."
        ),
        slots=["{industry}"],
    ),
    PromptTemplate(
        template_id="scam_type_indicators",
        template_text=(
            "What are known indicators of {scam_type} scams that differ from "
            "older pattern databases? Focus on evolving tactics from {time_period} "
            "and how scammers adapt to avoid detection."
        ),
        slots=["{scam_type}", "{time_period}"],
    ),
    PromptTemplate(
        template_id="platform_channels",
        template_text=(
            "What new platforms, channels, and distribution methods are scammers "
            "using to distribute fake {job_category} job postings? How do these "
            "differ from legitimate recruitment channels?"
        ),
        slots=["{job_category}"],
    ),
    PromptTemplate(
        template_id="enforcement_actions",
        template_text=(
            "What are the FTC's and other regulatory bodies' most recent enforcement "
            "actions against employment fraud in {region}? What patterns do these "
            "cases reveal about current scam tactics?"
        ),
        slots=["{region}"],
    ),
    PromptTemplate(
        template_id="evasion_detection",
        template_text=(
            "How are employment scammers currently evading {detection_method} detection? "
            "What counter-patterns or signals could catch these evasion tactics? "
            "Provide specific regex patterns and keyword lists."
        ),
        slots=["{detection_method}"],
    ),
]


# ---------------------------------------------------------------------------
# PromptOptimizer
# ---------------------------------------------------------------------------

class PromptOptimizer:
    """Tracks which prompt formats yield the most actionable patterns.

    Uses Thompson Sampling on prompt templates to auto-evolve toward
    the most productive research formats.  When Fathom is available,
    a ``FathomResearchStrategy`` arm is included alongside the local
    prompt templates.
    """

    def __init__(self) -> None:
        self.templates: list[PromptTemplate] = [
            PromptTemplate(
                template_id=t.template_id,
                template_text=t.template_text,
                slots=list(t.slots),
            )
            for t in _DEFAULT_TEMPLATES
        ]
        self._deep_research_strategy: object | None = None
        try:
            from sentinel._research_plugin import DeepResearchStrategy, plugin_available
            if plugin_available():
                self._deep_research_strategy = DeepResearchStrategy()
                logger.info("Deep research strategy arm registered in PromptOptimizer")
        except (ImportError, Exception):
            pass

    @property
    def deep_research_strategy(self) -> object | None:
        return self._deep_research_strategy

    def select_template(self) -> PromptTemplate:
        """Thompson Sampling: pick the template most likely to yield good results."""
        scores = [(t.sample(), t) for t in self.templates]

        if self._deep_research_strategy is not None and self._deep_research_strategy.available:
            dr_score = self._deep_research_strategy.sample()
            dr_template = PromptTemplate(
                template_id="deep_research",
                template_text="{scam_type}",
                slots=["{scam_type}"],
                alpha=self._deep_research_strategy.alpha,
                beta=self._deep_research_strategy.beta,
                total_patterns_extracted=self._deep_research_strategy.total_patterns_extracted,
                total_tokens_used=self._deep_research_strategy.total_tokens_used,
                uses=self._deep_research_strategy.uses,
            )
            scores.append((dr_score, dr_template))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def record_outcome(
        self,
        template_id: str,
        patterns_extracted: int,
        tokens_used: int,
    ) -> None:
        """Update template performance based on research results."""
        if template_id == "deep_research" and self._deep_research_strategy is not None:
            self._deep_research_strategy.record_outcome(patterns_extracted, tokens_used)
            return

        for t in self.templates:
            if t.template_id == template_id:
                t.uses += 1
                t.total_patterns_extracted += patterns_extracted
                t.total_tokens_used += tokens_used
                # Score: did we extract at least 1 pattern?
                if patterns_extracted > 0:
                    t.alpha += 1
                else:
                    t.beta += 1
                break

    def get_rankings(self) -> list[dict]:
        """Return templates ranked by Thompson mean."""
        ranked = sorted(self.templates, key=lambda t: t.mean, reverse=True)
        rankings = [
            {
                "template_id": t.template_id,
                "mean": round(t.mean, 3),
                "uses": t.uses,
                "patterns_extracted": t.total_patterns_extracted,
                "tokens_used": t.total_tokens_used,
                "efficiency": round(t.efficiency, 6),
            }
            for t in ranked
        ]
        if self._deep_research_strategy is not None:
            rankings.append(self._deep_research_strategy.to_dict())
        return rankings


# ---------------------------------------------------------------------------
# ResearchEngine
# ---------------------------------------------------------------------------

class ResearchEngine:
    """Autonomous research engine that discovers and integrates new fraud knowledge.

    Identifies detection weaknesses, generates targeted research prompts,
    executes research via the AI analyzer tier, extracts patterns, and
    integrates findings into the knowledge base.
    """

    # Minimum weakness priority to trigger research
    RESEARCH_THRESHOLD = 0.3

    # Default research budget per daemon cycle
    DEFAULT_BUDGET = 2

    def __init__(
        self,
        db: SentinelDB | None = None,
        analyzer_fn=None,
        research_budget: int = 2,
    ) -> None:
        self.db = db or SentinelDB()
        self._analyzer_fn = analyzer_fn  # callable(system_prompt, user_msg, max_tokens) -> str
        self.research_budget = research_budget
        self.optimizer = PromptOptimizer()

    # ------------------------------------------------------------------
    # 1. IDENTIFY WEAK AREAS
    # ------------------------------------------------------------------

    def identify_weak_areas(self) -> list[ResearchTopic]:
        """Analyze system performance to find areas needing research.

        Checks:
        - Signals with lowest precision (from user reports)
        - Score ranges with worst calibration
        - Job categories with fewest patterns
        - Most common false positive/negative patterns
        - Recently drifted signal distributions
        """
        topics: list[ResearchTopic] = []

        # --- Low-precision signals ---
        topics.extend(self._find_low_precision_signals())

        # --- False positive patterns ---
        topics.extend(self._find_fp_patterns())

        # --- False negative patterns ---
        topics.extend(self._find_fn_patterns())

        # --- Uncovered job categories ---
        topics.extend(self._find_uncovered_categories())

        # --- Calibration gaps ---
        topics.extend(self._find_calibration_gaps())

        # Merge duplicates by area, keeping highest priority
        merged: dict[str, ResearchTopic] = {}
        for t in topics:
            if t.area not in merged or t.priority > merged[t.area].priority:
                merged[t.area] = t

        # Enrich with last-researched timestamp from DB
        for area, topic in merged.items():
            history = self.db.get_top_research_topics(n=100)
            for h in history:
                if h.get("topic") == area:
                    topic.last_researched = h.get("last_researched")
                    break

        # Sort by priority descending
        result = sorted(merged.values(), key=lambda t: t.priority, reverse=True)
        return result

    def _find_low_precision_signals(self) -> list[ResearchTopic]:
        """Find signals with low precision from pattern stats."""
        topics: list[ResearchTopic] = []
        for status in ("active", "candidate"):
            patterns = self.db.get_patterns(status=status)
            for p in patterns:
                tp = p.get("true_positives", 0)
                fp = p.get("false_positives", 0)
                total = tp + fp
                if total < 5:
                    continue
                precision = tp / total
                if precision < 0.6:
                    priority = 1.0 - precision  # Lower precision -> higher priority
                    topics.append(ResearchTopic(
                        area=f"signal_{p['pattern_id']}",
                        priority=min(priority, 1.0),
                        reason=f"Signal '{p['name']}' has low precision ({precision:.0%} on {total} observations)",
                    ))
        return topics

    def _find_fp_patterns(self) -> list[ResearchTopic]:
        """Find common false-positive patterns from reports."""
        reports = self.db.get_reports(limit=200)
        fps = [r for r in reports if not r.get("is_scam") and r.get("our_prediction", 0) >= 0.5]
        if len(fps) >= 3:
            return [ResearchTopic(
                area="false_positives",
                priority=min(len(fps) / 20.0, 0.9),
                reason=f"{len(fps)} false positives found — need to refine detection",
            )]
        return []

    def _find_fn_patterns(self) -> list[ResearchTopic]:
        """Find common false-negative patterns from reports."""
        reports = self.db.get_reports(limit=200)
        fns = [r for r in reports if r.get("is_scam") and r.get("our_prediction", 0) < 0.5]
        if len(fns) >= 3:
            # Extract common words from FN reasons
            reasons = [r.get("reason", "") for r in fns if r.get("reason")]
            reason_words = Counter()
            for reason in reasons:
                words = set(re.findall(r"[a-z]{3,}", reason.lower()))
                reason_words.update(words)

            top_words = [w for w, _ in reason_words.most_common(3)]
            area = "missed_scams"
            if top_words:
                area = f"missed_scams_{'_'.join(top_words[:2])}"

            return [ResearchTopic(
                area=area,
                priority=min(len(fns) / 15.0, 0.95),
                reason=f"{len(fns)} missed scams — need new detection patterns",
            )]
        return []

    def _find_uncovered_categories(self) -> list[ResearchTopic]:
        """Identify scam categories with few or no patterns."""
        topics: list[ResearchTopic] = []
        known_categories = {
            "pig_butchering": "Pig butchering / romance-investment hybrid scams",
            "ai_generated_postings": "AI-generated realistic fake job postings",
            "task_scams": "Task-based scams (small tasks leading to investment traps)",
            "fake_recruiter": "Impersonation of real recruiters / companies",
            "crypto_job_scams": "Jobs requiring crypto wallet setup or transactions",
            "remote_onboarding_scam": "Fake remote onboarding with equipment purchase",
        }

        active_patterns = self.db.get_patterns(status="active")
        pattern_text = " ".join(
            f"{p.get('name', '')} {p.get('description', '')} {p.get('keywords_json', '')}"
            for p in active_patterns
        ).lower()

        for cat_id, description in known_categories.items():
            # Simple heuristic: check if any words from the category appear in patterns
            cat_words = set(cat_id.replace("_", " ").split())
            overlap = sum(1 for w in cat_words if w in pattern_text)
            coverage = overlap / len(cat_words) if cat_words else 1.0

            if coverage < 0.5:
                topics.append(ResearchTopic(
                    area=cat_id,
                    priority=round(0.7 * (1 - coverage), 2),
                    reason=f"Low coverage for '{description}' — {coverage:.0%} pattern overlap",
                ))

        return topics

    def _find_calibration_gaps(self) -> list[ResearchTopic]:
        """Find score ranges where calibration is poor."""
        from sentinel.flywheel import DetectionFlywheel
        try:
            fw = DetectionFlywheel(self.db)
            ece = fw.calibration_error()
            if ece > 0.15:
                return [ResearchTopic(
                    area="calibration",
                    priority=min(ece, 0.8),
                    reason=f"High calibration error (ECE={ece:.3f}) — scores may be misleading",
                )]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # 2. GENERATE RESEARCH PROMPTS
    # ------------------------------------------------------------------

    def generate_research_prompts(
        self, topics: list[ResearchTopic]
    ) -> list[ResearchPrompt]:
        """Create optimized prompts targeting the identified weak areas."""
        prompts: list[ResearchPrompt] = []

        for topic in topics[:self.research_budget]:
            template = self.optimizer.select_template()
            prompt_text = self._fill_template(template, topic)
            prompts.append(ResearchPrompt(
                topic=topic,
                prompt_text=prompt_text,
                expected_output_format="structured_json",
                max_tokens=1024,
                template_id=template.template_id,
            ))

        return prompts

    def _fill_template(self, template: PromptTemplate, topic: ResearchTopic) -> str:
        """Fill template slots with topic-appropriate values."""
        text = template.template_text

        # Build a slot-value mapping from the topic
        slot_values = {
            "{job_category}": self._infer_job_category(topic.area),
            "{time_period}": "2024-2026",
            "{industry}": self._infer_industry(topic.area),
            "{scam_type}": topic.area.replace("_", " "),
            "{region}": "United States",
            "{detection_method}": "keyword and regex-based",
        }

        for slot, value in slot_values.items():
            text = text.replace(slot, value)

        return text

    def _infer_job_category(self, area: str) -> str:
        """Infer a job category from a topic area."""
        category_map = {
            "data_entry": "data entry",
            "remote": "remote work",
            "software": "software engineering",
            "marketing": "marketing",
            "customer_service": "customer service",
            "administrative": "administrative",
        }
        for key, val in category_map.items():
            if key in area.lower():
                return val
        return "general employment"

    def _infer_industry(self, area: str) -> str:
        """Infer an industry from a topic area."""
        if "tech" in area.lower() or "software" in area.lower():
            return "technology"
        if "finance" in area.lower() or "crypto" in area.lower():
            return "financial services"
        return "general"

    # ------------------------------------------------------------------
    # 3. EXECUTE RESEARCH
    # ------------------------------------------------------------------

    def execute_research(self, prompt: ResearchPrompt) -> ResearchResult:
        """Execute a research prompt via the AI analyzer.

        Uses the configured analyzer function or falls back to
        the Anthropic client directly.  If a deep research plugin is
        available and selected, delegates to it instead.
        """
        if prompt.template_id == "deep_research":
            return self._execute_deep_research(prompt)

        raw_response = ""
        tokens_used = 0

        if self._analyzer_fn is not None:
            try:
                raw_response = self._analyzer_fn(
                    _RESEARCH_SYSTEM_PROMPT,
                    prompt.prompt_text,
                    prompt.max_tokens,
                )
                # Estimate tokens
                tokens_used = len(raw_response.split()) * 2
            except Exception as exc:
                logger.warning("Research AI call failed: %s", exc)
                raw_response = ""
        else:
            raw_response = self._call_ai_direct(prompt)
            tokens_used = len(raw_response.split()) * 2

        # Parse the response
        extracted = self.extract_patterns(raw_response)
        confidence = min(len(extracted) / 5.0, 1.0) if extracted else 0.0

        result = ResearchResult(
            topic=prompt.topic,
            raw_response=raw_response,
            extracted_patterns=extracted,
            confidence=confidence,
            prompt_template_id=prompt.template_id,
            tokens_used=tokens_used,
        )

        # Update prompt optimizer
        self.optimizer.record_outcome(
            template_id=prompt.template_id,
            patterns_extracted=len(extracted),
            tokens_used=tokens_used,
        )

        return result

    def _execute_deep_research(self, prompt: ResearchPrompt) -> ResearchResult:
        """Delegate research to the deep research plugin."""
        try:
            from sentinel._research_plugin import DeepResearchBridge

            bridge = DeepResearchBridge()
            dr_result = bridge.deep_research(
                query=f"employment fraud {prompt.topic.area.replace('_', ' ')}",
                domain="security",
                depth=2,
            )

            extracted = dr_result.extracted_patterns
            confidence = dr_result.confidence
            tokens_used = dr_result.report_length // 4

            result = ResearchResult(
                topic=prompt.topic,
                raw_response=dr_result.report_preview,
                extracted_patterns=extracted,
                confidence=confidence,
                prompt_template_id="deep_research",
                tokens_used=tokens_used,
            )

            self.optimizer.record_outcome(
                template_id="deep_research",
                patterns_extracted=len(extracted),
                tokens_used=tokens_used,
            )

            return result

        except (ImportError, Exception) as exc:
            logger.warning("Deep research fallback: %s", exc)
            return ResearchResult(
                topic=prompt.topic,
                raw_response="",
                extracted_patterns=[],
                confidence=0.0,
                prompt_template_id="deep_research",
                tokens_used=0,
            )

    def _call_ai_direct(self, prompt: ResearchPrompt) -> str:
        """Call the Anthropic API directly for research."""
        try:
            from sentinel.config import get_config
            config = get_config()
            if not config.ai_enabled:
                return ""

            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=config.ai_model_deep,
                max_tokens=prompt.max_tokens,
                system=_RESEARCH_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt.prompt_text}],
            )
            return next(
                (b.text for b in response.content if b.type == "text"), ""
            )
        except ImportError:
            logger.debug("anthropic not installed — skipping direct AI call")
            return ""
        except Exception as exc:
            logger.warning("Direct AI research call failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # 4. EXTRACT PATTERNS
    # ------------------------------------------------------------------

    def extract_patterns(self, raw_response: str) -> list[dict]:
        """Extract actionable patterns from AI research response.

        Parses structured JSON output and extracts:
        - New keyword patterns
        - New scam type categories
        - Evasion tactics
        - Signal weight adjustments
        """
        if not raw_response or not raw_response.strip():
            return []

        patterns: list[dict] = []

        # Try JSON parsing first
        parsed = self._try_parse_json(raw_response)
        if parsed:
            # Extract structured patterns
            for p in parsed.get("patterns", []):
                if isinstance(p, dict) and p.get("keyword"):
                    patterns.append({
                        "type": "keyword",
                        "keyword": p["keyword"],
                        "description": p.get("description", ""),
                        "category": p.get("category", "warning"),
                        "weight": float(p.get("weight", 0.5)),
                        "source": "research",
                    })

            # Extract evasion tactics as patterns
            for tactic in parsed.get("evasion_tactics", []):
                if isinstance(tactic, str) and len(tactic) > 5:
                    patterns.append({
                        "type": "evasion_tactic",
                        "keyword": tactic[:100],
                        "description": f"Evasion tactic: {tactic[:200]}",
                        "category": "warning",
                        "weight": 0.6,
                        "source": "research",
                    })

            # Extract weight adjustments
            for adj in parsed.get("weight_adjustments", []):
                if isinstance(adj, dict) and adj.get("signal_name"):
                    patterns.append({
                        "type": "weight_adjustment",
                        "signal_name": adj["signal_name"],
                        "direction": adj.get("direction", "unchanged"),
                        "reason": adj.get("reason", ""),
                        "source": "research",
                    })

            # Extract new categories
            for cat in parsed.get("new_categories", []):
                if isinstance(cat, str) and len(cat) > 2:
                    patterns.append({
                        "type": "new_category",
                        "keyword": cat,
                        "description": f"New scam category: {cat}",
                        "category": "red_flag",
                        "weight": 0.7,
                        "source": "research",
                    })
        else:
            # Fall back to text extraction
            patterns.extend(self._extract_from_text(raw_response))

        return patterns

    def _try_parse_json(self, text: str) -> dict | None:
        """Try to parse JSON from text, handling markdown code fences."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting from markdown code fence
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Try extracting any JSON object from the text
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def _extract_from_text(self, text: str) -> list[dict]:
        """Extract patterns from unstructured text response."""
        patterns: list[dict] = []

        # Look for bullet-pointed items that look like keywords/phrases
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            # Match lines starting with -, *, or numbers
            m = re.match(r"^[-*\d.]+\s+(.*)", line)
            if m:
                content = m.group(1).strip()
                # Skip very short or very long lines
                if 5 < len(content) < 200:
                    # Check if it looks like a keyword/phrase
                    if re.search(r"[a-zA-Z]{3,}", content):
                        patterns.append({
                            "type": "keyword",
                            "keyword": content[:100],
                            "description": f"Extracted from research: {content[:200]}",
                            "category": "warning",
                            "weight": 0.4,
                            "source": "research_text",
                        })

        return patterns[:20]  # Cap to prevent flooding

    # ------------------------------------------------------------------
    # 5. INTEGRATE FINDINGS
    # ------------------------------------------------------------------

    def integrate_findings(self, patterns: list[dict]) -> dict:
        """Add extracted patterns to the knowledge base.

        Creates candidate pattern entries for keyword-type findings
        and applies weight adjustments for existing signals.
        """
        new_patterns = 0
        adjustments = 0
        skipped = 0

        for p in patterns:
            ptype = p.get("type", "")

            if ptype in ("keyword", "evasion_tactic", "new_category"):
                keyword = p.get("keyword", "")
                if not keyword or len(keyword) < 3:
                    skipped += 1
                    continue

                # Check for duplicate keywords in existing patterns
                if self._keyword_exists(keyword):
                    skipped += 1
                    continue

                pattern_id = f"research_{uuid.uuid4().hex[:8]}"
                self.db.save_pattern({
                    "pattern_id": pattern_id,
                    "name": f"research_{keyword[:30].replace(' ', '_').lower()}",
                    "description": p.get("description", f"Discovered via research: {keyword}"),
                    "category": p.get("category", "warning"),
                    "regex": "",
                    "keywords_json": json.dumps([keyword]),
                    "alpha": 1.0,
                    "beta": 1.0,
                    "observations": 0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "status": "candidate",
                })
                new_patterns += 1

            elif ptype == "weight_adjustment":
                signal_name = p.get("signal_name", "")
                direction = p.get("direction", "")
                if signal_name and direction in ("increase", "decrease"):
                    # Apply a small adjustment
                    row = self.db.conn.execute(
                        "SELECT * FROM patterns WHERE pattern_id = ? OR name = ?",
                        (signal_name, signal_name),
                    ).fetchone()
                    if row:
                        pattern = dict(row)
                        old_alpha = pattern.get("alpha", 1.0)
                        if direction == "increase":
                            pattern["alpha"] = round(old_alpha * 1.05, 6)
                        else:
                            pattern["alpha"] = round(old_alpha * 0.95, 6)
                        self.db.save_pattern(pattern)
                        adjustments += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

        return {
            "new_patterns": new_patterns,
            "adjustments": adjustments,
            "skipped": skipped,
            "total_input": len(patterns),
        }

    def _keyword_exists(self, keyword: str) -> bool:
        """Check if a keyword already exists in any pattern."""
        keyword_lower = keyword.lower().strip()
        for status in ("active", "candidate"):
            for p in self.db.get_patterns(status=status):
                kw_raw = p.get("keywords") or p.get("keywords_json", "[]")
                if isinstance(kw_raw, str):
                    try:
                        kw_list = json.loads(kw_raw)
                    except (json.JSONDecodeError, TypeError):
                        kw_list = []
                else:
                    kw_list = kw_raw
                for existing in kw_list:
                    if existing.lower().strip() == keyword_lower:
                        return True
        return False

    # ------------------------------------------------------------------
    # 6. PRIORITIZE NEXT RESEARCH
    # ------------------------------------------------------------------

    def prioritize_next_research(
        self, history: list[ResearchResult] | None = None,
    ) -> list[ResearchTopic]:
        """Use past research value to allocate future research budget.

        Incorporates:
        - Which topics yielded the most adopted patterns
        - Recency decay (older research has lower priority)
        - Coverage gaps (never-researched topics get a bonus)
        """
        # Get DB-stored research history
        db_history = self.db.get_research_history(limit=50)

        # Build topic value map: topic -> avg precision impact
        topic_value: dict[str, float] = {}
        for run in db_history:
            topic = run.get("topic", "")
            delta = run.get("precision_delta", 0.0)
            if topic:
                if topic not in topic_value:
                    topic_value[topic] = []
                if isinstance(topic_value[topic], list):
                    topic_value[topic].append(delta)

        # Average the deltas
        topic_avg: dict[str, float] = {}
        for topic, deltas in topic_value.items():
            if isinstance(deltas, list) and deltas:
                topic_avg[topic] = sum(deltas) / len(deltas)

        # Identify current weak areas
        topics = self.identify_weak_areas()

        # Adjust priority based on historical value
        for topic in topics:
            if topic.area in topic_avg:
                avg_delta = topic_avg[topic.area]
                # Boost topics that historically improved detection
                if avg_delta > 0:
                    topic.priority = min(topic.priority + 0.2, 1.0)
                else:
                    topic.priority = max(topic.priority - 0.1, 0.1)

            # Bonus for never-researched topics
            if topic.last_researched is None:
                topic.priority = min(topic.priority + 0.15, 1.0)

        # Re-sort
        topics.sort(key=lambda t: t.priority, reverse=True)
        return topics

    # ------------------------------------------------------------------
    # FULL RESEARCH CYCLE
    # ------------------------------------------------------------------

    def run_cycle(self, max_prompts: int | None = None) -> list[ResearchResult]:
        """Execute one full research cycle: identify -> prompt -> research -> integrate.

        Returns list of ResearchResults for this cycle.
        """
        budget = max_prompts or self.research_budget
        results: list[ResearchResult] = []

        # Step 1: Identify weak areas
        topics = self.prioritize_next_research()

        # Filter to topics above threshold
        eligible = [t for t in topics if t.priority >= self.RESEARCH_THRESHOLD]
        if not eligible:
            logger.info("No research topics above threshold (%.2f)", self.RESEARCH_THRESHOLD)
            return results

        # Step 2: Generate prompts
        prompts = self.generate_research_prompts(eligible[:budget])

        # Step 3: Execute research
        for prompt in prompts:
            result = self.execute_research(prompt)
            results.append(result)

            # Step 4: Integrate findings
            if result.extracted_patterns:
                integration = self.integrate_findings(result.extracted_patterns)
                logger.info(
                    "Research on '%s': %d patterns extracted, %d integrated, %d skipped",
                    prompt.topic.area,
                    len(result.extracted_patterns),
                    integration["new_patterns"],
                    integration["skipped"],
                )

                # Step 5: Record to DB
                self.db.insert_research_run({
                    "topic": prompt.topic.area,
                    "prompt": prompt.prompt_text[:500],
                    "response_summary": result.raw_response[:500],
                    "patterns_extracted": len(result.extracted_patterns),
                    "patterns_adopted": integration["new_patterns"],
                    "precision_delta": 0.0,  # Measured later
                })

                # Update topic priority
                self.db.update_topic_priority(
                    topic=prompt.topic.area,
                    priority=prompt.topic.priority,
                    patterns_found=len(result.extracted_patterns),
                )

                # Publish to ecosystem
                publish_observation(
                    "success" if integration["new_patterns"] > 0 else "partial",
                    f"research/{prompt.topic.area}: "
                    f"{integration['new_patterns']} new patterns, "
                    f"{integration['adjustments']} adjustments",
                )

        return results

    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------

    def get_report(self) -> dict:
        """Return a summary of research engine status."""
        history = self.db.get_research_history(limit=50)
        topics = self.db.get_top_research_topics(n=10)

        total_patterns = sum(r.get("patterns_extracted", 0) for r in history)
        total_adopted = sum(r.get("patterns_adopted", 0) for r in history)
        avg_delta = 0.0
        if history:
            deltas = [r.get("precision_delta", 0.0) for r in history]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

        return {
            "total_research_runs": len(history),
            "total_patterns_extracted": total_patterns,
            "total_patterns_adopted": total_adopted,
            "adoption_rate": round(total_adopted / total_patterns, 3) if total_patterns > 0 else 0.0,
            "avg_precision_delta": round(avg_delta, 6),
            "top_topics": topics,
            "prompt_template_rankings": self.optimizer.get_rankings(),
        }
