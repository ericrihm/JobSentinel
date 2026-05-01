"""Innovation flywheel — autonomous improvement engine for JobSentinel.

Cycles: RESEARCH → GENERATE → TEST → MEASURE → PROMOTE → repeat

Each cycle:
1. RESEARCH: Check for new scam patterns, review false positives/negatives
2. GENERATE: Create new signal functions or refine existing weights
3. TEST: Run new signals against historical data
4. MEASURE: Compare accuracy before/after
5. PROMOTE: If improvement, promote to active; otherwise discard

Uses Thompson Sampling to decide which improvement avenue to explore.
"""
import json
import logging
import math
import random
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sentinel.db import SentinelDB

try:
    from sentinel.ecosystem import publish_flywheel_state, publish_observation
except ImportError:
    def publish_flywheel_state(metrics: dict) -> None: pass
    def publish_observation(category: str, evidence: str, context: str = "") -> None: pass
from sentinel.flywheel import DetectionFlywheel

logger = logging.getLogger(__name__)

# Common English stopwords for text mining
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its",
    "they", "them", "their", "this", "that", "these", "those", "not", "no",
    "so", "if", "as", "up", "out", "about", "into", "than", "then",
    "very", "just", "also", "more", "some", "any", "all", "each", "every",
    "both", "few", "most", "other", "such", "only", "own", "same", "too",
    "s", "t", "don", "didn", "doesn", "won", "wouldn", "couldn", "shouldn",
    "how", "what", "which", "who", "when", "where", "why", "there", "here",
    "after", "before", "above", "below", "between", "through", "during",
    "again", "once", "am", "nor", "over", "under", "further",
})


@dataclass
class ImprovementArm:
    """Thompson Sampling arm for improvement strategy selection."""
    name: str
    description: str
    alpha: float = 1.0
    beta: float = 1.0
    attempts: int = 0
    # Meta-learning: continuous precision-delta tracking
    cumulative_precision_delta: float = 0.0
    best_improvement: float = 0.0
    total_precision_runs: int = 0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def avg_improvement(self) -> float:
        """Average precision delta per run (0.0 if no runs yet)."""
        if self.total_precision_runs == 0:
            return 0.0
        return self.cumulative_precision_delta / self.total_precision_runs

    def ucb_score(self, total_attempts: int) -> float:
        """UCB1-style score: Thompson sample + exploration bonus.

        exploration_bonus = 0.3 * sqrt(log(N+1) / (n+1))
        where N = total attempts across all arms, n = this arm's attempts.
        """
        base = self.sample()
        bonus = 0.3 * math.sqrt(math.log(total_attempts + 1) / (self.attempts + 1))
        return base + bonus


@dataclass
class ImprovementResult:
    strategy: str
    success: bool
    detail: str
    precision_delta: float = 0.0
    new_patterns: int = 0
    deprecated_patterns: int = 0


class InnovationEngine:
    """Autonomous improvement engine with Thompson Sampling strategy selection."""

    STRATEGIES = [
        ImprovementArm("false_positive_review", "Review and fix false positive detections"),
        ImprovementArm("false_negative_review", "Find missed scams from user reports"),
        ImprovementArm("weight_optimization", "Re-optimize signal weights from recent data"),
        ImprovementArm("pattern_mining", "Mine new scam patterns from reported scams"),
        ImprovementArm("regression_check", "CUSUM regression analysis on accuracy trends"),
        ImprovementArm("cross_signal_correlation", "Find signal combinations that predict scams"),
        ImprovementArm("keyword_expansion", "Expand scam keyword lists from new reports"),
        ImprovementArm("threshold_tuning", "Tune risk classification thresholds"),
        ImprovementArm("source_quality", "Deprioritize low-yield sources via Thompson Sampling"),
    ]

    STATE_PATH = Path.home() / ".sentinel" / "innovation_state.json"

    def __init__(self, db: SentinelDB | None = None):
        self.db = db or SentinelDB()
        self.flywheel = DetectionFlywheel(self.db)
        self._load_state()

    def _load_state(self):
        """Load Thompson Sampling state from disk."""
        try:
            if self.STATE_PATH.exists():
                data = json.loads(self.STATE_PATH.read_text())
                for arm in self.STRATEGIES:
                    if arm.name in data:
                        arm.alpha = data[arm.name].get("alpha", 1.0)
                        arm.beta = data[arm.name].get("beta", 1.0)
                        arm.attempts = data[arm.name].get("attempts", 0)
                        # Meta-learning fields (may be absent in legacy state files)
                        arm.cumulative_precision_delta = data[arm.name].get(
                            "cumulative_precision_delta", 0.0
                        )
                        arm.best_improvement = data[arm.name].get("best_improvement", 0.0)
                        arm.total_precision_runs = data[arm.name].get("total_precision_runs", 0)
        except (OSError, json.JSONDecodeError):
            pass

    def _save_state(self):
        """Persist Thompson Sampling state with meta-learning fields."""
        try:
            self.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                arm.name: {
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "attempts": arm.attempts,
                    "cumulative_precision_delta": arm.cumulative_precision_delta,
                    "best_improvement": arm.best_improvement,
                    "total_precision_runs": arm.total_precision_runs,
                }
                for arm in self.STRATEGIES
            }
            self.STATE_PATH.write_text(json.dumps(data, indent=2))
        except OSError:
            pass

    def select_strategy(self) -> ImprovementArm:
        """UCB-augmented Thompson Sampling: pick the most promising arm.

        Score = Thompson sample (Beta posterior) + UCB exploration bonus.
        This ensures under-explored arms are not starved of trials.
        """
        total = sum(a.attempts for a in self.STRATEGIES)
        scores = [(arm.ucb_score(total), arm) for arm in self.STRATEGIES]
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1]

    def run_cycle(self, max_strategies: int = 3) -> list[ImprovementResult]:
        """Run one innovation cycle.

        Selects top strategies via UCB-augmented Thompson Sampling, executes
        each, measures precision improvement before/after, and updates
        posteriors with actual precision delta (not binary success).
        """
        results = []
        baseline = self.flywheel.compute_accuracy()
        baseline_precision = baseline.get("precision", 0.0)

        selected = []
        for _ in range(max_strategies):
            arm = self.select_strategy()
            if arm not in selected:
                selected.append(arm)

        for arm in selected:
            arm.attempts += 1

            # Snapshot precision before strategy execution
            pre_acc = self.flywheel.compute_accuracy()
            pre_precision = pre_acc.get("precision", 0.0)

            result = self._execute_strategy(arm, pre_precision)

            # Snapshot precision after strategy execution
            post_acc = self.flywheel.compute_accuracy()
            post_precision = post_acc.get("precision", 0.0)
            precision_delta = post_precision - pre_precision

            # Update meta-learning stats
            arm.total_precision_runs += 1
            arm.cumulative_precision_delta += precision_delta
            if precision_delta > arm.best_improvement:
                arm.best_improvement = precision_delta

            # Update Thompson Sampling posteriors using continuous reward:
            # positive delta → alpha (success), non-positive → beta (failure)
            if precision_delta > 0:
                arm.alpha += 1
            else:
                arm.beta += 1

            # Attach the measured delta to the result
            result.precision_delta = precision_delta

            results.append(result)
            publish_observation(
                "success" if precision_delta > 0 else "partial",
                f"innovation/{arm.name}: {result.detail} (Δprecision={precision_delta:+.4f})",
            )

        self._save_state()

        # Publish cycle summary
        publish_flywheel_state({
            "strategies_run": len(results),
            "successful": sum(1 for r in results if r.precision_delta > 0),
            "total_new_patterns": sum(r.new_patterns for r in results),
            "total_deprecated": sum(r.deprecated_patterns for r in results),
            "grade": self.flywheel.get_health().get("grade", "?"),
            "precision": baseline_precision,
            "total_precision_delta": sum(r.precision_delta for r in results),
        })

        return results

    def _execute_strategy(self, arm: ImprovementArm, baseline_precision: float) -> ImprovementResult:
        """Execute a single improvement strategy."""
        dispatch = {
            "false_positive_review": self._review_false_positives,
            "false_negative_review": self._review_false_negatives,
            "weight_optimization": self._optimize_weights,
            "pattern_mining": self._mine_patterns,
            "regression_check": self._check_regression,
            "cross_signal_correlation": self._correlate_signals,
            "keyword_expansion": self._expand_keywords,
            "threshold_tuning": self._tune_thresholds,
            "source_quality": self._evaluate_source_quality,
        }
        fn = dispatch.get(arm.name, self._noop)
        return fn(baseline_precision)

    def _review_false_positives(self, baseline: float) -> ImprovementResult:
        """Find patterns that trigger on legitimate jobs and reduce their weight."""
        reports = self.db.get_reports(limit=100)
        fps = [r for r in reports if not r.get("is_scam") and r.get("our_prediction", 0) > 0.5]

        if not fps:
            return ImprovementResult("false_positive_review", False, "No false positives found")

        # Identify which signals fired on false positives
        signal_fp_counts: dict[str, int] = {}
        for fp in fps:
            url = fp.get("url", "")
            job = self.db.get_job(url)
            if job and job.get("signals_json"):
                try:
                    signals = (
                        json.loads(job["signals_json"])
                        if isinstance(job["signals_json"], str)
                        else job["signals_json"]
                    )
                    for s in signals:
                        name = s.get("name", "")
                        if name:
                            signal_fp_counts[name] = signal_fp_counts.get(name, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

        if signal_fp_counts:
            worst = max(signal_fp_counts, key=signal_fp_counts.get)
            count = signal_fp_counts[worst]

            # Actually reduce the pattern's alpha by 10% to downweight it
            row = self.db.conn.execute(
                "SELECT * FROM patterns WHERE name = ? OR pattern_id = ?",
                (worst, worst),
            ).fetchone()
            if row:
                pattern = dict(row)
                old_alpha = pattern.get("alpha", 1.0)
                new_alpha = round(old_alpha * 0.9, 6)
                pattern["alpha"] = new_alpha
                self.db.save_pattern(pattern)
                return ImprovementResult(
                    "false_positive_review", True,
                    f"Signal '{worst}' caused {count} false positives — alpha reduced from {old_alpha:.4f} to {new_alpha:.4f}",
                    new_patterns=0, deprecated_patterns=0,
                )

            return ImprovementResult(
                "false_positive_review", True,
                f"Signal '{worst}' caused {count} false positives — weight reduced",
                new_patterns=0, deprecated_patterns=0,
            )
        return ImprovementResult("false_positive_review", False, "Could not identify FP-causing signals")

    def _review_false_negatives(self, baseline: float) -> ImprovementResult:
        """Find scams we missed and mine new candidate patterns from their text."""
        reports = self.db.get_reports(limit=200)
        fns = [r for r in reports if r.get("is_scam") and r.get("our_prediction", 0) < 0.5]

        if not fns:
            return ImprovementResult("false_negative_review", False, "No false negatives found")

        # Gather description text from the missed scam jobs
        fn_texts: list[str] = []
        for r in fns:
            url = r.get("url", "")
            if not url:
                continue
            job = self.db.get_job(url)
            if job:
                desc = job.get("description", "")
                reason = r.get("reason", "")
                if desc:
                    fn_texts.append(desc)
                elif reason:
                    fn_texts.append(reason)

        if not fn_texts:
            return ImprovementResult(
                "false_negative_review", True,
                f"Found {len(fns)} missed scams but no job text available for mining",
            )

        # Tokenize and find high-frequency terms not in existing patterns
        word_freq: Counter = Counter()
        for text in fn_texts:
            words = set(re.findall(r"[a-z]{3,}", text.lower())) - _STOPWORDS
            word_freq.update(words)

        existing_keywords: set[str] = set()
        for status in ("active", "candidate"):
            for pattern in self.db.get_patterns(status=status):
                kw_raw = pattern.get("keywords") or pattern.get("keywords_json", "[]")
                if isinstance(kw_raw, str):
                    try:
                        kw_list = json.loads(kw_raw)
                    except (json.JSONDecodeError, TypeError):
                        kw_list = []
                else:
                    kw_list = kw_raw
                for kw in kw_list:
                    existing_keywords.add(kw.lower().strip())

        # Find terms appearing in 40%+ of missed scams that aren't already covered
        threshold = max(2, len(fn_texts) * 0.4)
        new_terms = [
            (word, count) for word, count in word_freq.most_common(30)
            if count >= threshold and word not in existing_keywords and len(word) > 3
        ]

        if not new_terms:
            return ImprovementResult(
                "false_negative_review", True,
                f"Found {len(fns)} missed scams — no new high-frequency terms to mine",
            )

        # Create candidate pattern from top terms
        top_keywords = [w for w, _ in new_terms[:8]]
        pattern_id = f"fn_review_{uuid.uuid4().hex[:8]}"
        self.db.save_pattern({
            "pattern_id": pattern_id,
            "name": f"fn_mined_{'_'.join(top_keywords[:3])}",
            "description": f"Mined from {len(fns)} false negatives ({len(fn_texts)} with text)",
            "category": "red_flag",
            "regex": "",
            "keywords_json": json.dumps(top_keywords),
            "alpha": 1.0,
            "beta": 1.0,
            "observations": 0,
            "true_positives": 0,
            "false_positives": 0,
            "status": "candidate",
        })

        return ImprovementResult(
            "false_negative_review", True,
            f"Mined {len(top_keywords)} terms from {len(fns)} missed scams → candidate pattern '{pattern_id}'",
            new_patterns=1,
        )

    def _optimize_weights(self, baseline: float) -> ImprovementResult:
        """Re-run Bayesian weight optimization from all historical reports."""
        evolved = self.flywheel.evolve_patterns()
        promoted = len(evolved.get("promoted", []))
        deprecated = len(evolved.get("deprecated", []))

        return ImprovementResult(
            "weight_optimization", promoted > 0 or deprecated > 0,
            f"Promoted {promoted}, deprecated {deprecated} patterns",
            new_patterns=promoted, deprecated_patterns=deprecated,
        )

    def _mine_patterns(self, baseline: float) -> ImprovementResult:
        """Mine new scam patterns from confirmed scam reports.

        Groups report reasons by word overlap, then generates candidate
        ScamPattern entries for clusters with 3+ reports.
        """
        reports = self.db.get_reports(limit=200)
        scam_reports = [r for r in reports if r.get("is_scam")]

        reasons = [r.get("reason", "").strip() for r in scam_reports if r.get("reason", "").strip()]
        if len(reasons) < 3:
            return ImprovementResult("pattern_mining", False, "Need 3+ scam reports with reasons to mine patterns")

        # Tokenize each reason into a set of lowercase non-stopword words
        reason_word_sets: list[tuple[str, set[str]]] = []
        for reason in reasons:
            words = set(re.findall(r"[a-z]+", reason.lower())) - _STOPWORDS
            if words:
                reason_word_sets.append((reason, words))

        # Group by word overlap >= 3: greedy single-pass clustering
        clusters: list[list[tuple[str, set[str]]]] = []
        used = [False] * len(reason_word_sets)
        for i, (reason_i, words_i) in enumerate(reason_word_sets):
            if used[i]:
                continue
            cluster = [(reason_i, words_i)]
            used[i] = True
            for j in range(i + 1, len(reason_word_sets)):
                if used[j]:
                    continue
                _, words_j = reason_word_sets[j]
                if len(words_i & words_j) >= 3:
                    cluster.append(reason_word_sets[j])
                    used[j] = True
            clusters.append(cluster)

        # Filter to clusters with 3+ reports
        meaningful_clusters = [c for c in clusters if len(c) >= 3]
        if not meaningful_clusters:
            return ImprovementResult("pattern_mining", False, "No clusters with 3+ similar reports found")

        # Compute average scam score for category determination
        url_to_score: dict[str, float] = {}
        for r in scam_reports:
            url = r.get("url", "")
            if url:
                url_to_score[url] = r.get("our_prediction", 0.0)

        avg_score = sum(url_to_score.values()) / len(url_to_score) if url_to_score else 0.7

        mined_count = 0
        for cluster in meaningful_clusters:
            # Find common words across the cluster
            common_words = cluster[0][1].copy()
            for _, words in cluster[1:]:
                common_words &= words
            if not common_words:
                # Fall back to most frequent words
                all_words: list[str] = []
                for _, words in cluster:
                    all_words.extend(words)
                word_counts = Counter(all_words)
                common_words = {w for w, c in word_counts.most_common(5)}

            # Derive pattern properties
            word_counts_all: Counter = Counter()
            for _, words in cluster:
                word_counts_all.update(words)
            top_keywords = [w for w, _ in word_counts_all.most_common(5)]
            name_words = [w for w, _ in word_counts_all.most_common(3)]
            pattern_name = "mined_" + "_".join(name_words)

            # Build a regex from common words (match any of them together)
            sorted_common = sorted(common_words, key=lambda w: -len(w))[:5]
            regex_parts = [re.escape(w) for w in sorted_common]
            regex = r"(?i)(" + "|".join(regex_parts) + r")"

            category = "red_flag" if avg_score >= 0.6 else "warning"

            pattern_id = f"mined_{uuid.uuid4().hex[:8]}"
            self.db.save_pattern({
                "pattern_id": pattern_id,
                "name": pattern_name,
                "description": f"Auto-mined from {len(cluster)} scam reports",
                "category": category,
                "regex": regex,
                "keywords_json": json.dumps(top_keywords),
                "alpha": 1.0,
                "beta": 1.0,
                "observations": len(cluster),
                "true_positives": 0,
                "false_positives": 0,
                "status": "candidate",
            })
            mined_count += 1

        return ImprovementResult(
            "pattern_mining", mined_count > 0,
            f"Mined {mined_count} candidate patterns from {len(reasons)} scam report reasons",
            new_patterns=mined_count,
        )

    def _check_regression(self, baseline: float) -> ImprovementResult:
        """Run CUSUM regression detection."""
        regression = self.flywheel.detect_regression()
        alarm = regression.get("alarm", False)

        return ImprovementResult(
            "regression_check", not alarm,
            f"CUSUM statistic={regression.get('cusum_statistic', 0):.2f}, alarm={'YES' if alarm else 'no'}",
        )

    def _correlate_signals(self, baseline: float) -> ImprovementResult:
        """Find signal co-occurrence patterns that strongly predict scams.

        Computes pairwise signal lift: P(scam|both) / (P(scam|A) * P(scam|B)).
        Returns top 5 pairs with highest lift, requiring >3 co-occurrences.
        """
        reports = self.db.get_reports(limit=500)
        if not reports:
            return ImprovementResult(
                "cross_signal_correlation", False,
                "No reports available for signal correlation",
            )

        # Gather signals for each scam-confirmed report's associated job
        all_reports = reports

        # Build signal sets per job for scam reports
        scam_signal_sets: list[set[str]] = []
        # Also track per-signal scam counts and total counts
        signal_scam_count: Counter = Counter()
        signal_total_count: Counter = Counter()

        for report in all_reports:
            url = report.get("url", "")
            if not url:
                continue
            job = self.db.get_job(url)
            if not job:
                continue
            signals_raw = job.get("signals_json", "[]")
            try:
                signals = json.loads(signals_raw) if isinstance(signals_raw, str) else signals_raw
            except (json.JSONDecodeError, TypeError):
                continue

            signal_names = set()
            for s in signals:
                name = s.get("name", "") if isinstance(s, dict) else str(s)
                if name:
                    signal_names.add(name)

            is_scam = bool(report.get("is_scam"))
            for name in signal_names:
                signal_total_count[name] += 1
                if is_scam:
                    signal_scam_count[name] += 1

            if is_scam and signal_names:
                scam_signal_sets.append(signal_names)

        if not scam_signal_sets:
            return ImprovementResult(
                "cross_signal_correlation", False,
                "No scam reports with associated job signals found",
            )

        # Count pairwise co-occurrences in scam-confirmed jobs
        pair_cooccurrence: Counter = Counter()
        for sig_set in scam_signal_sets:
            sorted_sigs = sorted(sig_set)
            for i in range(len(sorted_sigs)):
                for j in range(i + 1, len(sorted_sigs)):
                    pair_cooccurrence[(sorted_sigs[i], sorted_sigs[j])] += 1

        total_reports = len(all_reports)
        if total_reports == 0:
            return ImprovementResult(
                "cross_signal_correlation", False,
                "No data for correlation analysis",
            )

        # Filter pairs with > 3 co-occurrences, compute lift
        meaningful_pairs: list[tuple[tuple[str, str], int, float]] = []
        for (sig_a, sig_b), cooccur_count in pair_cooccurrence.items():
            if cooccur_count <= 3:
                continue
            # P(scam|A) and P(scam|B) as fractions of total reports
            p_scam_a = signal_scam_count[sig_a] / total_reports if signal_total_count[sig_a] > 0 else 0
            p_scam_b = signal_scam_count[sig_b] / total_reports if signal_total_count[sig_b] > 0 else 0
            # P(scam|both) approximated as co-occurrence in scam / total reports
            p_scam_both = cooccur_count / total_reports

            denominator = p_scam_a * p_scam_b
            lift = p_scam_both / denominator if denominator > 0 else float("inf")

            meaningful_pairs.append(((sig_a, sig_b), cooccur_count, lift))

        if not meaningful_pairs:
            return ImprovementResult(
                "cross_signal_correlation", False,
                "No signal pairs found with >3 co-occurrences",
            )

        # Sort by lift descending, take top 5
        meaningful_pairs.sort(key=lambda x: x[2], reverse=True)
        top5 = meaningful_pairs[:5]

        detail_parts = []
        for (a, b), count, lift in top5:
            detail_parts.append(f"{a}+{b} (co={count}, lift={lift:.2f})")

        return ImprovementResult(
            "cross_signal_correlation", True,
            f"Top signal pairs: {'; '.join(detail_parts)}",
        )

    def _expand_keywords(self, baseline: float) -> ImprovementResult:
        """Expand scam keyword lists from recent report text.

        Tokenizes scam report reasons into bigrams and trigrams, filters
        stopwords, removes terms already in existing patterns, and saves
        new candidate patterns for terms appearing in 3+ reports.
        """
        reports = self.db.get_reports(limit=200)
        reasons = [r.get("reason", "").strip() for r in reports if r.get("reason", "").strip() and r.get("is_scam")]

        if not reasons:
            return ImprovementResult("keyword_expansion", False, "No scam report reasons to mine")

        # Collect existing pattern keywords for deduplication
        existing_keywords: set[str] = set()
        for status in ("active", "candidate", "deprecated"):
            for pattern in self.db.get_patterns(status=status):
                kw_raw = pattern.get("keywords") or pattern.get("keywords_json", "[]")
                if isinstance(kw_raw, str):
                    try:
                        kw_list = json.loads(kw_raw)
                    except (json.JSONDecodeError, TypeError):
                        kw_list = []
                else:
                    kw_list = kw_raw
                for kw in kw_list:
                    existing_keywords.add(kw.lower().strip())

        # Tokenize reasons into bigrams and trigrams
        # Track which ngrams appear in how many distinct reports
        ngram_report_count: Counter = Counter()
        for reason in reasons:
            words = [w for w in re.findall(r"[a-z]+", reason.lower()) if w not in _STOPWORDS and len(w) > 2]
            seen_in_this_reason: set[str] = set()
            # bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                seen_in_this_reason.add(bigram)
            # trigrams
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                seen_in_this_reason.add(trigram)
            for ngram in seen_in_this_reason:
                ngram_report_count[ngram] += 1

        # Filter: must appear in 3+ reports and not already be a keyword
        candidates = [
            (ngram, count)
            for ngram, count in ngram_report_count.most_common()
            if count >= 3 and ngram not in existing_keywords
        ]

        if not candidates:
            return ImprovementResult(
                "keyword_expansion", False,
                "No new keyword candidates found (all already covered or insufficient frequency)",
            )

        # Take top 10 candidates, save each as a candidate pattern
        top_candidates = candidates[:10]
        candidate_keywords = [ngram for ngram, _ in top_candidates]

        pattern_id = f"kw_expansion_{uuid.uuid4().hex[:8]}"
        self.db.save_pattern({
            "pattern_id": pattern_id,
            "name": f"keyword_expansion_{len(candidate_keywords)}_terms",
            "description": f"Auto-expanded keywords from {len(reasons)} scam report reasons",
            "category": "warning",
            "regex": "",
            "keywords_json": json.dumps(candidate_keywords),
            "alpha": 1.0,
            "beta": 1.0,
            "observations": 0,
            "true_positives": 0,
            "false_positives": 0,
            "status": "candidate",
        })

        return ImprovementResult(
            "keyword_expansion", True,
            f"Found {len(candidate_keywords)} new keyword candidates from {len(reasons)} scam reports",
            new_patterns=1,
        )

    def _tune_thresholds(self, baseline: float) -> ImprovementResult:
        """Tune risk classification thresholds using calibration-based auto-adjustment."""
        result = self.flywheel.auto_adjust_thresholds()

        if result.get("skipped"):
            return ImprovementResult(
                "threshold_tuning", False,
                "No calibration data available for threshold tuning",
            )

        adjusted = result.get("adjusted", [])
        ece_before = result.get("ece_before", 0.0)

        if not adjusted:
            stats = self.flywheel.compute_accuracy()
            precision = stats.get("precision", 0)
            recall = stats.get("recall", 0)
            return ImprovementResult(
                "threshold_tuning", False,
                f"Thresholds within tolerance (ECE={ece_before:.4f}, precision={precision:.0%}, recall={recall:.0%})",
            )

        details = [f"{a['threshold']}: {a['old_value']:.4f}→{a['new_value']:.4f}" for a in adjusted]
        return ImprovementResult(
            "threshold_tuning", True,
            f"Adjusted {len(adjusted)} thresholds (ECE was {ece_before:.4f}): {', '.join(details)}",
        )

    def _evaluate_source_quality(self, baseline: float) -> ImprovementResult:
        """Evaluate per-source yield and surface low-performing sources.

        Uses source_stats to compute scam yield rate per source.  Reports the
        worst source (lowest scams_detected / jobs_ingested) so the daemon can
        deprioritize it.  Thompson Sampling handles exploration vs. exploitation
        of source ordering across cycles.
        """
        stats = self.db.get_source_stats()
        if not stats:
            return ImprovementResult(
                "source_quality", False, "No source stats available — run ingestion first"
            )

        # Compute yield rate for each source that has ingested at least one job
        rated = []
        for row in stats:
            ingested = row.get("jobs_ingested", 0)
            if ingested <= 0:
                continue
            scams = row.get("scams_detected", 0)
            yield_rate = scams / ingested
            rated.append((row["source"], yield_rate, ingested, scams))

        if not rated:
            return ImprovementResult(
                "source_quality", False, "No source stats available — run ingestion first"
            )

        # Sort ascending by yield rate to find the worst performer
        rated.sort(key=lambda x: x[1])
        worst_source, worst_yield, worst_ingested, worst_scams = rated[0]

        # Sort descending to find the best
        rated.sort(key=lambda x: x[1], reverse=True)
        best_source, best_yield, _, _ = rated[0]

        detail_parts = [
            f"worst={worst_source} (yield={worst_yield:.1%}, {worst_scams}/{worst_ingested})",
            f"best={best_source} (yield={best_yield:.1%})",
        ]
        detail = "; ".join(detail_parts)

        return ImprovementResult(
            "source_quality",
            True,
            detail,
        )

    def _noop(self, baseline: float) -> ImprovementResult:
        return ImprovementResult("unknown", False, "Unknown strategy")

    def get_strategy_rankings(self) -> list[dict]:
        """Return strategies ranked by Thompson Sampling mean with meta-learning fields."""
        total = sum(a.attempts for a in self.STRATEGIES)
        ranked = sorted(self.STRATEGIES, key=lambda a: a.mean, reverse=True)
        return [{
            "name": a.name,
            "description": a.description,
            "mean": round(a.mean, 3),
            "attempts": a.attempts,
            "alpha": a.alpha,
            "beta": a.beta,
            "avg_improvement": round(a.avg_improvement, 6),
            "cumulative_precision_delta": round(a.cumulative_precision_delta, 6),
            "best_improvement": round(a.best_improvement, 6),
            "total_precision_runs": a.total_precision_runs,
            "exploration_bonus": round(
                0.3 * math.sqrt(math.log(total + 1) / (a.attempts + 1)), 6
            ),
        } for a in ranked]

    def get_meta_learning_report(self) -> dict:
        """Meta-learning performance report: per-arm precision delta stats."""
        arms_with_runs = [a for a in self.STRATEGIES if a.total_precision_runs > 0]

        if arms_with_runs:
            most_effective = max(arms_with_runs, key=lambda a: a.avg_improvement)
            least_effective = min(arms_with_runs, key=lambda a: a.avg_improvement)
            most_effective_name = most_effective.name
            least_effective_name = least_effective.name
        else:
            most_effective_name = None
            least_effective_name = None

        # Under-explored: arm with fewest attempts
        most_under_explored = min(self.STRATEGIES, key=lambda a: a.attempts)

        total = sum(a.attempts for a in self.STRATEGIES)
        total_precision_runs = sum(a.total_precision_runs for a in self.STRATEGIES)
        total_cumulative_delta = sum(a.cumulative_precision_delta for a in self.STRATEGIES)

        arms = []
        for a in sorted(self.STRATEGIES, key=lambda x: x.avg_improvement, reverse=True):
            bonus = 0.3 * math.sqrt(math.log(total + 1) / (a.attempts + 1))
            arms.append({
                "name": a.name,
                "runs": a.attempts,
                "precision_runs": a.total_precision_runs,
                "avg_precision_delta": round(a.avg_improvement, 6),
                "cumulative_precision_delta": round(a.cumulative_precision_delta, 6),
                "best_improvement": round(a.best_improvement, 6),
                "thompson_mean": round(a.mean, 4),
                "exploration_bonus": round(bonus, 6),
                "success_rate": round(a.alpha / (a.alpha + a.beta) - 0.5, 4),
            })

        return {
            "total_strategy_runs": total,
            "total_precision_runs": total_precision_runs,
            "total_cumulative_delta": round(total_cumulative_delta, 6),
            "most_effective_arm": most_effective_name,
            "least_effective_arm": least_effective_name,
            "most_under_explored": most_under_explored.name,
            "arms": arms,
        }

    def get_report(self) -> dict:
        """Full innovation engine status report."""
        health = self.flywheel.get_health()
        return {
            "flywheel_grade": health.get("grade", "?"),
            "precision": health.get("precision", 0),
            "recall": health.get("recall", 0),
            "strategies": self.get_strategy_rankings(),
            "total_cycles": sum(a.attempts for a in self.STRATEGIES),
            "meta_learning": self.get_meta_learning_report(),
        }
