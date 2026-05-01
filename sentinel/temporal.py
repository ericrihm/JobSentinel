"""Temporal Intelligence for scam evolution tracking.

Provides tools to understand *how* scam patterns change over time:

- ScamEvolutionTracker  — tracks pattern lifecycle and firing-rate trends
- TemporalAnomalyDetector — Z-score + seasonal decomposition on daily volumes
- PatternDrift          — KL-divergence between expected vs observed distributions
- PredictiveModel       — linear regression for scam-volume forecasting

Design constraints:
- stdlib only: math, statistics, collections, datetime
- @dataclass for all data classes
- No numpy / scipy / pandas
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sentinel.db import SentinelDB

# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------

def _today_iso() -> str:
    return datetime.now(UTC).date().isoformat()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _week_key(d: date) -> str:
    """Return ISO year-week string e.g. '2026-W17'."""
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _month_key(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def _to_date(value: str | date | datetime) -> date:
    """Coerce a date/datetime/ISO-string to a date object."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # Try progressively: bare date, datetime without tz, with tz suffix
    for fmt, length in [
        ("%Y-%m-%d", 10),
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y-%m-%dT%H:%M:%S", 19),  # re-tried with truncated tz strings
    ]:
        try:
            return datetime.strptime(value[:length], fmt).date()
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Cannot parse date from: {value!r}")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PatternLifecycle:
    """Lifecycle record for a single scam pattern.

    Tracks the pattern's journey through:
    emergence → growth → saturation → mutation → death
    """
    pattern_name: str
    first_seen: str = ""
    last_seen: str = ""
    total_observations: int = 0
    peak_rate: float = 0.0
    peak_week: str = ""
    # EMA of weekly firing rate
    ema_rate: float = 0.0
    ema_alpha: float = 0.2          # smoothing factor (0 < alpha <= 1)
    # Trend: positive = growing, negative = shrinking
    trend: float = 0.0
    status: str = "emerging"        # emerging | growing | saturated | mutating | dead


@dataclass
class ScamSeason:
    """A calendar period with elevated scam activity."""
    name: str
    description: str
    # Months as 1-based integers, e.g. [1, 2] for Jan/Feb
    peak_months: list[int] = field(default_factory=list)
    # Expected relative lift vs baseline (1.0 = no lift)
    lift_factor: float = 1.0


@dataclass
class AnomalyResult:
    """Result from TemporalAnomalyDetector for a single day."""
    date: str
    signal_name: str
    observed_volume: int
    expected_volume: float
    z_score: float
    is_anomaly: bool
    is_burst: bool
    seasonal_factor: float = 1.0
    message: str = ""


@dataclass
class DriftReport:
    """Result from PatternDrift comparison."""
    reference_start: str
    reference_end: str
    comparison_start: str
    comparison_end: str
    kl_divergence: float
    drift_detected: bool
    top_shifted_signals: list[dict] = field(default_factory=list)
    message: str = ""


@dataclass
class PredictionResult:
    """One-week-ahead scam volume prediction."""
    target_week: str
    predicted_volume: float
    lower_bound: float
    upper_bound: float
    confidence: float               # 0–1 based on R² of the fit
    slope: float                    # weekly trend in counts
    intercept: float
    n_weeks_used: int
    message: str = ""


# ---------------------------------------------------------------------------
# Known scam seasons
# ---------------------------------------------------------------------------

SCAM_SEASONS: list[ScamSeason] = [
    ScamSeason(
        name="tax_season",
        description="Tax refund and W-2 phishing scams spike Jan–April.",
        peak_months=[1, 2, 3, 4],
        lift_factor=1.45,
    ),
    ScamSeason(
        name="holiday_hiring",
        description="Fake seasonal jobs surge Nov–Dec.",
        peak_months=[11, 12],
        lift_factor=1.35,
    ),
    ScamSeason(
        name="graduation",
        description="Fake entry-level jobs targeting new graduates in May–June.",
        peak_months=[5, 6],
        lift_factor=1.25,
    ),
    ScamSeason(
        name="back_to_school",
        description="Fake tutoring / campus jobs peak Aug–Sep.",
        peak_months=[8, 9],
        lift_factor=1.15,
    ),
]


def current_scam_seasons(reference: date | None = None) -> list[ScamSeason]:
    """Return the seasons that are active for *reference* (default: today)."""
    ref = reference or datetime.now(UTC).date()
    return [s for s in SCAM_SEASONS if ref.month in s.peak_months]


def seasonal_lift_for_date(reference: date | None = None) -> float:
    """Multiplicative seasonal lift factor for a given date (≥ 1.0)."""
    seasons = current_scam_seasons(reference)
    if not seasons:
        return 1.0
    return max(s.lift_factor for s in seasons)


# ---------------------------------------------------------------------------
# ScamEvolutionTracker
# ---------------------------------------------------------------------------

class ScamEvolutionTracker:
    """Track how scam patterns change over time.

    Records firing-rate observations keyed to weekly or monthly windows and
    maintains an exponential moving average (EMA) of each pattern's rate.
    Detects lifecycle phase transitions using the EMA trend.

    Usage::

        tracker = ScamEvolutionTracker()
        tracker.record_observation("upfront_payment", count=12, total_jobs=100,
                                   observation_date=date.today())
        lifecycle = tracker.get_lifecycle("upfront_payment")
        print(lifecycle.status, lifecycle.ema_rate, lifecycle.trend)
    """

    # EMA smoothing factor (0 < alpha ≤ 1); higher = more reactive.
    DEFAULT_EMA_ALPHA: float = 0.3

    # Trend thresholds for lifecycle phase assignment
    GROWTH_THRESHOLD: float = 0.005     # +0.5 pp/week → growing
    DECAY_THRESHOLD: float = -0.005     # -0.5 pp/week → mutating/dying
    SATURATION_RATE: float = 0.20       # ≥ 20% firing rate → saturated
    DEAD_RATE: float = 0.005            # ≤ 0.5% firing rate → dead

    def __init__(self, db: SentinelDB | None = None, ema_alpha: float = DEFAULT_EMA_ALPHA) -> None:
        self.db = db
        self.ema_alpha = ema_alpha
        # pattern_name -> PatternLifecycle
        self._lifecycles: dict[str, PatternLifecycle] = {}
        # pattern_name -> deque of (week_key, rate) tuples (most recent last)
        self._rate_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=52))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_observation(
        self,
        pattern_name: str,
        count: int,
        total_jobs: int,
        observation_date: date | str | None = None,
    ) -> PatternLifecycle:
        """Record a weekly firing-rate observation for *pattern_name*.

        Args:
            pattern_name: Unique name of the scam signal/pattern.
            count:        Number of times the pattern fired in the window.
            total_jobs:   Total jobs scanned in the same window (denominator).
            observation_date: Date within the observation window (default: today).

        Returns:
            Updated PatternLifecycle for the pattern.
        """
        obs_date = _to_date(observation_date) if observation_date else datetime.now(UTC).date()
        rate = count / max(total_jobs, 1)
        week = _week_key(obs_date)
        date_iso = obs_date.isoformat()

        lc = self._lifecycles.setdefault(
            pattern_name,
            PatternLifecycle(pattern_name=pattern_name, first_seen=date_iso),
        )

        # Update first/last seen
        if not lc.first_seen or date_iso < lc.first_seen:
            lc.first_seen = date_iso
        if not lc.last_seen or date_iso > lc.last_seen:
            lc.last_seen = date_iso

        lc.total_observations += count

        # EMA update
        if lc.ema_rate == 0.0 and count == 0:
            # Stay at zero; don't initialise from empty
            prev_ema = 0.0
        elif lc.ema_rate == 0.0:
            # Bootstrap: first non-zero observation
            prev_ema = rate
        else:
            prev_ema = lc.ema_rate

        new_ema = self.ema_alpha * rate + (1.0 - self.ema_alpha) * prev_ema
        lc.trend = new_ema - prev_ema
        lc.ema_rate = new_ema

        # Peak tracking
        if rate > lc.peak_rate:
            lc.peak_rate = rate
            lc.peak_week = week

        # Lifecycle phase
        lc.status = self._classify_lifecycle(lc)

        # Store in rate history
        self._rate_history[pattern_name].append((week, rate))

        return lc

    def get_lifecycle(self, pattern_name: str) -> PatternLifecycle | None:
        """Return the current lifecycle record for *pattern_name*, or None."""
        return self._lifecycles.get(pattern_name)

    def all_lifecycles(self) -> dict[str, PatternLifecycle]:
        """Return a snapshot of all tracked lifecycle records."""
        return dict(self._lifecycles)

    def emerging_patterns(self, min_rate: float = 0.01, max_rate: float = 0.10) -> list[PatternLifecycle]:
        """Return patterns in the 'emerging' or 'growing' phase.

        These are early-warning signals — active but not yet widespread.

        Args:
            min_rate: Minimum EMA firing rate to be considered (filters noise).
            max_rate: Maximum EMA firing rate to be considered (below saturation).
        """
        return [
            lc for lc in self._lifecycles.values()
            if lc.status in ("emerging", "growing")
            and min_rate <= lc.ema_rate <= max_rate
        ]

    def rate_history(self, pattern_name: str) -> list[tuple[str, float]]:
        """Return the ordered list of (week_key, rate) observations."""
        return list(self._rate_history.get(pattern_name, []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_lifecycle(self, lc: PatternLifecycle) -> str:
        rate = lc.ema_rate
        trend = lc.trend

        if rate <= self.DEAD_RATE and lc.total_observations > 0:
            return "dead"
        if rate >= self.SATURATION_RATE:
            return "saturated"
        if trend >= self.GROWTH_THRESHOLD:
            return "growing"
        if trend <= self.DECAY_THRESHOLD:
            # Still alive but declining — likely mutating
            return "mutating" if rate > self.DEAD_RATE else "dead"
        if lc.total_observations > 0 and rate > self.DEAD_RATE:
            return "emerging"
        return "emerging"


# ---------------------------------------------------------------------------
# TemporalAnomalyDetector
# ---------------------------------------------------------------------------

class TemporalAnomalyDetector:
    """Z-score based anomaly detection on daily scam signal volumes.

    Features:
    - Maintains a rolling window of daily volumes per signal.
    - Seasonal decomposition: applies a weekday factor (Mon surge, Fri drop).
    - Burst detection: sudden spike relative to a short recent window.
    - Change-point detection via a cumulative-sum approach.

    Usage::

        detector = TemporalAnomalyDetector(window=14, z_threshold=2.5)
        result = detector.observe("upfront_payment", volume=25, obs_date=date.today())
        if result.is_anomaly:
            print(result.message)
    """

    # Multiplicative weekday seasonal factors (Mon=0 … Sun=6)
    WEEKDAY_FACTORS: list[float] = [1.12, 1.05, 1.00, 0.98, 0.90, 0.80, 0.75]

    def __init__(
        self,
        window: int = 14,
        z_threshold: float = 2.5,
        burst_multiplier: float = 3.0,
        burst_window: int = 3,
    ) -> None:
        """
        Args:
            window:           Rolling days to include in baseline statistics.
            z_threshold:      Z-score beyond which a day is flagged as anomalous.
            burst_multiplier: Volume must exceed burst_window mean * this to be a burst.
            burst_window:     Number of recent days to use for burst baseline.
        """
        self.window = window
        self.z_threshold = z_threshold
        self.burst_multiplier = burst_multiplier
        self.burst_window = burst_window
        # signal_name -> deque of (date_iso, volume) ordered oldest-first
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window + burst_window + 5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        signal_name: str,
        volume: int,
        obs_date: date | str | None = None,
    ) -> AnomalyResult:
        """Record *volume* for *signal_name* on *obs_date* and check for anomalies.

        Args:
            signal_name: Scam signal identifier.
            volume:      Raw daily count.
            obs_date:    Date of observation (default: today).

        Returns:
            AnomalyResult with z-score, anomaly flag, and burst flag.
        """
        d = _to_date(obs_date) if obs_date else datetime.now(UTC).date()
        date_iso = d.isoformat()

        # Seasonal adjustment: divide by weekday factor so Friday's naturally
        # lower volume isn't flagged as anomalously low.
        seasonal_factor = self.WEEKDAY_FACTORS[d.weekday()]
        adj_volume = volume / seasonal_factor

        hist = self._history[signal_name]
        hist.append((date_iso, volume))

        # Need at least 3 points for meaningful stats
        baseline_vols = [v for _, v in hist][:-1]   # exclude current point

        if len(baseline_vols) < 3:
            return AnomalyResult(
                date=date_iso,
                signal_name=signal_name,
                observed_volume=volume,
                expected_volume=float(volume),
                z_score=0.0,
                is_anomaly=False,
                is_burst=False,
                seasonal_factor=seasonal_factor,
                message="Insufficient history for anomaly detection.",
            )

        baseline_adj = [v / self.WEEKDAY_FACTORS[0] for v in baseline_vols]

        mean_adj = statistics.mean(baseline_adj)
        std_adj = statistics.pstdev(baseline_adj) if len(baseline_adj) >= 2 else 0.0

        if std_adj == 0.0:
            # Baseline is perfectly constant; use a large z if value deviates
            if adj_volume == mean_adj:
                z_score = 0.0
            else:
                # Use mean itself as scale to avoid div-by-zero; min scale of 1
                scale = max(abs(mean_adj), 1.0)
                z_score = (adj_volume - mean_adj) / scale * self.z_threshold * 2
        else:
            z_score = (adj_volume - mean_adj) / std_adj

        is_anomaly = abs(z_score) > self.z_threshold

        # Burst detection: compare to short recent window
        short_window = [v for _, v in hist][-(self.burst_window + 1):-1]
        if len(short_window) >= 2:
            recent_mean = statistics.mean(short_window)
            is_burst = (recent_mean > 0) and (volume > recent_mean * self.burst_multiplier)
        else:
            is_burst = False

        expected = mean_adj * seasonal_factor

        if is_burst:
            msg = f"Burst detected: {volume} vs recent mean {recent_mean:.1f} (×{self.burst_multiplier})."
        elif is_anomaly:
            direction = "spike" if z_score > 0 else "drop"
            msg = f"Anomaly ({direction}): z={z_score:.2f} on {date_iso} for '{signal_name}'."
        else:
            msg = f"Normal volume: z={z_score:.2f}."

        return AnomalyResult(
            date=date_iso,
            signal_name=signal_name,
            observed_volume=volume,
            expected_volume=round(expected, 2),
            z_score=round(z_score, 4),
            is_anomaly=is_anomaly,
            is_burst=is_burst,
            seasonal_factor=round(seasonal_factor, 4),
            message=msg,
        )

    def detect_changepoint(self, signal_name: str) -> dict:
        """Detect the most likely change-point in the stored history for *signal_name*.

        Uses a simple cumulative-sum (CUSUM) scan: iterates potential split
        points and finds the split that maximises the difference between the
        mean of the left and right halves (maximum likelihood under Gaussian).

        Returns:
            dict with keys: ``found`` (bool), ``changepoint_date`` (str | None),
            ``pre_mean``, ``post_mean``, ``magnitude``.
        """
        hist = list(self._history.get(signal_name, []))
        if len(hist) < 6:
            return {
                "found": False,
                "changepoint_date": None,
                "pre_mean": None,
                "post_mean": None,
                "magnitude": 0.0,
                "message": "Insufficient data for change-point detection (need ≥ 6 days).",
            }

        volumes = [v for _, v in hist]
        dates = [d for d, _ in hist]
        n = len(volumes)
        best_idx = -1
        best_magnitude = 0.0

        for i in range(2, n - 2):
            pre = volumes[:i]
            post = volumes[i:]
            pre_mean = statistics.mean(pre)
            post_mean = statistics.mean(post)
            magnitude = abs(post_mean - pre_mean)
            if magnitude > best_magnitude:
                best_magnitude = magnitude
                best_idx = i

        if best_idx < 0:
            return {
                "found": False,
                "changepoint_date": None,
                "pre_mean": None,
                "post_mean": None,
                "magnitude": 0.0,
                "message": "No change-point found.",
            }

        pre_mean = statistics.mean(volumes[:best_idx])
        post_mean = statistics.mean(volumes[best_idx:])
        overall_std = statistics.pstdev(volumes) if len(volumes) >= 2 else 1.0
        is_significant = best_magnitude > overall_std

        return {
            "found": is_significant,
            "changepoint_date": dates[best_idx],
            "pre_mean": round(pre_mean, 4),
            "post_mean": round(post_mean, 4),
            "magnitude": round(best_magnitude, 4),
            "message": (
                f"Change-point at {dates[best_idx]}: "
                f"pre_mean={pre_mean:.2f}, post_mean={post_mean:.2f}."
            ) if is_significant else "No significant change-point.",
        }

    def volume_history(self, signal_name: str) -> list[tuple[str, int]]:
        """Return ordered (date_iso, volume) history for *signal_name*."""
        return list(self._history.get(signal_name, []))


# ---------------------------------------------------------------------------
# PatternDrift
# ---------------------------------------------------------------------------

class PatternDrift:
    """Compare signal distributions between two time windows using KL-divergence.

    Detects when scam tactics shift — even if overall volume stays stable.
    For example, if fraudsters switch from "upfront_payment" to "crypto_payment",
    the total scam count may look normal while the *distribution* shifts.

    Usage::

        drift = PatternDrift()
        report = drift.compare(
            reference={"upfront_payment": 0.4, "salary_too_high": 0.3, "urgency": 0.3},
            current={"upfront_payment": 0.1, "crypto_payment": 0.5, "urgency": 0.4},
        )
        print(report.kl_divergence, report.drift_detected)
    """

    # KL-divergence threshold above which drift is reported
    DRIFT_THRESHOLD: float = 0.05

    def __init__(self, drift_threshold: float = DRIFT_THRESHOLD) -> None:
        self.drift_threshold = drift_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        reference: dict[str, float],
        current: dict[str, float],
        reference_start: str = "",
        reference_end: str = "",
        comparison_start: str = "",
        comparison_end: str = "",
    ) -> DriftReport:
        """Compare two signal distributions and return a DriftReport.

        Args:
            reference:  Dict of signal_name → rate/count in the baseline period.
            current:    Dict of signal_name → rate/count in the comparison period.
            reference_start/end:    ISO date strings labelling the baseline window.
            comparison_start/end:   ISO date strings labelling the current window.

        Returns:
            DriftReport with KL-divergence and per-signal deltas.
        """
        all_signals = sorted(set(reference) | set(current))
        if not all_signals:
            return DriftReport(
                reference_start=reference_start,
                reference_end=reference_end,
                comparison_start=comparison_start,
                comparison_end=comparison_end,
                kl_divergence=0.0,
                drift_detected=False,
                top_shifted_signals=[],
                message="No signals provided.",
            )

        p = self._normalise(reference, all_signals)
        q = self._normalise(current, all_signals)
        kl = self._kl_divergence(p, q)

        top_shifted = self._top_shifted(all_signals, p, q, top_n=10)
        drift_detected = kl > self.drift_threshold

        msg = (
            f"Drift detected (KL={kl:.4f} > {self.drift_threshold}). "
            f"Top shifted signal: {top_shifted[0]['signal'] if top_shifted else 'N/A'}."
        ) if drift_detected else f"No significant drift (KL={kl:.4f})."

        return DriftReport(
            reference_start=reference_start,
            reference_end=reference_end,
            comparison_start=comparison_start,
            comparison_end=comparison_end,
            kl_divergence=round(kl, 6),
            drift_detected=drift_detected,
            top_shifted_signals=top_shifted,
            message=msg,
        )

    def compare_from_db(
        self,
        db: SentinelDB,
        reference_days: int = 30,
        comparison_days: int = 7,
    ) -> DriftReport:
        """Pull signal_rate_history from *db* and compute drift.

        Args:
            db:               SentinelDB instance.
            reference_days:   How many older days form the baseline.
            comparison_days:  How many recent days form the current window.

        Returns:
            DriftReport (may have drift_detected=False if history is thin).
        """
        now = datetime.now(UTC)
        cutoff_recent = (now - timedelta(days=comparison_days)).isoformat()
        cutoff_baseline_start = (now - timedelta(days=reference_days + comparison_days)).isoformat()

        all_rows = db.get_signal_rate_history(limit=10_000)

        recent_rows = [r for r in all_rows if r.get("window_end", "") >= cutoff_recent]
        baseline_rows = [
            r for r in all_rows
            if cutoff_baseline_start <= r.get("window_end", "") < cutoff_recent
        ]

        def _aggregate(rows: list) -> dict[str, float]:
            counts: dict[str, float] = {}
            total = 0
            for r in rows:
                sig = r.get("signal_name", "")
                cnt = r.get("fire_count", 0)
                counts[sig] = counts.get(sig, 0) + cnt
                total += r.get("total_jobs", 0)
            denom = max(total, 1)
            return {sig: cnt / denom for sig, cnt in counts.items()}

        reference = _aggregate(baseline_rows)
        current = _aggregate(recent_rows)

        return self.compare(
            reference=reference,
            current=current,
            reference_start=cutoff_baseline_start[:10],
            reference_end=cutoff_recent[:10],
            comparison_start=cutoff_recent[:10],
            comparison_end=now.date().isoformat(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(rates: dict[str, float], signals: list[str]) -> list[float]:
        """Build a probability vector over *signals* from *rates*."""
        vec = [max(rates.get(s, 0.0), 0.0) for s in signals]
        total = sum(vec)
        eps = 1e-9
        if total == 0.0:
            n = len(signals)
            return [1.0 / n] * n
        return [(v + eps) / (total + eps * len(vec)) for v in vec]

    @staticmethod
    def _kl_divergence(p: list[float], q: list[float]) -> float:
        """KL(P || Q) = Σ p_i * log(p_i / q_i).  Both lists must be same length."""
        eps = 1e-10
        return sum(
            pi * math.log((pi + eps) / (qi + eps))
            for pi, qi in zip(p, q, strict=False)
            if pi > 0
        )

    @staticmethod
    def _top_shifted(
        signals: list[str],
        p: list[float],
        q: list[float],
        top_n: int = 10,
    ) -> list[dict]:
        deltas = [
            {
                "signal": sig,
                "baseline_share": round(pi, 6),
                "current_share": round(qi, 6),
                "delta": round(qi - pi, 6),
            }
            for sig, pi, qi in zip(signals, p, q, strict=False)
        ]
        deltas.sort(key=lambda x: abs(x["delta"]), reverse=True)
        return deltas[:top_n]


# ---------------------------------------------------------------------------
# PredictiveModel
# ---------------------------------------------------------------------------

class PredictiveModel:
    """Simple linear regression for next-week scam volume prediction.

    Fits a line through the (week_index, volume) pairs observed so far
    and extrapolates one step into the future.

    Usage::

        model = PredictiveModel()
        model.add_observation(week_key="2026-W15", volume=42)
        model.add_observation(week_key="2026-W16", volume=55)
        result = model.predict_next_week()
        print(result.predicted_volume, result.lower_bound, result.upper_bound)
    """

    MIN_WEEKS: int = 3     # minimum observations needed for a prediction

    def __init__(self) -> None:
        # Ordered list of (week_key, volume)
        self._observations: list[tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_observation(self, week_key: str, volume: float) -> None:
        """Append a weekly volume observation.

        Args:
            week_key: ISO week string, e.g. '2026-W17'.
            volume:   Total scam-related job count for that week.
        """
        self._observations.append((week_key, float(volume)))

    def add_observations(self, weekly_volumes: dict[str, float]) -> None:
        """Bulk-add a dict of {week_key: volume} observations (sorted by key)."""
        for wk in sorted(weekly_volumes):
            self.add_observation(wk, weekly_volumes[wk])

    def predict_next_week(self) -> PredictionResult:
        """Fit OLS linear regression and predict one week ahead.

        Returns a PredictionResult.  If fewer than MIN_WEEKS observations are
        available, returns a result with confidence=0.0 and a message.
        """
        n = len(self._observations)
        if n < self.MIN_WEEKS:
            return PredictionResult(
                target_week="",
                predicted_volume=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                confidence=0.0,
                slope=0.0,
                intercept=0.0,
                n_weeks_used=n,
                message=f"Insufficient data: {n} week(s) observed, need ≥ {self.MIN_WEEKS}.",
            )

        xs = list(range(n))
        ys = [v for _, v in self._observations]

        slope, intercept = self._ols(xs, ys)
        r2 = self._r_squared(xs, ys, slope, intercept)
        confidence = max(0.0, min(1.0, r2))

        next_x = n
        predicted = slope * next_x + intercept
        predicted = max(0.0, predicted)

        # Confidence interval: residual standard error * 1.96
        residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys, strict=False)]
        rse = statistics.pstdev(residuals) if len(residuals) >= 2 else 0.0

        margin = 1.96 * rse
        lower = max(0.0, predicted - margin)
        upper = predicted + margin

        # Compute target week key
        last_wk = self._observations[-1][0]
        target_wk = self._next_week_key(last_wk)

        return PredictionResult(
            target_week=target_wk,
            predicted_volume=round(predicted, 2),
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
            confidence=round(confidence, 4),
            slope=round(slope, 4),
            intercept=round(intercept, 4),
            n_weeks_used=n,
            message=f"Predicted {predicted:.1f} jobs for {target_wk} (R²={r2:.3f}).",
        )

    def predict_category_volumes(
        self,
        category_weekly: dict[str, dict[str, float]],
    ) -> dict[str, PredictionResult]:
        """Predict next-week volume for each category independently.

        Args:
            category_weekly: Dict of category → {week_key → volume}.

        Returns:
            Dict of category → PredictionResult.
        """
        results: dict[str, PredictionResult] = {}
        for cat, weekly in category_weekly.items():
            m = PredictiveModel()
            m.add_observations(weekly)
            results[cat] = m.predict_next_week()
        return results

    def observations(self) -> list[tuple[str, float]]:
        """Return a copy of the stored (week_key, volume) list."""
        return list(self._observations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
        """Ordinary least squares: returns (slope, intercept)."""
        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys, strict=False))
        sum_xx = sum(x * x for x in xs)
        denom = n * sum_xx - sum_x ** 2
        if denom == 0:
            return 0.0, sum_y / n
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    @staticmethod
    def _r_squared(xs: list[float], ys: list[float], slope: float, intercept: float) -> float:
        """Coefficient of determination R²."""
        if len(ys) < 2:
            return 0.0
        y_mean = statistics.mean(ys)
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=False))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    @staticmethod
    def _next_week_key(week_key: str) -> str:
        """Given '2026-W17', return '2026-W18' (handles year rollover)."""
        try:
            year_str, w_str = week_key.split("-W")
            year = int(year_str)
            week = int(w_str) + 1
            # ISO weeks: 52 or 53 per year; use Dec 28 trick
            last_week = date(year, 12, 28).isocalendar()[1]
            if week > last_week:
                week = 1
                year += 1
            return f"{year}-W{week:02d}"
        except (ValueError, AttributeError):
            return ""
