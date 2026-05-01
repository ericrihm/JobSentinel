# Task: Connect Flywheel Learned Weights to Scorer

## Task

Wire the flywheel's learned signal weights (from Thompson Sampling) into `scorer.py` so that `score_signals()` uses learned weights when available, falling back to static weights for new/unknown signals.

## Context

- `sentinel/scorer.py` contains `score_signals(signals: list[ScamSignal]) -> tuple[float, float]` which computes scam scores using each signal's `weight` field (set statically in `signals.py` signal extractors, e.g., `weight=0.95` for `upfront_payment`).
- `sentinel/flywheel.py` contains a `SignalWeightTracker` class that maintains Bayesian Beta(alpha, beta) posteriors per signal name. It learns from user reports via `update()` and provides `expected_weight()` and `sample()` methods.
- The `SignalWeightTracker` in `flywheel.py` is separate from the one in `scorer.py` (both files define their own version). The flywheel's version is the active learner.
- The DB stores pattern statistics (alpha, beta, observations) in the `patterns` table via `db.update_pattern_stats()`.
- Currently, learning happens (flywheel updates weights) but the scorer never reads those learned weights -- it always uses the static `signal.weight` values.

## What To Do

### 1. Add a function to load learned weights from DB

In `sentinel/db.py`, add a method to `SentinelDB`:

```python
def get_signal_weights(self) -> dict[str, float]:
    """Return learned signal weights as {signal_name: expected_weight}.
    
    Expected weight = alpha / (alpha + beta) from the patterns table.
    Only returns signals with observations > 0.
    """
```

Query the `patterns` table for all active patterns with `observations > 0`, compute `alpha / (alpha + beta)` for each, and return as a dict keyed by `pattern_id`.

### 2. Modify `score_signals()` in `sentinel/scorer.py`

Change the function signature to accept an optional weights dict:

```python
def score_signals(
    signals: list[ScamSignal],
    learned_weights: dict[str, float] | None = None,
) -> tuple[float, float]:
```

Inside the scoring loop:
- If `learned_weights` is provided and contains the signal's `name`, use the learned weight instead of `s.weight`.
- If the signal name is not in `learned_weights`, fall back to the static `s.weight`.
- Clamp all weights to `[0.001, 0.999]` to avoid log(0) errors.

### 3. Update callers of `score_signals()`

In `sentinel/scorer.py`, update `build_result()` to optionally pass learned weights through:

```python
def build_result(
    job: JobPosting,
    signals: list[ScamSignal],
    analysis_time_ms: float = 0.0,
    learned_weights: dict[str, float] | None = None,
) -> ValidationResult:
```

In `sentinel/analyzer.py`, in `analyze_job()`:
- Try to load learned weights from DB: call `SentinelDB().get_signal_weights()`.
- Pass them to `build_result()`.
- Wrap in try/except so failures fall back to static weights silently.

### 4. Add tests in `tests/test_flywheel_scorer.py`

Write tests covering:

- `score_signals()` without `learned_weights` behaves identically to current behavior (regression test).
- `score_signals()` with learned weights uses them for matching signals.
- Signals not in `learned_weights` fall back to their static weight.
- A signal with high learned weight (0.95) produces a higher scam score than the same signal with low learned weight (0.3).
- `db.get_signal_weights()` returns empty dict for a fresh DB.
- `db.get_signal_weights()` returns correct values after `update_pattern_stats()` calls.
- End-to-end: update pattern stats in DB, load weights, score signals -- verify weight propagation.

## Acceptance Criteria

- [ ] `score_signals()` accepts optional `learned_weights` parameter.
- [ ] When `learned_weights` is provided, matching signal weights are overridden.
- [ ] Signals not in `learned_weights` use their static weight.
- [ ] `db.get_signal_weights()` method exists and returns learned weights from the patterns table.
- [ ] `analyze_job()` attempts to load and use learned weights.
- [ ] All existing tests still pass (backward compatible -- `learned_weights=None` preserves old behavior).
- [ ] New tests verify weight propagation end-to-end.

## Constraints

- Do not change the default behavior of `score_signals()` when called without `learned_weights`.
- Do not remove or rename the static weights in `signals.py`.
- Keep DB access best-effort -- if loading weights fails, silently fall back to static weights.
- No new dependencies.

## Test Command

```bash
python -m pytest tests/test_flywheel_scorer.py tests/test_core.py -v
```
