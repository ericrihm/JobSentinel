# Task: Implement the Stub Innovation Strategies

## Task

Replace the three no-op stub strategies in `sentinel/innovation.py` with real implementations: `_correlate_signals`, `_expand_keywords`, and `_mine_patterns`.

## Context

- `sentinel/innovation.py` defines an `InnovationEngine` class with 8 Thompson Sampling strategies.
- Three of these strategies are effectively stubs that don't do real work:
  - `_correlate_signals` (line ~251): Returns a hardcoded success message without computing anything.
  - `_expand_keywords` (line ~258): Fetches report reasons but doesn't extract or propose new keywords.
  - `_mine_patterns` (line ~223): Fetches scam reports and reasons but doesn't generate candidate patterns.

- The DB has these relevant tables/methods:
  - `db.get_reports(limit)` returns `list[dict]` with keys: `url`, `is_scam`, `reason`, `our_prediction`, `was_correct`, `reported_at`.
  - `db.get_job(url)` returns a job dict with `signals_json` (JSON string of signal list).
  - `db.get_patterns(status)` returns `list[dict]` of patterns.
  - `db.save_pattern(pattern_data)` persists a pattern.
  - Job dicts have `signals_json` which is a JSON string like `[{"name": "upfront_payment", "category": "red_flag", "detail": "..."}]`.

- `sentinel/models.py` defines `ScamPattern` with fields: `pattern_id`, `name`, `description`, `category` (SignalCategory enum), `regex`, `keywords`, `alpha`, `beta`, `observations`, `true_positives`, `false_positives`, `status`.

## What To Do

### 1. Implement `_correlate_signals` (cross-signal correlation)

Replace the stub with real logic:

1. Fetch the last 200 reports from DB.
2. For each scam report (`is_scam=True`), load the corresponding job via `db.get_job(url)` and parse its `signals_json`.
3. Build a co-occurrence matrix: for each pair of signals that fired on the same scam job, increment a counter.
4. Compute a "pair strength" metric: `pair_count / min(individual_count_a, individual_count_b)`. This measures how often two signals co-occur relative to how often either appears alone.
5. Return the top 5 pairs sorted by pair strength.
6. Return `ImprovementResult` with `success=True` if any pairs found with strength > 0.3, listing the top pairs in `detail`. Return `success=False` if insufficient data (< 10 scam reports with signals).

### 2. Implement `_expand_keywords` (keyword expansion)

Replace the stub with real logic:

1. Fetch the last 100 scam reports (`is_scam=True`) from DB.
2. Collect all `reason` strings from these reports.
3. Tokenize reasons into words (lowercase, strip punctuation, remove stopwords). Use a hardcoded stopword list (the, a, an, is, was, were, are, been, be, have, has, had, do, does, did, will, would, could, should, may, might, shall, can, need, dare, ought, used, it, its, this, that, these, those, i, me, my, we, our, you, your, he, him, his, she, her, they, them, their, and, but, or, if, for, not, no, so, too, to, of, in, on, at, by, with, from, as, into, about, between, through, after, before, during, without, the, job, posting, scam, company, work, just).
4. Load existing pattern keywords from all active patterns in DB.
5. Find words that appear in 3+ report reasons but are NOT in any existing pattern's keyword list.
6. Return the top 10 new keyword candidates sorted by frequency.
7. Return `ImprovementResult` with `success=True` if new keywords found, listing them in `detail`. `success=False` if no new keywords discovered.

### 3. Implement `_mine_patterns` (pattern mining)

Replace the stub with real logic:

1. Fetch the last 200 scam reports (`is_scam=True`) from DB.
2. Collect all `reason` strings.
3. Group reasons by common phrases: extract 2-gram and 3-gram phrases from reasons, count their frequency.
4. Filter to n-grams appearing in 3+ different reports.
5. For each qualifying n-gram, generate a candidate `ScamPattern`:
   - `pattern_id`: `"mined_" + slugified_ngram`
   - `name`: Capitalized n-gram
   - `category`: `"warning"` (default for mined patterns)
   - `regex`: Simple regex matching the n-gram (case-insensitive)
   - `keywords`: The n-gram tokens as a list
   - `status`: `"candidate"` (not active until promoted by flywheel)
6. Save each candidate pattern to DB via `db.save_pattern()`.
7. Return `ImprovementResult` with `success=True` and `new_patterns` count if any candidates generated. Return `success=False` if insufficient data or no new patterns found.
8. Do NOT create duplicate patterns -- check existing pattern_ids before saving.

### 4. Add tests in `tests/test_innovation_strategies.py`

Write tests for each strategy:

**_correlate_signals:**
- With 0 scam reports, returns `success=False`.
- With 10+ scam reports that have signals, returns top pairs with strength values.
- Pairs are sorted by strength descending.

**_expand_keywords:**
- With 0 scam reports, returns `success=False`.
- With reports containing novel words, returns new keyword candidates.
- Words already in existing patterns are excluded.
- Stopwords are filtered out.

**_mine_patterns:**
- With < 5 scam reports, returns `success=False`.
- With 5+ reports sharing common phrases, generates candidate patterns.
- Candidate patterns are saved to DB with `status="candidate"`.
- Duplicate pattern_ids are not created on repeat runs.

For all tests, use a temp DB (via `tmp_path` fixture) and seed it with test data.

## Acceptance Criteria

- [ ] `_correlate_signals` computes real pairwise signal co-occurrence and returns top pairs.
- [ ] `_expand_keywords` extracts novel keywords from report reasons, excluding existing pattern keywords.
- [ ] `_mine_patterns` generates candidate ScamPattern objects from common report phrases and saves them to DB.
- [ ] All three strategies return `ImprovementResult` with meaningful `detail` strings.
- [ ] All existing tests pass (the stub strategies previously returned `success=True` unconditionally for `_correlate_signals`; the new tests should cover the real behavior).
- [ ] New tests cover each strategy with both sufficient and insufficient data.

## Constraints

- Use only Python stdlib for text processing (no nltk, spacy, etc.).
- N-gram extraction should be simple: split on whitespace, lowercase, take consecutive pairs/triples.
- Pattern mining generates `candidate` patterns -- they are NOT promoted to `active` (that's the flywheel's job).
- Do not modify other strategy implementations (`_review_false_positives`, `_optimize_weights`, etc.).

## Test Command

```bash
python -m pytest tests/test_innovation_strategies.py tests/test_advanced.py -v
```
