# Contributing to JobSentinel

Thanks for helping make job searching safer! Here's how to get started.

## Quick Start

```bash
git clone https://github.com/ericrihm/JobSentinel.git
cd JobSentinel
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Ways to Contribute

- **Report scam patterns** — Found a new scam type? Open an issue with the "scam-pattern" label
- **Improve signals** — Add or tune detection signals in `sentinel/signals.py` (Python) or `workers/src/signals.js` (JS)
- **Fix bugs** — Check issues labeled "good first issue" for beginner-friendly tasks
- **Add tests** — We have 2,789+ tests and want more. See `tests/` for patterns
- **Improve docs** — The website lives in `docs/` and is plain HTML/CSS/JS

## Architecture

```
sentinel/          # Python detection engine
├── signals.py     # 48 detection signals
├── adversarial.py # Anti-evasion normalization
├── flywheel.py    # Self-improving detection loop
└── db.py          # SQLite storage layer

workers/src/       # Cloudflare Worker API
└── signals.js     # 54 JS signals (shared with website + extension)

docs/              # Static website (GitHub Pages)
sentinel/web/extension/  # Chrome extension (MV3)
```

## Pull Request Guidelines

1. **One concern per PR** — Keep changes focused
2. **Tests required** — Add tests for new signals or logic changes
3. **Match existing style** — No linter config yet, just match what's there
4. **Signal changes need both engines** — If you change a signal in Python, update JS too (or flag it)

## Signal Contribution Guide

Adding a new signal? Here's the pattern:

**Python** (`sentinel/signals.py`):
```python
def check_my_signal(job: JobPosting) -> Optional[Signal]:
    if some_condition(job.description):
        return Signal(
            name="my_signal",
            category=SignalCategory.WARNING,
            weight=0.5,
            confidence=0.8,
            detail="Why this matters",
            evidence="matched text"
        )
    return None
```
Then add `check_my_signal` to `ALL_SIGNALS`.

**JavaScript** (`workers/src/signals.js`):
```javascript
function checkMySignal(job) {
  if (someCondition(job.description)) {
    return { name: 'my_signal', category: 'warning', weight: 0.50, confidence: 0.8, detail: '...', evidence: '...' };
  }
  return null;
}
```
Then add `checkMySignal` to `ALL_SIGNALS`.

## Code of Conduct

Be kind. We're all here to protect job seekers.
