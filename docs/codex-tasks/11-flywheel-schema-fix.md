# Task: Fix flywheel_metrics Table Schema Mismatch

## Task

The `flywheel_metrics` table in `db.py` is missing columns that `flywheel.py` tries to write. Fix the schema and add migration for existing databases.

## Context

- `sentinel/db.py` defines the `flywheel_metrics` table (line ~76) with 8 columns: id, cycle_ts, total_analyzed, true_positives, false_positives, precision, recall, signals_updated, patterns_evolved
- `sentinel/flywheel.py` `run_cycle()` writes additional fields: f1, accuracy, cycle_number, regression_alarm, cusum_statistic, patterns_promoted, patterns_deprecated
- These extra fields are silently dropped because the INSERT statement doesn't include them

## What To Do

### 1. Update CREATE TABLE schema in `db.py`

Add the missing columns to the `flywheel_metrics` table definition:

```sql
CREATE TABLE IF NOT EXISTS flywheel_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_ts TEXT,
    cycle_number INTEGER DEFAULT 0,
    total_analyzed INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    precision REAL DEFAULT 0.0,
    recall REAL DEFAULT 0.0,
    f1 REAL DEFAULT 0.0,
    accuracy REAL DEFAULT 0.0,
    signals_updated INTEGER DEFAULT 0,
    patterns_evolved INTEGER DEFAULT 0,
    patterns_promoted INTEGER DEFAULT 0,
    patterns_deprecated INTEGER DEFAULT 0,
    regression_alarm INTEGER DEFAULT 0,
    cusum_statistic REAL DEFAULT 0.0
);
```

### 2. Add migration for existing databases

In `SentinelDB.__init__()`, after `executescript(SCHEMA)`, add migration code:

```python
_FLYWHEEL_MIGRATIONS = [
    "ALTER TABLE flywheel_metrics ADD COLUMN f1 REAL DEFAULT 0.0",
    "ALTER TABLE flywheel_metrics ADD COLUMN accuracy REAL DEFAULT 0.0",
    "ALTER TABLE flywheel_metrics ADD COLUMN cycle_number INTEGER DEFAULT 0",
    "ALTER TABLE flywheel_metrics ADD COLUMN regression_alarm INTEGER DEFAULT 0",
    "ALTER TABLE flywheel_metrics ADD COLUMN cusum_statistic REAL DEFAULT 0.0",
    "ALTER TABLE flywheel_metrics ADD COLUMN patterns_promoted INTEGER DEFAULT 0",
    "ALTER TABLE flywheel_metrics ADD COLUMN patterns_deprecated INTEGER DEFAULT 0",
]
```

Run each in a try/except (SQLite silently handles "column already exists" via catching the OperationalError).

### 3. Update `save_flywheel_metrics()`

Ensure the INSERT statement includes all columns.

### 4. Add tests

- Test that new DB has all columns (create fresh, inspect PRAGMA table_info)
- Test save_flywheel_metrics with all fields set, then load and verify round-trip
- Test migration: create DB with old schema, then re-init and verify new columns exist

## Acceptance Criteria

- [ ] `flywheel_metrics` table includes all 16 columns
- [ ] Existing databases are migrated on next SentinelDB init
- [ ] `save_flywheel_metrics()` persists all fields
- [ ] Round-trip test passes
- [ ] All tests pass: `python -m pytest tests/ -v`

## Constraints

- Migrations must be idempotent (safe to run multiple times)
- Do not drop or recreate the table (existing data must be preserved)
- Use parameterized INSERT statements

## Test Command

```bash
python -m pytest tests/ -v
```
