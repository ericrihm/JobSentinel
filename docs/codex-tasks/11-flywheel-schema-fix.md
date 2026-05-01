# Task: Fix flywheel_metrics Table Schema Mismatch

## Task

Fix the `flywheel_metrics` table in `sentinel/db.py` so its schema includes all columns that `flywheel.py` `run_cycle()` writes, add a migration path for existing databases, and update `save_flywheel_metrics()` to persist all fields.

## Context

- `sentinel/db.py` defines the `flywheel_metrics` table schema in the `SCHEMA` string (lines 76-87):
  ```sql
  CREATE TABLE IF NOT EXISTS flywheel_metrics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      cycle_ts TEXT,
      total_analyzed INTEGER DEFAULT 0,
      true_positives INTEGER DEFAULT 0,
      false_positives INTEGER DEFAULT 0,
      precision REAL DEFAULT 0.0,
      recall REAL DEFAULT 0.0,
      signals_updated INTEGER DEFAULT 0,
      patterns_evolved INTEGER DEFAULT 0
  );
  ```

- `sentinel/flywheel.py` `run_cycle()` (line ~387) builds a metrics dict with these additional keys that are NOT in the schema:
  - `cycle_number` (int)
  - `f1` (float)
  - `accuracy` (float)
  - `patterns_promoted` (list of strings)
  - `patterns_deprecated` (list of strings)
  - `regression_alarm` (bool)
  - `cusum_statistic` (float)

- `db.save_flywheel_metrics()` (lines 389-411) only inserts the 8 columns from the original schema, so the extra fields from `run_cycle()` are silently dropped.

- The `get_stats()` method reads `precision` and `recall` from the last flywheel_metrics row, but not `f1` or other new fields.

## What To Do

### 1. Update the CREATE TABLE statement

In `sentinel/db.py`, update the `flywheel_metrics` table definition in the `SCHEMA` string to include all 16 columns:

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
    patterns_promoted TEXT DEFAULT '[]',
    patterns_deprecated TEXT DEFAULT '[]',
    regression_alarm INTEGER DEFAULT 0,
    cusum_statistic REAL DEFAULT 0.0
);
```

Notes:
- `patterns_promoted` and `patterns_deprecated` are stored as JSON strings (TEXT) since they are lists of pattern IDs.
- `regression_alarm` is stored as INTEGER (0/1 for boolean).

### 2. Add migration for existing databases

In `SentinelDB.__init__()`, after `self.conn.executescript(SCHEMA)` and `self.conn.commit()`, add a migration step:

```python
self._migrate_flywheel_metrics()
```

Implement the migration method:

```python
def _migrate_flywheel_metrics(self) -> None:
    """Add missing columns to flywheel_metrics for existing databases."""
    cursor = self.conn.execute("PRAGMA table_info(flywheel_metrics)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    
    migrations = [
        ("cycle_number", "INTEGER DEFAULT 0"),
        ("f1", "REAL DEFAULT 0.0"),
        ("accuracy", "REAL DEFAULT 0.0"),
        ("patterns_promoted", "TEXT DEFAULT '[]'"),
        ("patterns_deprecated", "TEXT DEFAULT '[]'"),
        ("regression_alarm", "INTEGER DEFAULT 0"),
        ("cusum_statistic", "REAL DEFAULT 0.0"),
    ]
    
    for col_name, col_def in migrations:
        if col_name not in existing_cols:
            self.conn.execute(
                f"ALTER TABLE flywheel_metrics ADD COLUMN {col_name} {col_def}"
            )
    
    self.conn.commit()
```

### 3. Update `save_flywheel_metrics()`

Replace the current INSERT statement to include all fields:

```python
def save_flywheel_metrics(self, metrics: dict) -> None:
    """Append a flywheel cycle snapshot."""
    promoted = metrics.get("patterns_promoted", [])
    deprecated = metrics.get("patterns_deprecated", [])
    if not isinstance(promoted, str):
        promoted = json.dumps(promoted)
    if not isinstance(deprecated, str):
        deprecated = json.dumps(deprecated)
    
    self.conn.execute(
        """
        INSERT INTO flywheel_metrics
            (cycle_ts, cycle_number, total_analyzed, true_positives, false_positives,
             precision, recall, f1, accuracy, signals_updated, patterns_evolved,
             patterns_promoted, patterns_deprecated, regression_alarm, cusum_statistic)
        VALUES
            (:cycle_ts, :cycle_number, :total_analyzed, :true_positives, :false_positives,
             :precision, :recall, :f1, :accuracy, :signals_updated, :patterns_evolved,
             :patterns_promoted, :patterns_deprecated, :regression_alarm, :cusum_statistic)
        """,
        {
            "cycle_ts": metrics.get("cycle_ts", _now_iso()),
            "cycle_number": metrics.get("cycle_number", 0),
            "total_analyzed": metrics.get("total_analyzed", 0),
            "true_positives": metrics.get("true_positives", 0),
            "false_positives": metrics.get("false_positives", 0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "signals_updated": metrics.get("signals_updated", 0),
            "patterns_evolved": metrics.get("patterns_evolved", 0),
            "patterns_promoted": promoted,
            "patterns_deprecated": deprecated,
            "regression_alarm": int(metrics.get("regression_alarm", False)),
            "cusum_statistic": metrics.get("cusum_statistic", 0.0),
        },
    )
    self.conn.commit()
```

### 4. Add a `get_flywheel_metrics()` method

Add a method to retrieve flywheel metrics for reading back:

```python
def get_flywheel_metrics(self, limit: int = 10) -> list[dict]:
    """Return the most recent flywheel cycle metrics, newest first."""
    rows = self.conn.execute(
        "SELECT * FROM flywheel_metrics ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        # Deserialize JSON list fields
        for list_field in ("patterns_promoted", "patterns_deprecated"):
            try:
                d[list_field] = json.loads(d.get(list_field) or "[]")
            except (json.JSONDecodeError, TypeError):
                d[list_field] = []
        d["regression_alarm"] = bool(d.get("regression_alarm", 0))
        results.append(d)
    return results
```

### 5. Add tests in `tests/test_flywheel_schema.py`

Write tests:

- **New database has all columns:**
  - Create a `SentinelDB` with `tmp_path`.
  - Run `PRAGMA table_info(flywheel_metrics)`.
  - Verify all 16 columns exist: id, cycle_ts, cycle_number, total_analyzed, true_positives, false_positives, precision, recall, f1, accuracy, signals_updated, patterns_evolved, patterns_promoted, patterns_deprecated, regression_alarm, cusum_statistic.

- **Round-trip persistence of all fields:**
  - Save a metrics dict with ALL fields populated (including `patterns_promoted=["pat_1", "pat_2"]` and `regression_alarm=True`).
  - Read it back via `get_flywheel_metrics()`.
  - Verify every field matches the input.

- **Migration from old schema:**
  - Create a DB manually with only the original 9 columns (use raw SQL to create the table, bypassing the `SCHEMA` string).
  - Close the connection.
  - Re-open with `SentinelDB()` (triggers migration).
  - Verify new columns exist via `PRAGMA table_info`.
  - Save metrics with all fields and verify round-trip works.

- **Backward compatibility (missing fields default correctly):**
  - Save metrics with only the original fields (no f1, no cycle_number, etc.).
  - Read back and verify defaults: f1=0.0, cycle_number=0, regression_alarm=False, patterns_promoted=[], patterns_deprecated=[], cusum_statistic=0.0.

- **patterns_promoted/deprecated serialization:**
  - Save metrics with `patterns_promoted=["pat_1", "pat_2"]` and `patterns_deprecated=["old_pat"]`.
  - Read back and verify they are Python lists, not JSON strings.

- **run_cycle integration:**
  - Create a `DetectionFlywheel` with temp DB.
  - Call `run_cycle()`.
  - Call `db.get_flywheel_metrics()`.
  - Verify the saved metrics contain `f1`, `cycle_number`, `regression_alarm`, `cusum_statistic` keys.

## Acceptance Criteria

- [ ] `flywheel_metrics` table schema includes all 16 columns.
- [ ] `save_flywheel_metrics()` persists all fields including f1, accuracy, cycle_number, patterns_promoted/deprecated, regression_alarm, cusum_statistic.
- [ ] Existing databases are migrated via ALTER TABLE on open.
- [ ] `get_flywheel_metrics()` method exists and correctly deserializes JSON list fields and boolean fields.
- [ ] Round-trip test passes for all fields.
- [ ] All existing tests still pass.

## Constraints

- SQLite ALTER TABLE only supports ADD COLUMN (not DROP or MODIFY). The migration must only add missing columns.
- Use `PRAGMA table_info` to check existing columns before altering.
- JSON-encode list fields (`patterns_promoted`, `patterns_deprecated`) as TEXT.
- Boolean fields (`regression_alarm`) are stored as INTEGER (0/1).
- Do not change `flywheel.py` -- only `db.py`.
- Migration must be idempotent (safe to run multiple times).

## Test Command

```bash
python -m pytest tests/test_flywheel_schema.py tests/test_core.py -v
```
