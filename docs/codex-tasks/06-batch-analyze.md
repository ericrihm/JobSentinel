# Task: Wire Up --file Batch Analysis in CLI

## Task

Add `--file` support to the `sentinel analyze` CLI command to enable batch analysis from JSON and CSV files.

## Context

- `sentinel/cli.py` defines the `analyze` Click command. Currently it only accepts a single `INPUT_TEXT` argument (URL or raw text).
- `CLAUDE.md` documents `sentinel analyze --file jobs.json` but this option does not exist yet.
- `sentinel/scanner.py` already has `load_jobs_from_file(path: str) -> list[JobPosting]` which supports JSON arrays, single JSON objects, and NDJSON format.
- `sentinel/analyzer.py` already has `batch_analyze(jobs: list[JobPosting], use_ai: bool = False) -> list[ValidationResult]`.
- The existing `analyze` command is defined at line ~69 of `cli.py`:
  ```python
  @main.command()
  @click.argument("input_text")
  @click.option("--title", ...)
  @click.option("--company", ...)
  @click.option("--no-ai", ...)
  ```

## What To Do

### 1. Add `--file` option to the analyze command

Modify the `analyze` command in `sentinel/cli.py`:

- Change `INPUT_TEXT` from a required argument to an optional argument: `@click.argument("input_text", required=False, default=None)`.
- Add `@click.option("--file", "file_path", type=click.Path(exists=True), default=None, help="Batch analyze jobs from a JSON or CSV file.")`.
- Validate that exactly one of `input_text` or `file_path` is provided. If neither or both, show an error.

### 2. Add CSV support to scanner

In `sentinel/scanner.py`, extend `load_jobs_from_file()` to also handle CSV format:

- If the file extension is `.csv`, parse it using `csv.DictReader` from stdlib.
- Map CSV column headers to `JobPosting` fields using the same `_FIELD_ALIASES` dict already in the file.
- Call `parse_job_json()` on each row dict (it already handles field alias mapping).
- Update the function docstring to mention CSV support.

### 3. Implement batch analysis flow

When `--file` is provided:

1. Call `load_jobs_from_file(file_path)` to get `list[JobPosting]`.
2. If the list is empty, print an error and exit.
3. Call `batch_analyze(jobs, use_ai=not no_ai)` to get results.
4. If `--json-output` flag is set, output the full results as JSON array.
5. Otherwise, output a summary table:

```
  Batch Analysis: jobs.json
  ──────────────────────────────────────
  Total jobs:     25
  Scam detected:   3 (12.0%)
  High risk:       2
  Suspicious:      4
  Low risk:        8
  Safe:           11
  
  Avg scam score: 0.23
  Analysis time:  142ms

  Flagged Jobs:
    [SCAM]  "Work From Home Data Entry" — ScamCo — 0.92
    [HIGH]  "Remote Assistant" — Unknown — 0.74
    [HIGH]  "Easy Money Online" — FastCash LLC — 0.68
```

6. The "Flagged Jobs" section shows only jobs with scam_score >= 0.6, sorted by score descending.
7. Persist all results to DB (best-effort, same pattern as single analyze).

### 4. Add tests in `tests/test_batch_analyze.py`

Write tests using Click's `CliRunner`:

- **JSON file batch:**
  - Create a temp JSON file with 3 job postings (1 scam, 1 legit, 1 ambiguous).
  - Run `sentinel analyze --file <path>`.
  - Verify output contains "Total jobs: 3" and "Scam detected:" line.

- **CSV file batch:**
  - Create a temp CSV file with headers: `title,company,description,url`.
  - Run `sentinel analyze --file <path>`.
  - Verify output contains job count.

- **JSON output mode:**
  - Run `sentinel --json-output analyze --file <path>`.
  - Verify output is valid JSON array.

- **Empty file:**
  - Create an empty JSON file (`[]`).
  - Verify the command prints an error and exits.

- **Mutual exclusivity:**
  - Run `sentinel analyze "some text" --file <path>` and verify it errors.
  - Run `sentinel analyze` (no args, no file) and verify it errors.

- **NDJSON file:**
  - Create a temp file with one JSON object per line.
  - Verify batch analysis works.

## Acceptance Criteria

- [ ] `sentinel analyze --file jobs.json` works and outputs a summary table.
- [ ] `sentinel analyze --file jobs.csv` works with CSV input.
- [ ] `sentinel --json-output analyze --file jobs.json` outputs JSON array.
- [ ] `INPUT_TEXT` and `--file` are mutually exclusive with clear error messages.
- [ ] Flagged jobs (score >= 0.6) are listed in the summary.
- [ ] All existing `analyze` command functionality (single URL/text) still works.
- [ ] All tests pass.

## Constraints

- Use `csv.DictReader` from stdlib for CSV parsing. Do not add pandas or other data libraries.
- Keep `INPUT_TEXT` as a positional argument (not an option) for backward compatibility.
- The batch flow should disable AI by default (`use_ai=False`) for cost reasons, unless `--no-ai` is explicitly NOT set.
- Do not modify `analyzer.py` or `scanner.py` signatures beyond adding CSV support to `load_jobs_from_file`.

## Test Command

```bash
python -m pytest tests/test_batch_analyze.py tests/test_core.py -v
```
