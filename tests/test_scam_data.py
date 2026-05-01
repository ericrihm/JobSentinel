"""Unit tests for sentinel/scam_data.py — scam data collection and ingestion."""

from __future__ import annotations

import csv
import io
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch, call

from sentinel.scam_data import (
    ScamDataCollector,
    _make_job_dict,
    _parse_salary_range,
    _synthetic_url,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collector() -> ScamDataCollector:
    return ScamDataCollector()


def _write_emfj_csv(rows: list[dict], path: str) -> None:
    """Write a minimal EMFJ CSV file for testing."""
    fieldnames = [
        "title", "location", "department", "salary_range", "company_profile",
        "description", "requirements", "benefits", "telecommuting",
        "has_company_logo", "has_questions", "employment_type",
        "required_experience", "required_education", "industry", "function",
        "fraudulent",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ===========================================================================
# _parse_salary_range helper
# ===========================================================================

class TestParseSalaryRange:
    def test_standard_range(self):
        lo, hi = _parse_salary_range("$50,000-$70,000")
        assert lo == 50000.0
        assert hi == 70000.0

    def test_single_value(self):
        lo, hi = _parse_salary_range("$80,000")
        assert lo == 80000.0
        assert hi == 80000.0

    def test_empty_string(self):
        assert _parse_salary_range("") == (0.0, 0.0)

    def test_non_numeric(self):
        assert _parse_salary_range("negotiable") == (0.0, 0.0)

    def test_euro_symbol(self):
        lo, hi = _parse_salary_range("€45000-€60000")
        assert lo == 45000.0
        assert hi == 60000.0


# ===========================================================================
# _make_job_dict helper
# ===========================================================================

class TestMakeJobDict:
    def test_required_fields_present(self):
        job = _make_job_dict(title="Test", company="Co", description="Desc")
        for key in ("url", "title", "company", "location", "description",
                    "salary_min", "salary_max", "source", "is_scam",
                    "scam_category", "posted_date", "scam_score", "confidence",
                    "risk_level", "analyzed_at", "signal_count", "signals_json",
                    "user_reported", "user_verdict"):
            assert key in job

    def test_is_scam_true(self):
        job = _make_job_dict(title="T", company="C", description="D", is_scam=True)
        assert job["is_scam"] is True
        assert job["user_verdict"] == "scam"

    def test_is_scam_false(self):
        job = _make_job_dict(title="T", company="C", description="D", is_scam=False)
        assert job["is_scam"] is False
        assert job["user_verdict"] == ""

    def test_synthetic_url_stable(self):
        job1 = _make_job_dict(title="Role", company="Firm", description="D", source="test")
        job2 = _make_job_dict(title="Role", company="Firm", description="D", source="test")
        assert job1["url"] == job2["url"]

    def test_synthetic_url_differs_for_different_inputs(self):
        job1 = _make_job_dict(title="A", company="X", description="D")
        job2 = _make_job_dict(title="B", company="Y", description="D")
        assert job1["url"] != job2["url"]


# ===========================================================================
# fetch_ftc_data
# ===========================================================================

class TestFetchFtcData:
    def setup_method(self):
        self.collector = _collector()

    def test_returns_list_of_dicts(self):
        # No CSV on disk → falls back to seed data
        with patch.dict("os.environ", {}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                data = self.collector.fetch_ftc_data()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_all_items_are_scam(self):
        with patch("pathlib.Path.exists", return_value=False):
            data = self.collector.fetch_ftc_data()
        assert all(item["is_scam"] is True for item in data)

    def test_limit_respected(self):
        with patch("pathlib.Path.exists", return_value=False):
            data = self.collector.fetch_ftc_data(limit=3)
        assert len(data) <= 3

    def test_seed_data_has_expected_fields(self):
        with patch("pathlib.Path.exists", return_value=False):
            data = self.collector.fetch_ftc_data()
        item = data[0]
        for key in ("title", "company", "description", "is_scam"):
            assert key in item

    def test_csv_path_env_var(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Job Title", "Company", "Description", "Location"])
            writer.writerow(["Scam Job", "Scam Co", "Pay upfront", "Remote"])
            tmp_path = f.name

        try:
            with patch.dict("os.environ", {"FTC_CSV_PATH": tmp_path}):
                data = self.collector.fetch_ftc_data()
            assert len(data) >= 1
            assert data[0]["title"] == "Scam Job"
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# load_kaggle_emfj
# ===========================================================================

class TestLoadKaggleEmfj:
    def setup_method(self):
        self.collector = _collector()

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.collector.load_kaggle_emfj("/nonexistent/path/emfj.csv")

    def test_loads_scam_and_legit(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        try:
            rows = [
                {"title": "Data Entry", "fraudulent": "1", "description": "Earn big"},
                {"title": "Software Engineer", "fraudulent": "0", "description": "Code"},
            ]
            _write_emfj_csv(rows, tmp_path)
            data = self.collector.load_kaggle_emfj(tmp_path)
            scam = [j for j in data if j["is_scam"]]
            legit = [j for j in data if not j["is_scam"]]
            assert len(scam) == 1
            assert len(legit) == 1
        finally:
            os.unlink(tmp_path)

    def test_fraudulent_truthy_values(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        try:
            rows = [
                {"title": "Job A", "fraudulent": "true"},
                {"title": "Job B", "fraudulent": "True"},
                {"title": "Job C", "fraudulent": "yes"},
                {"title": "Job D", "fraudulent": "0"},
            ]
            _write_emfj_csv(rows, tmp_path)
            data = self.collector.load_kaggle_emfj(tmp_path)
            scam_titles = {j["title"] for j in data if j["is_scam"]}
            assert "Job A" in scam_titles
            assert "Job B" in scam_titles
            assert "Job C" in scam_titles
            assert "Job D" not in scam_titles
        finally:
            os.unlink(tmp_path)

    def test_salary_range_parsed(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        try:
            rows = [{"title": "Analyst", "salary_range": "$50,000-$70,000", "fraudulent": "0"}]
            _write_emfj_csv(rows, tmp_path)
            data = self.collector.load_kaggle_emfj(tmp_path)
            assert data[0]["salary_min"] == 50000.0
            assert data[0]["salary_max"] == 70000.0
        finally:
            os.unlink(tmp_path)

    def test_emfj_specific_fields_preserved(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            tmp_path = f.name

        try:
            rows = [{
                "title": "Dev", "fraudulent": "0",
                "employment_type": "Full-time", "industry": "Tech",
                "required_experience": "Mid-Senior level", "telecommuting": "1",
            }]
            _write_emfj_csv(rows, tmp_path)
            data = self.collector.load_kaggle_emfj(tmp_path)
            item = data[0]
            assert item["employment_type"] == "Full-time"
            assert item["industry"] == "Tech"
            assert item["telecommuting"] is True
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# generate_scam_patterns
# ===========================================================================

class TestGenerateScamPatterns:
    def setup_method(self):
        self.collector = _collector()

    def test_returns_at_least_20_patterns(self):
        patterns = self.collector.generate_scam_patterns()
        assert len(patterns) >= 20

    def test_each_pattern_has_required_keys(self):
        patterns = self.collector.generate_scam_patterns()
        required = {"pattern_id", "name", "description", "category",
                    "keywords", "regex", "typical_weight", "alpha", "beta"}
        for p in patterns:
            missing = required - set(p.keys())
            assert not missing, f"Pattern {p.get('pattern_id')} missing keys: {missing}"

    def test_pattern_ids_are_unique(self):
        patterns = self.collector.generate_scam_patterns()
        ids = [p["pattern_id"] for p in patterns]
        assert len(ids) == len(set(ids)), "Duplicate pattern IDs found"

    def test_typical_weights_in_valid_range(self):
        patterns = self.collector.generate_scam_patterns()
        for p in patterns:
            assert 0.0 <= p["typical_weight"] <= 1.0, f"Invalid weight for {p['pattern_id']}"

    def test_categories_valid_values(self):
        patterns = self.collector.generate_scam_patterns()
        valid = {"red_flag", "warning", "positive", "structural", "ghost_job"}
        for p in patterns:
            assert p["category"] in valid, f"Invalid category {p['category']}"

    def test_regex_patterns_compile(self):
        import re
        patterns = self.collector.generate_scam_patterns()
        for p in patterns:
            try:
                re.compile(p["regex"])
            except re.error as exc:
                pytest.fail(f"Pattern {p['pattern_id']} has invalid regex: {exc}")

    def test_advance_fee_pattern_present(self):
        patterns = self.collector.generate_scam_patterns()
        ids = [p["pattern_id"] for p in patterns]
        assert "advance_fee_payment" in ids

    def test_keywords_are_lists(self):
        patterns = self.collector.generate_scam_patterns()
        for p in patterns:
            assert isinstance(p["keywords"], list), f"{p['pattern_id']} keywords not a list"


# ===========================================================================
# seed_database
# ===========================================================================

class TestSeedDatabase:
    def setup_method(self):
        self.collector = _collector()

    def test_returns_statistics_dict(self):
        mock_db = MagicMock()
        mock_db.conn.execute.return_value.fetchone.return_value = None
        mock_db.save_pattern = MagicMock()

        result = self.collector.seed_database(mock_db)

        assert "patterns_seeded" in result
        assert "patterns_new" in result
        assert "patterns_updated" in result
        assert "patterns_errors" in result
        assert "seeded_at" in result

    def test_all_patterns_seeded(self):
        mock_db = MagicMock()
        mock_db.conn.execute.return_value.fetchone.return_value = None
        mock_db.save_pattern = MagicMock()

        result = self.collector.seed_database(mock_db)
        expected = len(self.collector.generate_scam_patterns())
        assert result["patterns_seeded"] == expected
        assert result["patterns_new"] == expected
        assert result["patterns_updated"] == 0

    def test_existing_patterns_counted_as_updated(self):
        mock_db = MagicMock()
        # Simulate all patterns already existing
        mock_db.conn.execute.return_value.fetchone.return_value = {"pattern_id": "existing"}
        mock_db.save_pattern = MagicMock()

        result = self.collector.seed_database(mock_db)
        assert result["patterns_updated"] == result["patterns_seeded"]
        assert result["patterns_new"] == 0

    def test_db_error_counted_in_errors(self):
        mock_db = MagicMock()
        mock_db.conn.execute.return_value.fetchone.return_value = None
        mock_db.save_pattern.side_effect = Exception("DB write error")

        result = self.collector.seed_database(mock_db)
        assert result["patterns_errors"] > 0
        assert result["patterns_seeded"] == 0
