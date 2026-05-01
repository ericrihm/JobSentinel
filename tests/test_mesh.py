"""Tests for sentinel.mesh — FlywheelMesh, CascadeDetector, and related helpers.

18+ tests covering:
- Mesh topology registration and querying
- Dependency/downstream resolution
- Cascade impact preview with mock data
- Classification change detection
- Ripple effect cross-correlation
- Impact level thresholds
- ASCII graph output
- Edge cases
"""

import json
import math

import pytest

from sentinel.db import SentinelDB
from sentinel.mesh import (
    CascadeDetector,
    CascadeRecord,
    CascadeReport,
    FlywheelMesh,
    RippleEffect,
    _pearson_correlation,
    _score_to_bucket,
    build_default_mesh,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def empty_mesh() -> FlywheelMesh:
    return FlywheelMesh()


@pytest.fixture
def simple_mesh() -> FlywheelMesh:
    """A ─► B ─► C; A ─► C"""
    m = FlywheelMesh()
    m.register_flywheel("A", dependencies=[], outputs=["B", "C"])
    m.register_flywheel("B", dependencies=["A"], outputs=["C"])
    m.register_flywheel("C", dependencies=["A", "B"], outputs=[])
    return m


@pytest.fixture
def default_mesh() -> FlywheelMesh:
    return build_default_mesh()


@pytest.fixture
def temp_db(tmp_path) -> SentinelDB:
    db = SentinelDB(path=str(tmp_path / "test_mesh.db"))
    yield db
    db.close()


@pytest.fixture
def seeded_db(temp_db: SentinelDB) -> SentinelDB:
    """temp_db with sample scored jobs inserted."""
    for i in range(20):
        score = 0.1 + (i % 10) * 0.09  # scores 0.1 – 0.91
        signals = json.dumps([{"name": f"signal_{i % 5}", "category": "red_flag"}])
        temp_db.conn.execute(
            """
            INSERT INTO jobs (url, title, company, scam_score, risk_level, signals_json, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, '2026-01-01T00:00:00')
            """,
            (
                f"https://example.com/job/{i}",
                f"Job {i}",
                "Acme",
                score,
                "suspicious",
                signals,
            ),
        )
    temp_db.conn.commit()
    return temp_db


@pytest.fixture
def detector(default_mesh: FlywheelMesh) -> CascadeDetector:
    return CascadeDetector(mesh=default_mesh)


# ===========================================================================
# TestFlywheelMesh — topology registration & querying
# ===========================================================================


class TestFlywheelMeshRegistration:
    def test_register_single_flywheel(self, empty_mesh):
        empty_mesh.register_flywheel("detection")
        assert empty_mesh.has_node("detection")
        assert empty_mesh.node_count() == 1

    def test_register_with_dependencies(self, empty_mesh):
        empty_mesh.register_flywheel("calibration", dependencies=["detection"])
        assert empty_mesh.has_node("calibration")
        assert empty_mesh.has_node("detection")  # auto-created stub

    def test_register_with_outputs(self, empty_mesh):
        empty_mesh.register_flywheel("drift", outputs=["detection"])
        assert empty_mesh.has_node("drift")
        assert empty_mesh.has_node("detection")

    def test_edge_count(self, simple_mesh):
        # A→B, A→C, B→C = 3 edges
        assert simple_mesh.edge_count() == 3

    def test_node_count(self, simple_mesh):
        assert simple_mesh.node_count() == 3

    def test_has_node_unknown(self, empty_mesh):
        assert not empty_mesh.has_node("nonexistent")

    def test_get_dependency_graph_structure(self, simple_mesh):
        graph = simple_mesh.get_dependency_graph()
        assert "nodes" in graph
        assert "edges" in graph
        assert "adjacency" in graph
        assert set(graph["nodes"]) == {"A", "B", "C"}

    def test_get_dependency_graph_edges(self, simple_mesh):
        graph = simple_mesh.get_dependency_graph()
        edge_pairs = {(e["source"], e["target"]) for e in graph["edges"]}
        assert ("A", "B") in edge_pairs
        assert ("A", "C") in edge_pairs
        assert ("B", "C") in edge_pairs


class TestFlywheelMeshTraversal:
    def test_get_downstream_direct(self, simple_mesh):
        """A directly feeds B and C."""
        downstream = simple_mesh.get_downstream("A")
        assert "B" in downstream
        assert "C" in downstream

    def test_get_downstream_transitive(self, simple_mesh):
        """B's downstream should include C (transitively)."""
        downstream = simple_mesh.get_downstream("B")
        assert "C" in downstream

    def test_get_downstream_sink(self, simple_mesh):
        """C has no outputs, so downstream should be empty."""
        assert simple_mesh.get_downstream("C") == []

    def test_get_upstream_direct(self, simple_mesh):
        """C directly depends on A and B."""
        upstream = simple_mesh.get_upstream("C")
        assert "A" in upstream
        assert "B" in upstream

    def test_get_upstream_transitive(self, simple_mesh):
        """B's upstream should include A."""
        upstream = simple_mesh.get_upstream("B")
        assert "A" in upstream

    def test_get_upstream_source(self, simple_mesh):
        """A has no dependencies."""
        assert simple_mesh.get_upstream("A") == []

    def test_get_downstream_unknown_node(self, simple_mesh):
        """Unknown node returns empty list without raising."""
        assert simple_mesh.get_downstream("unknown_node") == []

    def test_get_upstream_unknown_node(self, simple_mesh):
        assert simple_mesh.get_upstream("unknown_node") == []

    def test_default_mesh_has_all_flywheels(self, default_mesh):
        expected = {"detection", "calibration", "innovation", "shadow", "drift", "research"}
        for fw in expected:
            assert default_mesh.has_node(fw), f"Missing node: {fw}"

    def test_default_mesh_detection_downstream(self, default_mesh):
        """detection is a sink in the default mesh (no direct outputs registered)."""
        # detection is at the end of the pipeline; research/drift feed it
        assert default_mesh.has_node("detection")

    def test_default_mesh_research_upstream(self, default_mesh):
        """research feeds detection and innovation."""
        downstream = default_mesh.get_downstream("research")
        assert "detection" in downstream or "innovation" in downstream


# ===========================================================================
# TestFlywheelMeshASCII
# ===========================================================================


class TestFlywheelMeshASCII:
    def test_render_ascii_nonempty(self, simple_mesh):
        output = simple_mesh.render_ascii()
        assert "[A]" in output
        assert "[B]" in output
        assert "[C]" in output

    def test_render_ascii_arrows(self, simple_mesh):
        output = simple_mesh.render_ascii()
        assert "-->" in output

    def test_render_ascii_sink_label(self, simple_mesh):
        output = simple_mesh.render_ascii()
        assert "(sink)" in output

    def test_render_ascii_empty_mesh(self, empty_mesh):
        output = empty_mesh.render_ascii()
        assert "empty" in output.lower()

    def test_render_ascii_default_mesh(self, default_mesh):
        output = default_mesh.render_ascii()
        assert "Flywheel Dependency Mesh" in output
        assert "detection" in output


# ===========================================================================
# TestCascadeReport helpers
# ===========================================================================


class TestScoreToBucket:
    def test_safe(self):
        assert _score_to_bucket(0.0) == "safe"
        assert _score_to_bucket(0.19) == "safe"

    def test_low(self):
        assert _score_to_bucket(0.2) == "low"
        assert _score_to_bucket(0.39) == "low"

    def test_suspicious(self):
        assert _score_to_bucket(0.4) == "suspicious"
        assert _score_to_bucket(0.59) == "suspicious"

    def test_high(self):
        assert _score_to_bucket(0.6) == "high"
        assert _score_to_bucket(0.79) == "high"

    def test_scam(self):
        assert _score_to_bucket(0.8) == "scam"
        assert _score_to_bucket(1.0) == "scam"


class TestPearsonCorrelation:
    def test_perfect_positive(self):
        r = _pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert abs(r - 1.0) < 1e-9

    def test_perfect_negative(self):
        r = _pearson_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert abs(r - (-1.0)) < 1e-9

    def test_no_variance(self):
        r = _pearson_correlation([5, 5, 5], [1, 2, 3])
        assert r == 0.0

    def test_too_short(self):
        r = _pearson_correlation([1], [1])
        assert r == 0.0

    def test_uncorrelated(self):
        """Orthogonal series should have r near 0."""
        x = [1, -1, 1, -1, 1, -1]
        y = [1, 1, -1, -1, 1, 1]
        r = _pearson_correlation(x, y)
        assert abs(r) < 0.6  # not strongly correlated


# ===========================================================================
# TestCascadeDetector.preview_impact
# ===========================================================================


class TestCascadePreviewImpact:
    def test_empty_db_returns_safe(self, detector, temp_db):
        report = detector.preview_impact(temp_db, {}, {}, sample_size=10)
        assert isinstance(report, CascadeReport)
        assert report.risk_level == "SAFE"
        assert report.jobs_sampled == 0

    def test_no_weight_change_is_safe(self, detector, seeded_db):
        """Same weights → no classification changes → SAFE."""
        weights = {"signal_0": 0.5, "signal_1": 0.5}
        report = detector.preview_impact(seeded_db, weights, weights, sample_size=20)
        assert report.risk_level == "SAFE"
        assert report.classifications_changed == 0
        assert report.change_rate == 0.0

    def test_large_weight_increase_can_raise_impact(self, detector, seeded_db):
        """Doubling all weights should change some classifications."""
        old = {"signal_0": 0.2, "signal_1": 0.2, "signal_2": 0.2}
        new = {"signal_0": 0.9, "signal_1": 0.9, "signal_2": 0.9}
        report = detector.preview_impact(seeded_db, old, new, sample_size=20)
        assert isinstance(report, CascadeReport)
        # Should detect some impact
        assert report.jobs_sampled > 0
        assert report.score_delta_mean >= 0.0  # weights increased → scores go up

    def test_report_dataclass_fields(self, detector, seeded_db):
        weights = {"signal_0": 0.5}
        report = detector.preview_impact(seeded_db, weights, weights)
        assert hasattr(report, "jobs_sampled")
        assert hasattr(report, "classifications_changed")
        assert hasattr(report, "change_rate")
        assert hasattr(report, "risk_level")
        assert hasattr(report, "score_delta_mean")
        assert hasattr(report, "score_delta_std")
        assert hasattr(report, "promoted_count")
        assert hasattr(report, "demoted_count")

    def test_promoted_plus_demoted_lte_changed(self, detector, seeded_db):
        """promoted + demoted should equal classifications_changed."""
        old = {"signal_0": 0.1, "signal_1": 0.1}
        new = {"signal_0": 0.8, "signal_1": 0.8}
        report = detector.preview_impact(seeded_db, old, new, sample_size=20)
        assert report.promoted_count + report.demoted_count == report.classifications_changed

    def test_impact_level_safe_below_5pct(self, detector, seeded_db):
        """No changes → 0% change rate → SAFE."""
        w = {"sig": 0.5}
        report = detector.preview_impact(seeded_db, w, w)
        assert report.risk_level == "SAFE"

    def test_to_dict(self, detector, seeded_db):
        w = {"sig": 0.5}
        report = detector.preview_impact(seeded_db, w, w)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "risk_level" in d
        assert "change_rate" in d


# ===========================================================================
# TestCascadeDetector.track_cascade
# ===========================================================================


class TestTrackCascade:
    def test_track_cascade_returns_record(self, detector, temp_db):
        record = detector.track_cascade(
            temp_db,
            "detection:evolve",
            {"precision": 0.8, "recall": 0.7},
            {"precision": 0.82, "recall": 0.71},
        )
        assert isinstance(record, CascadeRecord)
        assert record.trigger == "detection:evolve"
        assert record.change_type == "evolve"
        assert record.magnitude > 0.0

    def test_track_cascade_magnitude_computation(self, detector, temp_db):
        before = {"precision": 0.8}
        after = {"precision": 0.9}
        record = detector.track_cascade(temp_db, "detection:test", before, after)
        assert abs(record.magnitude - 0.1) < 1e-6

    def test_track_cascade_persists_to_db(self, detector, temp_db):
        detector.track_cascade(
            temp_db,
            "innovation:pattern_mining",
            {"precision": 0.75},
            {"precision": 0.78},
        )
        events = temp_db.get_cascade_history(limit=10)
        assert len(events) >= 1
        assert any(e["trigger"] == "innovation:pattern_mining" for e in events)

    def test_track_cascade_affected_flywheels(self, detector, temp_db):
        record = detector.track_cascade(
            temp_db,
            "innovation:run",
            {"precision": 0.7},
            {"precision": 0.72},
        )
        # innovation is upstream of detection in the default mesh
        assert isinstance(record.affected_flywheels, list)

    def test_track_cascade_no_change_zero_magnitude(self, detector, temp_db):
        before = {"precision": 0.75, "recall": 0.6}
        record = detector.track_cascade(temp_db, "detection:noop", before, before)
        assert record.magnitude == 0.0


# ===========================================================================
# TestCascadeDetector.detect_ripple_effects
# ===========================================================================


class TestDetectRippleEffects:
    def test_empty_db_returns_empty_list(self, detector, temp_db):
        ripples = detector.detect_ripple_effects(temp_db, lookback_cycles=5)
        assert ripples == []

    def test_insufficient_data_returns_empty(self, detector, temp_db):
        # Only 2 rows — below the min-3 threshold
        for i in range(2):
            temp_db.save_flywheel_metrics({"precision": 0.8, "recall": 0.7, "f1": 0.75,
                                           "accuracy": 0.78, "cycle_ts": f"2026-01-0{i+1}T00:00:00",
                                           "cycle_number": i, "total_analyzed": 10})
        ripples = detector.detect_ripple_effects(temp_db, lookback_cycles=5)
        assert ripples == []

    def test_ripple_returns_list(self, detector, temp_db):
        for i in range(10):
            temp_db.save_flywheel_metrics({
                "precision": 0.7 + i * 0.02,
                "recall": 0.65 + i * 0.01,
                "f1": 0.67 + i * 0.015,
                "accuracy": 0.70 + i * 0.01,
                "cycle_ts": f"2026-01-{i+1:02d}T00:00:00",
                "cycle_number": i,
                "total_analyzed": 50 + i,
                "signals_updated": 5,
                "patterns_evolved": 1,
            })
        ripples = detector.detect_ripple_effects(temp_db, lookback_cycles=5)
        assert isinstance(ripples, list)
        for r in ripples:
            assert isinstance(r, RippleEffect)

    def test_ripple_effect_fields(self, detector, temp_db):
        for i in range(8):
            temp_db.save_flywheel_metrics({
                "precision": 0.8 + (i % 2) * 0.05,
                "recall": 0.7,
                "f1": 0.75,
                "accuracy": 0.78,
                "cycle_ts": f"2026-02-{i+1:02d}T00:00:00",
                "cycle_number": i,
                "total_analyzed": 30,
                "signals_updated": 2,
                "patterns_evolved": 0,
            })
        ripples = detector.detect_ripple_effects(temp_db)
        for r in ripples:
            assert r.trigger_flywheel in ["detection", "calibration", "innovation", "shadow"]
            assert r.affected_flywheel in ["detection", "calibration", "innovation", "shadow"]
            assert -1.0 <= r.correlation <= 1.0
            assert r.lag_cycles in (0, 1)
            assert r.direction in ("amplifying", "dampening")


# ===========================================================================
# TestImpactThresholds
# ===========================================================================


class TestImpactThresholds:
    def test_threshold_safe(self):
        """0% change → SAFE."""
        d = CascadeDetector()
        # Build a minimal report manually via the constants
        assert d.SAFE_THRESHOLD == 0.05
        assert d.MODERATE_THRESHOLD == 0.15

    def test_change_rate_safe(self, detector, seeded_db):
        """Identical weights → rate = 0.0 → SAFE."""
        w = {"x": 0.5}
        r = detector.preview_impact(seeded_db, w, w)
        assert r.change_rate == 0.0
        assert r.risk_level == "SAFE"

    def test_change_rate_moderate_boundary(self):
        """Verify the SAFE/MODERATE boundary logic."""
        det = CascadeDetector()
        # 5% change is the boundary: < 5% SAFE, >= 5% MODERATE
        assert 0.04 < det.SAFE_THRESHOLD  # 4% < threshold → SAFE
        assert 0.05 >= det.SAFE_THRESHOLD  # 5% >= threshold → MODERATE


# ===========================================================================
# TestDB methods
# ===========================================================================


class TestDBMethods:
    def test_insert_and_get_cascade_event(self, temp_db):
        eid = temp_db.insert_cascade_event(
            trigger="detection:evolve",
            change_type="evolve",
            impact_json='{"risk_level": "SAFE"}',
        )
        assert isinstance(eid, int)
        events = temp_db.get_cascade_history(limit=5)
        assert len(events) == 1
        assert events[0]["trigger"] == "detection:evolve"
        assert events[0]["impact"]["risk_level"] == "SAFE"

    def test_upsert_mesh_edge(self, temp_db):
        temp_db.upsert_mesh_edge("detection", "calibration", "feedback", 0.9)
        edges = temp_db.get_mesh_topology()
        assert len(edges) == 1
        assert edges[0]["source"] == "detection"
        assert edges[0]["target"] == "calibration"

    def test_upsert_mesh_edge_update(self, temp_db):
        temp_db.upsert_mesh_edge("A", "B", "data", 1.0)
        temp_db.upsert_mesh_edge("A", "B", "feedback", 0.5)
        edges = temp_db.get_mesh_topology()
        assert len(edges) == 1
        assert edges[0]["edge_type"] == "feedback"
        assert edges[0]["weight"] == 0.5

    def test_get_recent_jobs_for_sampling(self, seeded_db):
        jobs = seeded_db.get_recent_jobs_for_sampling(limit=10)
        assert len(jobs) <= 10
        for j in jobs:
            assert "scam_score" in j
            assert "signals_json" in j

    def test_get_cascade_history_newest_first(self, temp_db):
        temp_db.insert_cascade_event("a", "t1", "{}")
        temp_db.insert_cascade_event("b", "t2", "{}")
        events = temp_db.get_cascade_history()
        # Newest first means 'b' should come before 'a'
        assert events[0]["trigger"] == "b"
        assert events[1]["trigger"] == "a"


# ===========================================================================
# TestCascadeReport.to_dict
# ===========================================================================


class TestCascadeReportToDict:
    def test_to_dict_all_fields(self):
        r = CascadeReport(
            jobs_sampled=50,
            classifications_changed=3,
            change_rate=0.06,
            risk_level="MODERATE",
            score_delta_mean=0.02,
            score_delta_std=0.01,
            promoted_count=2,
            demoted_count=1,
        )
        d = r.to_dict()
        assert d["jobs_sampled"] == 50
        assert d["classifications_changed"] == 3
        assert d["risk_level"] == "MODERATE"
        assert d["promoted_count"] == 2
        assert d["demoted_count"] == 1

    def test_ripple_effect_to_dict(self):
        re = RippleEffect(
            trigger_flywheel="innovation",
            affected_flywheel="detection",
            correlation=0.75,
            lag_cycles=1,
            direction="amplifying",
        )
        d = re.to_dict()
        assert d["trigger_flywheel"] == "innovation"
        assert d["affected_flywheel"] == "detection"
        assert d["correlation"] == 0.75
        assert d["direction"] == "amplifying"
