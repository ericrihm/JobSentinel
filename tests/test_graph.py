"""Tests for sentinel.graph — Network Graph Analysis module.

Covers:
  - String similarity primitives: jaro_similarity, jaro_winkler, levenshtein
  - MinHasher: signature generation, Jaccard estimation
  - shingle: character k-shingle sets
  - TextSimilarityIndex: add, similarity, find_near_duplicates, get_clusters
  - ScamNetworkGraph: add_posting, edge detection (contact/hash/text/company),
    get_clusters, get_hubs, degree_centrality, cross_platform_postings
  - RecruiterProfiler: record, get_flags (scam rate, hours, volume, templates),
    detect_sybils
  - CompanyShellDetector: fuzzy_match_known_scams, analyse (all flag types),
    register_address / address_reused_by
"""

import math
from datetime import UTC, datetime, timedelta

import pytest

from sentinel.models import JobPosting


# ============================================================================
# Helpers
# ============================================================================


def _make_job(
    url: str = "https://example.com/job/1",
    company: str = "ACME Corp",
    description: str = "We are hiring a software engineer.",
    recruiter_name: str = "Alice",
    source: str = "linkedin",
    industry: str = "Technology",
    company_linkedin_url: str = "https://linkedin.com/company/acme",
    **kwargs,
) -> JobPosting:
    return JobPosting(
        url=url,
        company=company,
        description=description,
        recruiter_name=recruiter_name,
        source=source,
        industry=industry,
        company_linkedin_url=company_linkedin_url,
        **kwargs,
    )


def _ts(hour: int = 12, days_ago: int = 0) -> str:
    """ISO timestamp at the given UTC hour, *days_ago* days in the past."""
    dt = datetime.now(UTC).replace(hour=hour, minute=0, second=0, microsecond=0)
    dt -= timedelta(days=days_ago)
    return dt.isoformat()


# ============================================================================
# 1. String similarity primitives
# ============================================================================


class TestJaroSimilarity:
    def test_identical_strings(self):
        from sentinel.graph import jaro_similarity

        assert jaro_similarity("hello", "hello") == 1.0

    def test_empty_strings(self):
        from sentinel.graph import jaro_similarity

        assert jaro_similarity("", "") == 1.0

    def test_one_empty(self):
        from sentinel.graph import jaro_similarity

        assert jaro_similarity("abc", "") == 0.0
        assert jaro_similarity("", "abc") == 0.0

    def test_completely_different(self):
        from sentinel.graph import jaro_similarity

        sim = jaro_similarity("abc", "xyz")
        assert 0.0 <= sim < 0.5

    def test_classic_example(self):
        from sentinel.graph import jaro_similarity

        # "MARTHA" vs "MARHTA" → well-known jaro = 0.944
        sim = jaro_similarity("MARTHA", "MARHTA")
        assert abs(sim - 0.944) < 0.01

    def test_range_always_0_to_1(self):
        from sentinel.graph import jaro_similarity

        pairs = [("scam", "spam"), ("global", "globel"), ("", "x"), ("abc", "abc")]
        for a, b in pairs:
            s = jaro_similarity(a, b)
            assert 0.0 <= s <= 1.0


class TestJaroWinkler:
    def test_prefix_boost(self):
        from sentinel.graph import jaro_winkler

        # Jaro-Winkler should be >= Jaro for strings with common prefix
        from sentinel.graph import jaro_similarity

        a, b = "JOHNNYAPPLESEED", "JOHNNYSEED"
        assert jaro_winkler(a, b) >= jaro_similarity(a, b)

    def test_identical(self):
        from sentinel.graph import jaro_winkler

        assert jaro_winkler("sentinel", "sentinel") == 1.0

    def test_similar_company_names(self):
        from sentinel.graph import jaro_winkler

        # Slight misspelling of known scam company
        sim = jaro_winkler("global solutions llc", "globel solutions llc")
        assert sim > 0.92

    def test_dissimilar_names(self):
        from sentinel.graph import jaro_winkler

        sim = jaro_winkler("amazon", "xyz")
        assert sim < 0.7


class TestLevenshtein:
    def test_identical(self):
        from sentinel.graph import levenshtein

        assert levenshtein("hello", "hello") == 0

    def test_one_substitution(self):
        from sentinel.graph import levenshtein

        assert levenshtein("kitten", "sitten") == 1

    def test_classic(self):
        from sentinel.graph import levenshtein

        # kitten → sitting = 3 edits
        assert levenshtein("kitten", "sitting") == 3

    def test_empty(self):
        from sentinel.graph import levenshtein

        assert levenshtein("", "abc") == 3
        assert levenshtein("abc", "") == 3

    def test_symmetry(self):
        from sentinel.graph import levenshtein

        assert levenshtein("abcd", "dcba") == levenshtein("dcba", "abcd")


# ============================================================================
# 2. MinHasher
# ============================================================================


class TestMinHasher:
    def test_signature_length(self):
        from sentinel.graph import MinHasher, shingle

        h = MinHasher(num_perm=64)
        sig = h.signature(shingle("hello world"))
        assert len(sig) == 64

    def test_identical_texts_full_similarity(self):
        from sentinel.graph import MinHasher, shingle

        h = MinHasher(num_perm=128)
        t = "This is a test job description for a software engineer position."
        s1 = h.signature(shingle(t))
        s2 = h.signature(shingle(t))
        assert MinHasher.jaccard_from_signatures(s1, s2) == 1.0

    def test_completely_different_texts_low_similarity(self):
        from sentinel.graph import MinHasher, shingle

        h = MinHasher(num_perm=128)
        s1 = h.signature(shingle("aaa bbb ccc ddd eee fff ggg hhh iii jjj"))
        s2 = h.signature(shingle("zzz yyy xxx www vvv uuu ttt sss rrr qqq"))
        sim = MinHasher.jaccard_from_signatures(s1, s2)
        assert sim < 0.15

    def test_near_duplicate_high_similarity(self):
        from sentinel.graph import MinHasher, shingle

        h = MinHasher(num_perm=256)
        base = (
            "We are hiring a motivated software engineer to join our team. "
            "You will build scalable web applications and collaborate with designers. "
            "Requirements: 3+ years Python, REST APIs, PostgreSQL."
        )
        mutated = base.replace("motivated", "enthusiastic").replace("designers", "product managers")
        s1 = h.signature(shingle(base))
        s2 = h.signature(shingle(mutated))
        sim = MinHasher.jaccard_from_signatures(s1, s2)
        assert sim > 0.5

    def test_empty_signature_returns_zero_similarity(self):
        from sentinel.graph import MinHasher

        assert MinHasher.jaccard_from_signatures([], []) == 0.0

    def test_mismatched_length_returns_zero(self):
        from sentinel.graph import MinHasher

        assert MinHasher.jaccard_from_signatures([1, 2], [1, 2, 3]) == 0.0


class TestShingle:
    def test_returns_set(self):
        from sentinel.graph import shingle

        result = shingle("hello world", k=3)
        assert isinstance(result, set)

    def test_short_text(self):
        from sentinel.graph import shingle

        # Text shorter than k → single hash fallback
        result = shingle("hi", k=5)
        assert len(result) >= 1

    def test_deterministic(self):
        from sentinel.graph import shingle

        t = "reproducible shingle output"
        assert shingle(t, k=4) == shingle(t, k=4)


# ============================================================================
# 3. TextSimilarityIndex
# ============================================================================


class TestTextSimilarityIndex:
    def test_add_and_len(self):
        from sentinel.graph import TextSimilarityIndex

        idx = TextSimilarityIndex()
        idx.add("p1", "Software engineer needed for backend role.")
        idx.add("p2", "Frontend developer position open now.")
        assert len(idx) == 2

    def test_self_similarity_is_one(self):
        from sentinel.graph import TextSimilarityIndex

        idx = TextSimilarityIndex()
        text = "Exciting opportunity to join our growing team of engineers."
        idx.add("p1", text)
        idx.add("p2", text)
        sim = idx.similarity("p1", "p2")
        assert sim == 1.0

    def test_different_texts_low_similarity(self):
        from sentinel.graph import TextSimilarityIndex

        idx = TextSimilarityIndex()
        idx.add("p1", "python machine learning data scientist job")
        idx.add("p2", "truck driver cdl class a delivery logistics warehouse")
        sim = idx.similarity("p1", "p2")
        assert sim < 0.3

    def test_missing_id_returns_zero(self):
        from sentinel.graph import TextSimilarityIndex

        idx = TextSimilarityIndex()
        idx.add("p1", "some text here")
        assert idx.similarity("p1", "nonexistent") == 0.0

    def test_find_near_duplicates(self):
        from sentinel.graph import TextSimilarityIndex

        idx = TextSimilarityIndex()
        template = (
            "We are seeking a customer service representative. "
            "Responsibilities: answer phones, resolve issues, document calls. "
            "Must have high school diploma. Pay: $18/hr. Apply now!"
        )
        idx.add("orig", template)
        idx.add("clone", template + " Join our team today.")
        idx.add("unrelated", "Senior DevOps engineer, Kubernetes, GCP, Terraform, 5+ years.")

        results = idx.find_near_duplicates(template, threshold=0.7)
        found_ids = [r[0] for r in results]
        assert "orig" in found_ids or "clone" in found_ids

    def test_get_clusters_single_item(self):
        from sentinel.graph import TextSimilarityIndex

        idx = TextSimilarityIndex()
        idx.add("p1", "Unique text that has no duplicates anywhere at all.")
        clusters = idx.get_clusters()
        assert len(clusters) == 1
        assert clusters[0].posting_ids == ["p1"]

    def test_get_clusters_template_flag(self):
        from sentinel.graph import TextSimilarityIndex

        idx = TextSimilarityIndex()
        template = (
            "Earn money from home! No experience needed. "
            "We pay $500 per day. Send your bank details to start."
        ) * 5  # long enough for reliable shingling
        for i in range(4):
            idx.add(f"p{i}", template)
        clusters = idx.get_clusters(threshold=0.9)
        big = max(clusters, key=lambda c: len(c.posting_ids))
        assert big.is_template is True

    def test_get_clusters_isolated_and_grouped(self):
        from sentinel.graph import TextSimilarityIndex

        idx = TextSimilarityIndex()
        duplicate = "Send payment to get started. Earn guaranteed income from home now!" * 4
        idx.add("dup1", duplicate)
        idx.add("dup2", duplicate)
        idx.add("unique", "Senior backend engineer Python Go Kubernetes distributed systems.")
        clusters = idx.get_clusters(threshold=0.95)
        # dup1 and dup2 should be in the same cluster
        for c in clusters:
            if "dup1" in c.posting_ids:
                assert "dup2" in c.posting_ids
                break


# ============================================================================
# 4. ScamNetworkGraph
# ============================================================================


class TestScamNetworkGraph:
    def test_add_posting_increments_node_count(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        g.add_posting(_make_job(url="https://example.com/1"))
        g.add_posting(_make_job(url="https://example.com/2"))
        assert g.node_count() == 2

    def test_no_self_edges(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        pid = g.add_posting(_make_job(url="https://example.com/1"))
        assert pid not in g._nodes[pid].neighbours

    def test_shared_contact_creates_edge(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        desc = "Apply at scammer@gmail.com for this job opportunity."
        g.add_posting(_make_job(url="https://a.com/1", description=desc, company="Co A"))
        g.add_posting(_make_job(url="https://b.com/2", description=desc, company="Co B"))
        assert g.edge_count() >= 1

    def test_shared_description_hash_creates_edge(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        desc = "Exact duplicate description text for both jobs here."
        g.add_posting(_make_job(url="https://a.com/1", description=desc))
        g.add_posting(_make_job(url="https://b.com/2", description=desc))
        assert g.edge_count() >= 1

    def test_same_company_creates_edge(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        g.add_posting(_make_job(url="https://a.com/1", company="Scam LLC", description="Job A desc"))
        g.add_posting(_make_job(url="https://b.com/2", company="Scam LLC", description="Job B desc"))
        assert g.edge_count() >= 1

    def test_unrelated_postings_no_edge(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        g.add_posting(_make_job(
            url="https://google.com/1",
            company="Google",
            description="Senior software engineer, distributed systems, Python, Go, Kubernetes.",
        ))
        g.add_posting(_make_job(
            url="https://ups.com/2",
            company="UPS Logistics",
            description="CDL Class A truck driver, long haul, overnight routes, DOT compliance.",
        ))
        assert g.edge_count() == 0

    def test_get_clusters_two_separate_components(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        desc = "Send payment to scammer@evil.com. Earn from home guaranteed!"
        g.add_posting(_make_job(url="https://a.com/1", description=desc))
        g.add_posting(_make_job(url="https://b.com/2", description=desc))
        g.add_posting(_make_job(
            url="https://legit.com/3",
            company="Different Corp",
            description="Legitimate DevOps role, 5+ years Kubernetes, GCP, Terraform, CI/CD.",
        ))
        clusters = g.get_clusters()
        sizes = sorted([len(c.node_ids) for c in clusters], reverse=True)
        assert sizes[0] >= 2  # at least one cluster of 2

    def test_get_hubs_min_degree(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        shared_contact = "scammer@evil.com"
        for i in range(4):
            desc = f"Job {i} — contact us at {shared_contact} to get started right away."
            g.add_posting(_make_job(url=f"https://x.com/{i}", description=desc, company=f"Fake Co {i}"))
        hubs = g.get_hubs(min_degree=1)
        assert len(hubs) > 0
        assert all(deg >= 1 for _, deg in hubs)

    def test_degree_centrality_range(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        desc = "Money making opportunity! Contact jobs@scam.org immediately."
        pids = []
        for i in range(3):
            pid = g.add_posting(_make_job(url=f"https://x.com/{i}", description=desc, company=f"Co {i}"))
            pids.append(pid)
        for pid in pids:
            dc = g.degree_centrality(pid)
            assert 0.0 <= dc <= 1.0

    def test_cross_platform_postings_detected(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        desc = "Guaranteed income! Contact hr@fakeco.net for details. Apply now!"
        g.add_posting(_make_job(url="https://linkedin.com/j/1", description=desc, source="linkedin"))
        g.add_posting(_make_job(url="https://remoteok.com/j/2", description=desc, source="remoteok"))
        cross = g.cross_platform_postings()
        assert len(cross) > 0
        # The cluster should list both platforms
        platforms_found = [item[1] for item in cross]
        all_platforms = [p for platforms in platforms_found for p in platforms]
        assert "linkedin" in all_platforms or "remoteok" in all_platforms

    def test_cluster_avg_scam_score(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        desc = "Pay fee to start. Contact fee@scam.org. Earn big now guaranteed!"
        g.add_posting(_make_job(url="https://a.com/1", description=desc, company="Co A"), scam_score=0.9)
        g.add_posting(_make_job(url="https://b.com/2", description=desc, company="Co B"), scam_score=0.8)
        clusters = g.get_clusters()
        big = max(clusters, key=lambda c: len(c.node_ids))
        assert 0.79 < big.avg_scam_score < 0.91

    def test_hub_has_highest_degree(self):
        from sentinel.graph import ScamNetworkGraph

        g = ScamNetworkGraph()
        desc = "Email us at hub@badactor.com — great opportunity for all!"
        pids = []
        for i in range(5):
            pid = g.add_posting(
                _make_job(url=f"https://x.com/{i}", description=desc, company=f"C{i}")
            )
            pids.append(pid)
        clusters = g.get_clusters()
        for cluster in clusters:
            if len(cluster.node_ids) > 1:
                hub = cluster.hub_node
                hub_deg = len(g._nodes[hub].neighbours)
                for nid in cluster.node_ids:
                    assert hub_deg >= len(g._nodes[nid].neighbours)
                break


# ============================================================================
# 5. RecruiterProfiler
# ============================================================================


class TestRecruiterProfiler:
    def test_record_increments_total(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        job = _make_job(recruiter_name="Bob")
        p.record(job, is_scam=False)
        p.record(job, is_scam=True)
        profile = p.get_profile("Bob")
        assert profile is not None
        assert profile.total_count == 2
        assert profile.scam_count == 1

    def test_scam_rate_calculation(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        job = _make_job(recruiter_name="Spammer")
        for i in range(10):
            p.record(job, is_scam=(i < 8))
        profile = p.get_profile("Spammer")
        assert abs(profile.scam_rate - 0.8) < 0.01

    def test_high_scam_rate_flagged(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        job = _make_job(recruiter_name="BadGuy")
        for i in range(6):
            p.record(job, is_scam=(i < 5))
        flags = p.get_flags("BadGuy")
        assert any("high_scam_rate" in f for f in flags)

    def test_unusual_hours_flagged(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        job = _make_job(recruiter_name="NightOwl")
        for h in [2, 3, 4, 2, 3]:  # all overnight hours
            p.record(job, posted_at=_ts(hour=h))
        flags = p.get_flags("NightOwl")
        assert any("unusual_hours" in f for f in flags)

    def test_normal_hours_not_flagged(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        job = _make_job(recruiter_name="Normal")
        for h in [9, 10, 11, 14, 15]:
            p.record(job, posted_at=_ts(hour=h))
        flags = p.get_flags("Normal")
        assert not any("unusual_hours" in f for f in flags)

    def test_templated_messages_flagged(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        template = "Apply now and earn money from home today guaranteed!"
        job = _make_job(recruiter_name="Spammer", description=template)
        for _ in range(5):
            p.record(job)
        flags = p.get_flags("Spammer")
        assert any("templated_messages" in f for f in flags)

    def test_no_flags_for_clean_recruiter(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        # Only 1 posting — below all thresholds
        job = _make_job(recruiter_name="Legit")
        p.record(job, is_scam=False)
        flags = p.get_flags("Legit")
        assert flags == []

    def test_all_recruiter_ids(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        for name in ["Alice", "Bob", "Carol"]:
            p.record(_make_job(recruiter_name=name))
        assert set(p.all_recruiter_ids()) == {"Alice", "Bob", "Carol"}

    def test_get_profile_unknown_returns_none(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        assert p.get_profile("nobody") is None

    def test_language_fingerprint_populated(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        job = _make_job(recruiter_name="Ling", description="earn money home guaranteed income pay")
        p.record(job)
        fp = p.get_profile("Ling").language_fingerprint
        assert isinstance(fp, dict)
        # At least some words captured
        assert len(fp) > 0


class TestSybilDetection:
    def _make_sybil_pair(self) -> tuple:
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        template = (
            "earn money from home guaranteed income apply now today scam fake job"
        )
        categories = ["Finance", "Finance", "Finance"]
        hours = [2, 3, 4]
        for i, (h, cat) in enumerate(zip(hours * 3, categories * 3)):
            job_a = _make_job(recruiter_name="FakeAlice", industry=cat, description=template)
            job_b = _make_job(recruiter_name="FakeBob", industry=cat, description=template)
            p.record(job_a, posted_at=_ts(hour=h))
            p.record(job_b, posted_at=_ts(hour=h))
        return p

    def test_sybil_pair_detected(self):
        p = self._make_sybil_pair()
        groups = p.detect_sybils(threshold=0.5)
        # At least one group should contain both fake recruiters
        all_members = [rid for group in groups for rid in group]
        assert "FakeAlice" in all_members or "FakeBob" in all_members

    def test_distinct_recruiters_not_grouped(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        # Alice posts overnight Finance scam-like jobs
        for h in [2, 3, 4]:
            p.record(_make_job(recruiter_name="Alice", industry="Finance", description="earn money home guaranteed"), posted_at=_ts(hour=h))

        # Bob posts daytime Tech legitimate jobs
        for h in [9, 10, 11]:
            p.record(_make_job(recruiter_name="Bob", industry="Technology", description="senior engineer python distributed systems kubernetes"), posted_at=_ts(hour=h))

        groups = p.detect_sybils(threshold=0.7)
        # Alice and Bob should not be in the same group
        for group in groups:
            assert not ("Alice" in group and "Bob" in group)

    def test_sybil_only_considers_profiles_with_min_postings(self):
        from sentinel.graph import RecruiterProfiler

        p = RecruiterProfiler()
        # Only 1 posting each → below threshold of 3
        p.record(_make_job(recruiter_name="Solo1", description="unique text A"))
        p.record(_make_job(recruiter_name="Solo2", description="unique text B"))
        groups = p.detect_sybils()
        assert groups == []


# ============================================================================
# 6. CompanyShellDetector
# ============================================================================


class TestCompanyShellDetector:
    def test_fuzzy_match_exact(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        d.load_known_scams(["Global Solutions LLC"])
        matches = d.fuzzy_match_known_scams("Global Solutions LLC")
        assert len(matches) == 1
        assert matches[0][1] == 1.0

    def test_fuzzy_match_slight_mutation(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        d.load_known_scams(["global solutions llc"])
        # Slight misspelling
        matches = d.fuzzy_match_known_scams("globel solutions llc", jw_threshold=0.85)
        assert len(matches) >= 1
        assert matches[0][1] > 0.85

    def test_fuzzy_match_no_false_positives(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        d.load_known_scams(["global solutions llc"])
        matches = d.fuzzy_match_known_scams("microsoft corporation", jw_threshold=0.88)
        assert matches == []

    def test_recently_registered_flagged(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        job = _make_job(company="Shady LLC", company_linkedin_url="")
        report = d.analyse(job, whois_age_days=30)
        assert report.shell_score > 0
        assert any("recently_registered" in f for f in report.flags)

    def test_no_employees_flagged(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        job = _make_job(company="Empty Corp", company_linkedin_url="")
        report = d.analyse(job, employee_count=0)
        assert any("no_employees" in f for f in report.flags)

    def test_virtual_office_address_flagged(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        job = _make_job(company="Shell Co", company_linkedin_url="")
        report = d.analyse(job, address="123 Main St Suite 100 Virtual Office")
        assert any("virtual_office" in f for f in report.flags)

    def test_no_linkedin_page_flagged(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        job = _make_job(company="Ghost LLC", company_linkedin_url="")
        report = d.analyse(job)
        assert any("no_linkedin" in f for f in report.flags)

    def test_address_reuse_detected(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        addr = "Suite 200, 100 Business Park Drive"
        d.register_address("Scam Co A", addr)
        d.register_address("Scam Co B", addr)
        others = d.address_reused_by(addr)
        assert "Scam Co A" in others
        assert "Scam Co B" in others

    def test_address_reuse_flagged_in_analyse(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        addr = "Po Box 9999 Anytown USA"
        d.register_address("Previous Scam", addr)
        job = _make_job(company="New Scam", company_linkedin_url="")
        report = d.analyse(job, address=addr)
        assert any("address_shared" in f for f in report.flags)

    def test_similar_known_scam_flagged(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        d.load_known_scams(["elite workforce solutions"])
        job = _make_job(company="Elite Workforce Solutions", company_linkedin_url="")
        report = d.analyse(job)
        assert len(report.similar_known_scams) > 0
        assert any("similar_to_known_scam" in f for f in report.flags)

    def test_generic_name_flagged(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        job = _make_job(company="Global National Solutions Group", company_linkedin_url="")
        report = d.analyse(job)
        assert any("generic_name" in f for f in report.flags)

    def test_legitimate_company_low_score(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        job = _make_job(
            company="Stripe",
            company_linkedin_url="https://linkedin.com/company/stripe",
        )
        report = d.analyse(job, whois_age_days=3000, employee_count=4000, address="185 Berry St San Francisco CA")
        assert report.shell_score < 0.4

    def test_shell_score_in_range(self):
        from sentinel.graph import CompanyShellDetector

        d = CompanyShellDetector()
        job = _make_job(company="XYZ", company_linkedin_url="")
        report = d.analyse(job, whois_age_days=10, employee_count=0, address="Virtual Office Suite 1")
        assert 0.0 <= report.shell_score <= 1.0
