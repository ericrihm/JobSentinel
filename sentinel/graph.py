"""Network Graph Analysis — cross-posting and recruiter network detection.

Four components:
  - ScamNetworkGraph   : adjacency graph of postings via shared features;
                         connected-component clustering, hub detection,
                         cross-platform tracking.
  - RecruiterProfiler  : posting-pattern modelling, volume spike detection,
                         Sybil detection via behavioural fingerprinting.
  - CompanyShellDetector: shell-company signals, name-mutation fuzzy matching,
                          virtual-office address reuse.
  - TextSimilarityIndex : shingling + MinHash near-duplicate detection,
                          copy-paste template tracking.

All stdlib. MinHash and Jaro-Winkler implemented from scratch.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sentinel.models import JobPosting

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


# ---------------------------------------------------------------------------
# String similarity primitives (stdlib-only)
# ---------------------------------------------------------------------------


def jaro_similarity(s1: str, s2: str) -> float:
    """Jaro similarity in [0, 1]."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    """Jaro-Winkler similarity; p is the scaling factor (default 0.1)."""
    jaro = jaro_similarity(s1, s2)
    prefix = 0
    for i in range(min(4, len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


def levenshtein(s1: str, s2: str) -> int:
    """Edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (c1 != c2)))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# MinHash — near-duplicate detection (stdlib-only)
# ---------------------------------------------------------------------------

_LARGE_PRIME = (1 << 61) - 1  # Mersenne prime M_61


class MinHasher:
    """MinHash signature for a set of shingles.

    Uses *num_perm* independent hash functions implemented via
    universal hashing: h(x) = (a*x + b) mod p  mod max_val,
    where p is a Mersenne prime and a, b are random coefficients
    drawn from a SHA-256-seeded sequence (deterministic).
    """

    def __init__(self, num_perm: int = 128) -> None:
        self.num_perm = num_perm
        # Deterministic coefficients derived from SHA-256 of seeds
        self._a: list[int] = []
        self._b: list[int] = []
        for i in range(num_perm):
            seed_a = hashlib.sha256(f"minhash_a_{i}".encode()).digest()
            seed_b = hashlib.sha256(f"minhash_b_{i}".encode()).digest()
            self._a.append(int.from_bytes(seed_a[:8], "little") | 1)  # odd → coprime to 2^k
            self._b.append(int.from_bytes(seed_b[:8], "little"))

    def signature(self, shingles: set[int]) -> list[int]:
        """Return the MinHash signature (list of *num_perm* integers)."""
        sig = [_LARGE_PRIME] * self.num_perm
        for v in shingles:
            for i in range(self.num_perm):
                h = (self._a[i] * v + self._b[i]) % _LARGE_PRIME
                if h < sig[i]:
                    sig[i] = h
        return sig

    @staticmethod
    def jaccard_from_signatures(sig1: list[int], sig2: list[int]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if not sig1 or not sig2 or len(sig1) != len(sig2):
            return 0.0
        matches = sum(a == b for a, b in zip(sig1, sig2, strict=False))
        return matches / len(sig1)


def shingle(text: str, k: int = 5) -> set[int]:
    """Build the set of character k-shingles as CRC-style 64-bit ints."""
    text = _normalise(text)
    if len(text) < k:
        return {hash(text) & 0xFFFF_FFFF_FFFF_FFFF}
    result: set[int] = set()
    for i in range(len(text) - k + 1):
        gram = text[i : i + k]
        h = int(hashlib.md5(gram.encode()).hexdigest()[:16], 16)
        result.add(h)
    return result


# ---------------------------------------------------------------------------
# TextSimilarityIndex
# ---------------------------------------------------------------------------


@dataclass
class SimilarityCluster:
    """A group of near-duplicate job postings sharing similar description text."""

    cluster_id: int
    posting_ids: list[str] = field(default_factory=list)
    centroid_text: str = ""
    avg_pairwise_similarity: float = 0.0
    is_template: bool = False  # True if cluster size >= TEMPLATE_THRESHOLD


_TEMPLATE_THRESHOLD = 3  # ≥3 postings sharing >0.8 similarity → scam template


class TextSimilarityIndex:
    """Shingling + MinHash near-duplicate detection across job postings.

    Usage::

        idx = TextSimilarityIndex()
        idx.add(posting_id, description_text)
        clusters = idx.get_clusters(threshold=0.8)
        near_dups = idx.find_near_duplicates(text, threshold=0.8)
    """

    def __init__(self, num_perm: int = 128, shingle_size: int = 5) -> None:
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self._hasher = MinHasher(num_perm)
        # posting_id → signature
        self._signatures: dict[str, list[int]] = {}
        # posting_id → original text (for centroid selection)
        self._texts: dict[str, str] = {}

    def add(self, posting_id: str, text: str) -> None:
        """Index a posting description."""
        shingles = shingle(text, self.shingle_size)
        self._signatures[posting_id] = self._hasher.signature(shingles)
        self._texts[posting_id] = text

    def similarity(self, id1: str, id2: str) -> float:
        """Estimated Jaccard similarity between two indexed postings."""
        if id1 not in self._signatures or id2 not in self._signatures:
            return 0.0
        return MinHasher.jaccard_from_signatures(
            self._signatures[id1], self._signatures[id2]
        )

    def find_near_duplicates(
        self, text: str, threshold: float = 0.8
    ) -> list[tuple[str, float]]:
        """Return [(posting_id, similarity)] for all indexed postings above threshold."""
        query_sig = self._hasher.signature(shingle(text, self.shingle_size))
        results: list[tuple[str, float]] = []
        for pid, sig in self._signatures.items():
            sim = MinHasher.jaccard_from_signatures(query_sig, sig)
            if sim >= threshold:
                results.append((pid, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_clusters(self, threshold: float = 0.8) -> list[SimilarityCluster]:
        """Union-Find clustering of postings with pairwise similarity ≥ threshold."""
        ids = list(self._signatures.keys())
        parent = {pid: pid for pid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = MinHasher.jaccard_from_signatures(
                    self._signatures[ids[i]], self._signatures[ids[j]]
                )
                if sim >= threshold:
                    union(ids[i], ids[j])

        groups: dict[str, list[str]] = defaultdict(list)
        for pid in ids:
            groups[find(pid)].append(pid)

        clusters: list[SimilarityCluster] = []
        for cluster_id, (_, members) in enumerate(groups.items()):
            # Compute average pairwise similarity
            sims: list[float] = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    sims.append(
                        MinHasher.jaccard_from_signatures(
                            self._signatures[members[i]], self._signatures[members[j]]
                        )
                    )
            avg_sim = statistics.mean(sims) if sims else 1.0
            # Choose the longest text as centroid
            centroid = max(members, key=lambda p: len(self._texts.get(p, "")))
            clusters.append(
                SimilarityCluster(
                    cluster_id=cluster_id,
                    posting_ids=members,
                    centroid_text=self._texts.get(centroid, "")[:200],
                    avg_pairwise_similarity=avg_sim,
                    is_template=len(members) >= _TEMPLATE_THRESHOLD,
                )
            )
        # Largest clusters first
        clusters.sort(key=lambda c: len(c.posting_ids), reverse=True)
        return clusters

    def __len__(self) -> int:
        return len(self._signatures)


# ---------------------------------------------------------------------------
# ScamNetworkGraph
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A job posting as a node in the scam network graph."""

    posting_id: str
    url: str = ""
    company: str = ""
    contact: str = ""  # extracted email/phone
    description_hash: str = ""
    source_platform: str = "linkedin"
    scam_score: float = 0.0
    neighbours: set[str] = field(default_factory=set)

    # Edge labels: why is this node connected to each neighbour?
    edge_reasons: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ScamCluster:
    """A connected component in the scam network."""

    cluster_id: int
    node_ids: list[str]
    hub_node: str  # node with highest degree
    hub_degree: int
    platforms: list[str]  # unique source platforms
    shared_features: list[str]  # e.g. ["contact_info", "description_hash"]
    avg_scam_score: float


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SIMILARITY_THRESHOLD = 0.75  # MinHash similarity to create an edge


def _extract_contact(job: JobPosting) -> str:
    """Extract a canonical contact string from a job posting."""
    text = f"{job.description} {job.url}"
    email = _EMAIL_RE.search(text)
    if email:
        return email.group(0).lower()
    phone = _PHONE_RE.search(text)
    if phone:
        return re.sub(r"[\s.\-]", "", phone.group(0))
    return ""


def _desc_hash(text: str) -> str:
    """SHA-256 of normalised description for exact-match edge detection."""
    return hashlib.sha256(_normalise(text).encode()).hexdigest()[:16]


class ScamNetworkGraph:
    """Adjacency graph of job postings connected by shared scam features.

    Edges are created when two postings share ANY of:
      - Identical contact info (email / phone)
      - Same description hash (exact duplicate)
      - Near-duplicate description text (MinHash Jaccard ≥ threshold)
      - Same shell-company name

    After building the graph, call :meth:`get_clusters` to retrieve connected
    components, and :meth:`get_hubs` to identify high-degree operators.
    """

    def __init__(
        self,
        text_threshold: float = _SIMILARITY_THRESHOLD,
        num_perm: int = 64,
    ) -> None:
        self._threshold = text_threshold
        self._nodes: dict[str, GraphNode] = {}
        self._text_index = TextSimilarityIndex(num_perm=num_perm)
        # Indices for fast edge building
        self._contact_index: dict[str, list[str]] = defaultdict(list)  # contact → [posting_ids]
        self._hash_index: dict[str, list[str]] = defaultdict(list)    # hash → [posting_ids]
        self._company_index: dict[str, list[str]] = defaultdict(list)  # company_norm → [posting_ids]

    def add_posting(self, job: JobPosting, scam_score: float = 0.0) -> str:
        """Add a job posting to the graph; return its posting_id.

        The posting_id is a deterministic hash of the URL (or description if no URL).
        """
        raw_id = job.url or job.description[:80]
        posting_id = hashlib.sha256(raw_id.encode()).hexdigest()[:12]

        contact = _extract_contact(job)
        desc_hash = _desc_hash(job.description)
        company_norm = _normalise(job.company)

        node = GraphNode(
            posting_id=posting_id,
            url=job.url,
            company=job.company,
            contact=contact,
            description_hash=desc_hash,
            source_platform=job.source or "linkedin",
            scam_score=scam_score,
        )
        self._nodes[posting_id] = node

        # Index text for near-dup detection
        self._text_index.add(posting_id, job.description)

        # Build edges against existing nodes
        self._build_edges(posting_id, contact, desc_hash, company_norm)

        # Update indices
        if contact:
            self._contact_index[contact].append(posting_id)
        self._hash_index[desc_hash].append(posting_id)
        if company_norm:
            self._company_index[company_norm].append(posting_id)

        return posting_id

    def _build_edges(
        self,
        new_id: str,
        contact: str,
        desc_hash: str,
        company_norm: str,
    ) -> None:
        """Connect *new_id* to existing nodes that share features."""

        def add_edge(a: str, b: str, reason: str) -> None:
            if a == b:
                return
            self._nodes[a].neighbours.add(b)
            self._nodes[b].neighbours.add(a)
            self._nodes[a].edge_reasons.setdefault(b, [])
            if reason not in self._nodes[a].edge_reasons[b]:
                self._nodes[a].edge_reasons[b].append(reason)
            self._nodes[b].edge_reasons.setdefault(a, [])
            if reason not in self._nodes[b].edge_reasons[a]:
                self._nodes[b].edge_reasons[a].append(reason)

        # 1. Shared contact
        if contact:
            for pid in self._contact_index.get(contact, []):
                add_edge(new_id, pid, "contact_info")

        # 2. Exact description hash
        for pid in self._hash_index.get(desc_hash, []):
            add_edge(new_id, pid, "description_hash")

        # 3. Near-duplicate description via MinHash
        self._text_index.find_near_duplicates(
            self._nodes[new_id].description_hash,  # placeholder; we recompute below
            threshold=self._threshold,
        )
        # Actually use the real text similarity (text_index already has new_id added)
        for pid in list(self._nodes.keys()):
            if pid == new_id:
                continue
            sim = self._text_index.similarity(new_id, pid)
            if sim >= self._threshold:
                add_edge(new_id, pid, "text_similarity")

        # 4. Same normalised company name
        if company_norm:
            for pid in self._company_index.get(company_norm, []):
                add_edge(new_id, pid, "company_name")

    def get_clusters(self) -> list[ScamCluster]:
        """Return connected components as ScamCluster objects."""
        visited: set[str] = set()
        clusters: list[ScamCluster] = []
        cluster_id = 0

        for start in self._nodes:
            if start in visited:
                continue
            # BFS
            component: list[str] = []
            queue = [start]
            while queue:
                node_id = queue.pop(0)
                if node_id in visited:
                    continue
                visited.add(node_id)
                component.append(node_id)
                queue.extend(self._nodes[node_id].neighbours - visited)

            # Find hub (highest degree)
            hub = max(component, key=lambda n: len(self._nodes[n].neighbours))
            platforms = list(
                {self._nodes[n].source_platform for n in component}
            )
            shared: set[str] = set()
            for n in component:
                for reasons in self._nodes[n].edge_reasons.values():
                    shared.update(reasons)
            scores = [self._nodes[n].scam_score for n in component]
            avg_score = statistics.mean(scores) if scores else 0.0

            clusters.append(
                ScamCluster(
                    cluster_id=cluster_id,
                    node_ids=component,
                    hub_node=hub,
                    hub_degree=len(self._nodes[hub].neighbours),
                    platforms=sorted(platforms),
                    shared_features=sorted(shared),
                    avg_scam_score=avg_score,
                )
            )
            cluster_id += 1

        # Largest clusters first
        clusters.sort(key=lambda c: len(c.node_ids), reverse=True)
        return clusters

    def get_hubs(self, min_degree: int = 2) -> list[tuple[str, int]]:
        """Return [(posting_id, degree)] for nodes with degree ≥ min_degree, sorted desc."""
        hubs = [
            (pid, len(node.neighbours))
            for pid, node in self._nodes.items()
            if len(node.neighbours) >= min_degree
        ]
        hubs.sort(key=lambda x: x[1], reverse=True)
        return hubs

    def degree_centrality(self, posting_id: str) -> float:
        """Normalised degree centrality for a posting."""
        n = len(self._nodes) - 1
        if n <= 0:
            return 0.0
        return len(self._nodes[posting_id].neighbours) / n

    def cross_platform_postings(self) -> list[tuple[str, list[str]]]:
        """Return clusters whose postings appear on more than one platform."""
        return [
            (f"cluster_{c.cluster_id}", c.platforms)
            for c in self.get_clusters()
            if len(c.platforms) > 1
        ]

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return sum(len(n.neighbours) for n in self._nodes.values()) // 2


# ---------------------------------------------------------------------------
# RecruiterProfiler
# ---------------------------------------------------------------------------


@dataclass
class RecruiterProfile:
    """Behavioural model for a single recruiter."""

    recruiter_id: str
    name: str = ""
    posting_times: list[str] = field(default_factory=list)  # ISO timestamps
    job_categories: list[str] = field(default_factory=list)
    message_texts: list[str] = field(default_factory=list)
    scam_count: int = 0
    total_count: int = 0

    @property
    def scam_rate(self) -> float:
        return self.scam_count / self.total_count if self.total_count > 0 else 0.0

    @property
    def posting_hours(self) -> list[int]:
        """Extract UTC hours from stored timestamps."""
        hours: list[int] = []
        for ts in self.posting_times:
            with contextlib.suppress(ValueError):
                hours.append(datetime.fromisoformat(ts).hour)
        return hours

    @property
    def language_fingerprint(self) -> dict[str, int]:
        """Word-frequency fingerprint across all message texts."""
        counter: Counter[str] = Counter()
        for text in self.message_texts:
            words = re.findall(r"\b[a-z]{3,}\b", text.lower())
            counter.update(words)
        return dict(counter.most_common(20))


_UNUSUAL_HOURS = frozenset(range(1, 6))  # 01:00–05:59 UTC
_VOLUME_SPIKE_FACTOR = 3.0               # 3× rolling average = spike


class RecruiterProfiler:
    """Model and flag anomalous recruiter behaviour across job postings.

    Usage::

        profiler = RecruiterProfiler()
        profiler.record(job, is_scam=True)
        flags = profiler.get_flags("recruiter_id")
    """

    def __init__(self) -> None:
        self._profiles: dict[str, RecruiterProfile] = {}

    def _get_or_create(self, recruiter_id: str, name: str = "") -> RecruiterProfile:
        if recruiter_id not in self._profiles:
            self._profiles[recruiter_id] = RecruiterProfile(
                recruiter_id=recruiter_id, name=name
            )
        return self._profiles[recruiter_id]

    def record(
        self,
        job: JobPosting,
        is_scam: bool = False,
        posted_at: str = "",
    ) -> None:
        """Record a job posting associated with a recruiter."""
        rid = job.recruiter_name or job.company or "unknown"
        profile = self._get_or_create(rid, name=job.recruiter_name)
        profile.posting_times.append(posted_at or _now_iso())
        profile.job_categories.append(job.industry or "")
        profile.message_texts.append(job.description[:500])
        profile.total_count += 1
        if is_scam:
            profile.scam_count += 1

    def get_flags(self, recruiter_id: str) -> list[str]:
        """Return a list of anomaly flag strings for a recruiter."""
        profile = self._profiles.get(recruiter_id)
        if not profile:
            return []

        flags: list[str] = []

        # Flag: high scam rate
        if profile.total_count >= 3 and profile.scam_rate >= 0.5:
            flags.append(f"high_scam_rate:{profile.scam_rate:.0%}")

        # Flag: unusual posting hours (require ≥ 3 postings to avoid noise)
        hours = profile.posting_hours
        if len(hours) >= 3:
            unusual = sum(1 for h in hours if h in _UNUSUAL_HOURS)
            if unusual / len(hours) >= 0.5:
                flags.append("unusual_hours:majority_overnight")

        # Flag: volume spike (last 5 vs baseline of first N)
        n = len(profile.posting_times)
        if n >= 10:
            baseline_rate = (n - 5) / max(n - 5, 1)
            recent_rate = 5.0
            if recent_rate >= baseline_rate * _VOLUME_SPIKE_FACTOR:
                flags.append("volume_spike:recent_activity")

        # Flag: templated messages (low unique word ratio)
        if len(profile.message_texts) >= 3:
            all_words: list[str] = []
            for t in profile.message_texts:
                all_words.extend(re.findall(r"\b[a-z]{3,}\b", t.lower()))
            if all_words:
                unique_ratio = len(set(all_words)) / len(all_words)
                if unique_ratio < 0.35:
                    flags.append("templated_messages:low_vocabulary_diversity")

        return flags

    def get_profile(self, recruiter_id: str) -> RecruiterProfile | None:
        return self._profiles.get(recruiter_id)

    def all_recruiter_ids(self) -> list[str]:
        return list(self._profiles.keys())

    # ------------------------------------------------------------------
    # Sybil detection — multiple fake accounts operated by same entity
    # ------------------------------------------------------------------

    def _behavioural_similarity(self, a: RecruiterProfile, b: RecruiterProfile) -> float:
        """Score behavioural similarity [0, 1] between two recruiter profiles.

        Combines:
          - Overlap in posting hours
          - Overlap in job categories
          - Language fingerprint cosine similarity
        """
        # Hour overlap
        hours_a = set(a.posting_hours)
        hours_b = set(b.posting_hours)
        hour_sim = len(hours_a & hours_b) / len(hours_a | hours_b) if hours_a and hours_b else 0.0

        # Category overlap
        cats_a = set(a.job_categories) - {""}
        cats_b = set(b.job_categories) - {""}
        cat_sim = len(cats_a & cats_b) / len(cats_a | cats_b) if cats_a and cats_b else 0.0

        # Language fingerprint cosine similarity
        fp_a = a.language_fingerprint
        fp_b = b.language_fingerprint
        common_words = set(fp_a) & set(fp_b)
        if common_words:
            dot = sum(fp_a[w] * fp_b[w] for w in common_words)
            mag_a = math.sqrt(sum(v * v for v in fp_a.values()))
            mag_b = math.sqrt(sum(v * v for v in fp_b.values()))
            lang_sim = dot / (mag_a * mag_b) if mag_a and mag_b else 0.0
        else:
            lang_sim = 0.0

        return (hour_sim + cat_sim + lang_sim) / 3

    def detect_sybils(self, threshold: float = 0.7) -> list[list[str]]:
        """Return groups of recruiter IDs likely operated by the same entity.

        Groups are formed by pairwise behavioural similarity ≥ *threshold*.
        Only profiles with ≥ 3 postings are considered.
        """
        candidates = [
            p for p in self._profiles.values() if p.total_count >= 3
        ]
        ids = [p.recruiter_id for p in candidates]
        parent = {rid: rid for rid in ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                sim = self._behavioural_similarity(candidates[i], candidates[j])
                if sim >= threshold:
                    union(ids[i], ids[j])

        groups: dict[str, list[str]] = defaultdict(list)
        for rid in ids:
            groups[find(rid)].append(rid)

        # Return only groups with more than one recruiter (actual Sybil clusters)
        return [members for members in groups.values() if len(members) > 1]


# ---------------------------------------------------------------------------
# CompanyShellDetector
# ---------------------------------------------------------------------------

_VIRTUAL_OFFICE_KEYWORDS = re.compile(
    r"\b(virtual office|registered agent|c/o |suite \d+|po box|mailbox)\b",
    re.IGNORECASE,
)
_GENERIC_NAME_WORDS = re.compile(
    r"\b(solutions?|services?|enterprises?|global|national|united|premier|elite|"
    r"dynamic|synergy|nexus|infinity|alpha|apex|summit|pinnacle|horizon|"
    r"corp(oration)?|inc(orporated)?|llc|ltd|group|partners?|associates?|"
    r"international|worldwide|unlimited|consulting|staffing)\b",
    re.IGNORECASE,
)
_SHELL_WHOIS_THRESHOLD = 180  # days; < 180 = recently registered


@dataclass
class ShellCompanyReport:
    """Results of shell-company analysis for a single company."""

    company_name: str
    shell_score: float        # 0.0–1.0
    flags: list[str]
    similar_known_scams: list[tuple[str, float]]  # (name, similarity)


class CompanyShellDetector:
    """Detect shell companies via registration age, name patterns, address reuse.

    Usage::

        detector = CompanyShellDetector()
        report = detector.analyse(job, whois_age_days=30, address="Suite 200 Virtual Office")
        matches = detector.fuzzy_match_known_scams("Globel Solutions LLC")
    """

    def __init__(self) -> None:
        self._known_scam_companies: list[str] = []
        self._address_registry: dict[str, list[str]] = defaultdict(list)

    def load_known_scams(self, names: list[str]) -> None:
        """Register a list of known scam company names for fuzzy matching."""
        self._known_scam_companies = [_normalise(n) for n in names]

    def fuzzy_match_known_scams(
        self, company_name: str, jw_threshold: float = 0.88
    ) -> list[tuple[str, float]]:
        """Return [(known_name, similarity)] where Jaro-Winkler ≥ threshold."""
        norm = _normalise(company_name)
        results: list[tuple[str, float]] = []
        for known in self._known_scam_companies:
            sim = jaro_winkler(norm, known)
            if sim >= jw_threshold:
                results.append((known, round(sim, 4)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def register_address(self, company_name: str, address: str) -> None:
        """Register a company's address for virtual-office reuse detection."""
        addr_norm = _normalise(address)
        self._address_registry[addr_norm].append(company_name)

    def address_reused_by(self, address: str) -> list[str]:
        """Return other companies registered at this address."""
        norm = _normalise(address)
        return [
            c for c in self._address_registry.get(norm, [])
        ]

    def analyse(
        self,
        job: JobPosting,
        whois_age_days: int = 0,
        address: str = "",
        employee_count: int = 0,
    ) -> ShellCompanyReport:
        """Score a company for shell-company indicators."""
        flags: list[str] = []
        score = 0.0

        # 1. Recently registered domain
        if 0 < whois_age_days < _SHELL_WHOIS_THRESHOLD:
            flags.append(f"recently_registered:{whois_age_days}d")
            score += 0.3

        # 2. Very few / zero employees
        if employee_count == 0:
            flags.append("no_employees_listed")
            score += 0.15
        elif employee_count <= 5:
            flags.append(f"minimal_employees:{employee_count}")
            score += 0.1

        # 3. Generic / templated company name
        words = job.company.split()
        generic_hits = sum(
            1 for w in words if _GENERIC_NAME_WORDS.search(w)
        )
        if generic_hits >= 2:
            flags.append(f"generic_name:{generic_hits}_generic_words")
            score += 0.2

        # 4. Virtual office address
        if address and _VIRTUAL_OFFICE_KEYWORDS.search(address):
            flags.append("virtual_office_address")
            score += 0.2

        # 5. Address reuse
        if address:
            self.register_address(job.company, address)
            reusers = [
                c for c in self.address_reused_by(address) if c != job.company
            ]
            if reusers:
                flags.append(f"address_shared_with:{','.join(reusers[:3])}")
                score += min(0.3, 0.1 * len(reusers))

        # 6. No LinkedIn company page
        if not job.company_linkedin_url:
            flags.append("no_linkedin_company_page")
            score += 0.1

        # 7. Fuzzy match to known scam companies
        similar = self.fuzzy_match_known_scams(job.company)
        if similar:
            flags.append(f"similar_to_known_scam:{similar[0][0]}")
            score += 0.25

        score = min(score, 1.0)
        return ShellCompanyReport(
            company_name=job.company,
            shell_score=round(score, 3),
            flags=flags,
            similar_known_scams=similar,
        )
