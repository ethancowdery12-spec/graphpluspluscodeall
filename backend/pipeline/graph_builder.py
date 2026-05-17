"""
Knowledge Graph Builder
NetworkX multigraph with PageRank, Louvain community detection,
entity resolution, versioning, graph persistence, and confidence tiers.
"""
import time
import math
import uuid
import hashlib
from typing import Optional, List, Dict, Any
import networkx as nx

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

# ─── Entity / Confidence Colours ─────────────────────────────────────────────

ENTITY_COLORS = {
    # Domain / knowledge entities
    "concept":      "#00d4ff",
    "model":        "#a855f7",
    "person":       "#f59e0b",
    "organization": "#10b981",
    "dataset":      "#f43f5e",
    "technique":    "#6366f1",
    "system":       "#ec4899",
    "paper":        "#84cc16",
    "codebase":     "#fb923c",
    # Code structure entities (new)
    "file":         "#94a3b8",
    "function":     "#34d399",
    "class":        "#f97316",
    "module":       "#818cf8",
    "interface":    "#67e8f9",
    "struct":       "#fbbf24",
    "trait":        "#c084fc",
    # Fallback
    "unknown":      "#4a566d",
}

# Confidence tier colours (used in frontend edge rendering)
TIER_COLORS = {
    "EXTRACTED": "#34d399",   # emerald — AST-parsed facts, deterministic
    "INFERRED":  "#60a5fa",   # blue    — LLM extracted, high-confidence
    "AMBIGUOUS": "#f59e0b",   # amber   — regex / low-confidence LLM
}

RELATION_TYPES = [
    "uses", "extends", "based_on", "introduced_by", "enables", "requires",
    "part_of", "evaluates_on", "outperforms", "trained_on", "combines",
    "is_a", "related_to", "derives_from", "implemented_in", "benchmarks",
    # Code-specific
    "defined_in", "imports", "calls", "inherits_from", "belongs_to",
    "nested_in", "decorated_with", "extends", "implements",
]


class GraphStore:
    """In-memory knowledge graph with persistence, confidence tiers, and export."""

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self._embeddings: Dict[str, List[float]] = {}
        self._entity_index: Dict[str, str] = {}   # normalized_name → node_id
        self._version = 0
        self._ingestion_log: List[dict] = []
        self._file_hashes: Dict[str, str] = {}    # sha256 → filepath
        self._metrics = {
            "total_queries":    0,
            "total_ingestions": 0,
            "cache_hits":       0,
            "avg_query_ms":     0,
        }

    # ─── Node Management ──────────────────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "concept",
                   metadata: dict = None, embedding: List[float] = None) -> str:
        """Add or update an entity node. Returns node_id."""
        normalized = name.lower().strip()
        if normalized in self._entity_index:
            node_id = self._entity_index[normalized]
            if metadata:
                self.G.nodes[node_id].update(metadata)
            return node_id

        node_id = hashlib.md5(normalized.encode()).hexdigest()[:12]
        self._entity_index[normalized] = node_id

        self.G.add_node(node_id, **{
            "id":         node_id,
            "label":      name,
            "type":       entity_type.lower(),
            "color":      ENTITY_COLORS.get(entity_type.lower(), ENTITY_COLORS["unknown"]),
            "created_at": time.time(),
            "version":    self._version,
            **(metadata or {}),
        })

        if embedding:
            self._embeddings[node_id] = embedding
        return node_id

    def add_relation(self, subject_id: str, predicate: str, object_id: str,
                     confidence: float = 1.0, source: str = "extraction",
                     confidence_tier: str = "INFERRED") -> str:
        """Add a directed relation edge with confidence tier."""
        if not self.G.has_node(subject_id) or not self.G.has_node(object_id):
            return None

        edge_id = f"{subject_id}_{predicate}_{object_id}"
        self.G.add_edge(subject_id, object_id, **{
            "id":               edge_id,
            "predicate":        predicate,
            "confidence":       confidence,
            "weight":           confidence,
            "confidence_tier":  confidence_tier,
            "source":           source,
            "created_at":       time.time(),
        })
        return edge_id

    def ingest_triples(self, triples: List[dict], source: str = "document") -> dict:
        """Batch ingest triples into the graph."""
        nodes_added = 0
        edges_added = 0
        self._version += 1

        for triple in triples:
            subj = triple.get("subject", "").strip()
            pred = triple.get("predicate", "related_to").strip()
            obj  = triple.get("object",  "").strip()
            conf = float(triple.get("confidence", 0.85))
            tier = triple.get("confidence_tier", "INFERRED")

            if not subj or not obj:
                continue

            s_type = _infer_entity_type(subj)
            o_type = _infer_entity_type(obj)

            prev_node_count = len(self._entity_index)
            s_id = self.add_entity(subj, s_type)
            o_id = self.add_entity(obj,  o_type)
            if len(self._entity_index) > prev_node_count:
                nodes_added += 1

            eid = self.add_relation(s_id, pred, o_id,
                                    confidence=conf, source=source,
                                    confidence_tier=tier)
            if eid:
                edges_added += 1

        self._metrics["total_ingestions"] += 1
        self._ingestion_log.append({
            "timestamp":   time.time(),
            "source":      source,
            "triples":     len(triples),
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "version":     self._version,
        })
        return {"nodes_added": nodes_added, "edges_added": edges_added,
                "version": self._version}

    # ─── File Hash Tracking ───────────────────────────────────────────────────

    def register_file_hash(self, sha256: str, filepath: str) -> None:
        self._file_hashes[sha256] = filepath

    def has_file_hash(self, sha256: str) -> bool:
        return sha256 in self._file_hashes

    # ─── Graph Analytics ──────────────────────────────────────────────────────

    def compute_pagerank(self) -> Dict[str, float]:
        if len(self.G) == 0:
            return {}
        try:
            pr = nx.pagerank(self.G, weight="weight")
            for node_id, rank in pr.items():
                if self.G.has_node(node_id):
                    self.G.nodes[node_id]["pagerank"] = rank
            return pr
        except Exception:
            return {}

    def compute_communities(self) -> Dict[str, int]:
        if len(self.G) == 0:
            return {}
        undirected = self.G.to_undirected()
        try:
            if HAS_LOUVAIN:
                partition = community_louvain.best_partition(undirected)
            else:
                communities = nx.algorithms.community.greedy_modularity_communities(undirected)
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
        except Exception:
            partition = {n: 0 for n in self.G.nodes()}

        for node_id, comm_id in partition.items():
            if self.G.has_node(node_id):
                self.G.nodes[node_id]["community"] = comm_id
        return partition

    # ─── Traversal ────────────────────────────────────────────────────────────

    def multi_hop_paths(self, start_ids: List[str], max_hops: int = 3,
                        top_k: int = 10, max_visited_per_start: int = 30) -> List[List[dict]]:
        """
        BFS multi-hop traversal from start nodes.
        Returns top-k paths by average edge confidence.
        max_visited_per_start caps BFS expansion per seed to avoid blowing up on dense graphs.
        """
        from collections import deque

        all_paths: List[tuple] = []
        seen_edge_sets: set = set()

        for start_id in start_ids:
            if not self.G.has_node(start_id):
                continue

            queue: deque = deque([(start_id, [])])
            visited_in_bfs: set = {start_id}

            while queue:
                node, path = queue.popleft()
                if len(path) >= max_hops:
                    continue

                for neighbor in self.G.successors(node):
                    edges = self.G.get_edge_data(node, neighbor) or {}
                    edge  = list(edges.values())[0] if edges else {
                        "predicate": "related_to", "confidence": 0.5,
                        "confidence_tier": "AMBIGUOUS",
                    }
                    hop = {
                        "from": {"id": node,     **self.G.nodes.get(node,     {})},
                        "edge": edge,
                        "to":   {"id": neighbor, **self.G.nodes.get(neighbor, {})},
                    }
                    new_path = path + [hop]

                    edge_key = frozenset(
                        (h["from"]["id"], h["to"]["id"]) for h in new_path
                    )
                    if edge_key not in seen_edge_sets:
                        seen_edge_sets.add(edge_key)
                        conf = sum(
                            h["edge"].get("confidence", 0.5) for h in new_path
                        ) / len(new_path)
                        all_paths.append((conf, new_path))

                    if neighbor not in visited_in_bfs and len(visited_in_bfs) < max_visited_per_start:
                        visited_in_bfs.add(neighbor)
                        queue.append((neighbor, new_path))

        all_paths.sort(key=lambda x: -x[0])
        return [p for _, p in all_paths[:top_k]]

    def semantic_neighbors(self, node_id: str, k: int = 5) -> List[str]:
        if node_id not in self._embeddings:
            return list(self.G.neighbors(node_id))[:k]
        query_emb = self._embeddings[node_id]
        scores = []
        for nid, emb in self._embeddings.items():
            if nid == node_id:
                continue
            score = _cosine_sim(query_emb, emb)
            scores.append((score, nid))
        scores.sort(key=lambda x: -x[0])
        return [nid for _, nid in scores[:k]]

    def find_nodes_by_name(self, name: str) -> List[str]:
        name_lower = name.lower()
        results = []
        for node_id, data in self.G.nodes(data=True):
            label = data.get("label", "").lower()
            if name_lower in label:
                results.append(node_id)
        return results

    # ─── Serialization ────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        """Serialize graph for D3.js frontend consumption."""
        self.compute_pagerank()
        self.compute_communities()

        nodes = []
        for node_id, data in self.G.nodes(data=True):
            degree = self.G.degree(node_id)
            nodes.append({
                "id":        node_id,
                "label":     data.get("label", node_id),
                "type":      data.get("type", "unknown"),
                "color":     data.get("color", ENTITY_COLORS["unknown"]),
                "pagerank":  round(data.get("pagerank", 0), 6),
                "community": data.get("community", 0),
                "degree":    degree,
                "size":      max(12, min(44, 12 + math.log1p(degree) * 8)),
            })

        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append({
                "source":          u,
                "target":          v,
                "predicate":       data.get("predicate", "related_to"),
                "confidence":      round(data.get("confidence", 1.0), 3),
                "weight":          data.get("weight", 1.0),
                "confidence_tier": data.get("confidence_tier", "INFERRED"),
            })

        return {"nodes": nodes, "edges": edges, "stats": self.get_stats()}

    def get_stats(self) -> dict:
        n = len(self.G.nodes())
        e = len(self.G.edges())

        # Confidence tier breakdown across all edges
        tier_counts: Dict[str, int] = {}
        for _, _, data in self.G.edges(data=True):
            tier = data.get("confidence_tier", "INFERRED")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        return {
            "node_count":           n,
            "edge_count":           e,
            "density":              round(nx.density(self.G), 6) if n > 1 else 0,
            "avg_degree":           round(sum(d for _, d in self.G.degree()) / max(n, 1), 2),
            "version":              self._version,
            "ingestion_count":      self._metrics["total_ingestions"],
            "query_count":          self._metrics["total_queries"],
            "file_count":           len(self._file_hashes),
            "entity_types":         _count_types(self.G),
            "confidence_breakdown": tier_counts,
        }

    def set_embedding(self, node_id: str, embedding: List[float]):
        self._embeddings[node_id] = embedding

    def increment_query_count(self):
        self._metrics["total_queries"] += 1


# ─── Module-level singleton ──────────────────────────────────────────────────

_graph_store: Optional[GraphStore] = None


def get_graph() -> GraphStore:
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
        # Try to restore persisted state before anything else
        try:
            from .persistence import load_graph
            load_graph(_graph_store)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[GraphStore] Persistence load skipped: {e}")
    return _graph_store


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-8)


def _infer_entity_type(name: str) -> str:
    name_lower = name.lower()

    # Code structure types (check first — more specific)
    if name_lower.endswith((".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".rb")):
        return "file"

    code_fn_hints = [".", "/"]   # module-qualified or path-like
    if any(h in name for h in code_fn_hints) and not name.startswith("http"):
        parts = name.split(".")
        last = parts[-1] if parts else ""
        if len(parts) >= 2 and last and last[0].islower() and last[0].isalpha():
            return "function"
        if len(parts) >= 2 and last and last[0].isupper():
            return "class"

    model_keywords = ["gpt", "bert", "llm", "transformer", "llama", "mistral",
                      "qwen", "claude", "gemini", "palm", "t5", "roberta", "xlnet"]
    person_keywords = ["et al", "vaswani", "devlin", "lecun", "hinton", "bengio",
                       "goodfellow", "team", "lab", "group"]
    org_keywords    = ["openai", "google", "meta", "microsoft", "deepmind", "hugging",
                       "anthropic", "university", "institute", "lab"]
    dataset_keywords= ["dataset", "corpus", "benchmark", "squad", "glue", "imagenet",
                       "wikipedia", "commoncrawl"]
    code_keywords   = ["library", "framework", "pytorch", "tensorflow", "jax", "numpy",
                       "scikit", "langchain", "llamaindex", "codebase", "repository"]
    technique_kw    = ["attention", "fine-tuning", "rlhf", "rag", "lora", "peft",
                       "quantization", "distillation", "training", "inference"]
    paper_kw        = ["paper", "arxiv", "publication", "survey", "study", "analysis"]

    for kw in model_keywords:
        if kw in name_lower: return "model"
    for kw in person_keywords:
        if kw in name_lower: return "person"
    for kw in org_keywords:
        if kw in name_lower: return "organization"
    for kw in dataset_keywords:
        if kw in name_lower: return "dataset"
    for kw in code_keywords:
        if kw in name_lower: return "codebase"
    for kw in technique_kw:
        if kw in name_lower: return "technique"
    for kw in paper_kw:
        if kw in name_lower: return "paper"
    return "concept"


def _count_types(G: nx.MultiDiGraph) -> dict:
    counts: dict = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts
