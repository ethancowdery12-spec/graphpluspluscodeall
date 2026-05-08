"""
Knowledge Graph Builder
NetworkX multigraph with PageRank, Louvain community detection,
entity resolution, and versioning.
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

# Entity type color palette (used by frontend)
ENTITY_COLORS = {
    "concept": "#00d4ff",
    "model": "#a855f7",
    "person": "#f59e0b",
    "organization": "#10b981",
    "dataset": "#f43f5e",
    "technique": "#6366f1",
    "system": "#ec4899",
    "paper": "#84cc16",
    "codebase": "#fb923c",
    "unknown": "#94a3b8",
}

RELATION_TYPES = [
    "uses", "extends", "based_on", "introduced_by", "enables", "requires",
    "part_of", "evaluates_on", "outperforms", "trained_on", "combines",
    "is_a", "related_to", "derives_from", "implemented_in", "benchmarks",
]


class GraphStore:
    """In-memory knowledge graph with export/import capabilities."""

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self._embeddings: Dict[str, List[float]] = {}
        self._entity_index: Dict[str, str] = {}  # normalized → node_id
        self._version = 0
        self._ingestion_log: List[dict] = []
        self._metrics = {
            "total_queries": 0,
            "total_ingestions": 0,
            "cache_hits": 0,
            "avg_query_ms": 0,
        }

    # ─── Node Management ───────────────────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "concept",
                   metadata: dict = None, embedding: List[float] = None) -> str:
        """Add or update an entity node. Returns node_id."""
        # Entity resolution: normalize name
        normalized = name.lower().strip()
        if normalized in self._entity_index:
            node_id = self._entity_index[normalized]
            # Update metadata if provided
            if metadata:
                self.G.nodes[node_id].update(metadata)
            return node_id

        node_id = hashlib.md5(normalized.encode()).hexdigest()[:12]
        self._entity_index[normalized] = node_id

        self.G.add_node(node_id, **{
            "id": node_id,
            "label": name,
            "type": entity_type.lower(),
            "color": ENTITY_COLORS.get(entity_type.lower(), ENTITY_COLORS["unknown"]),
            "created_at": time.time(),
            "version": self._version,
            **(metadata or {}),
        })

        if embedding:
            self._embeddings[node_id] = embedding

        return node_id

    def add_relation(self, subject_id: str, predicate: str, object_id: str,
                     confidence: float = 1.0, source: str = "extraction") -> str:
        """Add a directed relation edge."""
        if not self.G.has_node(subject_id) or not self.G.has_node(object_id):
            return None

        edge_id = f"{subject_id}_{predicate}_{object_id}"
        self.G.add_edge(subject_id, object_id, **{
            "id": edge_id,
            "predicate": predicate,
            "confidence": confidence,
            "weight": confidence,
            "source": source,
            "created_at": time.time(),
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
            obj = triple.get("object", "").strip()
            conf = float(triple.get("confidence", 0.85))

            if not subj or not obj:
                continue

            # Infer entity types
            s_type = _infer_entity_type(subj)
            o_type = _infer_entity_type(obj)

            s_id = self.add_entity(subj, s_type)
            o_id = self.add_entity(obj, o_type)

            new_node = s_id not in self._entity_index or o_id not in self._entity_index
            if new_node:
                nodes_added += 1

            eid = self.add_relation(s_id, pred, o_id, confidence=conf, source=source)
            if eid:
                edges_added += 1

        self._metrics["total_ingestions"] += 1
        self._ingestion_log.append({
            "timestamp": time.time(),
            "source": source,
            "triples": len(triples),
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "version": self._version,
        })

        return {"nodes_added": nodes_added, "edges_added": edges_added, "version": self._version}

    # ─── Graph Analytics ───────────────────────────────────────────────────────

    def compute_pagerank(self) -> Dict[str, float]:
        """Compute PageRank for all nodes."""
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
        """Detect communities using Louvain (or greedy fallback)."""
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

    # ─── Traversal ─────────────────────────────────────────────────────────────

    def multi_hop_paths(self, start_ids: List[str], max_hops: int = 3,
                        top_k: int = 5) -> List[List[dict]]:
        """BFS multi-hop traversal from start nodes, returns top-k paths."""
        all_paths = []
        visited_edges = set()

        for start_id in start_ids:
            if not self.G.has_node(start_id):
                continue
            try:
                paths = list(nx.all_simple_paths(
                    self.G.to_undirected(), start_id,
                    cutoff=max_hops
                ))[:20]
            except Exception:
                paths = []

            for path in paths:
                path_data = []
                path_score = 0
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edges = self.G.get_edge_data(u, v) or {}
                    edge = list(edges.values())[0] if edges else {"predicate": "related_to", "confidence": 0.5}
                    path_data.append({
                        "from": {"id": u, **self.G.nodes.get(u, {})},
                        "edge": edge,
                        "to": {"id": v, **self.G.nodes.get(v, {})},
                    })
                    path_score += edge.get("confidence", 0.5)

                if path_data:
                    avg_score = path_score / len(path_data)
                    all_paths.append((avg_score, path_data))

        # Sort by score, deduplicate, return top-k
        all_paths.sort(key=lambda x: -x[0])
        return [p for _, p in all_paths[:top_k]]

    def semantic_neighbors(self, node_id: str, k: int = 5) -> List[str]:
        """Get semantically similar nodes using embedding cosine similarity."""
        if node_id not in self._embeddings:
            # Fallback: return graph neighbors
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
        """Find node IDs by label (partial match)."""
        name_lower = name.lower()
        results = []
        for node_id, data in self.G.nodes(data=True):
            label = data.get("label", "").lower()
            if name_lower in label:
                results.append(node_id)
        return results

    # ─── Serialization ─────────────────────────────────────────────────────────

    def to_json(self) -> dict:
        """Serialize graph for frontend D3.js consumption."""
        self.compute_pagerank()
        self.compute_communities()

        nodes = []
        for node_id, data in self.G.nodes(data=True):
            degree = self.G.degree(node_id)
            nodes.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "type": data.get("type", "unknown"),
                "color": data.get("color", ENTITY_COLORS["unknown"]),
                "pagerank": round(data.get("pagerank", 0), 6),
                "community": data.get("community", 0),
                "degree": degree,
                "size": max(8, min(30, 8 + math.log1p(degree) * 5)),
            })

        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "predicate": data.get("predicate", "related_to"),
                "confidence": round(data.get("confidence", 1.0), 3),
                "weight": data.get("weight", 1.0),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": self.get_stats(),
        }

    def get_stats(self) -> dict:
        """Return graph topology stats."""
        n = len(self.G.nodes())
        e = len(self.G.edges())
        return {
            "node_count": n,
            "edge_count": e,
            "density": round(nx.density(self.G), 6) if n > 1 else 0,
            "avg_degree": round(sum(d for _, d in self.G.degree()) / max(n, 1), 2),
            "version": self._version,
            "ingestion_count": self._metrics["total_ingestions"],
            "query_count": self._metrics["total_queries"],
            "entity_types": _count_types(self.G),
        }

    def set_embedding(self, node_id: str, embedding: List[float]):
        self._embeddings[node_id] = embedding

    def increment_query_count(self):
        self._metrics["total_queries"] += 1


# ─── Module-level singleton ─────────────────────────────────────────────────

_graph_store: Optional[GraphStore] = None


def get_graph() -> GraphStore:
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store


# ─── Helpers ────────────────────────────────────────────────────────────────

def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-8)


def _infer_entity_type(name: str) -> str:
    name_lower = name.lower()
    model_keywords = ["gpt", "bert", "llm", "transformer", "llama", "mistral",
                      "qwen", "claude", "gemini", "palm", "t5", "roberta", "xlnet"]
    person_keywords = ["et al", "vaswani", "devlin", "lecun", "hinton", "bengio",
                       "goodfellow", "attention", "team", "lab", "group"]
    org_keywords = ["openai", "google", "meta", "microsoft", "deepmind", "hugging",
                    "anthropic", "university", "institute", "lab"]
    dataset_keywords = ["dataset", "corpus", "benchmark", "squad", "glue", "imagenet",
                        "wikipedia", "commoncrawl"]
    code_keywords = ["library", "framework", "pytorch", "tensorflow", "jax", "numpy",
                     "scikit", "langchain", "llamaindex", "codebase", "repository"]
    technique_keywords = ["attention", "fine-tuning", "rlhf", "rag", "lora", "peft",
                          "quantization", "distillation", "training", "inference"]
    paper_keywords = ["paper", "arxiv", "publication", "survey", "study", "analysis"]

    for kw in model_keywords:
        if kw in name_lower:
            return "model"
    for kw in person_keywords:
        if kw in name_lower:
            return "person"
    for kw in org_keywords:
        if kw in name_lower:
            return "organization"
    for kw in dataset_keywords:
        if kw in name_lower:
            return "dataset"
    for kw in code_keywords:
        if kw in name_lower:
            return "codebase"
    for kw in technique_keywords:
        if kw in name_lower:
            return "technique"
    for kw in paper_keywords:
        if kw in name_lower:
            return "paper"
    return "concept"


def _count_types(G: nx.MultiDiGraph) -> dict:
    counts = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts
